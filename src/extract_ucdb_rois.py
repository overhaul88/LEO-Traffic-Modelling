from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from rasterio.mask import mask
from shapely.geometry import mapping

DEFAULT_BASELINE_PATH = Path("data/processed/baseline_mollweide_1km_2022.tif")
DEFAULT_UCDB_PATH = Path("data/raw/ucdb/GHS_UCDB_GLOBE_R2024A.gpkg")
DEFAULT_LAYER = "GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A"
DEFAULT_SELECTION_PATH = Path("data/raw/roi_select.txt")
DEFAULT_OUTPUT_DIR = Path("data/outputs/roi")
DEFAULT_BUFFER_METERS = 10_000


@dataclass
class RequestedCity:
    order: int
    requested_name: str


@dataclass
class MatchRecord:
    order: int
    requested_name: str
    matched_name: str
    ucdb_id: int
    country: str
    area_km2: int
    population_total: float
    geometry_bounds: tuple[float, float, float, float]


def _path_arg(value: str) -> Path:
    return Path(value).expanduser().resolve()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "roi"


def clean_text(value: Any) -> Any:
    if isinstance(value, str):
        return value.lstrip("\ufeff").strip()
    return value


def clean_ucdb_dataframe(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cleaned = gdf.copy()
    cleaned.columns = [col.lstrip("\ufeff") for col in cleaned.columns]
    object_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    for col in object_columns:
        cleaned[col] = cleaned[col].map(clean_text)
    return cleaned


def load_requested_cities(selection_path: Path) -> list[RequestedCity]:
    requested = []
    seen: set[str] = set()

    for raw_line in selection_path.read_text().splitlines():
        name = raw_line.strip()
        if not name:
            continue
        if name in seen:
            raise ValueError(f"Duplicate city selection '{name}'.")
        seen.add(name)
        requested.append(
            RequestedCity(
                order=len(requested) + 1,
                requested_name=name,
            )
        )

    if not requested:
        raise ValueError(f"No city names found in selection file: {selection_path}")
    return requested


def load_ucdb_layer(ucdb_path: Path, layer: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(ucdb_path, layer=layer, engine="pyogrio")
    cleaned = clean_ucdb_dataframe(gdf)
    required = {"ID_UC_G0", "GC_UCN_MAI_2025", "GC_CNT_UNN_2025", "GC_UCA_KM2_2025", "GC_POP_TOT_2025", "geometry"}
    missing = required - set(cleaned.columns)
    if missing:
        raise ValueError(f"UCDB layer is missing required columns: {sorted(missing)}")
    return cleaned


def resolve_city_matches(
    ucdb_gdf: gpd.GeoDataFrame,
    requested_cities: list[RequestedCity],
) -> gpd.GeoDataFrame:
    rows: list[dict[str, Any]] = []
    unresolved: list[str] = []
    ambiguous: list[str] = []

    for city in requested_cities:
        matched = ucdb_gdf[ucdb_gdf["GC_UCN_MAI_2025"] == city.requested_name].copy()
        if len(matched) == 0:
            unresolved.append(city.requested_name)
            continue
        if len(matched) > 1:
            ambiguous.append(city.requested_name)
            continue

        row = matched.iloc[0]
        rows.append(
            {
                "order": city.order,
                "requested_name": city.requested_name,
                "matched_name": city.requested_name,
                "ucdb_id": int(row["ID_UC_G0"]),
                "country": str(row["GC_CNT_UNN_2025"]),
                "area_km2": int(row["GC_UCA_KM2_2025"]),
                "population_total": float(row["GC_POP_TOT_2025"]),
                "geometry": row.geometry,
            }
        )

    if unresolved or ambiguous:
        messages = []
        if unresolved:
            messages.append(f"Unresolved city names: {unresolved}")
        if ambiguous:
            messages.append(f"Ambiguous city names: {ambiguous}")
        raise ValueError("; ".join(messages))

    resolved = gpd.GeoDataFrame(rows, geometry="geometry", crs=ucdb_gdf.crs)
    resolved["geometry_bounds"] = resolved.geometry.bounds.apply(
        lambda row: (float(row["minx"]), float(row["miny"]), float(row["maxx"]), float(row["maxy"])),
        axis=1,
    )
    resolved = resolved.sort_values("order").reset_index(drop=True)
    return resolved


def inspect_inputs(
    baseline_path: Path,
    ucdb_path: Path,
    layer: str,
    selection_path: Path,
) -> dict[str, Any]:
    requested = load_requested_cities(selection_path)
    ucdb = load_ucdb_layer(ucdb_path, layer)
    matched = resolve_city_matches(ucdb, requested)

    with rasterio.open(baseline_path) as src:
        baseline = {
            "path": str(baseline_path),
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "resolution": [float(src.res[0]), float(src.res[1])],
            "bounds": [float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top)],
            "nodata": float(src.nodata) if src.nodata is not None else None,
            "dtype": src.dtypes[0],
        }

    records = [
        asdict(
            MatchRecord(
                order=int(row.order),
                requested_name=str(row.requested_name),
                matched_name=str(row.matched_name),
                ucdb_id=int(row.ucdb_id),
                country=str(row.country),
                area_km2=int(row.area_km2),
                population_total=float(row.population_total),
                geometry_bounds=tuple(float(v) for v in row.geometry_bounds),
            )
        )
        for row in matched.itertuples(index=False)
    ]

    return {
        "baseline": baseline,
        "ucdb": {"path": str(ucdb_path), "layer": layer, "crs": str(matched.crs), "feature_count": int(len(ucdb))},
        "selected_cities": records,
    }


def write_roi_outputs(
    baseline_path: Path,
    matched_rois: gpd.GeoDataFrame,
    output_dir: Path,
    buffer_meters: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    buffer_km = buffer_meters // 1000
    vector_records: list[dict[str, Any]] = []
    raster_records: list[dict[str, Any]] = []

    buffered = matched_rois.copy()
    buffered["geometry"] = buffered.geometry.buffer(buffer_meters)
    buffered["buffer_meters"] = buffer_meters
    buffered["buffer_km"] = buffer_km
    vector_output = output_dir / f"ucdb_rois_buffer{buffer_km}km.gpkg"
    buffered.to_file(vector_output, layer="ucdb_rois", driver="GPKG", engine="pyogrio")

    with rasterio.open(baseline_path) as src:
        if src.crs != CRS.from_user_input(buffered.crs):
            raise ValueError(f"CRS mismatch: baseline={src.crs}, ucdb={buffered.crs}")

        base_profile = src.profile.copy()
        nodata = src.nodata

        for row in buffered.itertuples(index=False):
            prefix = f"{int(row.order):02d}_{slugify(row.matched_name)}_ucdb_buffer{buffer_km}km"
            baseline_output = output_dir / f"{prefix}_baseline.tif"
            mask_output = output_dir / f"{prefix}_mask.tif"
            geometry = row.geometry
            geometries = [mapping(geometry)]

            clipped, transform = mask(
                src,
                geometries,
                crop=True,
                nodata=nodata,
                filled=True,
                all_touched=False,
            )
            clipped_band = clipped[0]
            mask_array = geometry_mask(
                geometries,
                out_shape=clipped_band.shape,
                transform=transform,
                invert=True,
                all_touched=False,
            ).astype(np.uint8)

            baseline_profile = base_profile.copy()
            baseline_profile.update(
                {
                    "height": clipped_band.shape[0],
                    "width": clipped_band.shape[1],
                    "transform": transform,
                    "count": 1,
                    "compress": "lzw",
                    "BIGTIFF": "IF_SAFER",
                }
            )
            with rasterio.open(baseline_output, "w", **baseline_profile) as dst:
                dst.write(clipped_band, 1)

            mask_profile = {
                "driver": "GTiff",
                "height": mask_array.shape[0],
                "width": mask_array.shape[1],
                "count": 1,
                "dtype": "uint8",
                "crs": src.crs,
                "transform": transform,
                "nodata": 0,
                "compress": "lzw",
                "BIGTIFF": "IF_SAFER",
            }
            with rasterio.open(mask_output, "w", **mask_profile) as dst:
                dst.write(mask_array, 1)

            valid = np.not_equal(clipped_band, nodata) if nodata is not None else np.isfinite(clipped_band)
            raster_records.append(
                {
                    "order": int(row.order),
                    "requested_name": str(row.requested_name),
                    "matched_name": str(row.matched_name),
                    "country": str(row.country),
                    "ucdb_id": int(row.ucdb_id),
                    "buffer_meters": buffer_meters,
                    "baseline_path": str(baseline_output),
                    "mask_path": str(mask_output),
                    "width": int(clipped_band.shape[1]),
                    "height": int(clipped_band.shape[0]),
                    "bounds": [
                        float(transform.c),
                        float(transform.f + clipped_band.shape[0] * transform.e),
                        float(transform.c + clipped_band.shape[1] * transform.a),
                        float(transform.f),
                    ],
                    "pixel_count_inside_mask": int(mask_array.sum()),
                    "valid_baseline_pixels": int(valid.sum()),
                    "baseline_sum": float(clipped_band[valid].astype(np.float64).sum()) if valid.any() else 0.0,
                }
            )
            vector_records.append(
                {
                    "order": int(row.order),
                    "requested_name": str(row.requested_name),
                    "matched_name": str(row.matched_name),
                    "country": str(row.country),
                    "ucdb_id": int(row.ucdb_id),
                    "area_km2": int(row.area_km2),
                    "population_total": float(row.population_total),
                    "buffer_meters": buffer_meters,
                }
            )

    manifest = {
        "baseline_path": str(baseline_path),
        "vector_output": str(vector_output),
        "roi_count": len(raster_records),
        "rois": raster_records,
    }
    manifest_path = output_dir / "roi_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    summary_df = pd.DataFrame(vector_records)
    summary_output = output_dir / "roi_manifest.csv"
    summary_df.to_csv(summary_output, index=False)

    return manifest


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract buffered UCDB ROIs from the Mollweide baseline raster.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Resolve and report selected UCDB city matches.")
    build_parser = subparsers.add_parser("build", help="Extract buffered ROI rasters and masks.")

    for subparser in (inspect_parser, build_parser):
        subparser.add_argument("--baseline", type=_path_arg, default=DEFAULT_BASELINE_PATH)
        subparser.add_argument("--ucdb", type=_path_arg, default=DEFAULT_UCDB_PATH)
        subparser.add_argument("--layer", default=DEFAULT_LAYER)
        subparser.add_argument("--selection", type=_path_arg, default=DEFAULT_SELECTION_PATH)
        subparser.add_argument("--output-dir", type=_path_arg, default=DEFAULT_OUTPUT_DIR)
        subparser.add_argument("--buffer-meters", type=int, default=DEFAULT_BUFFER_METERS)

    inspect_parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    inspection = inspect_inputs(
        baseline_path=args.baseline,
        ucdb_path=args.ucdb,
        layer=args.layer,
        selection_path=args.selection,
    )
    if args.command == "inspect":
        if args.json:
            print(json.dumps(inspection, indent=2))
        else:
            print(json.dumps(inspection, indent=2))
        return

    requested = load_requested_cities(args.selection)
    ucdb = load_ucdb_layer(args.ucdb, args.layer)
    matched = resolve_city_matches(ucdb, requested)
    manifest = write_roi_outputs(
        baseline_path=args.baseline,
        matched_rois=matched,
        output_dir=args.output_dir,
        buffer_meters=args.buffer_meters,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
