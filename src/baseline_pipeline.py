from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject

MOLLWEIDE_CRS = CRS.from_string("ESRI:54009")
DEFAULT_NODATA = -9999.0
DEFAULT_HIST_BINS = 65536


@dataclass
class RasterStats:
    path: str
    crs: str
    width: int
    height: int
    resolution: tuple[float, float]
    bounds: tuple[float, float, float, float]
    nodata: float | None
    dtype: str
    valid_cells: int
    zero_cells: int
    min: float | None
    max: float | None
    total_sum: float


@dataclass
class TargetGrid:
    crs: str
    width: int
    height: int
    transform: tuple[float, float, float, float, float, float]
    bounds: tuple[float, float, float, float]
    resolution_m: int


@dataclass
class NormalizationBounds:
    lower_percentile: float
    upper_percentile: float
    lower_value: float
    upper_value: float
    log_min: float
    log_max: float
    valid_cells: int


def _path_arg(value: str) -> Path:
    return Path(value).expanduser().resolve()


def inspect_raster(path: Path) -> RasterStats:
    with rasterio.open(path) as src:
        nodata = src.nodata
        min_value: float | None = None
        max_value: float | None = None
        total_sum = 0.0
        valid_cells = 0
        zero_cells = 0

        for _, window in src.block_windows(1):
            arr = src.read(1, window=window)
            valid = _valid_mask(arr, nodata)
            if not valid.any():
                continue

            values = arr[valid].astype(np.float64, copy=False)
            valid_cells += int(values.size)
            zero_cells += int(np.count_nonzero(values == 0))
            total_sum += float(values.sum())

            window_min = float(values.min())
            window_max = float(values.max())
            min_value = window_min if min_value is None else min(min_value, window_min)
            max_value = window_max if max_value is None else max(max_value, window_max)

        return RasterStats(
            path=str(path),
            crs=str(src.crs),
            width=src.width,
            height=src.height,
            resolution=(float(src.res[0]), float(src.res[1])),
            bounds=(
                float(src.bounds.left),
                float(src.bounds.bottom),
                float(src.bounds.right),
                float(src.bounds.top),
            ),
            nodata=float(nodata) if nodata is not None else None,
            dtype=src.dtypes[0],
            valid_cells=valid_cells,
            zero_cells=zero_cells,
            min=min_value,
            max=max_value,
            total_sum=total_sum,
        )


def _valid_mask(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    if nodata is None:
        return np.isfinite(arr)
    if math.isnan(nodata):
        return ~np.isnan(arr)
    return np.not_equal(arr, nodata)


def derive_target_grid(reference_path: Path, resolution_m: int) -> TargetGrid:
    with rasterio.open(reference_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            MOLLWEIDE_CRS,
            src.width,
            src.height,
            *src.bounds,
            resolution=resolution_m,
        )
        bounds = (
            float(transform.c),
            float(transform.f + height * transform.e),
            float(transform.c + width * transform.a),
            float(transform.f),
        )
        return TargetGrid(
            crs=str(MOLLWEIDE_CRS),
            width=width,
            height=height,
            transform=(transform.a, transform.b, transform.c, transform.d, transform.e, transform.f),
            bounds=bounds,
            resolution_m=resolution_m,
        )


def _grid_transform(grid: TargetGrid) -> Affine:
    a, b, c, d, e, f = grid.transform
    return Affine(a, b, c, d, e, f)


def make_raster_profile(grid: TargetGrid, nodata: float, compress: str = "lzw") -> dict[str, Any]:
    return {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 1,
        "dtype": "float32",
        "crs": MOLLWEIDE_CRS,
        "transform": _grid_transform(grid),
        "nodata": nodata,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": compress,
        "BIGTIFF": "IF_SAFER",
    }


def reproject_to_temp(
    src_path: Path,
    dst_path: Path,
    grid: TargetGrid,
    nodata: float,
    warp_mem_limit_mb: int,
    threads: int,
) -> RasterStats:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    profile = make_raster_profile(grid, nodata)

    with rasterio.open(src_path) as src, rasterio.open(dst_path, "w", **profile) as dst:
        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst.transform,
            dst_crs=dst.crs,
            dst_nodata=nodata,
            resampling=Resampling.sum,
            init_dest_nodata=True,
            num_threads=threads,
            warp_mem_limit=warp_mem_limit_mb,
        )

    return inspect_raster(dst_path)


def compute_normalization_bounds(
    raster_path: Path,
    lower_percentile: float,
    upper_percentile: float,
    hist_bins: int,
) -> NormalizationBounds:
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        log_min = math.inf
        log_max = -math.inf
        valid_cells = 0

        for _, window in src.block_windows(1):
            arr = src.read(1, window=window)
            valid = _valid_mask(arr, nodata)
            if not valid.any():
                continue

            values = arr[valid].astype(np.float64, copy=False)
            logs = np.log1p(values)
            valid_cells += int(logs.size)
            log_min = min(log_min, float(logs.min()))
            log_max = max(log_max, float(logs.max()))

        if valid_cells == 0:
            return NormalizationBounds(
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
                lower_value=0.0,
                upper_value=0.0,
                log_min=0.0,
                log_max=0.0,
                valid_cells=0,
            )

        if math.isclose(log_min, log_max):
            return NormalizationBounds(
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
                lower_value=log_min,
                upper_value=log_max,
                log_min=log_min,
                log_max=log_max,
                valid_cells=valid_cells,
            )

        histogram = np.zeros(hist_bins, dtype=np.int64)
        scale = (hist_bins - 1) / (log_max - log_min)

        for _, window in src.block_windows(1):
            arr = src.read(1, window=window)
            valid = _valid_mask(arr, nodata)
            if not valid.any():
                continue

            logs = np.log1p(arr[valid].astype(np.float64, copy=False))
            indices = np.floor((logs - log_min) * scale).astype(np.int64)
            np.clip(indices, 0, hist_bins - 1, out=indices)
            histogram += np.bincount(indices, minlength=hist_bins)

    lower_value = percentile_from_histogram(histogram, log_min, log_max, lower_percentile)
    upper_value = percentile_from_histogram(histogram, log_min, log_max, upper_percentile)

    return NormalizationBounds(
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        lower_value=lower_value,
        upper_value=upper_value,
        log_min=log_min,
        log_max=log_max,
        valid_cells=valid_cells,
    )


def percentile_from_histogram(
    histogram: np.ndarray,
    log_min: float,
    log_max: float,
    percentile: float,
) -> float:
    total = int(histogram.sum())
    if total == 0:
        return 0.0

    if percentile <= 0:
        return log_min
    if percentile >= 100:
        return log_max

    target = math.ceil((percentile / 100.0) * total)
    cumulative = np.cumsum(histogram, dtype=np.int64)
    index = int(np.searchsorted(cumulative, target, side="left"))
    if histogram.size == 1:
        return log_min

    width = (log_max - log_min) / histogram.size
    return log_min + index * width


def normalize_valid_values(
    values: np.ndarray,
    lower_value: float,
    upper_value: float,
) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32, copy=False)

    normalized = np.zeros(values.shape, dtype=np.float32)
    if not math.isfinite(lower_value) or not math.isfinite(upper_value) or upper_value <= lower_value:
        return normalized

    np.log1p(values, out=values)
    normalized = ((values - lower_value) / (upper_value - lower_value)).astype(np.float32, copy=False)
    np.clip(normalized, 0.0, 1.0, out=normalized)
    return normalized


def normalize_block(
    data: np.ndarray,
    nodata: float,
    bounds: NormalizationBounds,
) -> tuple[np.ndarray, np.ndarray]:
    valid = _valid_mask(data, nodata)
    normalized = np.zeros(data.shape, dtype=np.float32)
    if valid.any():
        normalized[valid] = normalize_valid_values(
            data[valid].astype(np.float64, copy=True),
            bounds.lower_value,
            bounds.upper_value,
        )
    return normalized, valid


def fuse_arrays(
    wp_data: np.ndarray,
    ls_data: np.ndarray,
    wp_bounds: NormalizationBounds,
    ls_bounds: NormalizationBounds,
    nodata: float,
    worldpop_weight: float,
    landscan_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    wp_norm, wp_valid = normalize_block(wp_data, nodata, wp_bounds)
    ls_norm, ls_valid = normalize_block(ls_data, nodata, ls_bounds)

    out = np.full(wp_data.shape, nodata, dtype=np.float32)
    valid = wp_valid | ls_valid
    overlap = wp_valid & ls_valid
    wp_only = wp_valid & ~ls_valid
    ls_only = ls_valid & ~wp_valid

    out[wp_only] = wp_norm[wp_only]
    out[ls_only] = ls_norm[ls_only]
    if overlap.any():
        out[overlap] = (worldpop_weight * wp_norm[overlap]) + (landscan_weight * ls_norm[overlap])

    return out, valid


def build_baseline(
    worldpop_path: Path,
    landscan_path: Path,
    tmp_dir: Path,
    output_path: Path,
    resolution_m: int,
    worldpop_weight: float,
    landscan_weight: float,
    warp_mem_limit_mb: int,
    threads: int,
    lower_percentile: float,
    upper_percentile: float,
    hist_bins: int,
    nodata: float,
    keep_intermediates: bool,
) -> dict[str, Any]:
    grid = derive_target_grid(landscan_path, resolution_m)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    wp_tmp = tmp_dir / f"worldpop_mollweide_{resolution_m}m.tif"
    ls_tmp = tmp_dir / f"landscan_mollweide_{resolution_m}m.tif"

    raw_worldpop_stats = inspect_raster(worldpop_path)
    raw_landscan_stats = inspect_raster(landscan_path)
    wp_temp_stats = reproject_to_temp(worldpop_path, wp_tmp, grid, nodata, warp_mem_limit_mb, threads)
    ls_temp_stats = reproject_to_temp(landscan_path, ls_tmp, grid, nodata, warp_mem_limit_mb, threads)

    wp_bounds = compute_normalization_bounds(wp_tmp, lower_percentile, upper_percentile, hist_bins)
    ls_bounds = compute_normalization_bounds(ls_tmp, lower_percentile, upper_percentile, hist_bins)

    output_profile = make_raster_profile(grid, nodata)
    with rasterio.open(wp_tmp) as wp_src, rasterio.open(ls_tmp) as ls_src, rasterio.open(
        output_path,
        "w",
        **output_profile,
    ) as out_src:
        for _, window in out_src.block_windows(1):
            wp_arr = wp_src.read(1, window=window)
            ls_arr = ls_src.read(1, window=window)
            fused, valid = fuse_arrays(
                wp_arr,
                ls_arr,
                wp_bounds,
                ls_bounds,
                nodata,
                worldpop_weight,
                landscan_weight,
            )
            if not valid.any():
                fused.fill(nodata)
            out_src.write(fused, 1, window=window)

    baseline_stats = inspect_raster(output_path)
    manifest = {
        "target_grid": asdict(grid),
        "parameters": {
            "worldpop_weight": worldpop_weight,
            "landscan_weight": landscan_weight,
            "nodata": nodata,
            "resolution_m": resolution_m,
            "warp_mem_limit_mb": warp_mem_limit_mb,
            "threads": threads,
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile,
            "hist_bins": hist_bins,
        },
        "inputs": {
            "worldpop": asdict(raw_worldpop_stats),
            "landscan": asdict(raw_landscan_stats),
        },
        "intermediates": {
            "worldpop_mollweide": asdict(wp_temp_stats),
            "landscan_mollweide": asdict(ls_temp_stats),
        },
        "normalization": {
            "worldpop": asdict(wp_bounds),
            "landscan": asdict(ls_bounds),
        },
        "output": asdict(baseline_stats),
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if not keep_intermediates:
        for temp_path in (wp_tmp, ls_tmp):
            if temp_path.exists():
                temp_path.unlink()

    return manifest


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the fused Mollweide baseline raster.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Report metadata and global stats for input rasters.")
    inspect_parser.add_argument(
        "--worldpop",
        type=_path_arg,
        default=Path("data/raw/worldpop/worldpop-global-2022.tif"),
    )
    inspect_parser.add_argument(
        "--landscan",
        type=_path_arg,
        default=Path("data/raw/landscan/landscan-global-2022.tif"),
    )
    inspect_parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text.")

    build_parser = subparsers.add_parser("build", help="Reproject, normalize, and fuse the baseline raster.")
    build_parser.add_argument(
        "--worldpop",
        type=_path_arg,
        default=Path("data/raw/worldpop/worldpop-global-2022.tif"),
    )
    build_parser.add_argument(
        "--landscan",
        type=_path_arg,
        default=Path("data/raw/landscan/landscan-global-2022.tif"),
    )
    build_parser.add_argument("--tmp-dir", type=_path_arg, default=Path("data/processed/tmp"))
    build_parser.add_argument(
        "--output",
        type=_path_arg,
        default=Path("data/processed/baseline_mollweide_1km_2022.tif"),
    )
    build_parser.add_argument("--resolution-m", type=int, default=1000)
    build_parser.add_argument("--worldpop-weight", type=float, default=0.5)
    build_parser.add_argument("--landscan-weight", type=float, default=0.5)
    build_parser.add_argument("--warp-mem-mb", type=int, default=512)
    build_parser.add_argument("--threads", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    build_parser.add_argument("--lower-percentile", type=float, default=1.0)
    build_parser.add_argument("--upper-percentile", type=float, default=99.5)
    build_parser.add_argument("--hist-bins", type=int, default=DEFAULT_HIST_BINS)
    build_parser.add_argument("--nodata", type=float, default=DEFAULT_NODATA)
    build_parser.add_argument("--keep-intermediates", action="store_true")

    return parser


def _print_stats(stats: RasterStats) -> None:
    print(json.dumps(asdict(stats), indent=2))


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    if args.command == "inspect":
        stats = [inspect_raster(args.worldpop), inspect_raster(args.landscan)]
        if args.json:
            print(json.dumps([asdict(item) for item in stats], indent=2))
        else:
            for item in stats:
                _print_stats(item)
        return

    total_weight = args.worldpop_weight + args.landscan_weight
    if total_weight <= 0:
        raise SystemExit("WorldPop and LandScan weights must sum to a positive value.")

    manifest = build_baseline(
        worldpop_path=args.worldpop,
        landscan_path=args.landscan,
        tmp_dir=args.tmp_dir,
        output_path=args.output,
        resolution_m=args.resolution_m,
        worldpop_weight=args.worldpop_weight / total_weight,
        landscan_weight=args.landscan_weight / total_weight,
        warp_mem_limit_mb=args.warp_mem_mb,
        threads=args.threads,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        hist_bins=args.hist_bins,
        nodata=args.nodata,
        keep_intermediates=args.keep_intermediates,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
