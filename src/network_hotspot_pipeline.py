from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.transform import xy as transform_xy
from shapely.geometry import LineString, box

try:
    from src.create_hotspots import HotspotClusterConfig, HotspotClusterResult, HotspotClusterer, ROIGridData
    from src.extract_ucdb_rois import slugify
except ModuleNotFoundError:
    from create_hotspots import HotspotClusterConfig, HotspotClusterResult, HotspotClusterer, ROIGridData
    from extract_ucdb_rois import slugify

MOLLWEIDE_CRS = CRS.from_string("ESRI:54009")
WGS84_CRS = CRS.from_epsg(4326)
DEFAULT_ROI_MANIFEST_PATH = Path("data/outputs/roi/roi_manifest.json")
DEFAULT_ROI_VECTOR_PATH = Path("data/outputs/roi/ucdb_rois_buffer10km.gpkg")
DEFAULT_OUTPUT_ROOT = Path("data/outputs/network")
DEFAULT_CACHE_DIR = Path("data/cache/osmnx")
DEFAULT_NETWORK_TYPE = "drive"
DEFAULT_ROI_LAYER = "ucdb_rois"


@dataclass(frozen=True)
class NetworkHotspotConfig:
    roi_manifest_path: Path = DEFAULT_ROI_MANIFEST_PATH
    roi_vector_path: Path = DEFAULT_ROI_VECTOR_PATH
    roi_vector_layer: str = DEFAULT_ROI_LAYER
    output_root: Path = DEFAULT_OUTPUT_ROOT
    cache_dir: Path = DEFAULT_CACHE_DIR
    network_type: str = DEFAULT_NETWORK_TYPE
    simplify: bool = True
    retain_all: bool = True
    truncate_by_edge: bool = True
    hotspot_config: HotspotClusterConfig = field(
        default_factory=lambda: HotspotClusterConfig(
            hotspot_quantile=0.5,
            n_clusters=1000,
            random_state=42,
            saturation_eps=1e-6,
            local_window_size=3,
            min_weight=1.0,
            batch_size=2048,
        )
    )


@dataclass(frozen=True)
class CityContext:
    order: int
    requested_name: str
    matched_name: str
    slug: str
    country: str
    ucdb_id: int
    buffer_meters: int
    baseline_path: Path
    mask_path: Path
    roi_polygon_mollweide: Any
    roi_bounds_mollweide: tuple[float, float, float, float]
    output_dir: Path


def _path_arg(value: str) -> Path:
    return Path(value).expanduser().resolve()


class NetworkHotspotPipeline:
    def __init__(self, config: NetworkHotspotConfig | None = None) -> None:
        self.config = config or NetworkHotspotConfig()
        self.clusterer = HotspotClusterer(self.config.hotspot_config)

    @staticmethod
    def _resolve_project_root(project_root: Path | None = None) -> Path:
        if project_root is not None:
            return project_root.expanduser().resolve()
        root = Path.cwd().resolve()
        if (root / "data").exists():
            return root
        if (root.parent / "data").exists():
            return root.parent
        raise ValueError("Could not infer project root containing the data directory.")

    @staticmethod
    def load_manifest(manifest_path: Path) -> dict[str, Any]:
        return json.loads(manifest_path.read_text())

    def _resolved_config_paths(self, project_root: Path | None = None) -> tuple[Path, Path, Path, Path]:
        root = self._resolve_project_root(project_root)
        manifest_path = (root / self.config.roi_manifest_path).resolve()
        vector_path = (root / self.config.roi_vector_path).resolve()
        output_root = (root / self.config.output_root).resolve()
        cache_dir = (root / self.config.cache_dir).resolve()
        return manifest_path, vector_path, output_root, cache_dir

    def load_roi_vectors(self, project_root: Path | None = None) -> gpd.GeoDataFrame:
        _, vector_path, _, _ = self._resolved_config_paths(project_root)
        return gpd.read_file(vector_path, layer=self.config.roi_vector_layer, engine="pyogrio")

    def resolve_city_context(self, city_name: str, project_root: Path | None = None) -> CityContext:
        manifest_path, _, output_root, _ = self._resolved_config_paths(project_root)
        manifest = self.load_manifest(manifest_path)
        matches = [record for record in manifest["rois"] if record["matched_name"] == city_name]
        if not matches:
            raise KeyError(f"City not found in ROI manifest: {city_name}")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous city in ROI manifest: {city_name}")
        record = matches[0]

        roi_vectors = self.load_roi_vectors(project_root)
        roi_matches = roi_vectors[roi_vectors["matched_name"] == city_name]
        if len(roi_matches) != 1:
            raise ValueError(f"Expected exactly one ROI polygon for {city_name}, found {len(roi_matches)}.")
        row = roi_matches.iloc[0]

        root = self._resolve_project_root(project_root)
        slug = f"{int(record['order']):02d}_{slugify(record['matched_name'])}"
        return CityContext(
            order=int(record["order"]),
            requested_name=str(record["requested_name"]),
            matched_name=str(record["matched_name"]),
            slug=slug,
            country=str(record["country"]),
            ucdb_id=int(record["ucdb_id"]),
            buffer_meters=int(record["buffer_meters"]),
            baseline_path=(root / Path(record["baseline_path"])).resolve(),
            mask_path=(root / Path(record["mask_path"])).resolve(),
            roi_polygon_mollweide=row.geometry,
            roi_bounds_mollweide=tuple(float(v) for v in row.geometry.bounds),
            output_dir=(output_root / slug).resolve(),
        )

    @staticmethod
    def load_roi_data(context: CityContext) -> ROIGridData:
        return HotspotClusterer.load_roi_from_paths(
            baseline_path=context.baseline_path,
            mask_path=context.mask_path,
            matched_name=context.matched_name,
        )

    @staticmethod
    def _roi_polygon_wgs84(geometry: Any) -> Any:
        geo = gpd.GeoSeries([geometry], crs=MOLLWEIDE_CRS)
        return geo.to_crs(WGS84_CRS).iloc[0]

    @staticmethod
    def _cell_geometries(rows: np.ndarray, cols: np.ndarray, transform: Affine) -> list[Any]:
        x_left = transform.c + cols * transform.a
        x_right = x_left + transform.a
        y_top = transform.f + rows * transform.e
        y_bottom = y_top + transform.e
        min_y = np.minimum(y_top, y_bottom)
        max_y = np.maximum(y_top, y_bottom)
        return [
            box(float(left), float(bottom), float(right), float(top))
            for left, right, bottom, top in zip(x_left, x_right, min_y, max_y)
        ]

    def build_grid_gdf(self, roi_data: ROIGridData, hotspot_result: HotspotClusterResult) -> gpd.GeoDataFrame:
        rows, cols = np.where(roi_data.roi_mask == 1)
        xs, ys = transform_xy(roi_data.transform, rows, cols, offset="center")
        geometries = self._cell_geometries(rows.astype(np.int64), cols.astype(np.int64), roi_data.transform)

        baseline_values = roi_data.baseline[rows, cols].astype(np.float64)
        baseline_valid = roi_data.valid_mask[rows, cols]
        baseline_export = np.where(baseline_valid, baseline_values, np.nan)
        candidate_mask = hotspot_result.candidate_mask[rows, cols]
        cluster_ids = hotspot_result.cluster_grid[rows, cols]
        cluster_series = pd.Series(pd.array(cluster_ids, dtype="Int64"))
        cluster_series.loc[~candidate_mask] = pd.NA

        return gpd.GeoDataFrame(
            {
                "row": rows.astype(int),
                "col": cols.astype(int),
                "cell_id": (rows.astype(np.int64) * roi_data.baseline.shape[1] + cols.astype(np.int64)),
                "center_x": np.asarray(xs, dtype=np.float64),
                "center_y": np.asarray(ys, dtype=np.float64),
                "in_roi": True,
                "baseline_valid": baseline_valid.astype(bool),
                "baseline_value": baseline_export,
                "score": hotspot_result.score_map[rows, cols].astype(np.float64),
                "score_rank": hotspot_result.score_rank_map[rows, cols].astype(np.float64),
                "local_score": hotspot_result.local_score_map[rows, cols].astype(np.float64),
                "is_hotspot_candidate": candidate_mask.astype(bool),
                "cluster_id": cluster_series,
            },
            geometry=geometries,
            crs=roi_data.crs,
        )

    def download_drive_graph(self, roi_polygon_mollweide: Any, cache_dir: Path) -> nx.MultiDiGraph:
        cache_dir.mkdir(parents=True, exist_ok=True)
        ox.settings.use_cache = True
        ox.settings.cache_folder = str(cache_dir)
        polygon_wgs84 = self._roi_polygon_wgs84(roi_polygon_mollweide)
        graph = ox.graph_from_polygon(
            polygon_wgs84,
            network_type=self.config.network_type,
            simplify=self.config.simplify,
            retain_all=self.config.retain_all,
            truncate_by_edge=self.config.truncate_by_edge,
        )
        if graph.number_of_nodes() == 0:
            bounds = tuple(float(v) for v in polygon_wgs84.bounds)
            raise ValueError(f"OSMnx returned an empty graph for ROI bounds={bounds}.")
        return graph

    @staticmethod
    def project_graph_to_mollweide(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        return ox.project_graph(graph, to_crs=MOLLWEIDE_CRS)

    @staticmethod
    def graph_component_count(graph: nx.MultiDiGraph) -> int:
        return int(nx.number_weakly_connected_components(graph))

    def snap_cluster_centers(
        self,
        graph: nx.MultiDiGraph,
        cluster_centers: gpd.GeoDataFrame,
    ) -> tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        if cluster_centers.empty:
            raise ValueError("No hotspot centers available to snap.")

        center_x = cluster_centers.geometry.x.to_numpy()
        center_y = cluster_centers.geometry.y.to_numpy()
        node_ids, distances = ox.distance.nearest_nodes(
            graph,
            X=center_x,
            Y=center_y,
            return_dist=True,
        )

        nodes_gdf = ox.graph_to_gdfs(graph, nodes=True, edges=False)
        snapped_lookup = nodes_gdf.reindex(node_ids)
        if snapped_lookup["geometry"].isna().any():
            raise ValueError("Failed to resolve one or more snapped graph nodes.")

        mapping_df = cluster_centers.drop(columns="geometry").copy()
        mapping_df["node_id"] = pd.Series(node_ids, index=mapping_df.index)
        mapping_df["hotspot_x"] = center_x
        mapping_df["hotspot_y"] = center_y
        mapping_df["node_x"] = snapped_lookup["x"].to_numpy(dtype=np.float64)
        mapping_df["node_y"] = snapped_lookup["y"].to_numpy(dtype=np.float64)
        mapping_df["snap_distance_m"] = np.asarray(distances, dtype=np.float64)

        snapped_nodes_gdf = gpd.GeoDataFrame(
            mapping_df.copy(),
            geometry=snapped_lookup.geometry.to_numpy(),
            crs=cluster_centers.crs,
        )
        snap_lines_gdf = gpd.GeoDataFrame(
            mapping_df.copy(),
            geometry=[
                LineString([(float(hx), float(hy)), (float(nx_), float(ny_))])
                for hx, hy, nx_, ny_ in zip(
                    mapping_df["hotspot_x"],
                    mapping_df["hotspot_y"],
                    mapping_df["node_x"],
                    mapping_df["node_y"],
                )
            ],
            crs=cluster_centers.crs,
        )
        return mapping_df, snapped_nodes_gdf, snap_lines_gdf

    @staticmethod
    def aggregate_node_demand(mapping_df: pd.DataFrame) -> pd.DataFrame:
        return (
            mapping_df.groupby("node_id", as_index=False)
            .agg(
                hotspot_count=("cluster_id", "size"),
                sum_pixel_count=("pixel_count", "sum"),
                sum_mean_local_score=("mean_local_score", "sum"),
                sum_weight_sum=("weight_sum", "sum"),
            )
            .sort_values(["sum_mean_local_score", "hotspot_count"], ascending=[False, False])
            .reset_index(drop=True)
        )

    @staticmethod
    def _write_layer(gdf: gpd.GeoDataFrame, path: Path, layer: str) -> None:
        gdf.to_file(path, layer=layer, driver="GPKG", engine="pyogrio")

    def write_outputs(
        self,
        context: CityContext,
        roi_data: ROIGridData,
        hotspot_result: HotspotClusterResult,
        grid_gdf: gpd.GeoDataFrame,
        graph: nx.MultiDiGraph,
        mapping_df: pd.DataFrame,
        cluster_centers_gdf: gpd.GeoDataFrame,
        snapped_nodes_gdf: gpd.GeoDataFrame,
        snap_lines_gdf: gpd.GeoDataFrame,
    ) -> dict[str, Path]:
        context.output_dir.mkdir(parents=True, exist_ok=True)
        graphml_path = context.output_dir / "network.graphml"
        network_gpkg_path = context.output_dir / "network.gpkg"
        grid_gpkg_path = context.output_dir / "grid.gpkg"
        hotspots_gpkg_path = context.output_dir / "hotspots.gpkg"
        cluster_summary_csv = context.output_dir / "hotspot_cluster_summary.csv"
        hotspot_node_map_csv = context.output_dir / "hotspot_node_map.csv"
        node_hotspot_demand_csv = context.output_dir / "node_hotspot_demand.csv"
        manifest_path = context.output_dir / "manifest.json"

        ox.save_graphml(graph, filepath=graphml_path)
        ox.save_graph_geopackage(graph, filepath=network_gpkg_path)

        if grid_gpkg_path.exists():
            grid_gpkg_path.unlink()
        if hotspots_gpkg_path.exists():
            hotspots_gpkg_path.unlink()

        self._write_layer(grid_gdf, grid_gpkg_path, "grid_cells")
        self._write_layer(cluster_centers_gdf, hotspots_gpkg_path, "cluster_centers")
        self._write_layer(snapped_nodes_gdf, hotspots_gpkg_path, "snapped_nodes")
        self._write_layer(snap_lines_gdf, hotspots_gpkg_path, "snap_lines")

        hotspot_result.cluster_summary.to_csv(cluster_summary_csv, index=False)
        mapping_df.to_csv(hotspot_node_map_csv, index=False)
        node_hotspot_demand = self.aggregate_node_demand(mapping_df)
        node_hotspot_demand.to_csv(node_hotspot_demand_csv, index=False)

        node_count = int(graph.number_of_nodes())
        edge_count = int(graph.number_of_edges())
        manifest = {
            "city": context.matched_name,
            "order": context.order,
            "country": context.country,
            "ucdb_id": context.ucdb_id,
            "baseline_path": str(context.baseline_path),
            "mask_path": str(context.mask_path),
            "roi_bounds_mollweide": [float(v) for v in context.roi_bounds_mollweide],
            "roi_area_km2": float(context.roi_polygon_mollweide.area / 1_000_000.0),
            "graph_crs": str(graph.graph.get("crs")),
            "graph_node_count": node_count,
            "graph_edge_count": edge_count,
            "graph_component_count": self.graph_component_count(graph),
            "grid_cell_count": int(len(grid_gdf)),
            "valid_baseline_cell_count": int(roi_data.valid_mask.sum()),
            "requested_cluster_count": int(self.config.hotspot_config.n_clusters),
            "actual_cluster_count": int(len(hotspot_result.cluster_summary)),
            "hotspot_candidate_count": int(hotspot_result.candidate_count),
            "snap_distance_m": {
                "min": float(mapping_df["snap_distance_m"].min()),
                "max": float(mapping_df["snap_distance_m"].max()),
                "mean": float(mapping_df["snap_distance_m"].mean()),
                "median": float(mapping_df["snap_distance_m"].median()),
                "p95": float(mapping_df["snap_distance_m"].quantile(0.95)),
            },
            "outputs": {
                "graphml": str(graphml_path),
                "network_gpkg": str(network_gpkg_path),
                "grid_gpkg": str(grid_gpkg_path),
                "hotspots_gpkg": str(hotspots_gpkg_path),
                "cluster_summary_csv": str(cluster_summary_csv),
                "hotspot_node_map_csv": str(hotspot_node_map_csv),
                "node_hotspot_demand_csv": str(node_hotspot_demand_csv),
            },
            "hotspot_config": hotspot_result.to_dict()["config"],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

        return {
            "graphml": graphml_path,
            "network_gpkg": network_gpkg_path,
            "grid_gpkg": grid_gpkg_path,
            "hotspots_gpkg": hotspots_gpkg_path,
            "cluster_summary_csv": cluster_summary_csv,
            "hotspot_node_map_csv": hotspot_node_map_csv,
            "node_hotspot_demand_csv": node_hotspot_demand_csv,
            "manifest_json": manifest_path,
        }

    def inspect_city(self, city_name: str, project_root: Path | None = None) -> dict[str, Any]:
        context = self.resolve_city_context(city_name, project_root=project_root)
        roi_data = self.load_roi_data(context)
        hotspot_result = self.clusterer.fit(roi_data)
        mask_cells = int((roi_data.roi_mask == 1).sum())

        return {
            "city": context.matched_name,
            "country": context.country,
            "ucdb_id": context.ucdb_id,
            "baseline_path": str(context.baseline_path),
            "mask_path": str(context.mask_path),
            "output_dir": str(context.output_dir),
            "roi_bounds_mollweide": [float(v) for v in context.roi_bounds_mollweide],
            "roi_area_km2": float(context.roi_polygon_mollweide.area / 1_000_000.0),
            "raster": {
                "shape": [int(roi_data.baseline.shape[0]), int(roi_data.baseline.shape[1])],
                "crs": roi_data.crs.to_string() if roi_data.crs is not None else None,
                "transform": list(roi_data.transform.to_gdal()),
                "nodata": float(roi_data.nodata) if roi_data.nodata is not None else None,
                "mask_cell_count": mask_cells,
                "valid_baseline_cell_count": int(roi_data.valid_mask.sum()),
            },
            "hotspots": {
                "candidate_count": int(hotspot_result.candidate_count),
                "actual_cluster_count": int(len(hotspot_result.cluster_summary)),
                "threshold": float(hotspot_result.threshold),
                "config": hotspot_result.to_dict()["config"],
            },
        }

    def build(self, city_name: str, project_root: Path | None = None) -> dict[str, Any]:
        context = self.resolve_city_context(city_name, project_root=project_root)
        roi_data = self.load_roi_data(context)
        hotspot_result = self.clusterer.fit(roi_data)
        grid_gdf = self.build_grid_gdf(roi_data, hotspot_result)
        _, _, _, cache_dir = self._resolved_config_paths(project_root)
        graph_wgs84 = self.download_drive_graph(context.roi_polygon_mollweide, cache_dir=cache_dir)
        graph_mollweide = self.project_graph_to_mollweide(graph_wgs84)
        cluster_centers_gdf = hotspot_result.cluster_centers_gdf(weighted=True)
        mapping_df, snapped_nodes_gdf, snap_lines_gdf = self.snap_cluster_centers(graph_mollweide, cluster_centers_gdf)
        outputs = self.write_outputs(
            context=context,
            roi_data=roi_data,
            hotspot_result=hotspot_result,
            grid_gdf=grid_gdf,
            graph=graph_mollweide,
            mapping_df=mapping_df,
            cluster_centers_gdf=cluster_centers_gdf,
            snapped_nodes_gdf=snapped_nodes_gdf,
            snap_lines_gdf=snap_lines_gdf,
        )
        return {
            "city": context.matched_name,
            "output_dir": str(context.output_dir),
            "outputs": {name: str(path) for name, path in outputs.items()},
        }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract an OSM road graph, build an equal-area ROI grid, and snap ROI hotspots to graph nodes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Resolve a city ROI and report hotspot/grid metadata.")
    build_parser = subparsers.add_parser("build", help="Build graph, grid, hotspot, and snap outputs for a city.")

    for subparser in (inspect_parser, build_parser):
        subparser.add_argument("--city", required=True)
        subparser.add_argument("--roi-manifest", type=_path_arg, default=DEFAULT_ROI_MANIFEST_PATH)
        subparser.add_argument("--roi-vector", type=_path_arg, default=DEFAULT_ROI_VECTOR_PATH)
        subparser.add_argument("--output-root", type=_path_arg, default=DEFAULT_OUTPUT_ROOT)
        subparser.add_argument("--cache-dir", type=_path_arg, default=DEFAULT_CACHE_DIR)
        subparser.add_argument("--network-type", default=DEFAULT_NETWORK_TYPE)
        subparser.add_argument("--n-clusters", type=int, default=1000)
        subparser.add_argument("--hotspot-quantile", type=float, default=0.5)
        subparser.add_argument("--local-window-size", type=int, default=3)
        subparser.add_argument("--batch-size", type=int, default=2048)
        subparser.add_argument("--random-state", type=int, default=42)
        subparser.add_argument("--saturation-eps", type=float, default=1e-6)
        subparser.add_argument("--min-weight", type=float, default=1.0)
        subparser.add_argument("--json", action="store_true")

    return parser


def config_from_args(args: argparse.Namespace) -> NetworkHotspotConfig:
    return NetworkHotspotConfig(
        roi_manifest_path=args.roi_manifest,
        roi_vector_path=args.roi_vector,
        output_root=args.output_root,
        cache_dir=args.cache_dir,
        network_type=args.network_type,
        hotspot_config=HotspotClusterConfig(
            hotspot_quantile=args.hotspot_quantile,
            n_clusters=args.n_clusters,
            random_state=args.random_state,
            saturation_eps=args.saturation_eps,
            local_window_size=args.local_window_size,
            min_weight=args.min_weight,
            batch_size=args.batch_size,
        ),
    )


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    pipeline = NetworkHotspotPipeline(config_from_args(args))

    if args.command == "inspect":
        payload = pipeline.inspect_city(args.city)
    else:
        payload = pipeline.build(args.city)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
