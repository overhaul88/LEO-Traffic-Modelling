from __future__ import annotations

import argparse
import io
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xyzservices.providers as xyz
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from PIL import Image
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds

try:
    from src.agent_simulation import (
        _LW_BY_CLASS,
        _MAJOR_ROAD_CLASSES,
        _edge_segments,
        _in_event,
        build_default_scenarios,
        build_edge_util_arrays,
        build_hotspot_demand_table,
        build_routing_graph,
        run_simulation,
        simulation_summary,
        SimulationConfig,
    )
    from src.extract_ucdb_rois import (
        DEFAULT_BASELINE_PATH,
        DEFAULT_BUFFER_METERS,
        DEFAULT_LAYER,
        DEFAULT_UCDB_PATH,
        RequestedCity,
        load_ucdb_layer,
        resolve_city_matches,
        slugify,
        write_roi_outputs,
    )
    from src.network_hotspot_pipeline import DEFAULT_CACHE_DIR, NetworkHotspotConfig, NetworkHotspotPipeline
except ModuleNotFoundError:
    from agent_simulation import (
        _LW_BY_CLASS,
        _MAJOR_ROAD_CLASSES,
        _edge_segments,
        _in_event,
        build_default_scenarios,
        build_edge_util_arrays,
        build_hotspot_demand_table,
        build_routing_graph,
        run_simulation,
        simulation_summary,
        SimulationConfig,
    )
    from extract_ucdb_rois import (
        DEFAULT_BASELINE_PATH,
        DEFAULT_BUFFER_METERS,
        DEFAULT_LAYER,
        DEFAULT_UCDB_PATH,
        RequestedCity,
        load_ucdb_layer,
        resolve_city_matches,
        slugify,
        write_roi_outputs,
    )
    from network_hotspot_pipeline import DEFAULT_CACHE_DIR, NetworkHotspotConfig, NetworkHotspotPipeline

MOLLWEIDE_CRS = CRS.from_string("ESRI:54009")
WGS84_CRS = CRS.from_epsg(4326)
WEB_MERCATOR_CRS = CRS.from_epsg(3857)
WEB_MERCATOR_HALF_WORLD = 20_037_508.342789244
DEFAULT_OUTPUT_ROOT = Path("data/outputs/map_overlay")
DEFAULT_PROVIDER = "OpenStreetMap.Mapnik"
DEFAULT_USER_AGENT = "leotm-map-overlay/1.0 (+https://www.openstreetmap.org/copyright)"


@dataclass(frozen=True)
class TileRange:
    zoom: int
    x_min: int
    x_max: int
    y_min: int
    y_max: int

    @property
    def tile_count(self) -> int:
        return (self.x_max - self.x_min + 1) * (self.y_max - self.y_min + 1)


@dataclass(frozen=True)
class BasemapArtifact:
    image_path: Path
    metadata_path: Path
    extent_mollweide: tuple[float, float, float, float]
    padded_bounds_mollweide: tuple[float, float, float, float]
    provider_name: str
    attribution: str
    tile_range: TileRange


def _path_arg(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _resolve_project_root(project_root: Path | None = None) -> Path:
    if project_root is not None:
        return project_root.expanduser().resolve()
    for candidate in [Path.cwd().resolve(), Path(__file__).resolve().parent.parent]:
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate
    return Path.cwd().resolve()


def _load_network_bundle(manifest_path: Path, roi_vector_path: Path, city_name: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    roi = gpd.read_file(roi_vector_path, layer="ucdb_rois", engine="pyogrio")
    roi = roi.loc[roi["matched_name"] == city_name, ["matched_name", "country", "geometry"]].copy()
    if roi.empty:
        raise ValueError(f"Could not locate ROI geometry for {city_name!r} in {roi_vector_path}.")
    outputs = manifest["outputs"]
    return {
        "manifest": manifest,
        "roi": roi,
        "hotspot_node_map": pd.read_csv(outputs["hotspot_node_map_csv"]),
        "node_hotspot_demand": pd.read_csv(outputs["node_hotspot_demand_csv"]),
        "cluster_summary": pd.read_csv(outputs["cluster_summary_csv"]),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def pad_bounds(bounds: tuple[float, float, float, float], pad_fraction: float) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bounds
    width = max(maxx - minx, 1.0)
    height = max(maxy - miny, 1.0)
    pad_x = width * pad_fraction
    pad_y = height * pad_fraction
    return (
        float(minx - pad_x),
        float(miny - pad_y),
        float(maxx + pad_x),
        float(maxy + pad_y),
    )


def lonlat_to_tile_indices(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    lon = ((lon + 180.0) % 360.0) - 180.0
    n = 2**zoom
    x = int(math.floor((lon + 180.0) / 360.0 * n))
    lat_rad = math.radians(lat)
    y = int(
        math.floor(
            (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
        )
    )
    return x, y


def tile_bounds_3857(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    n = 2**zoom
    tile_span = (2.0 * WEB_MERCATOR_HALF_WORLD) / n
    left = -WEB_MERCATOR_HALF_WORLD + x * tile_span
    right = left + tile_span
    top = WEB_MERCATOR_HALF_WORLD - y * tile_span
    bottom = top - tile_span
    return (float(left), float(bottom), float(right), float(top))


def choose_zoom(bounds_3857: tuple[float, float, float, float], long_side_px: int, max_zoom: int) -> int:
    minx, miny, maxx, maxy = bounds_3857
    width = max(maxx - minx, 1.0)
    height = max(maxy - miny, 1.0)
    desired_res = max(width, height) / max(long_side_px, 1)
    world_res_0 = (2.0 * WEB_MERCATOR_HALF_WORLD) / 256.0
    zoom = int(math.floor(math.log2(world_res_0 / desired_res)))
    return max(0, min(max_zoom, zoom))


def tile_range_for_bounds(bounds_wgs84: tuple[float, float, float, float], zoom: int) -> TileRange:
    west, south, east, north = bounds_wgs84
    x0, y1 = lonlat_to_tile_indices(west, south, zoom)
    x1, y0 = lonlat_to_tile_indices(east, north, zoom)
    n = 2**zoom
    return TileRange(
        zoom=zoom,
        x_min=max(0, min(n - 1, min(x0, x1))),
        x_max=max(0, min(n - 1, max(x0, x1))),
        y_min=max(0, min(n - 1, min(y0, y1))),
        y_max=max(0, min(n - 1, max(y0, y1))),
    )


def _sanitize_provider_name(provider_name: str) -> str:
    return provider_name.lower().replace(".", "_")


def _resolve_tile_provider(provider_name: str) -> Any:
    current = xyz
    for token in provider_name.split("."):
        if not hasattr(current, token):
            raise ValueError(f"Unknown tile provider {provider_name!r}.")
        current = getattr(current, token)
    return current


def fetch_tile(
    provider: Any,
    tile_range: TileRange,
    x: int,
    y: int,
    cache_dir: Path,
    session: requests.Session,
    user_agent: str,
    timeout_s: float = 30.0,
) -> Image.Image:
    n = 2**tile_range.zoom
    if not (0 <= y < n):
        raise ValueError(f"Tile y={y} is outside the legal range for z={tile_range.zoom}.")
    x_wrapped = x % n
    provider_cache = cache_dir / _sanitize_provider_name(provider.name) / str(tile_range.zoom) / str(x_wrapped)
    provider_cache.mkdir(parents=True, exist_ok=True)
    cache_path = provider_cache / f"{y}.png"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGBA")

    url = provider.build_url(x=x_wrapped, y=y, z=tile_range.zoom)
    response = session.get(url, headers={"User-Agent": user_agent}, timeout=timeout_s)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGBA")
    image.save(cache_path)
    return image


def build_tile_mosaic(
    provider: Any,
    tile_range: TileRange,
    cache_dir: Path,
    session: requests.Session,
    user_agent: str,
) -> tuple[Image.Image, tuple[float, float, float, float]]:
    x_count = tile_range.x_max - tile_range.x_min + 1
    y_count = tile_range.y_max - tile_range.y_min + 1
    mosaic = Image.new("RGBA", (x_count * 256, y_count * 256))
    for x in range(tile_range.x_min, tile_range.x_max + 1):
        for y in range(tile_range.y_min, tile_range.y_max + 1):
            tile = fetch_tile(provider, tile_range, x, y, cache_dir, session, user_agent)
            mosaic.paste(tile, ((x - tile_range.x_min) * 256, (y - tile_range.y_min) * 256))
    left, _, _, top = tile_bounds_3857(tile_range.x_min, tile_range.y_min, tile_range.zoom)
    _, bottom, right, _ = tile_bounds_3857(tile_range.x_max, tile_range.y_max, tile_range.zoom)
    return mosaic, (left, bottom, right, top)


def warp_basemap_to_mollweide(
    mosaic: Image.Image,
    mosaic_bounds_3857: tuple[float, float, float, float],
    padded_bounds_mollweide: tuple[float, float, float, float],
    long_side_px: int,
) -> np.ndarray:
    minx, miny, maxx, maxy = padded_bounds_mollweide
    width_m = max(maxx - minx, 1.0)
    height_m = max(maxy - miny, 1.0)
    if width_m >= height_m:
        dst_width = max(1, int(round(long_side_px)))
        dst_height = max(1, int(round(long_side_px * height_m / width_m)))
    else:
        dst_height = max(1, int(round(long_side_px)))
        dst_width = max(1, int(round(long_side_px * width_m / height_m)))

    src = np.asarray(mosaic, dtype=np.uint8)
    src_transform = from_bounds(
        mosaic_bounds_3857[0],
        mosaic_bounds_3857[1],
        mosaic_bounds_3857[2],
        mosaic_bounds_3857[3],
        src.shape[1],
        src.shape[0],
    )
    dst_transform = from_bounds(minx, miny, maxx, maxy, dst_width, dst_height)
    dst = np.zeros((4, dst_height, dst_width), dtype=np.uint8)

    for band_idx in range(4):
        reproject(
            source=src[:, :, band_idx],
            destination=dst[band_idx],
            src_transform=src_transform,
            src_crs=WEB_MERCATOR_CRS,
            dst_transform=dst_transform,
            dst_crs=MOLLWEIDE_CRS,
            resampling=Resampling.bilinear,
            dst_nodata=0,
        )
    return np.moveaxis(dst, 0, -1)


def load_basemap_artifact(image_path: Path) -> np.ndarray:
    return np.asarray(Image.open(image_path).convert("RGBA"), dtype=np.uint8)


def build_warped_basemap(
    roi_bounds_mollweide: tuple[float, float, float, float],
    basemap_dir: Path,
    provider_name: str,
    cache_dir: Path,
    pad_fraction: float,
    long_side_px: int,
    max_tiles: int,
    tile_zoom: int | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    force: bool = False,
    session: requests.Session | None = None,
) -> BasemapArtifact:
    basemap_dir.mkdir(parents=True, exist_ok=True)
    image_path = basemap_dir / "warped_basemap.png"
    metadata_path = basemap_dir / "warped_basemap.json"
    if image_path.exists() and metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text())
        tile_range = TileRange(**metadata["tile_range"])
        return BasemapArtifact(
            image_path=image_path,
            metadata_path=metadata_path,
            extent_mollweide=tuple(metadata["extent_mollweide"]),
            padded_bounds_mollweide=tuple(metadata["padded_bounds_mollweide"]),
            provider_name=metadata["provider_name"],
            attribution=metadata["attribution"],
            tile_range=tile_range,
        )

    provider = _resolve_tile_provider(provider_name)
    max_zoom = int(provider.get("max_zoom", 19))
    padded_bounds = pad_bounds(roi_bounds_mollweide, pad_fraction)
    bounds_wgs84 = transform_bounds(MOLLWEIDE_CRS, WGS84_CRS, *padded_bounds, densify_pts=64)
    bounds_3857 = transform_bounds(WGS84_CRS, WEB_MERCATOR_CRS, *bounds_wgs84, densify_pts=64)

    zoom = tile_zoom if tile_zoom is not None else choose_zoom(bounds_3857, long_side_px, max_zoom)
    tile_range = tile_range_for_bounds(bounds_wgs84, zoom)
    while tile_range.tile_count > max_tiles and zoom > 0:
        zoom -= 1
        tile_range = tile_range_for_bounds(bounds_wgs84, zoom)
    if tile_range.tile_count > max_tiles:
        raise ValueError(
            f"Tile coverage is still too large ({tile_range.tile_count} tiles) at z={zoom}. "
            "Increase --max-tiles or lower the requested zoom."
        )

    session = session or requests.Session()
    mosaic, mosaic_bounds_3857 = build_tile_mosaic(provider, tile_range, cache_dir, session, user_agent)
    warped = warp_basemap_to_mollweide(mosaic, mosaic_bounds_3857, padded_bounds, long_side_px)
    Image.fromarray(warped, mode="RGBA").save(image_path)

    metadata = {
        "provider_name": provider.name,
        "attribution": provider.get("attribution", provider.name),
        "extent_mollweide": [float(v) for v in padded_bounds],
        "padded_bounds_mollweide": [float(v) for v in padded_bounds],
        "bounds_wgs84": [float(v) for v in bounds_wgs84],
        "bounds_3857": [float(v) for v in bounds_3857],
        "tile_range": asdict(tile_range),
        "mosaic_bounds_3857": [float(v) for v in mosaic_bounds_3857],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return BasemapArtifact(
        image_path=image_path,
        metadata_path=metadata_path,
        extent_mollweide=tuple(metadata["extent_mollweide"]),
        padded_bounds_mollweide=tuple(metadata["padded_bounds_mollweide"]),
        provider_name=metadata["provider_name"],
        attribution=metadata["attribution"],
        tile_range=tile_range,
    )


def save_single_map_animation(
    result: dict[str, Any],
    roi_gdf: gpd.GeoDataFrame,
    basemap_rgba: np.ndarray,
    basemap_extent: tuple[float, float, float, float],
    attribution: str,
    output_path: Path,
    edge_scope: str = "major",
    fps: int = 5,
) -> dict[str, Any]:
    all_egdf, maj_egdf, ticks, all_umat, maj_umat, vmax = build_edge_util_arrays(result, display_crs="ESRI:54009")
    if edge_scope == "all":
        edge_gdf = all_egdf
        umat = all_umat
    else:
        edge_gdf = maj_egdf
        umat = maj_umat

    if edge_gdf.empty:
        raise ValueError(f"No edges available for edge_scope={edge_scope!r}.")

    segments = _edge_segments(edge_gdf)
    widths = np.array([_LW_BY_CLASS.get(rc, 0.32) for rc in edge_gdf["road_class"]], dtype=np.float64)
    if edge_scope == "major":
        widths = np.maximum(widths * 1.8, 0.8)

    norm = Normalize(vmin=0.0, vmax=max(vmax, 0.1))
    xmin, ymin, xmax, ymax = basemap_extent

    fig, ax = plt.subplots(figsize=(10.5, 10.5), dpi=90, constrained_layout=True)
    ax.set_facecolor("#0F1114")
    ax.imshow(basemap_rgba, extent=[xmin, xmax, ymin, ymax], origin="upper", zorder=0)
    roi_gdf.boundary.plot(ax=ax, color="#F3F4F6", linewidth=0.8, alpha=0.8, zorder=2)

    line_collection = LineCollection(
        segments,
        linewidths=widths,
        norm=norm,
        cmap="hot_r",
        alpha=0.92,
        zorder=3,
    )
    line_collection.set_array(umat[0])
    ax.add_collection(line_collection)

    sm = ScalarMappable(norm=norm, cmap="hot_r")
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Utilization (load / cap-per-tick)", fontsize=9)

    title = ax.set_title("", fontsize=12, color="white", pad=10)
    subtitle = ax.text(
        0.01,
        0.99,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#F59E0B",
        bbox={"facecolor": "#111827", "alpha": 0.72, "pad": 3, "edgecolor": "none"},
    )
    attribution_text = ax.text(
        0.01,
        0.01,
        attribution,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color="#E5E7EB",
        bbox={"facecolor": "#111827", "alpha": 0.66, "pad": 2, "edgecolor": "none"},
    )
    attribution_text.set_text(attribution)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_axis_off()

    scenario = result["scenario"]
    city_name = result["city_name"]
    tick_minutes = result["tick_minutes"]

    def _frame_text(frame_index: int) -> tuple[str, str]:
        tick = ticks[frame_index]
        wall_minutes = tick * tick_minutes
        scenario_name = scenario["name"]
        title_text = f"{city_name} - {scenario_name} traffic flow"
        event_active = _in_event(tick, scenario)
        subtitle_text = f"tick={tick}  |  t={wall_minutes} min"
        if event_active:
            subtitle_text += "  |  event active"
        return title_text, subtitle_text

    def _update(frame_index: int) -> tuple[Any, ...]:
        line_collection.set_array(umat[frame_index])
        title_text, subtitle_text = _frame_text(frame_index)
        title.set_text(title_text)
        subtitle.set_text(subtitle_text)
        subtitle.set_color("#F59E0B" if _in_event(ticks[frame_index], scenario) else "#E5E7EB")
        return line_collection, title, subtitle

    animation = FuncAnimation(
        fig,
        _update,
        frames=len(ticks),
        interval=max(80, 1000 // max(fps, 1)),
        blit=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

    return {
        "frames": len(ticks),
        "ticks": [int(t) for t in ticks],
        "vmax": float(max(vmax, 0.1)),
        "edge_scope": edge_scope,
    }


def run_warmup_bottleneck_scan(
    graph: Any,
    node_gdf: gpd.GeoDataFrame,
    edge_gdf: gpd.GeoDataFrame,
    demand_table: pd.DataFrame,
    attractor_table: pd.DataFrame,
    args: argparse.Namespace,
) -> list[tuple[int, int]]:
    warmup_cfg = SimulationConfig(
        city_name=args.city,
        n_agents=args.n_agents,
        n_ticks=args.replan_interval + 5,
        tick_minutes=args.tick_minutes,
        random_seed=args.random_seed,
        sample_every=1,
        position_sample_size=0,
        replan_interval=args.replan_interval,
        capacity_scale=args.capacity_scale,
        congestion_alpha=args.congestion_alpha,
        congestion_beta=args.congestion_beta,
        diurnal_enabled=False,
        sim_start_hour=args.sim_start_hour,
    )
    warmup_result = run_simulation(
        graph,
        node_gdf,
        edge_gdf,
        demand_table,
        attractor_table,
        warmup_cfg,
        {"name": "warmup", "events": []},
    )
    post_replan_ticks = [tick for tick in warmup_result["edge_history"] if tick >= args.replan_interval]
    aggregate_loads: dict[tuple[int, int], float] = {}
    for tick in post_replan_ticks:
        for edge, load in warmup_result["edge_history"][tick].items():
            aggregate_loads[edge] = aggregate_loads.get(edge, 0.0) + float(load)

    ranked: list[tuple[tuple[int, int], float, str]] = []
    for (u, v), load in aggregate_loads.items():
        if not graph.has_edge(u, v):
            continue
        road_class = graph[u][v].get("road_class", "other")
        if road_class not in _MAJOR_ROAD_CLASSES:
            continue
        ranked.append(((u, v), load, road_class))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [edge for edge, _, _ in ranked[:8]]


def _discover_network_manifest(network_output_root: Path) -> Path:
    manifests = sorted(network_output_root.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No network manifest found under {network_output_root}")
    if len(manifests) > 1:
        raise ValueError(f"Expected one network manifest under {network_output_root}, found {len(manifests)}.")
    return manifests[0]


def inspect_city(args: argparse.Namespace) -> None:
    project_root = _resolve_project_root(args.project_root)
    output_dir = (project_root / args.output_root / slugify(args.city)).resolve()
    roi_dir = output_dir / "roi"
    network_root = output_dir / "network"
    network_manifest = next(iter(sorted(network_root.glob("*/manifest.json"))), None)
    payload = {
        "city": args.city,
        "project_root": str(project_root),
        "output_dir": str(output_dir),
        "roi_manifest_exists": (roi_dir / "roi_manifest.json").exists(),
        "network_manifest": str(network_manifest) if network_manifest else None,
        "basemap_exists": (output_dir / "basemap" / "warped_basemap.png").exists(),
        "animation_exists": any((output_dir / "animation").glob("*.gif")),
    }
    print(json.dumps(payload, indent=2))


def run_pipeline(args: argparse.Namespace) -> None:
    project_root = _resolve_project_root(args.project_root)
    city_slug = slugify(args.city)
    output_dir = (project_root / args.output_root / city_slug).resolve()
    roi_dir = output_dir / "roi"
    network_output_root = output_dir / "network"
    basemap_dir = output_dir / "basemap"
    simulation_dir = output_dir / "simulation"
    animation_dir = output_dir / "animation"
    tile_cache_dir = (project_root / args.tile_cache_dir).resolve()

    print(f"Project root: {project_root}")
    print(f"City: {args.city}")
    print(f"Output dir: {output_dir}")

    print("\nPreparing ROI artifacts ...")
    roi_manifest_path = roi_dir / "roi_manifest.json"
    roi_vector_path = roi_dir / f"ucdb_rois_buffer{args.buffer_meters // 1000}km.gpkg"
    if args.force_roi or not roi_manifest_path.exists() or not roi_vector_path.exists():
        ucdb = load_ucdb_layer((project_root / args.ucdb).resolve(), args.layer)
        matched = resolve_city_matches(
            ucdb,
            [RequestedCity(order=1, requested_name=args.city)],
        )
        write_roi_outputs(
            baseline_path=(project_root / args.baseline).resolve(),
            matched_rois=matched,
            output_dir=roi_dir,
            buffer_meters=args.buffer_meters,
        )
    print(f"  ROI manifest: {roi_manifest_path}")

    print("\nBuilding hotspot/network artifacts ...")
    network_pipeline = NetworkHotspotPipeline(
        NetworkHotspotConfig(
            roi_manifest_path=roi_manifest_path,
            roi_vector_path=roi_vector_path,
            output_root=network_output_root,
            cache_dir=(project_root / args.osmnx_cache_dir).resolve(),
            network_type=args.network_type,
        )
    )
    network_manifest_path = None
    if args.force_network:
        payload = network_pipeline.build(args.city, project_root=project_root)
        network_manifest_path = Path(payload["outputs"]["manifest_json"]).resolve()
    else:
        try:
            network_manifest_path = _discover_network_manifest(network_output_root)
        except FileNotFoundError:
            payload = network_pipeline.build(args.city, project_root=project_root)
            network_manifest_path = Path(payload["outputs"]["manifest_json"]).resolve()
    print(f"  Network manifest: {network_manifest_path}")

    print("\nLoading network bundle ...")
    bundle = _load_network_bundle(network_manifest_path, roi_vector_path, args.city)
    graphml_path = Path(bundle["manifest"]["outputs"]["graphml"]).resolve()
    roi = bundle["roi"].to_crs(MOLLWEIDE_CRS)

    print("Building routing graph ...")
    routing_graph, routing_nodes, routing_edges = build_routing_graph(graphml_path)

    print("\nBuilding demand tables ...")
    full_demand_table, attractor_table, demand_summary = build_hotspot_demand_table(
        routing_graph,
        bundle["node_hotspot_demand"],
        attractor_top_k=args.attractor_top_k,
    )
    print(demand_summary.to_string())

    high_bet_edges: list[tuple[int, int]] | None = None
    if args.scenario != "baseline":
        print(f"\nWarmup ({args.n_agents} agents, {args.replan_interval + 5} ticks) ...")
        high_bet_edges = run_warmup_bottleneck_scan(
            routing_graph,
            routing_nodes,
            routing_edges,
            full_demand_table,
            attractor_table,
            args,
        )

    scenario_catalog = build_default_scenarios(
        routing_edges,
        attractor_table,
        n_ticks=args.n_ticks,
        high_betweenness_edges=high_bet_edges,
    )
    if args.scenario == "baseline":
        scenario = scenario_catalog["baseline"]
    else:
        scenario = scenario_catalog[args.scenario]

    print(f"\nRunning scenario {scenario['name']!r} ...")
    sim_config = SimulationConfig(
        city_name=args.city,
        n_agents=args.n_agents,
        n_ticks=args.n_ticks,
        tick_minutes=args.tick_minutes,
        random_seed=args.random_seed,
        sample_every=args.sample_every,
        position_sample_size=0,
        replan_interval=args.replan_interval,
        capacity_scale=args.capacity_scale,
        congestion_alpha=args.congestion_alpha,
        congestion_beta=args.congestion_beta,
        diurnal_enabled=not args.no_diurnal,
        sim_start_hour=args.sim_start_hour,
    )
    scenario_result = run_simulation(
        routing_graph,
        routing_nodes,
        routing_edges,
        full_demand_table,
        attractor_table,
        sim_config,
        scenario,
    )
    summary = simulation_summary(scenario_result)
    print("\nSummary:")
    print(summary.to_string())

    simulation_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = simulation_dir / f"metrics_{scenario['name']}.csv"
    summary_path = simulation_dir / f"summary_{scenario['name']}.json"
    scenario_result["metrics"].to_csv(metrics_path, index=False)
    summary_path.write_text(json.dumps(_json_ready(summary.to_dict()), indent=2))

    print("\nBuilding warped OSM basemap ...")
    basemap_artifact = build_warped_basemap(
        roi_bounds_mollweide=tuple(bundle["manifest"]["roi_bounds_mollweide"]),
        basemap_dir=basemap_dir,
        provider_name=args.tile_provider,
        cache_dir=tile_cache_dir,
        pad_fraction=args.pad_fraction,
        long_side_px=args.basemap_long_side_px,
        max_tiles=args.max_tiles,
        tile_zoom=args.tile_zoom,
        force=args.force_basemap,
    )
    basemap_rgba = load_basemap_artifact(basemap_artifact.image_path)

    print("\nExporting animation ...")
    animation_path = animation_dir / f"traffic_flow_{scenario['name']}.gif"
    animation_meta = save_single_map_animation(
        result=scenario_result,
        roi_gdf=roi,
        basemap_rgba=basemap_rgba,
        basemap_extent=basemap_artifact.extent_mollweide,
        attribution=basemap_artifact.attribution,
        output_path=animation_path,
        edge_scope=args.edge_scope,
        fps=args.fps,
    )

    run_manifest = {
        "city": args.city,
        "city_slug": city_slug,
        "scenario": scenario["name"],
        "project_root": str(project_root),
        "output_dir": str(output_dir),
        "roi_manifest": str(roi_manifest_path),
        "roi_vector": str(roi_vector_path),
        "network_manifest": str(network_manifest_path),
        "graphml": str(graphml_path),
        "metrics_csv": str(metrics_path),
        "summary_json": str(summary_path),
        "animation_gif": str(animation_path),
        "basemap": {
            "image_path": str(basemap_artifact.image_path),
            "metadata_path": str(basemap_artifact.metadata_path),
            "provider_name": basemap_artifact.provider_name,
            "attribution": basemap_artifact.attribution,
            "tile_range": asdict(basemap_artifact.tile_range),
            "extent_mollweide": [float(v) for v in basemap_artifact.extent_mollweide],
        },
        "animation": animation_meta,
        "simulation": {
            "n_agents": args.n_agents,
            "n_ticks": args.n_ticks,
            "tick_minutes": args.tick_minutes,
            "random_seed": args.random_seed,
            "sample_every": args.sample_every,
            "replan_interval": args.replan_interval,
            "attractor_top_k": args.attractor_top_k,
            "capacity_scale": args.capacity_scale,
            "congestion_alpha": args.congestion_alpha,
            "congestion_beta": args.congestion_beta,
            "diurnal_enabled": not args.no_diurnal,
            "sim_start_hour": args.sim_start_hour,
        },
    }
    run_manifest_path = output_dir / "run_manifest.json"
    run_manifest_path.write_text(json.dumps(_json_ready(run_manifest), indent=2))
    print(f"\nRun manifest: {run_manifest_path}")
    print(f"Animation: {animation_path}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ROI->network->simulation pipeline and render one map-overlay traffic animation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect expected output locations for a city.")
    run_parser = subparsers.add_parser("run", help="Run the downstream pipeline and export one animation.")

    for subparser in (inspect_parser, run_parser):
        subparser.add_argument("--city", required=True)
        subparser.add_argument("--project-root", type=_path_arg, default=None)
        subparser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    run_parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    run_parser.add_argument("--ucdb", type=Path, default=DEFAULT_UCDB_PATH)
    run_parser.add_argument("--layer", default=DEFAULT_LAYER)
    run_parser.add_argument("--buffer-meters", type=int, default=DEFAULT_BUFFER_METERS)
    run_parser.add_argument("--network-type", default="drive")
    run_parser.add_argument("--osmnx-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    run_parser.add_argument("--tile-cache-dir", type=Path, default=Path("data/cache/tiles"))
    run_parser.add_argument("--tile-provider", default=DEFAULT_PROVIDER)
    run_parser.add_argument("--tile-zoom", type=int, default=None)
    run_parser.add_argument("--max-tiles", type=int, default=64)
    run_parser.add_argument("--pad-fraction", type=float, default=0.05)
    run_parser.add_argument("--basemap-long-side-px", type=int, default=1600)
    run_parser.add_argument("--edge-scope", choices=["major", "all"], default="major")
    run_parser.add_argument("--fps", type=int, default=5)
    run_parser.add_argument("--scenario", choices=["baseline", "capacity_drop", "edge_closure", "hotspot_surge"], default="capacity_drop")
    run_parser.add_argument("--n-agents", type=int, default=1500)
    run_parser.add_argument("--n-ticks", type=int, default=80)
    run_parser.add_argument("--tick-minutes", type=int, default=6)
    run_parser.add_argument("--random-seed", type=int, default=42)
    run_parser.add_argument("--sample-every", type=int, default=2)
    run_parser.add_argument("--attractor-top-k", type=int, default=150)
    run_parser.add_argument("--replan-interval", type=int, default=10)
    run_parser.add_argument("--capacity-scale", type=float, default=0.50)
    run_parser.add_argument("--congestion-alpha", type=float, default=1.8)
    run_parser.add_argument("--congestion-beta", type=float, default=2.8)
    run_parser.add_argument("--no-diurnal", action="store_true")
    run_parser.add_argument("--sim-start-hour", type=float, default=6.0)
    run_parser.add_argument("--force-roi", action="store_true")
    run_parser.add_argument("--force-network", action="store_true")
    run_parser.add_argument("--force-basemap", action="store_true")

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.command == "inspect":
        inspect_city(args)
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()
