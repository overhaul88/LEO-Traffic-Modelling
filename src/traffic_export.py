#!/usr/bin/env python3
"""
src/traffic_export.py — Synthetic traffic dataset export.

Turns the transient per-tick output of the agent simulation
(``UrbanFlowModel.edge_history`` — ``dict[tick] -> {(u, v): load}``) into a
persisted, reusable *data product*: a grid-aligned spatio-temporal traffic
tensor registered to the **same Mollweide 1 km grid as the input population
rasters**.  This is the deliverable that lets the project actually "generate
synthetic traffic data for any area", and it is the input the downstream
ConvLSTM hotspot forecaster (``src/forecast_hotspots.py``) consumes.

For each sampled tick the per-edge agent crossings are aggregated to the grid
cell containing the edge centroid, producing a ``(T, C, H, W)`` float32 tensor
with two channels:

    channel 0  volume        — total agent-crossings summed per cell per tick
    channel 1  utilization   — cell volume / cell capacity-per-tick

The grid geometry (transform / shape / CRS) is read from the ROI-clipped
baseline raster referenced by the network manifest, so the tensor is pixel-
aligned with ``data/processed/baseline_*`` and the per-city ROI rasters.

Outputs (under ``<output_dir>/dataset/``):

    traffic_tensor_<scenario>.npy   (T, C, H, W) float32  — ML-ready
    traffic_tensor_<scenario>.tif   multiband GeoTIFF (volume, one band/tick)
    dataset_manifest.json           grid + temporal metadata, channel names
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import Affine, rowcol

CHANNEL_NAMES = ['volume', 'utilization']
NODATA = -1.0


# ── grid geometry ──────────────────────────────────────────────────────────────

class ROIGrid:
    """Mollweide grid (transform/shape/CRS) the traffic tensor is aligned to."""

    def __init__(self, transform: Affine, height: int, width: int, crs: Any):
        self.transform = transform
        self.height    = int(height)
        self.width     = int(width)
        self.crs       = crs

    @classmethod
    def from_raster(cls, raster_path: str | Path) -> 'ROIGrid':
        with rasterio.open(raster_path) as src:
            return cls(src.transform, src.height, src.width, src.crs)

    def rowcol(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised (x, y) Mollweide → (row, col) integer pixel indices."""
        rows, cols = rowcol(self.transform, xs, ys, op=np.floor)
        return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def roi_grid_from_bundle(bundle: dict) -> ROIGrid:
    """Resolve the ROI grid from a city bundle's network manifest."""
    baseline_path = bundle['manifest']['baseline_path']
    return ROIGrid.from_raster(baseline_path)


# ── edge → cell precompute ──────────────────────────────────────────────────────

def _edge_cell_map(
    edge_gdf,
    grid: ROIGrid,
    cap_per_tick: np.ndarray | None = None,
) -> tuple[dict[tuple[int, int], int], np.ndarray]:
    """
    Map each (u, v) edge to the flat cell index (row * W + col) of its centroid.

    Returns (uv_to_cell, cell_capacity) where uv_to_cell skips out-of-bounds
    edges (graph may extend past the ROI), and cell_capacity[flat] is the summed
    per-tick capacity of all in-bounds edges in that cell (for the utilization
    channel), clipped to a small positive floor.
    """
    centroids = edge_gdf.geometry.centroid
    xs = np.asarray(centroids.x.values, dtype=float)
    ys = np.asarray(centroids.y.values, dtype=float)
    rows, cols = grid.rowcol(xs, ys)

    in_bounds = (
        (rows >= 0) & (rows < grid.height) &
        (cols >= 0) & (cols < grid.width)
    )
    flat_all = rows * grid.width + cols

    us = edge_gdf['u'].to_numpy()
    vs = edge_gdf['v'].to_numpy()

    n_cells      = grid.height * grid.width
    cell_capacity = np.zeros(n_cells, dtype=np.float64)
    uv_to_cell: dict[tuple[int, int], int] = {}
    for i in range(len(edge_gdf)):
        if not in_bounds[i]:
            continue
        flat = int(flat_all[i])
        uv_to_cell[(int(us[i]), int(vs[i]))] = flat
        if cap_per_tick is not None:
            cell_capacity[flat] += float(cap_per_tick[i])

    np.maximum(cell_capacity, 0.1, out=cell_capacity)
    return uv_to_cell, cell_capacity


# ── tensor construction ─────────────────────────────────────────────────────────

def build_grid_tensor(result: dict, grid: ROIGrid) -> tuple[np.ndarray, list[int]]:
    """
    Build a (T, C, H, W) float32 tensor from result['edge_history'].

    T = number of sampled ticks, C = len(CHANNEL_NAMES), (H, W) = ROI grid.
    """
    edge_gdf = result['edge_gdf']
    tph      = 60.0 / result['tick_minutes']
    cs       = result['capacity_scale']
    cap_per_tick = (edge_gdf['base_capacity'].to_numpy(dtype=float) * cs / tph)

    uv_to_cell, cell_capacity = _edge_cell_map(edge_gdf, grid, cap_per_tick)

    ticks   = sorted(result['edge_history'].keys())
    n_cells = grid.height * grid.width
    volume  = np.zeros((len(ticks), n_cells), dtype=np.float64)

    for t_i, tick in enumerate(ticks):
        frame = volume[t_i]
        for (u, v), load in result['edge_history'][tick].items():
            cell = uv_to_cell.get((int(u), int(v)))
            if cell is not None:
                frame[cell] += float(load)

    util = volume / cell_capacity[None, :]

    T = len(ticks)
    tensor = np.empty((T, len(CHANNEL_NAMES), grid.height, grid.width), dtype=np.float32)
    tensor[:, 0] = volume.reshape(T, grid.height, grid.width).astype(np.float32)
    tensor[:, 1] = util.reshape(T, grid.height, grid.width).astype(np.float32)
    return tensor, ticks


# ── persistence ─────────────────────────────────────────────────────────────────

def _write_geotiff(path: Path, volume_thw: np.ndarray, grid: ROIGrid) -> None:
    """Write the volume channel as a multiband GeoTIFF (one band per tick)."""
    T = volume_thw.shape[0]
    profile = {
        'driver'   : 'GTiff',
        'width'    : grid.width,
        'height'   : grid.height,
        'count'    : max(T, 1),
        'dtype'    : 'float32',
        'crs'      : grid.crs,
        'transform': grid.transform,
        'nodata'   : NODATA,
        'tiled'    : True,
        'blockxsize': 256,
        'blockysize': 256,
        'compress' : 'lzw',
        'BIGTIFF'  : 'IF_SAFER',
    }
    with rasterio.open(path, 'w', **profile) as dst:
        for t in range(T):
            dst.write(volume_thw[t].astype(np.float32), t + 1)


def save_traffic_dataset(
    result: dict,
    bundle: dict,
    output_dir: Path,
    config: Any,
    grid: ROIGrid | None = None,
) -> dict[str, Any]:
    """
    Build and persist the synthetic traffic tensor for a single scenario run.

    Returns a small summary dict (paths + shape) and writes/updates
    ``dataset_manifest.json`` in ``<output_dir>/dataset/``.
    """
    grid = grid or roi_grid_from_bundle(bundle)
    scenario_name = result['scenario']['name']

    tensor, ticks = build_grid_tensor(result, grid)
    T, C, H, W = tensor.shape

    dataset_dir = Path(output_dir) / 'dataset'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    npy_path = dataset_dir / f'traffic_tensor_{scenario_name}.npy'
    tif_path = dataset_dir / f'traffic_tensor_{scenario_name}.tif'
    np.save(npy_path, tensor)
    _write_geotiff(tif_path, tensor[:, 0], grid)

    tick_minutes  = result['tick_minutes']
    start_hour    = getattr(config, 'sim_start_hour', 6.0)
    hours = [float((start_hour + t * tick_minutes / 60.0) % 24.0) for t in ticks]

    a, b, c, d, e, f = (grid.transform.a, grid.transform.b, grid.transform.c,
                        grid.transform.d, grid.transform.e, grid.transform.f)

    entry = {
        'scenario'    : scenario_name,
        'shape_TCHW'  : [int(T), int(C), int(H), int(W)],
        'channels'    : CHANNEL_NAMES,
        'ticks'       : [int(t) for t in ticks],
        'hour_of_day' : hours,
        'npy'         : str(npy_path.resolve()),
        'geotiff'     : str(tif_path.resolve()),
    }

    manifest_path = dataset_dir / 'dataset_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            'city'             : result.get('city_name'),
            'crs'              : str(grid.crs),
            'grid_height'      : int(H),
            'grid_width'       : int(W),
            'transform'        : [a, b, c, d, e, f],
            'tick_minutes'     : int(tick_minutes),
            'sim_start_hour'   : float(start_hour),
            'diurnal_enabled'  : bool(getattr(config, 'diurnal_enabled', False)),
            'n_agents'         : int(getattr(config, 'n_agents', 0)),
            'channel_names'    : CHANNEL_NAMES,
            'scenarios'        : {},
        }
    manifest['scenarios'][scenario_name] = entry
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f'  Saved traffic tensor [{scenario_name}] '
          f'{tensor.shape} -> {npy_path}')
    return {
        'npy'          : npy_path,
        'geotiff'      : tif_path,
        'manifest'     : manifest_path,
        'shape'        : (T, C, H, W),
    }
