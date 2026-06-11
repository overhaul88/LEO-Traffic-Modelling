#!/usr/bin/env python3
"""
src/agent_simulation.py — Agent-based urban traffic simulation.

Turns a saved hotspot-network graph into a traffic simulation: agents spawn at
hotspot-weighted road nodes, route to hotspot destinations, and dwell before
re-travelling.  Routing trees are rebuilt periodically using scipy Dijkstra
with BPR congestion costs.  Comparative runs (baseline vs. scenario) expose
the effect of policy shocks on throughput and edge utilisation.

Usage
-----
    # Inspect city bundle and scenario catalog without running
    python src/agent_simulation.py inspect --city Mumbai

    # Full run with defaults (saves plots + CSVs to data/outputs/simulation/)
    python src/agent_simulation.py run --city Mumbai

    # Custom run
    python src/agent_simulation.py run --city Mumbai \\
        --n-agents 1500 --n-ticks 80 --scenario capacity_drop \\
        --capacity-scale 0.50 --output-dir data/outputs/simulation/test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra
from shapely.geometry import LineString, Point

try:
    from src.network_hotspot_pipeline import NetworkHotspotPipeline
except ModuleNotFoundError:
    from network_hotspot_pipeline import NetworkHotspotPipeline

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
plt.rcParams['figure.dpi'] = 140
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11


# ── road-class tables ─────────────────────────────────────────────────────────

ROAD_CLASS_PRIORITY = [
    'motorway', 'motorway_link', 'trunk', 'trunk_link',
    'primary', 'primary_link', 'secondary', 'secondary_link',
    'tertiary', 'tertiary_link', 'residential', 'living_street',
    'service', 'unclassified',
]

BASE_CAPACITY_BY_CLASS: dict[str, float] = {
    'motorway': 220.0, 'motorway_link': 180.0, 'trunk': 180.0, 'trunk_link': 150.0,
    'primary': 140.0, 'primary_link': 120.0, 'secondary': 100.0, 'secondary_link': 85.0,
    'tertiary': 70.0, 'tertiary_link': 60.0, 'residential': 38.0, 'living_street': 24.0,
    'service': 20.0, 'unclassified': 30.0, 'other': 28.0,
}

ROAD_PENALTY_BY_CLASS: dict[str, float] = {
    'motorway': 1.00, 'motorway_link': 1.03, 'trunk': 1.04, 'trunk_link': 1.08,
    'primary': 1.10, 'primary_link': 1.14, 'secondary': 1.20, 'secondary_link': 1.24,
    'tertiary': 1.34, 'tertiary_link': 1.40, 'residential': 1.62, 'living_street': 1.80,
    'service': 1.95, 'unclassified': 1.55, 'other': 1.70,
}

PROFILE_LIBRARY: dict[str, dict] = {
    'commuter': {
        'share': 0.55, 'speed_kph': 28.0,
        'attraction_power': 1.05, 'dwell_mean_ticks': 8, 'dwell_std_ticks': 2,
    },
    'errand': {
        'share': 0.30, 'speed_kph': 22.0,
        'attraction_power': 0.95, 'dwell_mean_ticks': 4, 'dwell_std_ticks': 1,
    },
    'explorer': {
        'share': 0.15, 'speed_kph': 18.0,
        'attraction_power': 0.75, 'dwell_mean_ticks': 2, 'dwell_std_ticks': 1,
    },
}

_MAJOR_ROAD_CLASSES = {
    'motorway', 'motorway_link', 'trunk', 'trunk_link',
    'primary', 'primary_link', 'secondary', 'secondary_link',
}


# ── helper utilities ──────────────────────────────────────────────────────────

def _project_root_from_path(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / 'src').exists() and (p / 'data').exists():
            return p
    return start


def _highway_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    text = str(value).strip().lower()
    if not text or text == 'nan':
        return []
    text = text.strip('[]')
    result = []
    for part in text.split(','):
        token = part.strip().strip('"\'')
        if token:
            result.append(token)
    return result


def _canonical_road_class(value: Any) -> str:
    tokens = _highway_tokens(value)
    for candidate in ROAD_CLASS_PRIORITY:
        if candidate in tokens:
            return candidate
    return tokens[0] if tokens else 'other'


def _parse_lane_count(value: Any) -> float:
    if value is None:
        return 1.0
    if isinstance(value, (int, float)):
        return max(float(value), 1.0)
    text = str(value).strip().lower()
    if not text or text == 'nan':
        return 1.0
    parts = []
    for chunk in text.replace(';', ',').split(','):
        chunk = chunk.strip().strip('[]').strip('"\'')
        try:
            parts.append(float(chunk))
        except ValueError:
            continue
    return max(statistics.mean(parts), 1.0) if parts else 1.0


def _rounded(frame: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    out = frame.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].round(decimals)
    return out


# ── data loading ──────────────────────────────────────────────────────────────

def load_city_bundle(project_root: Path, city_name: str) -> dict:
    pipeline = NetworkHotspotPipeline()
    context  = pipeline.resolve_city_context(city_name, project_root=project_root)
    manifest = json.loads((context.output_dir / 'manifest.json').read_text())
    return {
        'context'            : context,
        'manifest'           : manifest,
        'hotspot_node_map'   : pd.read_csv(manifest['outputs']['hotspot_node_map_csv']),
        'node_hotspot_demand': pd.read_csv(manifest['outputs']['node_hotspot_demand_csv']),
        'cluster_summary'    : pd.read_csv(manifest['outputs']['cluster_summary_csv']),
        'roi'                : pipeline.load_roi_vectors(project_root=project_root).loc[
            lambda df: df['matched_name'] == city_name,
            ['matched_name', 'country', 'geometry'],
        ].copy(),
    }


# ── graph construction ────────────────────────────────────────────────────────

def build_routing_graph(
    graphml_path: Path | str,
) -> tuple[nx.DiGraph, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load GraphML, deduplicate multi-edges (keep cheapest), retain largest SCC.
    Returns (graph, node_gdf, edge_gdf) in Mollweide (ESRI:54009).
    """
    mollweide_crs = 'ESRI:54009'
    print('  Loading GraphML (one-time ~10 s) ...')
    t0    = time.perf_counter()
    G_ml  = ox.load_graphml(str(graphml_path))
    print(f'  Loaded in {time.perf_counter()-t0:.1f}s: '
          f'{G_ml.number_of_nodes()} nodes, {G_ml.number_of_edges()} edges')

    G_di = nx.DiGraph()
    for n, data in G_ml.nodes(data=True):
        x, y = float(data['x']), float(data['y'])
        G_di.add_node(int(n), x=x, y=y, geometry=Point(x, y))

    for u, v, data in G_ml.edges(data=True):
        u, v      = int(u), int(v)
        if u not in G_di or v not in G_di:
            continue
        road_class   = _canonical_road_class(data.get('highway'))
        lane_count   = _parse_lane_count(data.get('lanes'))
        length_m     = max(float(data.get('length', 1.0) or 1.0), 1.0)
        base_cap     = BASE_CAPACITY_BY_CLASS.get(road_class, BASE_CAPACITY_BY_CLASS['other']) * max(lane_count, 1.0)
        road_penalty = ROAD_PENALTY_BY_CLASS.get(road_class, ROAD_PENALTY_BY_CLASS['other'])
        base_cost    = length_m * road_penalty
        geom = data.get('geometry') or LineString(
            [G_di.nodes[u]['geometry'], G_di.nodes[v]['geometry']]
        )
        attrs = dict(
            length_m=length_m, road_class=road_class, lane_count=lane_count,
            base_capacity=base_cap, road_penalty=road_penalty,
            base_cost=base_cost, geometry=geom,
            highway=data.get('highway'), name=data.get('name'),
        )
        if G_di.has_edge(u, v):
            if base_cost < G_di[u][v]['base_cost']:
                G_di[u][v].update(attrs)
        else:
            G_di.add_edge(u, v, **attrs)

    largest_scc = max(nx.strongly_connected_components(G_di), key=len)
    graph = G_di.subgraph(largest_scc).copy()
    print(f'  SCC: {graph.number_of_nodes()} nodes '
          f'({100*graph.number_of_nodes()/G_di.number_of_nodes():.1f}%), '
          f'{graph.number_of_edges()} edges')

    node_rows = [
        {'node_id': int(nid), 'x': d['x'], 'y': d['y'], 'geometry': d['geometry']}
        for nid, d in graph.nodes(data=True)
    ]
    edge_rows = [
        {
            'u': int(u), 'v': int(v),
            'length_m': d['length_m'], 'road_class': d['road_class'],
            'lane_count': d['lane_count'], 'base_capacity': d['base_capacity'],
            'road_penalty': d['road_penalty'], 'base_cost': d['base_cost'],
            'geometry': d['geometry'],
        }
        for u, v, d in graph.edges(data=True)
    ]
    node_gdf = gpd.GeoDataFrame(node_rows, geometry='geometry', crs=mollweide_crs)
    edge_gdf = gpd.GeoDataFrame(edge_rows, geometry='geometry', crs=mollweide_crs)
    return graph, node_gdf, edge_gdf


# ── demand tables ─────────────────────────────────────────────────────────────

def build_hotspot_demand_table(
    graph: nx.DiGraph,
    node_hotspot_demand: pd.DataFrame,
    attractor_top_k: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Returns (full_table, attractor_table, summary).
    full_table      — all hotspot nodes in SCC, used for spawn origins.
    attractor_table — top-K by demand weight, used as destination candidates.
    """
    source = node_hotspot_demand.copy()
    source['node_id'] = source['node_id'].astype(int)
    source = (
        source[source['node_id'].isin(graph.nodes)]
        .sort_values('sum_weight_sum', ascending=False)
        .reset_index(drop=True)
    )
    total_w         = float(source['sum_weight_sum'].sum())
    source['weight'] = source['sum_weight_sum'] / total_w
    full_table = source.reset_index(drop=True)

    if attractor_top_k is not None:
        att = source.head(int(attractor_top_k)).copy()
        att['weight'] = att['sum_weight_sum'] / float(att['sum_weight_sum'].sum())
        attractor_table = att.reset_index(drop=True)
    else:
        attractor_table = full_table.copy()

    summary = pd.Series({
        'hotspot_nodes_in_scc'      : int(len(source)),
        'attractor_nodes'           : int(len(attractor_table)),
        'total_hotspot_weight'      : round(float(total_w), 3),
        'attractor_weight_coverage' : round(
            float(attractor_table['sum_weight_sum'].sum() / total_w), 3
        ),
    }, name='value')
    return full_table, attractor_table, summary


# ── scenario catalog ──────────────────────────────────────────────────────────

def build_default_scenarios(
    edge_gdf: gpd.GeoDataFrame,
    attractor_table: pd.DataFrame,
    n_ticks: int,
    high_betweenness_edges: list[tuple[int, int]] | None = None,
) -> dict[str, dict]:
    """
    Build the four default scenarios: baseline, capacity_drop, edge_closure,
    hotspot_surge.  When high_betweenness_edges is provided the first two
    scenarios target those edges (genuine routing bottlenecks); otherwise falls
    back to a capacity-heuristic selection.
    """
    focal_node = int(
        attractor_table.sort_values('sum_weight_sum', ascending=False).iloc[0]['node_id']
    )
    start_tick = max(4, n_ticks // 4)
    end_tick   = max(start_tick + 8, int(n_ticks * 0.75))

    if high_betweenness_edges and len(high_betweenness_edges) >= 4:
        rc_lookup = {
            (int(r.u), int(r.v)): r.road_class
            for r in edge_gdf.itertuples(index=False)
            if hasattr(r, 'road_class')
        }
        bet_art = [
            (u, v) for u, v in high_betweenness_edges
            if rc_lookup.get((u, v)) in _MAJOR_ROAD_CLASSES
        ]
        edge_keys    = (bet_art[:8] if len(bet_art) >= 4 else high_betweenness_edges[:8])
        chosen_class = rc_lookup.get(edge_keys[0], 'arterial') if edge_keys else 'arterial'
        source       = 'betweenness'
    else:
        arterials = edge_gdf[edge_gdf['road_class'].isin(_MAJOR_ROAD_CLASSES)].copy()
        if arterials.empty:
            arterials = edge_gdf.copy()
        top_art = None
        for tier in [
            {'trunk', 'trunk_link'}, {'primary', 'primary_link'},
            {'secondary', 'secondary_link'}, {'motorway', 'motorway_link'},
        ]:
            subset = arterials[
                arterials['road_class'].isin(tier) & (arterials['base_capacity'] >= 40.0)
            ]
            if len(subset) >= 4:
                top_art = (
                    subset.sort_values('base_capacity', ascending=True)
                    .drop_duplicates(subset=['u', 'v']).head(8)
                )
                break
        if top_art is None:
            top_art = arterials.sort_values('base_capacity', ascending=True).head(8)
        edge_keys    = [(int(r.u), int(r.v)) for r in top_art.itertuples(index=False)]
        chosen_class = top_art.iloc[0]['road_class']
        source       = 'capacity-heuristic'

    edge_key = edge_keys[0]
    return {
        'baseline': {
            'name': 'baseline',
            'description': 'No policy shock. Agents react only to endogenous congestion.',
            'events': [],
            '_edge_source': source,
        },
        'capacity_drop': {
            'name': 'capacity_drop',
            'description': (
                f'Capacity reduced to 25 % on {chosen_class} corridor '
                f'({source}, ticks {start_tick}–{end_tick}).'
            ),
            'events': [{
                'kind': 'capacity_drop', 'tick_start': start_tick, 'tick_end': end_tick,
                'edges': [list(e) for e in edge_keys], 'capacity_factor': 0.25,
            }],
            'highlight_edge': edge_key,
            '_edge_source': source,
        },
        'edge_closure': {
            'name': 'edge_closure',
            'description': (
                f'{chosen_class.title()} corridor closed entirely '
                f'({source}, ticks {start_tick}–{end_tick}).'
            ),
            'events': [{
                'kind': 'edge_closure', 'tick_start': start_tick, 'tick_end': end_tick,
                'edges': [list(e) for e in edge_keys[:3]],
            }],
            'highlight_edge': edge_key,
            '_edge_source': source,
        },
        'hotspot_surge': {
            'name': 'hotspot_surge',
            'description': (
                'Destination attraction surges 3× at the strongest hotspot node.'
            ),
            'events': [{
                'kind': 'hotspot_surge', 'tick_start': start_tick, 'tick_end': end_tick,
                'node_id': focal_node, 'multiplier': 3.0,
            }],
            'highlight_node': focal_node,
            '_edge_source': source,
        },
    }


def scenario_table(catalog: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {
            'scenario'   : name,
            'description': s['description'],
            'event_count': len(s['events']),
            'edge_source': s.get('_edge_source', ''),
        }
        for name, s in catalog.items()
    ])


# ── routing engine ────────────────────────────────────────────────────────────

class RoutingEngine:
    """
    Routing-tree engine backed by scipy.sparse.csgraph.dijkstra.

    Running Dijkstra from destination D on the transposed (reversed) graph
    yields predecessor arrays pred[v] = next hop from v toward D in the
    original graph.  next_hop() is O(1) after a rebuild.
    """

    def __init__(self, graph: nx.DiGraph, node_index_map: dict[int, int]):
        self.graph          = graph
        self.node_index_map = node_index_map
        self.index_node_map = {i: nid for nid, i in node_index_map.items()}
        self.n              = len(node_index_map)
        self._pred : dict[int, np.ndarray] = {}
        self._epoch = -1

    def rebuild(
        self,
        dyn_costs: dict[tuple[int, int], float],
        dest_node_ids: list[int],
        epoch: int,
    ) -> float:
        t0 = time.perf_counter()
        rows, cols, data = [], [], []
        for (u, v), cost in dyn_costs.items():
            ui = self.node_index_map.get(u)
            vi = self.node_index_map.get(v)
            if ui is None or vi is None:
                continue
            rows.append(ui); cols.append(vi); data.append(float(cost))

        if not data:
            self._pred  = {}
            self._epoch = epoch
            return 0.0

        adj   = csr_matrix((data, (rows, cols)), shape=(self.n, self.n))
        adj_T = adj.T.tocsr()

        valid      = [(d, self.node_index_map[d]) for d in dest_node_ids if d in self.node_index_map]
        if not valid:
            self._pred  = {}
            self._epoch = epoch
            return 0.0

        dest_ids = [d for d, _ in valid]
        dest_idx = np.array([i for _, i in valid], dtype=np.int32)

        _, pred_matrix = scipy_dijkstra(
            adj_T, indices=dest_idx, directed=True, return_predecessors=True
        )
        self._pred  = {did: pred_matrix[k] for k, did in enumerate(dest_ids)}
        self._epoch = epoch
        return time.perf_counter() - t0

    def next_hop(self, current_node: int, dest_node: int) -> int | None:
        if current_node == dest_node:
            return None
        pred_arr = self._pred.get(dest_node)
        if pred_arr is None:
            return None
        curr_idx = self.node_index_map.get(current_node)
        if curr_idx is None:
            return None
        pred_idx = int(pred_arr[curr_idx])
        if pred_idx < 0:
            return None
        return self.index_node_map.get(pred_idx)


# ── dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class TrafficProfile:
    name            : str
    share           : float
    speed_kph       : float
    attraction_power: float
    dwell_mean_ticks: int
    dwell_std_ticks : int

    def meters_per_tick(self, tick_minutes: int) -> float:
        return self.speed_kph * 1000.0 / 60.0 * tick_minutes


@dataclass
class DiurnalProfile:
    """
    Time-of-day activity curve λ(hour) ∈ [0, 1] driving how likely an agent is
    to (re-)depart on a given tick.  Default is a bimodal AM/PM commute rhythm
    (two Gaussians over a low overnight floor), normalised so the daily peak is
    1.0.  ``activity_at`` is the only thing the model needs.
    """
    am_peak_hour : float = 8.0
    pm_peak_hour : float = 18.0
    am_width_h   : float = 2.2
    pm_width_h   : float = 2.6
    pm_strength  : float = 1.10   # PM rush slightly stronger than AM
    night_floor  : float = 0.06   # minimum activity at 03:00-ish

    def _raw(self, hour: float) -> float:
        am = math.exp(-0.5 * ((hour - self.am_peak_hour) / self.am_width_h) ** 2)
        pm = self.pm_strength * math.exp(
            -0.5 * ((hour - self.pm_peak_hour) / self.pm_width_h) ** 2
        )
        return self.night_floor + am + pm

    def activity_at(self, hour: float) -> float:
        hour = hour % 24.0
        peak = self.night_floor + max(1.0, self.pm_strength)
        return max(0.0, min(1.0, self._raw(hour) / peak))


@dataclass
class SimulationConfig:
    city_name            : str
    n_agents             : int
    n_ticks              : int
    tick_minutes         : int
    random_seed          : int
    sample_every         : int
    position_sample_size : int
    replan_interval      : int
    capacity_scale       : float = 0.50
    congestion_alpha     : float = 1.8
    congestion_beta      : float = 2.8
    diurnal_enabled      : bool  = True
    sim_start_hour       : float = 6.0

    def hour_at_tick(self, tick: int) -> float:
        """Wall-clock hour-of-day for a given tick."""
        return (self.sim_start_hour + tick * self.tick_minutes / 60.0) % 24.0


@dataclass
class TrafficAgent:
    unique_id        : int
    profile_name     : str
    current_node     : int
    destination_node : int   = None
    state            : str   = 'travelling'
    dwell_remaining  : int   = 0
    next_node        : int   = None
    edge_progress_m  : float = 0.0
    travel_ticks            : int   = 0
    trips_completed         : int   = 0
    total_distance_m        : float = 0.0
    total_dwell_ticks       : int   = 0
    trip_tick_counter       : int   = 0
    reroutes                : int   = 0
    completed_trip_durations: list  = field(default_factory=list)


# ── data collector ────────────────────────────────────────────────────────────

class _DataCollector:
    """Minimal model-reporter data collector (no Mesa dependency)."""

    def __init__(self, model_reporters: dict | None = None):
        self._reporters = model_reporters or {}
        self._records: list[dict] = []

    def collect(self, model: 'UrbanFlowModel') -> None:
        row = {}
        for name, reporter in self._reporters.items():
            row[name] = reporter(model) if callable(reporter) else getattr(model, reporter)
        self._records.append(row)

    def get_model_vars_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)


# ── simulation model ──────────────────────────────────────────────────────────

class UrbanFlowModel:
    """
    Agent-based urban traffic model.

    Agents cycle: spawn → travel (routing-tree guided) → arrive → dwell → travel.
    Routing trees are rebuilt at each plan epoch (every replan_interval ticks and
    at scenario event boundaries) so congestion and policy shocks affect routing.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        node_gdf: gpd.GeoDataFrame,
        edge_gdf: gpd.GeoDataFrame,
        demand_table: pd.DataFrame,
        attractor_table: pd.DataFrame,
        config: SimulationConfig,
        scenario: dict,
        profiles: dict[str, TrafficProfile],
    ):
        self.graph           = graph
        self.node_gdf        = node_gdf
        self.edge_gdf        = edge_gdf
        self.demand_table    = demand_table.reset_index(drop=True)
        self.attractor_table = attractor_table.reset_index(drop=True)
        self.config          = config
        self.scenario        = scenario
        self.profiles        = profiles
        self.rng             = random.Random(config.random_seed)
        self.diurnal         = DiurnalProfile()

        node_id_list         = list(graph.nodes())
        self.node_index_map  = {nid: i for i, nid in enumerate(node_id_list)}
        self.index_node_map  = {i: nid for nid, i in self.node_index_map.items()}

        self.routing_engine  = RoutingEngine(graph, self.node_index_map)

        self.tick              = 0
        self.plan_epoch        = 0
        self.agents: list[TrafficAgent] = []
        self.edge_loads        = Counter()
        self.prev_edge_loads   = Counter()
        self.dyn_costs: dict   = {}
        self.route_queries_this_tick  = 0
        self.route_failures_this_tick = 0
        self.completed_trip_durations: list[int] = []

        self._replan_ticks = set(range(0, config.n_ticks + 1, config.replan_interval))
        self._replan_ticks.add(0)
        for event in scenario.get('events', []):
            self._replan_ticks.add(event['tick_start'])
            self._replan_ticks.add(event['tick_end'])

        self._spawn_nodes      = self.demand_table['node_id'].astype(int).tolist()
        self._spawn_weights    = self.demand_table['weight'].to_numpy(dtype=float)
        self._att_nodes        = self.attractor_table['node_id'].astype(int).tolist()
        self._att_weights_base = self.attractor_table['weight'].to_numpy(dtype=float)
        self._profile_names    = list(profiles.keys())
        self._profile_weights  = [profiles[n].share for n in self._profile_names]

        self.position_history: dict[int, gpd.GeoDataFrame] = {}
        self.edge_history: dict[int, dict]                  = {}

        self.data_collector = _DataCollector(model_reporters={
            'tick'            : lambda m: m.tick,
            'hour'            : lambda m: m.config.hour_at_tick(m.tick),
            'diurnal_lambda'  : lambda m: m._diurnal_lambda(),
            'active_travelers': lambda m: sum(a.state == 'travelling' for a in m.agents),
            'dwellers'        : lambda m: sum(a.state == 'dwelling'   for a in m.agents),
            'completed_trips' : lambda m: sum(a.trips_completed        for a in m.agents),
            'route_queries'   : lambda m: m.route_queries_this_tick,
            'route_failures'  : lambda m: m.route_failures_this_tick,
            'mean_trip_time'  : lambda m: (
                float(np.mean(m.completed_trip_durations))
                if m.completed_trip_durations else float('nan')
            ),
            'mean_distance_m' : lambda m: (
                float(np.mean([a.total_distance_m for a in m.agents]))
                if m.agents else float('nan')
            ),
            'max_utilization' : lambda m: m._max_utilization(),
            'mean_utilization': lambda m: m._mean_utilization(),
            'overloaded_edges': lambda m: sum(v > 1.0 for v in m._utilization_values()),
        })

        self._update_dyn_costs()
        elapsed = self.routing_engine.rebuild(self.dyn_costs, self._att_nodes, self.plan_epoch)
        self.plan_epoch = 1
        print(f'  Initial routing trees ({len(self._att_nodes)} dests): {elapsed:.2f}s')
        self._spawn_agents()
        self._capture_snapshot(force=True)
        self.data_collector.collect(self)

    # ── event helpers ─────────────────────────────────────────────────────────

    def _active_events(self) -> list[dict]:
        return [
            e for e in self.scenario.get('events', [])
            if e['tick_start'] <= self.tick < e['tick_end']
        ]

    @staticmethod
    def _event_edges(event: dict) -> set[tuple[int, int]]:
        if 'edges' in event:
            return {(int(e[0]), int(e[1])) for e in event['edges']}
        if 'edge' in event:
            return {(int(event['edge'][0]), int(event['edge'][1]))}
        return set()

    # ── dynamic cost computation ───────────────────────────────────────────────

    def _update_dyn_costs(self) -> None:
        tph          = 60.0 / self.config.tick_minutes
        cap_factors  : dict[tuple, float] = {}
        closed_edges : set[tuple]         = set()

        for event in self._active_events():
            if event['kind'] == 'capacity_drop':
                for e in self._event_edges(event):
                    cap_factors[e] = cap_factors.get(e, 1.0) * event.get('capacity_factor', 1.0)
            elif event['kind'] == 'edge_closure':
                closed_edges.update(self._event_edges(event))

        cs    = self.config.capacity_scale
        alpha = self.config.congestion_alpha
        beta  = self.config.congestion_beta

        self.dyn_costs = {}
        for u, v, d in self.graph.edges(data=True):
            edge = (u, v)
            if edge in closed_edges:
                continue
            eff_cap      = d['base_capacity'] * cap_factors.get(edge, 1.0)
            cap_per_tick = max(eff_cap * cs / tph, 0.1)
            load         = self.prev_edge_loads.get(edge, 0)
            util         = load / cap_per_tick
            congestion   = 1.0 + alpha * (max(util, 0.0) ** beta)
            self.dyn_costs[edge] = d['base_cost'] * congestion

    # ── utilisation metrics ────────────────────────────────────────────────────

    def _utilization_values(self) -> list[float]:
        tph  = 60.0 / self.config.tick_minutes
        cs   = self.config.capacity_scale
        vals = []
        for (u, v), load in self.prev_edge_loads.items():
            if self.graph.has_edge(u, v):
                cap_pt = max(self.graph[u][v]['base_capacity'] * cs / tph, 0.1)
                vals.append(load / cap_pt)
        return vals

    def _max_utilization(self) -> float:
        v = self._utilization_values()
        return max(v) if v else 0.0

    def _mean_utilization(self) -> float:
        v = self._utilization_values()
        return float(np.mean(v)) if v else 0.0

    # ── destination sampling ───────────────────────────────────────────────────

    def _destination_weights(
        self, profile_name: str, exclude_node: int | None = None
    ) -> tuple[list | None, list | None]:
        profile = self.profiles[profile_name]
        eff_w   = self._att_weights_base.copy()

        for event in self._active_events():
            if event['kind'] == 'hotspot_surge':
                surge_id = int(event['node_id'])
                mul      = float(event.get('multiplier', 1.0))
                for i, nid in enumerate(self._att_nodes):
                    if nid == surge_id:
                        eff_w[i] *= mul

        eff_w = eff_w ** profile.attraction_power
        if exclude_node is not None:
            excl  = int(exclude_node)
            eff_w = np.array([0.0 if nid == excl else w
                              for nid, w in zip(self._att_nodes, eff_w)])

        total = float(eff_w.sum())
        if total <= 0.0:
            return None, None
        return self._att_nodes, (eff_w / total).tolist()

    def _sample_destination(
        self, profile_name: str, exclude_node: int | None = None
    ) -> int | None:
        nodes, weights = self._destination_weights(profile_name, exclude_node)
        if nodes is None:
            return None
        return int(self.rng.choices(nodes, weights=weights, k=1)[0])

    # ── agent spawning ─────────────────────────────────────────────────────────

    def _spawn_agents(self) -> None:
        # Stagger first departures across the opening day so the active
        # population ramps with the diurnal curve instead of all 25k leaving at
        # t=0.  Agents begin dwelling with an offset uniform over a day; the
        # λ-gate in _step_dwelling then shapes the realised departures.
        ticks_per_day = max(1, round(24 * 60 / self.config.tick_minutes))
        stagger = self.config.diurnal_enabled
        for uid in range(self.config.n_agents):
            pname  = self.rng.choices(self._profile_names, weights=self._profile_weights, k=1)[0]
            origin = int(self.rng.choices(self._spawn_nodes, weights=self._spawn_weights, k=1)[0])
            dest   = self._sample_destination(pname, exclude_node=origin) or origin
            agent  = TrafficAgent(
                unique_id=uid, profile_name=pname,
                current_node=origin, destination_node=dest,
            )
            if stagger:
                agent.state           = 'dwelling'
                agent.dwell_remaining = self.rng.randint(1, ticks_per_day)
            self.agents.append(agent)

    # ── agent stepping ─────────────────────────────────────────────────────────

    def _diurnal_lambda(self) -> float:
        """Current departure-activity multiplier λ(t) ∈ [0, 1]."""
        if not self.config.diurnal_enabled:
            return 1.0
        return self.diurnal.activity_at(self.config.hour_at_tick(self.tick))

    def _dwell_ticks(self, profile_name: str) -> int:
        p = self.profiles[profile_name]
        return max(1, int(round(self.rng.gauss(p.dwell_mean_ticks, p.dwell_std_ticks))))

    def _mark_arrival(self, agent: TrafficAgent) -> None:
        agent.state           = 'dwelling'
        agent.trips_completed += 1
        agent.dwell_remaining  = self._dwell_ticks(agent.profile_name)
        agent.completed_trip_durations.append(agent.trip_tick_counter)
        self.completed_trip_durations.append(agent.trip_tick_counter)
        agent.trip_tick_counter = 0
        agent.next_node         = None
        agent.edge_progress_m   = 0.0

    def _step_dwelling(self, agent: TrafficAgent) -> None:
        agent.total_dwell_ticks += 1
        agent.dwell_remaining   -= 1
        if agent.dwell_remaining <= 0:
            # Diurnal gate: defer departure during low-activity hours so agents
            # accumulate overnight and release during the AM/PM peaks.
            if self.config.diurnal_enabled and self.rng.random() > self._diurnal_lambda():
                agent.dwell_remaining = 1
                return
            dest = self._sample_destination(agent.profile_name, exclude_node=agent.current_node)
            if dest is None:
                agent.dwell_remaining = 1
                return
            agent.destination_node = dest
            agent.state            = 'travelling'

    def _step_travelling(self, agent: TrafficAgent) -> None:
        if agent.destination_node is None:
            dest = self._sample_destination(agent.profile_name, exclude_node=agent.current_node)
            if dest is None:
                agent.travel_ticks += 1
                return
            agent.destination_node = dest

        profile       = self.profiles[agent.profile_name]
        budget        = profile.meters_per_tick(self.config.tick_minutes)
        failure_streak = 0

        while budget > 1e-6:
            if agent.current_node == agent.destination_node and agent.next_node is None:
                self._mark_arrival(agent)
                return

            if agent.next_node is None:
                nh = self.routing_engine.next_hop(agent.current_node, agent.destination_node)
                self.route_queries_this_tick += 1
                if nh is None:
                    self.route_failures_this_tick += 1
                    failure_streak += 1
                    if failure_streak > 8:
                        break
                    new_dest = self._sample_destination(
                        agent.profile_name, exclude_node=agent.current_node
                    )
                    if new_dest is None:
                        break
                    agent.reroutes        += 1
                    agent.destination_node = new_dest
                    continue
                failure_streak  = 0
                agent.next_node = nh
                agent.edge_progress_m = 0.0

            u, v = agent.current_node, agent.next_node
            if not self.graph.has_edge(u, v):
                agent.next_node = None
                agent.reroutes  += 1
                continue

            edge_len  = self.graph[u][v]['length_m']
            remaining = edge_len - agent.edge_progress_m

            if budget >= remaining:
                budget                -= remaining
                agent.total_distance_m += remaining
                self.edge_loads[(u, v)] += 1
                agent.current_node    = v
                agent.next_node       = None
                agent.edge_progress_m = 0.0
                if agent.current_node == agent.destination_node:
                    self._mark_arrival(agent)
                    return
            else:
                agent.edge_progress_m  += budget
                agent.total_distance_m += budget
                self.edge_loads[(u, v)] += 1
                budget = 0.0

        agent.travel_ticks     += 1
        agent.trip_tick_counter += 1

    # ── snapshot capture ───────────────────────────────────────────────────────

    def _agent_position(self, agent: TrafficAgent) -> Point:
        if agent.next_node is None:
            return self.graph.nodes[agent.current_node]['geometry']
        u, v = agent.current_node, agent.next_node
        if not self.graph.has_edge(u, v):
            return self.graph.nodes[agent.current_node]['geometry']
        geom     = self.graph[u][v]['geometry']
        edge_len = max(self.graph[u][v]['length_m'], 1e-6)
        frac     = min(max(agent.edge_progress_m / edge_len, 0.0), 1.0)
        return geom.interpolate(frac, normalized=True)

    def _capture_snapshot(self, force: bool = False) -> None:
        if (not force
                and self.tick % self.config.sample_every != 0
                and self.tick != self.config.n_ticks):
            return
        self.edge_history[self.tick] = dict(self.edge_loads)
        n_sample = min(self.config.position_sample_size, len(self.agents))
        if n_sample == 0:
            self.position_history[self.tick] = gpd.GeoDataFrame(
                columns=['tick', 'agent_id', 'profile_name', 'state',
                         'current_node', 'destination_node', 'geometry'],
                geometry='geometry', crs=self.node_gdf.crs,
            )
            return
        sampled = (
            self.rng.sample(self.agents, n_sample)
            if n_sample < len(self.agents) else list(self.agents)
        )
        rows = [
            {
                'tick': self.tick, 'agent_id': a.unique_id,
                'profile_name': a.profile_name, 'state': a.state,
                'current_node': a.current_node, 'destination_node': a.destination_node,
                'geometry': self._agent_position(a),
            }
            for a in sampled
        ]
        self.position_history[self.tick] = gpd.GeoDataFrame(
            rows, geometry='geometry', crs=self.node_gdf.crs
        )

    # ── main tick ──────────────────────────────────────────────────────────────

    def step(self) -> None:
        if self.tick in self._replan_ticks:
            self._update_dyn_costs()
            self.routing_engine.rebuild(self.dyn_costs, self._att_nodes, self.plan_epoch)
            self.plan_epoch += 1

        self.prev_edge_loads          = Counter(self.edge_loads)
        self.edge_loads               = Counter()
        self.route_queries_this_tick  = 0
        self.route_failures_this_tick = 0

        for agent in self.agents:
            if agent.state == 'dwelling':
                self._step_dwelling(agent)
            else:
                self._step_travelling(agent)

        self._capture_snapshot()
        self.data_collector.collect(self)
        self.tick += 1


# ── runner ────────────────────────────────────────────────────────────────────

def _build_profiles() -> dict[str, TrafficProfile]:
    return {name: TrafficProfile(name=name, **cfg) for name, cfg in PROFILE_LIBRARY.items()}


def run_simulation(
    graph: nx.DiGraph,
    node_gdf: gpd.GeoDataFrame,
    edge_gdf: gpd.GeoDataFrame,
    demand_table: pd.DataFrame,
    attractor_table: pd.DataFrame,
    config: SimulationConfig,
    scenario: dict,
) -> dict:
    print(f"\nScenario: {scenario['name']!r}")
    model = UrbanFlowModel(
        graph, node_gdf, edge_gdf,
        demand_table, attractor_table,
        config, scenario, _build_profiles(),
    )
    t0 = time.perf_counter()
    for i in range(config.n_ticks):
        model.step()
        if (i + 1) % 20 == 0:
            n_trips = sum(a.trips_completed for a in model.agents)
            n_dwell = sum(a.state == 'dwelling' for a in model.agents)
            print(f'  tick {i+1:3d}/{config.n_ticks}  '
                  f'completed={n_trips}  dwellers={n_dwell}  '
                  f'elapsed={time.perf_counter()-t0:.1f}s')
    print(f'  Finished in {time.perf_counter()-t0:.1f}s')

    metrics           = model.data_collector.get_model_vars_dataframe()
    metrics['scenario'] = scenario['name']

    # Release heavy per-run state that nothing downstream consumes.  The result
    # is held for the whole program (baseline_result stays alive while the
    # scenario model is built), so keeping these alive doubles peak memory and
    # OOM-kills large cities.  Plots/metrics only need the histories below.
    #   • routing_engine._pred — dense (n_dests × n_nodes) int32 predecessor
    #     matrix from scipy Dijkstra (~478 MB for 300 dests × 400k nodes).
    #   • agents — n_agents TrafficAgent objects (final stats already in metrics).
    model.routing_engine._pred = {}
    model.agents              = []
    model.edge_loads          = Counter()
    model.prev_edge_loads     = Counter()
    model.dyn_costs           = {}

    return {
        'model'           : model,
        'metrics'         : metrics,
        'edge_history'    : model.edge_history,
        'position_history': model.position_history,
        'edge_gdf'        : edge_gdf,
        'node_gdf'        : node_gdf,
        'scenario'        : scenario,
        'tick_minutes'    : config.tick_minutes,
        'capacity_scale'  : config.capacity_scale,
        'city_name'       : config.city_name,
        'config'          : config,
    }


def simulation_summary(result: dict) -> pd.Series:
    metrics = result['metrics']
    final   = metrics.iloc[-1]
    return pd.Series({
        'scenario'              : result['scenario']['name'],
        'ticks'                 : int(metrics['tick'].max()),
        'final_active_travelers': int(final['active_travelers']),
        'final_dwellers'        : int(final['dwellers']),
        'completed_trips'       : int(final['completed_trips']),
        'mean_trip_time_ticks'  : (
            round(float(final['mean_trip_time']), 2)
            if not math.isnan(float(final['mean_trip_time'])) else float('nan')
        ),
        'peak_max_utilization'  : round(float(metrics['max_utilization'].max()), 3),
        'peak_overloaded_edges' : int(metrics['overloaded_edges'].max()),
        'route_queries_total'   : int(metrics['route_queries'].sum()),
        'route_failures_total'  : int(metrics['route_failures'].sum()),
    }, name=result['scenario']['name'])


# ── visualisation helpers ─────────────────────────────────────────────────────

def _nearest_sampled_tick(history: dict, target_tick: int) -> int | None:
    available = sorted(history.keys())
    return min(available, key=lambda t: abs(t - target_tick)) if available else None


def _edge_snapshot_gdf(
    result: dict, tick: int, display_crs: str | None = None
) -> tuple[gpd.GeoDataFrame, int]:
    tick  = _nearest_sampled_tick(result['edge_history'], tick)
    tph   = 60.0 / result['tick_minutes']
    cs    = result['capacity_scale']
    frame = result['edge_gdf'][['u', 'v', 'road_class', 'base_capacity', 'geometry']].copy()
    load_dict = result['edge_history'].get(tick, {})
    if load_dict:
        load_rows = pd.DataFrame([
            {'u': int(u), 'v': int(v), 'edge_load': float(ld)}
            for (u, v), ld in load_dict.items()
        ])
        frame = frame.merge(load_rows, on=['u', 'v'], how='left')
        frame['edge_load'] = frame['edge_load'].fillna(0.0)
    else:
        frame['edge_load'] = 0.0
    frame['utilization'] = frame['edge_load'] / (
        (frame['base_capacity'] * cs).clip(lower=0.1) / tph
    )
    gdf = gpd.GeoDataFrame(frame, geometry='geometry', crs=result['edge_gdf'].crs)
    if display_crs:
        return gdf.to_crs(display_crs), tick
    return gdf, tick


def _position_snapshot_gdf(
    result: dict, tick: int, display_crs: str | None = None
) -> tuple[gpd.GeoDataFrame, int]:
    tick = _nearest_sampled_tick(result['position_history'], tick)
    gdf  = result['position_history'][tick].copy()
    if display_crs:
        return gdf.to_crs(display_crs), tick
    return gdf, tick


# ── flow visualisation ────────────────────────────────────────────────────────

_LW_BY_CLASS = {
    'motorway': 1.8, 'motorway_link': 1.1, 'trunk': 1.5, 'trunk_link': 1.0,
    'primary': 1.1, 'primary_link': 0.85, 'secondary': 0.75, 'secondary_link': 0.65,
    'tertiary': 0.55, 'tertiary_link': 0.48, 'residential': 0.38,
    'living_street': 0.32, 'service': 0.30, 'unclassified': 0.38, 'other': 0.32,
}
_CMAP_FLOW   = 'hot_r'
_CMAP_DIFF   = 'RdBu_r'
_CMAP_HEAT   = 'YlOrRd'
_EVENT_COLOR = '#FF8C00'


def _edge_segments(gdf: gpd.GeoDataFrame) -> list:
    segs = []
    for geom in gdf.geometry:
        try:
            if geom.geom_type == 'LineString':
                segs.append(np.array(geom.coords))
            elif geom.geom_type == 'MultiLineString':
                segs.append(np.array(geom.geoms[0].coords))
            else:
                segs.append(np.array([[0., 0.], [0., 0.]]))
        except Exception:
            segs.append(np.array([[0., 0.], [0., 0.]]))
    return segs


def _lw_array(gdf: gpd.GeoDataFrame) -> np.ndarray:
    return np.array([_LW_BY_CLASS.get(rc, 0.32) for rc in gdf['road_class']])


def _in_event(tick: int, scenario: dict) -> bool:
    return any(e['tick_start'] <= tick < e['tick_end']
               for e in scenario.get('events', []))


def build_edge_util_arrays(
    result: dict,
    display_crs: str | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, list, np.ndarray, np.ndarray, float]:
    """
    Pre-compute a (n_ticks × n_edges) float32 utilisation matrix.
    Returns (all_egdf, maj_egdf, ticks, all_umat, maj_umat, vmax).
    """
    tph  = 60.0 / result['tick_minutes']
    cs   = result['capacity_scale']
    egdf = result['edge_gdf'].copy()
    if display_crs:
        egdf = egdf.to_crs(display_crs)
    egdf = egdf.reset_index(drop=True)

    cap_arr     = (egdf['base_capacity'].values * cs / tph).clip(min=0.1)
    edge_to_idx = {(int(r.u), int(r.v)): i
                   for i, r in enumerate(egdf.itertuples(index=False))}

    ticks     = sorted(result['edge_history'].keys())
    umat      = np.zeros((len(ticks), len(egdf)), dtype=np.float32)
    for t_i, tick in enumerate(ticks):
        for (u, v), load in result['edge_history'][tick].items():
            idx = edge_to_idx.get((int(u), int(v)))
            if idx is not None:
                umat[t_i, idx] = load / cap_arr[idx]

    nz   = umat[umat > 0]
    vmax = float(np.percentile(nz, 95)) if len(nz) else 1.0

    mmask = egdf['road_class'].isin(_MAJOR_ROAD_CLASSES).values
    return (egdf,
            egdf[mmask].reset_index(drop=True),
            ticks,
            umat,
            umat[:, mmask],
            vmax)


# ── output: static plots ──────────────────────────────────────────────────────

def save_timeseries(
    baseline_result: dict,
    scenario_result: dict,
    scenario_catalog: dict,
    scenario_name: str,
    output_path: Path,
) -> None:
    import seaborn as sns
    sns.set_theme(style='whitegrid', context='talk')

    city_name    = baseline_result['city_name']
    tick_minutes = baseline_result['tick_minutes']
    n_agents     = baseline_result['config'].n_agents
    n_ticks      = baseline_result['config'].n_ticks

    b_metrics    = baseline_result['metrics'].copy()
    s_metrics    = scenario_result['metrics'].copy()
    metrics_long = pd.concat([b_metrics, s_metrics], ignore_index=True)

    fig, axes = plt.subplots(2, 3, figsize=(22, 10), constrained_layout=True)

    sns.lineplot(data=metrics_long, x='tick', y='active_travelers', hue='scenario', ax=axes[0, 0])
    axes[0, 0].set_title('Active Travelers')

    sns.lineplot(data=metrics_long, x='tick', y='dwellers', hue='scenario', ax=axes[0, 1])
    axes[0, 1].set_title('Dwellers (arrived & resting)')

    sns.lineplot(data=metrics_long, x='tick', y='completed_trips', hue='scenario', ax=axes[0, 2])
    axes[0, 2].set_title('Cumulative Completed Trips')

    sns.lineplot(data=metrics_long, x='tick', y='max_utilization', hue='scenario', ax=axes[1, 0])
    axes[1, 0].set_title('Peak Edge Utilisation')
    axes[1, 0].set_ylabel('load / cap-per-tick')

    sns.lineplot(data=metrics_long, x='tick', y='mean_trip_time', hue='scenario', ax=axes[1, 1])
    axes[1, 1].set_title('Mean Trip Time (ticks)')

    sns.lineplot(data=metrics_long, x='tick', y='overloaded_edges', hue='scenario', ax=axes[1, 2])
    axes[1, 2].set_title('Overloaded Edges per Tick')

    for event in scenario_catalog[scenario_name].get('events', []):
        for ax in axes.flatten():
            ax.axvline(event['tick_start'], color='orange', ls='--', lw=1, alpha=0.7)
            ax.axvline(event['tick_end'],   color='orange', ls=':',  lw=1, alpha=0.7)

    plt.suptitle(
        f'{city_name} — Agent Flow Simulation '
        f'({n_agents} agents, {n_ticks} ticks, {tick_minutes} min/tick)',
        fontsize=14,
    )
    fig.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_path}')


def save_spatial_peak(
    baseline_result: dict,
    scenario_result: dict,
    roi_gdf: gpd.GeoDataFrame,
    attractor_pts: gpd.GeoDataFrame,
    scenario_name: str,
    display_crs: str,
    output_path: Path,
) -> None:
    b_metrics = baseline_result['metrics']
    s_metrics = scenario_result['metrics']
    b_peak    = int(b_metrics.loc[b_metrics['max_utilization'].idxmax(), 'tick'])
    s_peak    = int(s_metrics.loc[s_metrics['max_utilization'].idxmax(), 'tick'])

    b_edges, b_tick = _edge_snapshot_gdf(baseline_result, b_peak, display_crs=display_crs)
    s_edges, s_tick = _edge_snapshot_gdf(scenario_result, s_peak, display_crs=display_crs)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9), constrained_layout=True)

    roi_gdf.boundary.plot(ax=axes[0], color='black', linewidth=1.0)
    b_edges.plot(ax=axes[0], column='utilization', cmap='viridis', linewidth=0.8,
                 legend=True, legend_kwds={'label': 'Utilisation (load/cap-per-tick)'})
    attractor_pts.plot(ax=axes[0], color='#E45756', markersize=12, alpha=0.8, zorder=5)
    axes[0].set_title(f'Baseline — edge utilisation at tick {b_tick}')
    axes[0].set_axis_off()

    roi_gdf.boundary.plot(ax=axes[1], color='black', linewidth=1.0)
    s_edges.plot(ax=axes[1], column='utilization', cmap='magma', linewidth=0.8,
                 legend=True, legend_kwds={'label': 'Utilisation (load/cap-per-tick)'})
    attractor_pts.plot(ax=axes[1], color='#E45756', markersize=12, alpha=0.8, zorder=5)
    axes[1].set_title(f'{scenario_name} — edge utilisation at tick {s_tick}')
    axes[1].set_axis_off()

    fig.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_path}')


def save_agent_positions(
    baseline_result: dict,
    scenario_result: dict,
    roi_gdf: gpd.GeoDataFrame,
    routing_edges: gpd.GeoDataFrame,
    scenario_name: str,
    display_crs: str,
    output_path: Path,
) -> None:
    b_metrics = baseline_result['metrics']
    s_metrics = scenario_result['metrics']
    b_peak    = int(b_metrics.loc[b_metrics['max_utilization'].idxmax(), 'tick'])
    s_peak    = int(s_metrics.loc[s_metrics['max_utilization'].idxmax(), 'tick'])

    b_pos, b_pos_tick = _position_snapshot_gdf(baseline_result, b_peak, display_crs=display_crs)
    s_pos, s_pos_tick = _position_snapshot_gdf(scenario_result, s_peak, display_crs=display_crs)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9), constrained_layout=True)

    for ax, pos, tick, label in [
        (axes[0], b_pos, b_pos_tick, 'Baseline'),
        (axes[1], s_pos, s_pos_tick, scenario_name),
    ]:
        roi_gdf.boundary.plot(ax=ax, color='black', linewidth=1.0)
        routing_edges.plot(ax=ax, color='#C7C7C7', linewidth=0.25, alpha=0.5)
        if not pos.empty:
            pos.plot(ax=ax, column='state', categorical=True, markersize=20, legend=True)
        ax.set_title(f'{label} — agent positions at tick {tick}')
        ax.set_axis_off()

    fig.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_path}')


def save_flow_snapshots(
    egdf: gpd.GeoDataFrame,
    ticks: list,
    b_umat: np.ndarray,
    s_umat: np.ndarray,
    scenario: dict,
    roi_gdf: gpd.GeoDataFrame,
    attractor_pts_gdf: gpd.GeoDataFrame,
    vmax_shared: float,
    city_name: str,
    scenario_name: str,
    tick_minutes: int,
    output_path: Path,
    n_cols: int = 6,
) -> None:
    tick_to_i  = {t: i for i, t in enumerate(ticks)}
    snap_ticks = [
        _nearest_sampled_tick({t: None for t in ticks}, int(round(sf)))
        for sf in np.linspace(0, ticks[-1], n_cols)
    ]
    segs = _edge_segments(egdf)
    lw   = _lw_array(egdf)
    norm = Normalize(vmin=0, vmax=vmax_shared)
    xmin, ymin, xmax, ymax = roi_gdf.total_bounds

    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 4.0, 8.5), constrained_layout=True)

    for col, snap_t in enumerate(snap_ticks):
        t_i      = tick_to_i[snap_t]
        wall_min = snap_t * tick_minutes
        in_ev    = _in_event(snap_t, scenario)

        for row, (umat, row_label) in enumerate([(b_umat, 'Baseline'), (s_umat, scenario_name)]):
            ax = axes[row, col]
            ax.set_facecolor('#FFF3CD' if (row == 1 and in_ev) else 'white')
            roi_gdf.boundary.plot(ax=ax, color='#444444', linewidth=0.7, zorder=2)
            lc = LineCollection(segs, linewidths=lw, norm=norm, cmap=_CMAP_FLOW, alpha=0.88, zorder=3)
            lc.set_array(umat[t_i])
            ax.add_collection(lc)
            attractor_pts_gdf.plot(ax=ax, color='#E45756', markersize=5, alpha=0.65, zorder=5)
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            ax.set_aspect('equal'); ax.set_axis_off()
            if col == 0:
                ax.text(-0.03, 0.5, row_label, transform=ax.transAxes,
                        fontsize=10, fontweight='bold', ha='right', va='center', rotation=90)
            tick_str = f't={snap_t} · {wall_min} min'
            if in_ev and row == 1:
                tick_str += ' ⚡'
            ax.set_title(tick_str, fontsize=8.5, pad=3,
                         color=_EVENT_COLOR if (in_ev and row == 1) else '#222222')

    sm = ScalarMappable(norm=norm, cmap=_CMAP_FLOW)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.012, pad=0.02)
    cb.set_label('Utilisation (load / cap-per-tick)', fontsize=9)
    fig.suptitle(
        f'{city_name}  —  Traffic flow evolution: Baseline vs {scenario_name}',
        fontsize=12, y=1.01,
    )
    fig.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_path}')


def save_corridor_heatmap(
    egdf: gpd.GeoDataFrame,
    ticks: list,
    b_umat: np.ndarray,
    s_umat: np.ndarray,
    scenario: dict,
    city_name: str,
    scenario_name: str,
    output_path: Path,
    n_top: int = 25,
) -> None:
    mean_s  = s_umat.mean(axis=0)
    top_idx = np.argsort(mean_s)[-n_top:][::-1]
    b_top   = b_umat[:, top_idx].T
    s_top   = s_umat[:, top_idx].T
    vmax_h  = max(float(b_top.max()), float(s_top.max()), 0.01)

    ylabels = [
        f"{egdf.iloc[i]['road_class']}  {int(egdf.iloc[i]['u'])}→{int(egdf.iloc[i]['v'])}"
        for i in top_idx
    ]
    step  = max(1, len(ticks) // 8)
    xlabs = [str(t) if j % step == 0 else '' for j, t in enumerate(ticks)]

    ev_starts = [j for j, t in enumerate(ticks)
                 if any(e['tick_start'] == t for e in scenario.get('events', []))]
    ev_ends   = [j for j, t in enumerate(ticks)
                 if any(e['tick_end'] == t for e in scenario.get('events', []))]

    fig, axes = plt.subplots(2, 1, figsize=(22, max(6, n_top * 0.40)), constrained_layout=True)
    for ax, mat, title in zip(axes, [b_top, s_top], ['Baseline', scenario_name]):
        im = ax.imshow(mat, aspect='auto', cmap=_CMAP_HEAT,
                       vmin=0, vmax=vmax_h, interpolation='nearest')
        ax.set_yticks(range(n_top))
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_xticks(range(len(ticks)))
        ax.set_xticklabels(xlabs, fontsize=8)
        ax.set_xlabel('Tick', fontsize=9)
        ax.set_title(f'{title} — top {n_top} corridors by mean utilisation', fontsize=11)
        for j in ev_starts:
            ax.axvline(j - 0.5, color=_EVENT_COLOR, lw=1.8, ls='--', alpha=0.9)
        for j in ev_ends:
            ax.axvline(j - 0.5, color=_EVENT_COLOR, lw=1.8, ls=':',  alpha=0.9)
        fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01).set_label('Utilisation', fontsize=8)

    fig.suptitle(f'Corridor Utilisation Heatmap — {city_name}', fontsize=13)
    fig.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_path}')


def save_flow_delta(
    egdf: gpd.GeoDataFrame,
    ticks: list,
    b_umat: np.ndarray,
    s_umat: np.ndarray,
    baseline_metrics: pd.DataFrame,
    scenario_metrics: pd.DataFrame,
    scenario: dict,
    roi_gdf: gpd.GeoDataFrame,
    city_name: str,
    scenario_name: str,
    tick_minutes: int,
    output_path: Path,
) -> None:
    merge = baseline_metrics[['tick', 'max_utilization']].merge(
        scenario_metrics[['tick', 'max_utilization']], on='tick', suffixes=('_b', '_s'),
    )
    merge['abs_diff'] = (merge['max_utilization_s'] - merge['max_utilization_b']).abs()
    peak_tick = int(merge.loc[merge['abs_diff'].idxmax(), 'tick'])
    t_i = {t: i for i, t in enumerate(ticks)}.get(
        _nearest_sampled_tick({t: None for t in ticks}, peak_tick), 0
    )

    diff = s_umat[t_i].astype(float) - b_umat[t_i].astype(float)
    nz   = np.abs(diff[diff != 0])
    dmax = float(np.percentile(nz, 97)) if len(nz) else 1.0
    norm = TwoSlopeNorm(vcenter=0, vmin=-dmax, vmax=dmax)

    segs = _edge_segments(egdf)
    lw   = _lw_array(egdf)
    xmin, ymin, xmax, ymax = roi_gdf.total_bounds
    in_ev = _in_event(peak_tick, scenario)

    fig, ax = plt.subplots(figsize=(14, 12), constrained_layout=True)
    roi_gdf.boundary.plot(ax=ax, color='#444444', linewidth=0.8, zorder=2)
    lc = LineCollection(segs, linewidths=lw, norm=norm, cmap=_CMAP_DIFF, alpha=0.88, zorder=3)
    lc.set_array(diff)
    ax.add_collection(lc)

    sm = ScalarMappable(norm=norm, cmap=_CMAP_DIFF)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label('Δ utilisation  (scenario − baseline)', fontsize=10)

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal'); ax.set_axis_off()
    event_str = '  ·  ⚡ event active' if in_ev else ''
    ax.set_title(
        f'Flow redistribution at tick {peak_tick} ({peak_tick * tick_minutes} min)'
        f'{event_str}\n'
        f'{scenario_name}  vs  baseline  ·  net Δ util sum = {diff.sum():+.2f}',
        fontsize=12,
    )
    fig.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_path}')


def save_animation(
    b_emaj: gpd.GeoDataFrame,
    s_emaj: gpd.GeoDataFrame,
    ticks: list,
    b_umat_maj: np.ndarray,
    s_umat_maj: np.ndarray,
    scenario: dict,
    roi_gdf: gpd.GeoDataFrame,
    city_name: str,
    scenario_name: str,
    tick_minutes: int,
    vmax_shared: float,
    output_path: Path,
    fps: int = 5,
) -> None:
    norm   = Normalize(vmin=0, vmax=vmax_shared)
    xmin, ymin, xmax, ymax = roi_gdf.total_bounds

    b_segs = _edge_segments(b_emaj)
    s_segs = _edge_segments(s_emaj)

    fig, (ax_b, ax_s) = plt.subplots(1, 2, figsize=(20, 8), dpi=80, constrained_layout=True)
    fig.patch.set_facecolor('#111111')
    for ax in (ax_b, ax_s):
        ax.set_facecolor('#111111')
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal'); ax.set_axis_off()
        roi_gdf.boundary.plot(ax=ax, color='#555555', linewidth=0.7, zorder=2)

    lc_b = LineCollection(b_segs, linewidths=_lw_array(b_emaj), norm=norm,
                          cmap=_CMAP_FLOW, alpha=0.92, zorder=3)
    lc_b.set_array(b_umat_maj[0])
    ax_b.add_collection(lc_b)

    lc_s = LineCollection(s_segs, linewidths=_lw_array(s_emaj), norm=norm,
                          cmap=_CMAP_FLOW, alpha=0.92, zorder=3)
    lc_s.set_array(s_umat_maj[0])
    ax_s.add_collection(lc_s)

    sm = ScalarMappable(norm=norm, cmap=_CMAP_FLOW)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=[ax_b, ax_s], fraction=0.015, pad=0.02)
    cb.set_label('Utilisation', fontsize=10, color='white')
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

    ttl_b = ax_b.set_title('', fontsize=11, color='white', pad=5)
    ttl_s = ax_s.set_title('', fontsize=11, color='white', pad=5)
    fig.suptitle(f'{city_name}  ·  Agent Traffic Flow', color='white', fontsize=12)

    def _update(frame: int):
        t     = ticks[frame]
        wall  = t * tick_minutes
        in_ev = _in_event(t, scenario)
        lc_b.set_array(b_umat_maj[frame])
        lc_s.set_array(s_umat_maj[frame])
        ttl_b.set_text(f'Baseline  ·  t={t}  ({wall} min)')
        ttl_s.set_text(
            f'{scenario_name}  ·  t={t}  ({wall} min)' + ('  ⚡' if in_ev else '')
        )
        ttl_s.set_color(_EVENT_COLOR if in_ev else 'white')

    anim = FuncAnimation(fig, _update, frames=len(ticks),
                         interval=max(80, 1000 // fps), blit=False)
    print(f'  Exporting {len(ticks)} frames to GIF ...')
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ── main orchestrator ─────────────────────────────────────────────────────────

def _find_project_root() -> Path:
    for candidate in [Path.cwd(), Path(__file__).resolve().parent.parent]:
        if (candidate / 'src').exists() and (candidate / 'data').exists():
            return candidate
    return Path.cwd()


def _resolve_output_dir(args: argparse.Namespace, city_slug: str) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    project_root = _find_project_root() if not getattr(args, 'project_root', None) else Path(args.project_root)
    return project_root / 'data' / 'outputs' / 'simulation' / city_slug


def do_inspect(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root) if args.project_root else _find_project_root()
    print(f'Project root: {project_root}')
    print(f'Loading city bundle for {args.city!r} ...')
    bundle = load_city_bundle(project_root, args.city)
    manifest = bundle['manifest']

    info = {
        'city'             : manifest['city'],
        'country'          : manifest['country'],
        'roi_area_km2'     : manifest['roi_area_km2'],
        'graph_nodes'      : manifest['graph_node_count'],
        'graph_edges'      : manifest['graph_edge_count'],
        'hotspot_clusters' : manifest['actual_cluster_count'],
        'demand_csv_rows'  : len(bundle['node_hotspot_demand']),
        'manifest_path'    : str(bundle['context'].output_dir / 'manifest.json'),
    }
    print(json.dumps(info, indent=2))


def do_run(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root) if args.project_root else _find_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Slugify city name the same way the pipeline does (simple lower + replace)
    city_slug = args.city.lower().replace(' ', '_')
    output_dir = _resolve_output_dir(args, city_slug)
    output_dir.mkdir(parents=True, exist_ok=True)

    display_crs = args.display_crs

    # ── load data ─────────────────────────────────────────────────────────────
    print(f'Loading city bundle for {args.city!r} ...')
    bundle = load_city_bundle(project_root, args.city)

    print('\nBuilding routing graph ...')
    graphml_path = bundle['manifest']['outputs']['graphml']
    routing_graph, routing_nodes_raw, routing_edges_raw = build_routing_graph(graphml_path)

    print('\nBuilding demand tables ...')
    full_demand_table, attractor_table, demand_summary = build_hotspot_demand_table(
        routing_graph, bundle['node_hotspot_demand'],
        attractor_top_k=args.attractor_top_k,
    )
    print(demand_summary.to_string())

    # ── warmup: identify genuine routing bottlenecks ──────────────────────────
    print(f'\nWarmup ({args.n_agents} agents, {args.replan_interval + 5} ticks) ...')
    warmup_cfg = SimulationConfig(
        city_name=args.city, n_agents=args.n_agents,
        n_ticks=args.replan_interval + 5, tick_minutes=args.tick_minutes,
        random_seed=args.random_seed, sample_every=1, position_sample_size=0,
        replan_interval=args.replan_interval,
        capacity_scale=args.capacity_scale,
        congestion_alpha=args.congestion_alpha,
        congestion_beta=args.congestion_beta,
        diurnal_enabled=False,   # warmup needs steady demand for a clean bottleneck signal
    )
    warmup_result = run_simulation(
        routing_graph, routing_nodes_raw, routing_edges_raw,
        full_demand_table, attractor_table, warmup_cfg,
        {'name': 'warmup', 'events': []},
    )

    post_replan = [t for t in warmup_result['edge_history'] if t >= args.replan_interval]
    warmup_loads: Counter = Counter()
    for t in post_replan:
        warmup_loads.update(warmup_result['edge_history'][t])

    warmup_ranked = []
    for (u, v), total_load in warmup_loads.items():
        if not routing_graph.has_edge(u, v):
            continue
        rc = routing_graph[u][v].get('road_class', 'other')
        if rc not in _MAJOR_ROAD_CLASSES:
            continue
        warmup_ranked.append(((u, v), total_load, rc))
    warmup_ranked.sort(key=lambda x: x[1], reverse=True)
    high_bet_edges = [e for e, _, _ in warmup_ranked[:8]]

    tph = 60.0 / args.tick_minutes
    print('  Top arterial edges by post-replan load:')
    for e, load, rc in warmup_ranked[:6]:
        cap_pt = routing_graph[e[0]][e[1]]['base_capacity'] * args.capacity_scale / tph
        print(f'    {e}  rc={rc:<14}  load={load:4d}  cap_pt={cap_pt:.1f}')

    del warmup_result, warmup_loads, warmup_ranked

    # ── scenario catalog ──────────────────────────────────────────────────────
    scenario_catalog = build_default_scenarios(
        routing_edges_raw, attractor_table,
        n_ticks=args.n_ticks,
        high_betweenness_edges=high_bet_edges,
    )
    print('\nScenario catalog:')
    print(scenario_table(scenario_catalog).to_string(index=False))

    # ── simulation runs ───────────────────────────────────────────────────────
    sim_config = SimulationConfig(
        city_name=args.city, n_agents=args.n_agents, n_ticks=args.n_ticks,
        tick_minutes=args.tick_minutes, random_seed=args.random_seed,
        sample_every=args.sample_every, position_sample_size=args.position_sample_size,
        replan_interval=args.replan_interval,
        capacity_scale=args.capacity_scale,
        congestion_alpha=args.congestion_alpha,
        congestion_beta=args.congestion_beta,
        diurnal_enabled=not args.no_diurnal,
        sim_start_hour=args.sim_start_hour,
    )

    print('\n--- Running simulations ---')
    baseline_result = run_simulation(
        routing_graph, routing_nodes_raw, routing_edges_raw,
        full_demand_table, attractor_table, sim_config, scenario_catalog['baseline'],
    )
    scenario_result = run_simulation(
        routing_graph, routing_nodes_raw, routing_edges_raw,
        full_demand_table, attractor_table, sim_config, scenario_catalog[args.scenario],
    )

    summary_df = pd.DataFrame([
        simulation_summary(baseline_result),
        simulation_summary(scenario_result),
    ]).reset_index(drop=True)
    print('\nSummary:')
    print(_rounded(summary_df, 3).to_string(index=False))

    # ── save CSVs ─────────────────────────────────────────────────────────────
    baseline_result['metrics'].to_csv(output_dir / 'metrics_baseline.csv', index=False)
    scenario_result['metrics'].to_csv(output_dir / f'metrics_{args.scenario}.csv', index=False)
    summary_df.to_csv(output_dir / 'summary.csv', index=False)
    print(f'\nCSVs saved to {output_dir}')

    # ── synthetic traffic dataset (grid-aligned tensor) ───────────────────────
    # Built here while edge_history is still intact (the viz phase below frees it).
    if not args.no_export_tensor:
        print('\nExporting synthetic traffic tensor ...')
        try:
            from src.traffic_export import roi_grid_from_bundle, save_traffic_dataset
        except ModuleNotFoundError:
            from traffic_export import roi_grid_from_bundle, save_traffic_dataset
        grid = roi_grid_from_bundle(bundle)
        print(f'  ROI grid: {grid.height}×{grid.width} cells @ 1km ({grid.crs})')
        save_traffic_dataset(baseline_result, bundle, output_dir, sim_config, grid=grid)
        save_traffic_dataset(scenario_result, bundle, output_dir, sim_config, grid=grid)

    # ── display projections ───────────────────────────────────────────────────
    routing_edges_disp = routing_edges_raw.to_crs(display_crs)
    roi                = bundle['roi'].to_crs(display_crs)
    attractor_pts      = routing_nodes_raw.to_crs(display_crs)
    attractor_pts      = attractor_pts[attractor_pts['node_id'].isin(attractor_table['node_id'])].copy()

    b_metrics = baseline_result['metrics']
    s_metrics = scenario_result['metrics']

    # ── static plots ──────────────────────────────────────────────────────────
    print('\nSaving plots ...')

    save_timeseries(
        baseline_result, scenario_result, scenario_catalog, args.scenario,
        output_dir / 'timeseries.png',
    )
    save_spatial_peak(
        baseline_result, scenario_result, roi, attractor_pts,
        args.scenario, display_crs,
        output_dir / 'spatial_peak.png',
    )
    save_agent_positions(
        baseline_result, scenario_result, roi, routing_edges_disp,
        args.scenario, display_crs,
        output_dir / 'agent_positions.png',
    )

    # ── flow visualisations ───────────────────────────────────────────────────
    print('\nBuilding utilisation matrices ...')
    (b_eall, b_emaj_v, b_ticks_v, b_umat_all, b_umat_maj_v, b_vmax) = \
        build_edge_util_arrays(baseline_result, display_crs)
    baseline_result['edge_history'] = {}   # consumed into b_umat_all; free it
    (s_eall, s_emaj_v, s_ticks_v, s_umat_all, s_umat_maj_v, s_vmax) = \
        build_edge_util_arrays(scenario_result, display_crs)
    scenario_result['edge_history'] = {}   # consumed into s_umat_all; free it
    vmax_sh = max(b_vmax, s_vmax)
    print(f'  {len(b_eall)} edges × {len(b_ticks_v)} ticks | vmax={vmax_sh:.2f}')
    print(f'  {len(b_emaj_v)} major-road edges for animation')

    save_flow_snapshots(
        b_eall, b_ticks_v, b_umat_all, s_umat_all,
        scenario_catalog[args.scenario], roi, attractor_pts,
        vmax_sh, args.city, args.scenario, args.tick_minutes,
        output_dir / 'flow_snapshots.png',
        n_cols=6,
    )
    save_corridor_heatmap(
        b_eall, b_ticks_v, b_umat_all, s_umat_all,
        scenario_catalog[args.scenario],
        args.city, args.scenario,
        output_dir / 'corridor_heatmap.png',
        n_top=25,
    )
    save_flow_delta(
        b_eall, b_ticks_v, b_umat_all, s_umat_all,
        b_metrics, s_metrics,
        scenario_catalog[args.scenario], roi,
        args.city, args.scenario, args.tick_minutes,
        output_dir / 'flow_delta.png',
    )

    if not args.no_animation:
        save_animation(
            b_emaj_v, s_emaj_v, b_ticks_v, b_umat_maj_v, s_umat_maj_v,
            scenario_catalog[args.scenario], roi,
            args.city, args.scenario, args.tick_minutes,
            vmax_sh,
            output_dir / 'animation.gif',
            fps=5,
        )

    # ── run params manifest ───────────────────────────────────────────────────
    params = {
        'city'                : args.city,
        'n_agents'            : args.n_agents,
        'n_ticks'             : args.n_ticks,
        'tick_minutes'        : args.tick_minutes,
        'random_seed'         : args.random_seed,
        'sample_every'        : args.sample_every,
        'position_sample_size': args.position_sample_size,
        'attractor_top_k'     : args.attractor_top_k,
        'replan_interval'     : args.replan_interval,
        'scenario'            : args.scenario,
        'capacity_scale'      : args.capacity_scale,
        'congestion_alpha'    : args.congestion_alpha,
        'congestion_beta'     : args.congestion_beta,
        'diurnal_enabled'     : not args.no_diurnal,
        'sim_start_hour'      : args.sim_start_hour,
        'export_tensor'       : not args.no_export_tensor,
        'display_crs'         : display_crs,
        'output_dir'          : str(output_dir),
    }
    (output_dir / 'run_params.json').write_text(json.dumps(params, indent=2))
    print(f'\nRun params saved to {output_dir / "run_params.json"}')
    print(f'\nAll outputs written to: {output_dir}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='agent_simulation',
        description='Agent-based urban traffic simulation on a hotspot-network graph.',
    )
    sub = p.add_subparsers(dest='command', required=True)

    # ── inspect ───────────────────────────────────────────────────────────────
    inspect_p = sub.add_parser('inspect', help='Show city bundle metadata without running.')
    inspect_p.add_argument('--city', required=True)
    inspect_p.add_argument('--project-root', default=None)

    # ── run ───────────────────────────────────────────────────────────────────
    run_p = sub.add_parser('run', help='Run simulation and save all outputs.')
    run_p.add_argument('--city',                  required=True,
                       help='City name as it appears in the ROI manifest (e.g. "Mumbai").')
    run_p.add_argument('--project-root',          default=None,
                       help='Path to project root. Auto-detected if omitted.')
    run_p.add_argument('--n-agents',              type=int,   default=1500,
                       help='Number of agents to simulate. (default: 1500)')
    run_p.add_argument('--n-ticks',               type=int,   default=80,
                       help='Number of simulation ticks. (default: 80)')
    run_p.add_argument('--tick-minutes',          type=int,   default=6,
                       help='Real-world minutes per tick. (default: 6)')
    run_p.add_argument('--random-seed',           type=int,   default=42)
    run_p.add_argument('--sample-every',          type=int,   default=2,
                       help='Record edge loads every N ticks. (default: 2)')
    run_p.add_argument('--position-sample-size',  type=int,   default=60,
                       help='Agents to sample for position snapshots per tick. (default: 60)')
    run_p.add_argument('--attractor-top-k',       type=int,   default=150,
                       help='Top-K demand nodes used as destinations. (default: 150)')
    run_p.add_argument('--replan-interval',       type=int,   default=10,
                       help='Ticks between routing-tree rebuilds. (default: 10)')
    run_p.add_argument('--scenario',              default='capacity_drop',
                       choices=['capacity_drop', 'edge_closure', 'hotspot_surge'],
                       help='Scenario to compare against baseline. (default: capacity_drop)')
    run_p.add_argument('--capacity-scale',        type=float, default=0.50,
                       help='Compress real-world capacity to simulation agent counts. (default: 0.50)')
    run_p.add_argument('--congestion-alpha',      type=float, default=1.8,
                       help='BPR congestion α coefficient. (default: 1.8)')
    run_p.add_argument('--congestion-beta',       type=float, default=2.8,
                       help='BPR congestion β exponent. (default: 2.8)')
    run_p.add_argument('--display-crs',           default='EPSG:4326',
                       help='CRS for output maps. (default: EPSG:4326)')
    run_p.add_argument('--output-dir',            default=None,
                       help='Output directory. Defaults to data/outputs/simulation/<city>.')
    run_p.add_argument('--no-animation',          action='store_true',
                       help='Skip HTML animation export (faster).')
    run_p.add_argument('--no-diurnal',            action='store_true',
                       help='Disable diurnal demand modulation (constant demand, legacy behaviour).')
    run_p.add_argument('--sim-start-hour',        type=float, default=6.0,
                       help='Wall-clock hour-of-day at tick 0, for diurnal demand. (default: 6.0)')
    run_p.add_argument('--no-export-tensor',      action='store_true',
                       help='Skip writing the grid-aligned synthetic traffic tensor dataset.')

    return p


def main() -> None:
    parser = _make_parser()
    args   = parser.parse_args()

    if args.command == 'inspect':
        do_inspect(args)
    else:
        do_run(args)


if __name__ == '__main__':
    main()
