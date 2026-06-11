from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.transform import xy as transform_xy
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ROIGridData:
    baseline: np.ndarray
    roi_mask: np.ndarray
    valid_mask: np.ndarray
    transform: Affine
    crs: CRS
    nodata: float | None
    baseline_path: Path
    mask_path: Path
    matched_name: str | None = None


@dataclass(frozen=True)
class HotspotClusterConfig:
    hotspot_quantile: float = 0.85
    n_clusters: int = 20
    random_state: int = 42
    saturation_eps: float = 1e-6
    local_window_size: int = 3
    min_weight: float = 1.0
    batch_size: int = 1024

    def __post_init__(self) -> None:
        if not 0.0 < self.hotspot_quantile < 1.0:
            raise ValueError("hotspot_quantile must be between 0 and 1.")
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be >= 1.")
        if self.saturation_eps <= 0:
            raise ValueError("saturation_eps must be > 0.")
        if self.local_window_size < 1 or self.local_window_size % 2 == 0:
            raise ValueError("local_window_size must be a positive odd integer.")
        if self.min_weight <= 0:
            raise ValueError("min_weight must be > 0.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")


@dataclass(frozen=True)
class HotspotClusterResult:
    matched_name: str | None
    config: HotspotClusterConfig
    threshold: float
    candidate_count: int
    score_map: np.ndarray
    score_rank_map: np.ndarray
    local_score_map: np.ndarray
    candidate_mask: np.ndarray
    cluster_grid: np.ndarray
    cluster_points: pd.DataFrame
    cluster_summary: pd.DataFrame
    transform: Affine
    crs: CRS
    baseline_path: Path
    mask_path: Path

    def cluster_centers_gdf(self, weighted: bool = False) -> gpd.GeoDataFrame:
        centers = self.cluster_summary.copy()
        x_col = "weighted_center_x" if weighted else "center_x"
        y_col = "weighted_center_y" if weighted else "center_y"
        geometry = gpd.points_from_xy(centers[x_col], centers[y_col], crs=self.crs)
        return gpd.GeoDataFrame(centers, geometry=geometry, crs=self.crs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched_name": self.matched_name,
            "threshold": self.threshold,
            "candidate_count": self.candidate_count,
            "config": {
                "hotspot_quantile": self.config.hotspot_quantile,
                "n_clusters": self.config.n_clusters,
                "random_state": self.config.random_state,
                "saturation_eps": self.config.saturation_eps,
                "local_window_size": self.config.local_window_size,
                "min_weight": self.config.min_weight,
                "batch_size": self.config.batch_size,
            },
            "cluster_count": int(len(self.cluster_summary)),
            "baseline_path": str(self.baseline_path),
            "mask_path": str(self.mask_path),
            "crs": self.crs.to_string() if self.crs is not None else None,
        }


class HotspotClusterer:
    def __init__(self, config: HotspotClusterConfig | None = None) -> None:
        self.config = config or HotspotClusterConfig()

    @staticmethod
    def load_manifest(manifest_path: Path) -> dict[str, Any]:
        return json.loads(manifest_path.read_text())

    @staticmethod
    def find_roi_record(manifest: dict[str, Any], city_name: str) -> dict[str, Any]:
        for record in manifest["rois"]:
            if record["matched_name"] == city_name:
                return record
        raise KeyError(f"City not found in manifest: {city_name}")

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
    def load_roi_from_paths(
        baseline_path: Path,
        mask_path: Path,
        matched_name: str | None = None,
    ) -> ROIGridData:
        baseline_path = baseline_path.expanduser().resolve()
        mask_path = mask_path.expanduser().resolve()

        with rasterio.open(baseline_path) as src:
            baseline = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
            baseline_shape = baseline.shape

        with rasterio.open(mask_path) as src:
            roi_mask = src.read(1).astype(np.uint8)
            if src.transform != transform:
                raise ValueError("Mask transform does not match baseline transform.")
            if src.crs != crs:
                raise ValueError("Mask CRS does not match baseline CRS.")
            if roi_mask.shape != baseline_shape:
                raise ValueError("Mask shape does not match baseline shape.")

        valid_mask = (baseline != nodata) & (roi_mask == 1) if nodata is not None else (np.isfinite(baseline) & (roi_mask == 1))
        if not valid_mask.any():
            raise ValueError(f"No valid ROI pixels found for baseline={baseline_path} mask={mask_path}.")

        return ROIGridData(
            baseline=baseline,
            roi_mask=roi_mask,
            valid_mask=valid_mask,
            transform=transform,
            crs=crs,
            nodata=nodata,
            baseline_path=baseline_path,
            mask_path=mask_path,
            matched_name=matched_name,
        )

    def load_roi_from_manifest(
        self,
        manifest_path: Path,
        city_name: str,
        project_root: Path | None = None,
    ) -> ROIGridData:
        manifest = self.load_manifest(manifest_path.expanduser().resolve())
        record = self.find_roi_record(manifest, city_name)
        root = self._resolve_project_root(project_root)
        baseline_path = root / Path(record["baseline_path"])
        mask_path = root / Path(record["mask_path"])
        return self.load_roi_from_paths(baseline_path, mask_path, matched_name=record["matched_name"])

    def run_from_manifest(
        self,
        manifest_path: Path,
        city_name: str,
        project_root: Path | None = None,
    ) -> HotspotClusterResult:
        roi_data = self.load_roi_from_manifest(
            manifest_path=manifest_path,
            city_name=city_name,
            project_root=project_root,
        )
        return self.fit(roi_data)

    @staticmethod
    def percentile_rank_map(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        ranked = np.full(values.shape, np.nan, dtype=np.float32)
        valid_values = values[valid_mask]
        pct = pd.Series(valid_values).rank(method="average", pct=True).to_numpy(dtype=np.float32)
        ranked[valid_mask] = pct
        return ranked

    def local_mean(self, values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        filled = np.where(valid_mask, values, 0.0).astype(np.float32)
        valid_float = valid_mask.astype(np.float32)
        h, w = values.shape
        pad = self.config.local_window_size // 2
        sum_arr = np.zeros((h, w), dtype=np.float32)
        count_arr = np.zeros((h, w), dtype=np.float32)
        padded_values = np.pad(filled, pad, mode="constant")
        padded_valid = np.pad(valid_float, pad, mode="constant")

        for dy in range(self.config.local_window_size):
            for dx in range(self.config.local_window_size):
                sum_arr += padded_values[dy : dy + h, dx : dx + w]
                count_arr += padded_valid[dy : dy + h, dx : dx + w]

        local_mean = np.divide(sum_arr, count_arr, out=np.zeros_like(sum_arr), where=count_arr > 0)
        local_mean[~valid_mask] = np.nan
        return local_mean

    def saturation_score_map(self, baseline: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        score = np.full(baseline.shape, np.nan, dtype=np.float32)
        score[valid_mask] = -np.log10(np.maximum(1.0 - baseline[valid_mask], self.config.saturation_eps))
        return score

    def _candidate_mask(self, local_score_map: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, float]:
        local_values = local_score_map[valid_mask]
        threshold = float(np.quantile(local_values, self.config.hotspot_quantile))
        candidate_mask = valid_mask & (local_score_map >= threshold)
        return candidate_mask, threshold

    def fit(self, roi_data: ROIGridData) -> HotspotClusterResult:
        score_map = self.saturation_score_map(roi_data.baseline, roi_data.valid_mask)
        score_rank_map = self.percentile_rank_map(score_map, roi_data.valid_mask)
        local_score_map = self.local_mean(score_map, roi_data.valid_mask)
        candidate_mask, threshold = self._candidate_mask(local_score_map, roi_data.valid_mask)

        candidate_rows, candidate_cols = np.where(candidate_mask)
        candidate_count = len(candidate_rows)
        if candidate_count == 0:
            raise ValueError("No hotspot candidates found. Lower hotspot_quantile and rerun.")

        cluster_count = max(1, min(self.config.n_clusters, candidate_count))
        features = np.column_stack(
            [
                candidate_cols.astype(np.float32),
                candidate_rows.astype(np.float32),
                score_map[candidate_mask].astype(np.float32),
                local_score_map[candidate_mask].astype(np.float32),
            ]
        )
        scaled = StandardScaler().fit_transform(features)
        weights = np.clip(local_score_map[candidate_mask].astype(np.float32), self.config.min_weight, None)

        model = MiniBatchKMeans(
            n_clusters=cluster_count,
            random_state=self.config.random_state,
            n_init="auto",
            batch_size=self.config.batch_size,
        )
        labels = model.fit_predict(scaled, sample_weight=weights)

        cluster_grid = np.full(score_map.shape, -1, dtype=np.int32)
        cluster_grid[candidate_mask] = labels.astype(np.int32)

        xs, ys = transform_xy(roi_data.transform, candidate_rows, candidate_cols, offset="center")
        cluster_points = pd.DataFrame(
            {
                "cluster_id": labels.astype(int),
                "row": candidate_rows.astype(int),
                "col": candidate_cols.astype(int),
                "x": np.asarray(xs, dtype=np.float64),
                "y": np.asarray(ys, dtype=np.float64),
                "score": score_map[candidate_mask].astype(np.float64),
                "local_score": local_score_map[candidate_mask].astype(np.float64),
            }
        )

        cluster_summary = (
            cluster_points.groupby("cluster_id", as_index=False)
            .agg(
                pixel_count=("cluster_id", "size"),
                mean_score=("score", "mean"),
                max_score=("score", "max"),
                mean_local_score=("local_score", "mean"),
                weight_sum=("local_score", "sum"),
                center_row=("row", "mean"),
                center_col=("col", "mean"),
                center_x=("x", "mean"),
                center_y=("y", "mean"),
            )
            .sort_values(["mean_local_score", "pixel_count"], ascending=[False, False])
            .reset_index(drop=True)
        )
        weighted_centers = (
            cluster_points.assign(
                weighted_row=cluster_points["row"] * cluster_points["local_score"],
                weighted_col=cluster_points["col"] * cluster_points["local_score"],
                weighted_x=cluster_points["x"] * cluster_points["local_score"],
                weighted_y=cluster_points["y"] * cluster_points["local_score"],
            )
            .groupby("cluster_id", as_index=False)
            .agg(
                weighted_row_sum=("weighted_row", "sum"),
                weighted_col_sum=("weighted_col", "sum"),
                weighted_x_sum=("weighted_x", "sum"),
                weighted_y_sum=("weighted_y", "sum"),
            )
        )
        cluster_summary = cluster_summary.merge(weighted_centers, on="cluster_id", how="left")
        cluster_summary["weighted_center_row"] = cluster_summary["weighted_row_sum"] / cluster_summary["weight_sum"]
        cluster_summary["weighted_center_col"] = cluster_summary["weighted_col_sum"] / cluster_summary["weight_sum"]
        cluster_summary["weighted_center_x"] = cluster_summary["weighted_x_sum"] / cluster_summary["weight_sum"]
        cluster_summary["weighted_center_y"] = cluster_summary["weighted_y_sum"] / cluster_summary["weight_sum"]
        cluster_summary = cluster_summary.drop(
            columns=[
                "weighted_row_sum",
                "weighted_col_sum",
                "weighted_x_sum",
                "weighted_y_sum",
            ]
        )

        return HotspotClusterResult(
            matched_name=roi_data.matched_name,
            config=self.config,
            threshold=threshold,
            candidate_count=candidate_count,
            score_map=score_map,
            score_rank_map=score_rank_map,
            local_score_map=local_score_map,
            candidate_mask=candidate_mask,
            cluster_grid=cluster_grid,
            cluster_points=cluster_points,
            cluster_summary=cluster_summary,
            transform=roi_data.transform,
            crs=roi_data.crs,
            baseline_path=roi_data.baseline_path,
            mask_path=roi_data.mask_path,
        )

    @staticmethod
    def save_cluster_outputs(
        result: HotspotClusterResult,
        output_dir: Path,
        name_prefix: str,
    ) -> dict[str, Path]:
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / f"{name_prefix}_cluster_summary.csv"
        gpkg_path = output_dir / f"{name_prefix}_cluster_centers.gpkg"
        json_path = output_dir / f"{name_prefix}_cluster_metadata.json"

        result.cluster_summary.to_csv(csv_path, index=False)
        result.cluster_centers_gdf().to_file(gpkg_path, layer="cluster_centers", driver="GPKG")
        json_path.write_text(json.dumps(result.to_dict(), indent=2))

        return {
            "summary_csv": csv_path,
            "centers_gpkg": gpkg_path,
            "metadata_json": json_path,
        }
