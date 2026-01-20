# density/kde.py
# -*- coding: utf-8 -*-
"""PDV流儀のKDE密度推定（ボクセル重心→RoIグリッド→Ball Query→KDE）"""

from dataclasses import dataclass
from typing import Tuple, Optional
import math
import numpy as np
import open3d as o3d

from utility import aabb_of_points, voxel_index

@dataclass
class KDEGridResult:
    grid_points: np.ndarray
    kde_values: np.ndarray
    neighbor_counts: np.ndarray
    radius: float
    sigma: float
    grid_U: int
    roi_min: np.ndarray
    roi_max: np.ndarray

class PDVKDEDensityCalculator:
    """PDVの流儀を踏襲した密度推定（簡易版）"""
    def __init__(self, voxel_size: Tuple[float, float, float], grid_U: int, radius: float, sigma: float):
        self.voxel_size = np.asarray(voxel_size, dtype=np.float64)
        self.grid_U = int(grid_U)
        self.radius = float(radius)
        self.sigma = float(sigma)
        if self.grid_U <= 0:
            raise ValueError("grid_Uは正の整数である必要がある")
        if self.radius <= 0 or self.sigma <= 0:
            raise ValueError("radiusとsigmaは正である必要がある")

    # --- ボクセル重心 ---
    def compute_voxel_centroids(self, points: np.ndarray):
        pmin, _ = aabb_of_points(points)
        origin = pmin.astype(np.float64)
        ijk = voxel_index(points, origin, self.voxel_size)
        uniq, inv = np.unique(ijk, axis=0, return_inverse=True)
        sums = np.zeros((uniq.shape[0], 3), dtype=np.float64)
        np.add.at(sums, inv, points)
        cnt = np.bincount(inv, minlength=uniq.shape[0]).astype(np.float64)
        cnt[cnt == 0] = 1.0
        centroids = (sums / cnt[:, None]).astype(np.float64)
        return centroids, uniq.astype(np.int64), origin

    # --- グリッド生成 ---
    def generate_grid(self, roi_min: np.ndarray, roi_max: np.ndarray) -> np.ndarray:
        U = self.grid_U
        mins = roi_min.astype(np.float64); maxs = roi_max.astype(np.float64)
        step = (maxs - mins) / U
        xs = np.arange(U, dtype=np.float64) + 0.5
        grid = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), axis=-1)
        grid = mins[None,None,None,:] + grid * step[None,None,None,:]
        return grid.reshape(-1, 3)

    # --- KDE ---
    def kde_on_grid(self, centroids: np.ndarray, grid_points: np.ndarray):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(centroids)
        kdt = o3d.geometry.KDTreeFlann(pc)

        r, s = self.radius, self.sigma
        s3 = s ** 3
        norm = 1.0 / ((2.0 * math.pi) ** 1.5)

        kde = np.zeros((grid_points.shape[0],), dtype=np.float64)
        nb  = np.zeros_like(kde, dtype=np.int64)
        for i, g in enumerate(grid_points):
            count, idx, d2 = kdt.search_radius_vector_3d(g, r)
            nb[i] = int(count)
            if count == 0:
                continue
            d2 = np.asarray(d2, dtype=np.float64)
            kern = np.exp(-0.5 * d2 / (s * s)) * norm
            kde[i] = float(np.sum(kern) / (count * s3))
        return kde, nb

    # --- 実行 ---
    def compute(self, points: np.ndarray, roi: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> KDEGridResult:
        centroids, _, _ = self.compute_voxel_centroids(points)
        if roi is None:
            roi_min, roi_max = aabb_of_points(points)
        else:
            roi_min, roi_max = roi
        roi_min = roi_min - self.radius * 0.5
        roi_max = roi_max + self.radius * 0.5
        grid = self.generate_grid(roi_min, roi_max)
        kde_vals, nb_counts = self.kde_on_grid(centroids, grid)
        return KDEGridResult(grid, kde_vals, nb_counts, self.radius, self.sigma, self.grid_U, roi_min, roi_max)
