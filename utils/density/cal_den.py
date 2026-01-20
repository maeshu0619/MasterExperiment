"""ボクセル単位の密度計算
このモジュールは、点群データのボクセル単位の密度を計算するためのクラスを提供します。
ボクセル密度は、各ボクセル内の点の数をその体積で割った値です。"""

import glob
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from utility import aabb_of_points, voxel_index, voxel_center_from_index

@dataclass
class VoxelDensityResult:
    centers: np.ndarray  # (M,3)
    counts: np.ndarray   # (M,)
    density: np.ndarray  # (M,) count / voxel_volume
    index_ijk: np.ndarray  # (M,3) 各ボクセルの整数インデックス
    voxel_size: np.ndarray  # (3,)
    origin: np.ndarray      # (3,)


class VoxelDensityCalculator:
    """ボクセル単位の密度（単純な点数/体積）を計算する"""

    def __init__(self, voxel_size: Tuple[float, float, float]):
        self.voxel_size = np.array(voxel_size, dtype=np.float64)
        if np.any(self.voxel_size <= 0):
            raise ValueError("voxel_sizeは正である必要がある")

    def compute(self, points: np.ndarray, origin: Optional[np.ndarray] = None) -> VoxelDensityResult:
        # AABBに基づく原点設定（未指定ならmin）
        if origin is None:
            pmin, _ = aabb_of_points(points)
            origin = pmin
        origin = origin.astype(np.float64)

        # 各点のボクセルインデックス
        ijk = voxel_index(points, origin, self.voxel_size)

        # ユニークなボクセルごとにカウント
        # uniqueの戻り値：uniq_indicesは各ユニーク行の代表、inverseは元配列→ユニーク行のマッピング
        uniq, inverse, counts = np.unique(ijk, axis=0, return_inverse=True, return_counts=True)

        # ボクセル中心座標
        centers = voxel_center_from_index(uniq, origin, self.voxel_size)

        # 体積と密度
        vol = float(np.prod(self.voxel_size))
        density = counts.astype(np.float64) / vol

        return VoxelDensityResult(
            centers=centers,
            counts=counts.astype(np.int64),
            density=density,
            index_ijk=uniq.astype(np.int64),
            voxel_size=self.voxel_size.copy(),
            origin=origin.copy(),
        )