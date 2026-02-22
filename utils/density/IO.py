"""
PointCloudIOクラス
点群の読み込みと書き出しを担当します。
このクラスは、Open3Dライブラリを使用して点群データを処理します。
"""

import numpy as np
from pathlib import Path   
import open3d as o3d
from typing import Tuple, Union
import os

from utility import ensure_dir, aabb_of_points, voxel_index, voxel_center_from_index


class PointCloudIO:
    """点群の読み込み/書き出しを担当"""

    @staticmethod
    def load_points(path: Path) -> np.ndarray:
        """PCD/PLY等をOpen3Dで読み込み、Nx3のnumpy配列を返す
        ・座標以外の属性（法線・色・強度など）は無視
        """
        pcd = o3d.io.read_point_cloud(str(path))
        if pcd is None or len(pcd.points) == 0:
            raise ValueError(f"点群が空、または読み込めない: {path}")
        pts = np.asarray(pcd.points, dtype=np.float64)
        return pts

    @staticmethod
    def save_points_with_scalar(points: np.ndarray, scalar: np.ndarray, out_ply: Path) -> None:
        """各点にスカラー値（例：密度）を持つ点群をPLYで保存
        ・可視化しやすいようにスカラーを0-1で正規化して色付け（グレースケール）
        """
        ensure_dir(out_ply)
        s = scalar.astype(np.float64)
        if s.size == 0:
            s = np.zeros((points.shape[0],), dtype=np.float64)
        s_min, s_max = float(np.min(s)), float(np.max(s))
        if s_max > s_min:
            sn = (s - s_min) / (s_max - s_min)
        else:
            sn = np.zeros_like(s)
        colors = np.stack([sn, sn, sn], axis=1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        if not o3d.io.write_point_cloud(str(out_ply), pcd, write_ascii=False):
            raise IOError(f"PLYの書き出しに失敗: {out_ply}")