import open3d as o3d
import numpy as np
import os
from glob import glob

# ===============================
# 入出力ルート
# ===============================
input_root = "../data/train/video"
output_root = "../data/train/video_scaled"

quant_bits = 10

# ===============================
# PCC 用スケーリング関数（完全版）
# ===============================
def scale_pointcloud_for_pcc(points: np.ndarray, quant_bits: int = 10):
    """
    ・形状を歪ませない（等方スケール）
    ・上下反転を補正
    ・PCC 用整数グリッドへ変換
    """
    assert points.ndim == 2 and points.shape[1] == 3

    # ---------- 座標系補正（Y反転） ----------
    points = points.copy()
    points[:, 1] *= -1   # 上下反転補正

    # ---------- バウンディングボックス ----------
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)

    size = max_xyz - min_xyz
    max_range = np.max(size)

    if max_range == 0:
        max_range = 1.0

    # ---------- 等方正規化 ----------
    points_01 = (points - min_xyz) / max_range

    # ---------- PCC グリッド化 ----------
    max_coord = (1 << quant_bits) - 1
    scaled_points = np.round(points_01 * max_coord).astype(np.float64)

    return scaled_points

# ===============================
# すべての ply を再帰的に取得
# ===============================
ply_files = glob(os.path.join(input_root, "**", "*.ply"), recursive=True)

print(f"[INFO] Found {len(ply_files)} PLY files")

# ===============================
# 順番に処理
# ===============================
for ply_path in ply_files:

    rel_path = os.path.relpath(ply_path, input_root)
    output_path = os.path.join(output_root, rel_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[PROCESS] {rel_path}")

    # ---------- 読み込み ----------
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)

    # ---------- スケーリング ----------
    scaled_points = scale_pointcloud_for_pcc(points, quant_bits)

    # ---------- 書き込み ----------
    pcd_scaled = o3d.geometry.PointCloud(pcd)
    pcd_scaled.points = o3d.utility.Vector3dVector(scaled_points)

    o3d.io.write_point_cloud(output_path, pcd_scaled)

print("[INFO] All point clouds scaled successfully.")
