import os
import numpy as np
import open3d as o3d
from pathlib import Path

# ---------- 設定 ----------
PCD_PATH = r"F:\Experiments\MasterEx\pcd-dataset\PU-GAN\complex\AncientTurtl_aligned.pcd"
VOXEL_SIZE = 0.05
DS_RATIO = 1.0 / 10.0
OUT_DIR = Path("down_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- ユーティリティ ----------
def numpy_to_pcd(xyz: np.ndarray, color=None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        col = np.tile(np.array(color, dtype=np.float64)[None, :], (xyz.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(col)
    return pcd

def fps_indices(xyz: np.ndarray, n_samples: int, seed: int = 0) -> np.ndarray:
    N = xyz.shape[0]
    n = int(min(max(1, n_samples), N))
    rng = np.random.default_rng(seed)
    idxs = np.empty(n, dtype=np.int64)
    idxs[0] = rng.integers(0, N)

    dists = np.full(N, np.inf, dtype=np.float64)
    last = idxs[0]
    for i in range(1, n):
        diff = xyz - xyz[last]
        d = np.linalg.norm(diff, axis=1)
        dists = np.minimum(dists, d)
        last = int(np.argmax(dists))
        idxs[i] = last
    return idxs

def voxel_index(points: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.floor((points - origin) / voxel_size).astype(np.int32)

"""
# ---------- 点群読み込み ----------
pcd = o3d.io.read_point_cloud(PCD_PATH)
if pcd.is_empty():
    raise RuntimeError(f"点群が読み込めない: {PCD_PATH}")
pts = np.asarray(pcd.points, dtype=np.float64)
N = pts.shape[0]

# ---------- Voxel分割 ----------
pmin = np.min(pts, axis=0)
v_indices = voxel_index(pts, pmin, VOXEL_SIZE)

# ---------- occupied voxel（点が存在するボクセル）を収集 ----------
voxel_to_indices = {}
for i, idx in enumerate(v_indices):
    key = tuple(idx)
    voxel_to_indices.setdefault(key, []).append(i)

# ---------- occupied voxel からランダムに 1〜3 個選択 ----------
rng = np.random.default_rng(42)
occupied_keys = list(voxel_to_indices.keys())
select_k = min(3, len(occupied_keys))
chosen_keys = rng.choice(occupied_keys, size=select_k, replace=False)

# ---------- FPS ダウンサンプリング（選ばれたボクセルのみ） ----------
kept_points = []
downsampled_points = []

for key, indices in voxel_to_indices.items():
    voxel_pts = pts[indices]
    if key in chosen_keys:
        k = max(10, int(len(voxel_pts) * DS_RATIO))
        idx = fps_indices(voxel_pts, k, seed=rng.integers(0, 10000))
        downsampled_points.append(voxel_pts[idx])
    else:
        kept_points.append(voxel_pts)

# ---------- 統合 ----------
final_pts = np.concatenate(downsampled_points + kept_points, axis=0)
pcd_out = numpy_to_pcd(final_pts, color=(1.0, 0.6, 0.2))

# ---------- 保存 ----------
out_path = OUT_DIR / "partial_voxel_fps.pcd"
o3d.io.write_point_cloud(str(out_path), pcd_out, write_ascii=True)
print(f"保存しました: {out_path}")

# ---------- 表示 ----------
o3d.visualization.draw_geometries([pcd_out], window_name="Partially FPS Sampled", width=1024, height=768)
"""