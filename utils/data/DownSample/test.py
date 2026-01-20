import os
import numpy as np
import open3d as o3d

# ---------- 設定 ----------
# Windowsパスは raw 文字列にする
PCD_PATH = r"F:\Experiments\MasterEx\pcd-dataset\PU-GAN\complex\AncientTurtl_aligned.pcd"
# ダウンサンプリング率（1/30）
DS_RATIO = 1.0 / 30.0

# ---------- ユーティリティ ----------
def numpy_to_pcd(xyz: np.ndarray, color=None) -> o3d.geometry.PointCloud:
    """numpy(N,3) → Open3D PointCloud。color=(r,g,b)を与えると単色着色"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    if color is not None:
        col = np.tile(np.array(color, dtype=np.float64)[None, :], (xyz.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(col)
    return pcd

def fps_indices(xyz: np.ndarray, n_samples: int, seed: int = 0) -> np.ndarray:
    """Farthest Point Sampling のインデックス列を返す（O(N * n_samples)）"""
    N = xyz.shape[0]
    n = int(min(max(1, n_samples), N))
    rng = np.random.default_rng(seed)
    idxs = np.empty(n, dtype=np.int64)
    idxs[0] = rng.integers(0, N)

    dists = np.full(N, np.inf, dtype=np.float64)
    last = idxs[0]
    for i in range(1, n):
        # 前回選んだ点からの距離で最短距離を更新
        diff = xyz - xyz[last]
        d = np.einsum('ij,ij->i', diff, diff) ** 0.5  # L2距離
        dists = np.minimum(dists, d)
        last = int(np.argmax(dists))
        idxs[i] = last
    return idxs

# ---------- 読み込み ----------
pcd = o3d.io.read_point_cloud(PCD_PATH)
if pcd.is_empty():
    raise RuntimeError(f"点群が読み込めない: {PCD_PATH}")

pts = np.asarray(pcd.points, dtype=np.float64)
N = pts.shape[0]
if N == 0:
    raise RuntimeError("点数が0である")

# 表示用にオリジナルを薄いグレーに
pcd_orig = numpy_to_pcd(pts, color=(0.75, 0.75, 0.75))

# ---------- ランダムサンプリング ----------
# Open3Dのrandom_down_sampleは比率指定（0<ratio<=1）
ratio = max(1.0 / N, min(1.0, DS_RATIO))
pcd_rand = pcd.random_down_sample(ratio)  # 色は自前で付け直す
pcd_rand.colors = o3d.utility.Vector3dVector(np.tile(np.array([0.3, 0.8, 0.4]), (len(pcd_rand.points), 1)))

# ---------- FPS（約1/30点を選択） ----------
k = max(1, int(round(N * DS_RATIO)))
idx_fps = fps_indices(pts, k, seed=0)
pcd_fps = pcd.select_by_index(idx_fps.tolist())
pcd_fps.colors = o3d.utility.Vector3dVector(np.tile(np.array([0.95, 0.35, 0.35]), (len(pcd_fps.points), 1)))

print(f"総点数 N={N}, ランダム={len(pcd_rand.points)}, FPS={len(pcd_fps.points)}")

# ---------- 表示（3ウィンドウを順に） ----------
"""
o3d.visualization.draw_geometries([pcd_orig], window_name="Original", width=1024, height=768)
o3d.visualization.draw_geometries([pcd_rand], window_name="Random Downsample (~1/30)", width=1024, height=768)
o3d.visualization.draw_geometries([pcd_fps],  window_name="FPS Downsample (~1/30)",    width=1024, height=768)
"""

os.makedirs("down_data", exist_ok=True)
o3d.io.write_point_cloud("down_data/random_downsample.pcd", pcd_rand, write_ascii=True)
o3d.io.write_point_cloud("down_data/fps_downsample.pcd", pcd_fps, write_ascii=True)

print("ダウンサンプリング結果を down_data/ に保存しました。")
# ---------- 終了 ----------