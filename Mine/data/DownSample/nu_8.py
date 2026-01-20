"""
【システム概要】

1. 入力点群を voxel_size でボクセル分割する
2. 占有ボクセルから NUM_CENTERS 個の中心ボクセルを FPS で選択する
3. 各中心ボクセルを中心とした球状範囲（BALL_RADIUS）内のボクセルを取得
4. 対象ボクセル内の点群をすべて同一倍率（DOWNSAMPLE_RATIO）でダウンサンプリング
5. PAINT_TARGET_VOXELS が True の場合、対象ボクセルの点を赤色に変更
6. 点の削減・色変更後の点群を PLY として保存

7. NUM_CENTERS を変えて複数のダウンサンプリング点群を生成
   （例: ds-1, ds-2, ..., ds-NUM_CENTERS
"""

import os
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from collections import defaultdict

# =========================
# 設定パラメータ
# =========================
DATASET_NAME = "LongDress"
INPUT_NAME = "gt"

VOXEL_SIZE = 5.0                 # ボクセルサイズ
NUM_CENTERS = 10                   # ダウンサンプリング箇所数
BALL_RADIUS = 8                   # 球状範囲（ボクセル単位）
DOWNSAMPLE_RATIO = 0.05            # 残す割合（全ボクセル共通）
PAINT_TARGET_VOXELS = True        # True: 赤く塗る / False: 色保持

# =========================
# FPS（中心ボクセル選択用）
# =========================
def farthest_point_sampling(points, K):
    N = points.shape[0]
    selected = [np.random.randint(N)]
    dist = np.full(N, np.inf)

    for _ in range(1, K):
        last = points[selected[-1]]
        d = np.sum((points - last) ** 2, axis=1)
        dist = np.minimum(dist, d)
        selected.append(np.argmax(dist))

    return selected

INPUT_PLY = f"../Dataset/ground/{DATASET_NAME}/{INPUT_NAME}.ply"

# =========================
# 1. 点群読み込み
# =========================
pcd = o3d.io.read_point_cloud(INPUT_PLY)
pts = np.asarray(pcd.points)
cols = np.asarray(pcd.colors) if pcd.has_colors() else None

# =========================
# 2. ボクセル分割
# =========================
voxel_idx = np.floor(pts / VOXEL_SIZE).astype(int)
voxel_dict = defaultdict(list)
for i, v in enumerate(voxel_idx):
    voxel_dict[tuple(v)].append(i)

occupied_voxels = list(voxel_dict.keys())
vox_coords = np.array(occupied_voxels, dtype=float)

# =========================
# 3. 中心ボクセル選択
# =========================
fps_idx = farthest_point_sampling(vox_coords, min(NUM_CENTERS, len(vox_coords)))
center_voxels = [occupied_voxels[i] for i in fps_idx]

print("中心ボクセル:")
for v in center_voxels:
    print(" ", v)
    
for a in range(NUM_CENTERS):
    OUTPUT_PLY = f"../Dataset/nonuniform4/{DATASET_NAME}/ds-{a+1}.ply"
    print(center_voxels) 
    center_voxels_a = center_voxels[:a+1]
    print(center_voxels_a)
    # =========================
    # 4. 点保持マスク
    # =========================
    keep_mask = np.ones(len(pts), dtype=bool)

    # =========================
    # 5. 球状ダウンサンプリング
    # =========================
    for cv in center_voxels_a:
        cv = np.array(cv)

        for vox in occupied_voxels:
            v = np.array(vox)
            if np.linalg.norm(v - cv) > BALL_RADIUS:
                continue

            idx = np.array(voxel_dict[vox], dtype=int)
            idx = idx[keep_mask[idx]]

            if len(idx) <= 1:
                continue

            keep_n = max(1, int(len(idx) * DOWNSAMPLE_RATIO))
            if keep_n >= len(idx):
                continue

            keep_local = np.random.choice(idx, keep_n, replace=False)
            drop_local = np.setdiff1d(idx, keep_local)
            keep_mask[drop_local] = False

            if PAINT_TARGET_VOXELS and cols is not None:
                cols[keep_local] = np.array([1.0, 0.0, 0.0])  # 赤

    # =========================
    # 6. 出力点群生成
    # =========================
    new_pts = pts[keep_mask]
    new_cols = cols[keep_mask] if cols is not None else None

    print(f"出力点数: {len(new_pts)}")

    # =========================
    # 7. PLY 保存
    # =========================
    os.makedirs(os.path.dirname(OUTPUT_PLY), exist_ok=True)

    if new_cols is not None:
        new_cols = (new_cols * 255).astype(np.uint8)
        vertex = np.array(
            [(x, y, z, r, g, b) for (x, y, z), (r, g, b) in zip(new_pts, new_cols)],
            dtype=[('x','f4'),('y','f4'),('z','f4'),
                ('red','u1'),('green','u1'),('blue','u1')]
        )
    else:
        vertex = np.array(
            [tuple(p) for p in new_pts],
            dtype=[('x','f4'),('y','f4'),('z','f4')]
        )

    PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write(OUTPUT_PLY)
    print(f"[OK] Saved -> {OUTPUT_PLY}\n")
