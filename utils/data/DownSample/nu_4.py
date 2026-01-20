"""
ランダムに中心ボクセルを選ぶ
中心ボクセルのダウンサンプリング倍率 base_ratio（例：0.1〜0.4） をランダム生成
近傍階層ごとに倍率を
　→ base_ratio → base_ratio×1.5 → base_ratio×2.0 ... のように増加
マンハッタン距離で同じ階層にある近傍ボクセルを取得
各ボクセルで ratio に基づきランダムに点を残して削除
全点数が指定目標に達したら終了
"""


import os
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from collections import defaultdict

# =========================
#  設定
# =========================
datasetname = "LongDress"
inputname = "gt"
num = 20
outputname = f"ds-{num}"
input_ply = f"../Dataset/ground/{datasetname}/{inputname}.ply"
output_ply = f"../Dataset/nonuniform2/{datasetname}/{outputname}.ply"

voxel_size = 8.0  # ボクセルサイズ
min_ratio = 0.05  # 1番疎にする倍率の下限
max_ratio = 0.5   # ランダム倍率の上限（中心ボクセルはここから基準を生成）
neighbor_levels = 3  # 近傍階層の深さ

max_centers = 3  # 一度に選ぶ中心ボクセルの最大数
cnt1 = 0
total_processed = 0

os.makedirs(os.path.dirname(output_ply), exist_ok=True)

# =========================
#  1. 点群読み込み
# =========================
pcd = o3d.io.read_point_cloud(input_ply)
pts = np.asarray(pcd.points)
N_total = pts.shape[0]
target_num_points = N_total / num

print(f"入力点数: {N_total}")
print(f"目標点数: {target_num_points}")

if N_total <= target_num_points:
    new_pts = pts

else:
    # =========================
    #  2. ボクセルインデックスの計算
    # =========================
    voxel_idx = np.floor(pts / voxel_size).astype(int)
    # print(f"voxel_idx.shape: {voxel_idx.shape}")

    # =========================
    #  3. 各ボクセルに点を入れる
    # =========================
    voxel_dict = defaultdict(list)
    for i, vid in enumerate(voxel_idx):
        voxel_dict[tuple(vid)].append(i)

    occupied_voxels = list(voxel_dict.keys())

    num_voxels = len(voxel_dict)
    print(f"占有ボクセル数: {num_voxels}")

    # =========================
    #  4. 全点の保持マスク
    # =========================
    keep_mask = np.ones(N_total, dtype=bool)

    # =========================
    #  5. 目標点数までランダムにスパース化を繰り返す
    # =========================
    while keep_mask.sum() > target_num_points:

        # ランダムに max_centers 個だけ選ぶ
        selected_centers = np.random.choice(
            len(occupied_voxels), 
            size=min(max_centers, len(occupied_voxels)), 
            replace=False
        )

        for center_idx in selected_centers:

            if keep_mask.sum() <= target_num_points:
                break

            center_voxel = occupied_voxels[center_idx]

            # ---- 中心ボクセルのランダム倍率を決定 ----
            base_ratio = np.random.uniform(min_ratio, max_ratio)

            # ---- 近傍階層ごとに倍率を増やす ----
            for level in range(neighbor_levels + 1):

                ratio = min(base_ratio * (1 + 0.5 * level), 1.0)

                # マンハッタン距離 level の近傍ボクセルを取得
                neighbors = [
                    (center_voxel[0] + dx,
                    center_voxel[1] + dy,
                    center_voxel[2] + dz)
                    for dx in range(-level, level + 1)
                    for dy in range(-level, level + 1)
                    for dz in range(-level, level + 1)
                    if abs(dx) + abs(dy) + abs(dz) == level
                ]

                # --- 近傍ボクセルの処理 ---
                for vox in neighbors:

                    if vox not in voxel_dict:
                        continue

                    pts_idx = np.array(voxel_dict[vox])
                    pts_idx = pts_idx[keep_mask[pts_idx]]

                    N_v = len(pts_idx)
                    if N_v <= 1:
                        continue

                    keep_num = max(1, int(N_v * ratio))
                    if keep_num >= N_v:
                        continue

                    # 残す点と消す点を決定
                    keep_local = np.random.choice(pts_idx, keep_num, replace=False)
                    drop_local = np.setdiff1d(pts_idx, keep_local)

                    keep_mask[drop_local] = False
                    total_processed += 1

                    if keep_mask.sum() <= target_num_points:
                        break

        cnt1 += 1
        
    new_pts = pts[keep_mask]

print(f"")
print(f"総処理回数: {cnt1}")
print(f"総処理中心ボクセル数: {max_centers}")
print(f"総処理点数: {total_processed}")
print(f"最終出力点数: {new_pts.shape[0]}")

# =========================
#  6. PLY 保存
# =========================
props = [('x','f4'),('y','f4'),('z','f4')]
vertex = np.array([tuple(p) for p in new_pts], dtype=props)

el = PlyElement.describe(vertex, 'vertex')
PlyData([el], text=True).write(output_ply)

print(f"[OK] Saved: {output_ply}")
