"""
・num_select_voxelsの数値を一定にせずに、初めに指定した点群数になるまでダウンサンプルするボクセルを選び続ける 
・全体の点群数が指定した数値以下になったら終了する
"""


import os
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from collections import defaultdict

# ===== 設定 =====
datasetname = "LongDress"
name = "gt"
input_ply = f"../Dataset/ground/{datasetname}/{name}.ply"
output_ply = f"../Dataset/nonuniform/{datasetname}/{name}.ply"

voxel_size = 8.0           # ボクセルサイズ
ratio = 0.3                # スパース化倍率（対象ボクセル内の点をratio倍だけ残す）
target_num_points = 382911  # 最終的に残したい点数の上限（例）

os.makedirs(os.path.dirname(output_ply), exist_ok=True)

# ===== 1. Open3Dで点群を読み込む =====
pcd = o3d.io.read_point_cloud(input_ply)
pts = np.asarray(pcd.points)

# RGB の有無をチェック
has_color = False
if len(np.asarray(pcd.colors)) == pts.shape[0]:
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    has_color = True
else:
    colors = None

N_total = pts.shape[0]
print(f"入力点数: {N_total}")

# 既に目標以下なら何もしないで保存
if N_total <= target_num_points:
    new_pts = pts
    if has_color:
        new_colors = colors
else:
    # ===== 2. ボクセルインデックス計算 =====
    voxel_idx = np.floor(pts / voxel_size).astype(int)

    # ===== 3. 占有ボクセルごとに点をまとめる =====
    voxel_dict = defaultdict(list)
    for i, vid in enumerate(voxel_idx):
        voxel_dict[tuple(vid)].append(i)

    occupied_voxels = list(voxel_dict.keys())

    # ===== 4. 全点を保持するためのマスクを用意 =====
    keep_mask = np.ones(N_total, dtype=bool)

    # ===== 5. 目標点数以下になるまでボクセルを選んでスパース化 =====
    #  ボクセルを何周かする可能性があるので while ループを回す
    while keep_mask.sum() > target_num_points:
        changed = False  # この周回で点数が減ったかどうか
        
        # ボクセルの順番をシャッフル
        perm = np.random.permutation(len(occupied_voxels))
        
        for idx_in_list in perm:
            if keep_mask.sum() <= target_num_points:
                break  # 目標以下になったら即終了
            
            voxel_key = occupied_voxels[idx_in_list]
            point_ids = np.array(voxel_dict[voxel_key], dtype=int)
            
            # すでに削除されている点は除外
            point_ids = point_ids[keep_mask[point_ids]]
            N_voxel = len(point_ids)
            
            # 点が1個以下、もしくは ratio で減らす余地がない場合はスキップ
            if N_voxel <= 1:
                continue
            
            # このボクセル内で残す点数
            keep_num = max(1, int(N_voxel * ratio))
            if keep_num >= N_voxel:
                continue  # 減らせないのでスキップ
            
            # 残す点を選ぶ
            keep_local = np.random.choice(point_ids, keep_num, replace=False)
            # 削除する点 = それ以外
            drop_local = np.setdiff1d(point_ids, keep_local, assume_unique=True)
            
            # マスク更新（削除）
            keep_mask[drop_local] = False
            changed = True
        
        # 1周しても何も減らせなかったら終了（無限ループ防止）
        if not changed:
            print("これ以上スパース化できないため終了")
            break

    # 最終的に残す点群
    new_pts = pts[keep_mask]
    if has_color:
        new_colors = colors[keep_mask]

    print(f"出力点数: {new_pts.shape[0]} (target <= {target_num_points})")

# ===== 6. PLYフォーマットに整形 =====
new_props = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
if has_color:
    new_props += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

values = []
for i in range(new_pts.shape[0]):
    row = [new_pts[i, 0], new_pts[i, 1], new_pts[i, 2]]
    if has_color:
        row += [new_colors[i, 0], new_colors[i, 1], new_colors[i, 2]]
    values.append(tuple(row))

vertex_new = np.array(values, dtype=new_props)

el = PlyElement.describe(vertex_new, 'vertex')
PlyData([el], text=True).write(output_ply)

print(f"[OK] Saved: {output_ply}")
