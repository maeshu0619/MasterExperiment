"""
・入力の点群データをVoxel化して、部分的にランダムな領域にダウンサンプリングをほどこす 
・選択するVoxelは点群が存在する箇所のみを選択する（何もないボクセルは選ばない） 
・ボクセルの個数と大きさ、ダウンサンプリングの倍率は任意に選択できるものとする
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

voxel_size = 8.0          # 好きなボクセルサイズに変更
num_select_voxels = 5000  # ダウンサンプルするボクセル個数
ratio = 0.3               # 選ばれたボクセル内の点を30%残す

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


# ===== 2. ボクセルインデックス計算 =====
voxel_idx = np.floor(pts / voxel_size).astype(int)

# ===== 3. 占有ボクセルごとに点をまとめる =====
voxel_dict = defaultdict(list)
for i, vid in enumerate(voxel_idx):
    voxel_dict[tuple(vid)].append(i)

occupied_voxels = list(voxel_dict.keys())

# ===== 4. ランダムに部分ボクセルを選択 =====
num_selected = min(num_select_voxels, len(occupied_voxels))
selected_voxels = set(
    np.random.choice(len(occupied_voxels), num_selected, replace=False)
)


# ===== 5. 選ばれたボクセル内のみスパース化 =====
keep_ids = []

for i, voxel_key in enumerate(occupied_voxels):

    point_ids = voxel_dict[voxel_key]

    # このボクセルが「スパース化対象」なら ratio 倍残す
    if i in selected_voxels:
        N = len(point_ids)
        keep = max(1, int(N * ratio))
        chosen = np.random.choice(point_ids, keep, replace=False)
        keep_ids.extend(chosen.tolist())
    else:
        # 非対象ボクセルは全点残す
        keep_ids.extend(point_ids)

keep_ids = np.array(keep_ids)

new_pts = pts[keep_ids]
if has_color:
    new_colors = colors[keep_ids]


# ===== 6. PLYフォーマットに整形（あなたが貼ったコードと同形式） =====
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
