"""
【システム概要（簡易箇条書き）】

1. 入力点群を voxel_size によってボクセル分割する
2. 占有ボクセルの中から 3〜5 個の中心ボクセルをランダムに選ぶ（これで終了条件）
3. 各中心ボクセルごとに
      ・ダウンサンプリング倍率 base_ratio をランダムに生成（例 0.05〜0.4）
      ・間引く範囲（最大階層）もランダムに決定（例 2〜6階層）
4. ボクセル座標 (vx,vy,vz) に基づき、
      中心ボクセルから距離 level のマンハッタン近傍ボクセルを取得
5. 各近傍レベルごとに倍率を変化させながら点を間引く
      ratio = base_ratio × (1 + α × level)  ※ α もランダム
6. 点群の属性情報（色）も座標と同時に保持したまま PLY 出力する
"""

import os
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from collections import defaultdict

def farthest_point_sampling(points, K):
    """
    points: (N, 3) の numpy array （ボクセル座標）
    K: 選びたい個数
    """
    N = points.shape[0]
    centers = []

    # ランダムに1つ目を選ぶ
    first = np.random.randint(N)
    centers.append(first)

    # 各点との最短距離を保持
    dist = np.full(N, np.inf)

    for _ in range(1, K):
        # すでに選んだ中心との距離を更新
        last_center = centers[-1]
        diff = points - points[last_center]
        d = np.sum(diff * diff, axis=1)
        dist = np.minimum(dist, d)

        # 最も遠い点を次の中心に選ぶ
        next_center = np.argmax(dist)
        centers.append(next_center)

    return centers

# =========================
#  設定
# =========================
datasetname = "LongDress"
inputname = "gt"
num = 5

input_ply  = f"../Dataset/ground/{datasetname}/{inputname}.ply"

voxel_size = 10.0  # ボクセルサイズ

min_ratio = 0.01  # 最小削減倍率
max_ratio = 0.2   # 最大削減倍率（中心ボクセル基準）

min_levels = 6    # 削る最小近傍階層
max_levels = 12    # 削る最大近傍階層


# =========================
# 1. 点群読み込み
# =========================
pcd = o3d.io.read_point_cloud(input_ply)
pts = np.asarray(pcd.points)
has_color = pcd.has_colors()

if has_color:
    cols = np.asarray(pcd.colors)   # (N,3) in [0,1]

N_total = pts.shape[0]
print(f"入力点数: {N_total}")

# =========================
# 2. ボクセルインデックス計算
# =========================
voxel_idx = np.floor(pts / voxel_size).astype(int)

# =========================
# 3. ボクセル辞書作成
# =========================
voxel_dict = defaultdict(list)
for i, vid in enumerate(voxel_idx):
    voxel_dict[tuple(vid)].append(i)

occupied_voxels = list(voxel_dict.keys())
print(f"占有ボクセル数: {len(occupied_voxels)}")

# =========================
# 4. センターボクセル 3〜5個ランダム選択（これで終了）
# =========================
# num_centers = np.random.randint(3, 6)
num_centers = num
num_centers = min(num_centers, len(occupied_voxels))

center_indices = np.random.choice(
    len(occupied_voxels),
    size=num_centers,
    replace=False
)

# ボクセル座標 → FPS 用座標 (floatに変換してもOK)
vox_coords = np.array(occupied_voxels, dtype=float)

K = num_centers  # 3〜5 など
fps_idx = farthest_point_sampling(vox_coords, K)
center_voxels = [occupied_voxels[i] for i in fps_idx]


print("選択された中心ボクセル:")
for cv in center_voxels:
    print("  ", cv)

# =========================
# 5. 各センターボクセルに対してランダム設定を生成
# =========================
center_settings = {}
for cv in center_voxels:
    # base_ratio = np.random.uniform(min_ratio, max_ratio)
    # levels = np.random.randint(min_levels, max_levels + 1)
    base_ratio = 0.05
    levels = 10

    alpha = np.random.uniform(0.3, 0.8)

    center_settings[cv] = {
        "base_ratio": base_ratio,
        "levels": levels,
        "alpha": alpha
    }

ini_num = num
for k in range(num):
    print(f"\n===== 実行 {k+1}/{ini_num} =====")
    outputname = f"ds-{num}"
    output_ply = f"../Dataset/nonuniform2/{datasetname}/{outputname}.ply"
    os.makedirs(os.path.dirname(output_ply), exist_ok=True)
    print("\n中心ごとのランダム設定:", center_settings)

    # =========================
    # 6. 点保持マスク
    # =========================
    keep_mask = np.ones(N_total, dtype=bool)

    # =========================
    # 7. センターボクセル周辺だけを削る処理（1回で終了）
    # =========================
    for cv in center_voxels:

        base_ratio = center_settings[cv]["base_ratio"]
        max_lv = center_settings[cv]["levels"]
        alpha = center_settings[cv]["alpha"]

        print(f"\n中心 {cv} を処理（levels={max_lv}, base={base_ratio:.3f}）")

        for level in range(max_lv + 1):

            ratio = min(base_ratio * (1 + alpha * level), 1.0)

            neighbors = [
                (cv[0] + dx, cv[1] + dy, cv[2] + dz)
                for dx in range(-level, level + 1)
                for dy in range(-level, level + 1)
                for dz in range(-level, level + 1)
                if abs(dx) + abs(dy) + abs(dz) == level
            ]

            for vox in neighbors:

                if vox not in voxel_dict:
                    continue

                pts_idx = np.array(voxel_dict[vox], dtype=int)
                pts_idx = pts_idx[keep_mask[pts_idx]]

                N_v = len(pts_idx)
                if N_v <= 1:
                    continue

                keep_num = max(1, int(N_v * ratio))
                if keep_num >= N_v:
                    continue

                keep_local = np.random.choice(pts_idx, keep_num, replace=False)
                drop_local = np.setdiff1d(pts_idx, keep_local)

                keep_mask[drop_local] = False

    # =========================
    # 8. 出力点群作成
    # =========================
    new_pts = pts[keep_mask]
    print(f"\n最終点数: {new_pts.shape[0]}")

    if has_color:
        new_cols = cols[keep_mask]
        new_cols = (new_cols * 255).astype(np.uint8)

    # =========================
    # 9. PLY 保存（座標＋色）
    # =========================
    if has_color:
        props = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ]
        vertex = np.array(
            [(x, y, z, r, g, b) for (x, y, z), (r, g, b) in zip(new_pts, new_cols)],
            dtype=props
        )
    else:
        props = [('x','f4'),('y','f4'),('z','f4')]
        vertex = np.array([tuple(p) for p in new_pts], dtype=props)

    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=True).write(output_ply)

    print(f"[OK] Saved → {output_ply}")

    if k == num:
        break
    # 中心ボクセルを1つ削除（両方から削除する必要がある）
    remove_cv = center_voxels.pop(0)
    center_settings.pop(remove_cv)
    num -= 1

