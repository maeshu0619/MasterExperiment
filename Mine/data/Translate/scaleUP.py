import open3d as o3d
import numpy as np
import os

# ===============================
# 入出力パスの指定
# ===============================
input_ply_path = "../Dataset/ground/CWI/gt.ply"
output_ply_path = "../Dataset/ground/CWI/gt.ply"

# ===============================
# PLY 読み込み
# ===============================
pcd = o3d.io.read_point_cloud(input_ply_path)
points = np.asarray(pcd.points)  # float, 正規化済み想定


def scale_pointcloud_for_pcc(
    points: np.ndarray,
    quant_bits: int = 10
):
    """
    PCC / Octree 用に点群を整数スケールへ変換する関数
    """

    assert points.ndim == 2 and points.shape[1] == 3

    # 各軸の min / max
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)

    # ゼロ割防止
    scale = max_xyz - min_xyz
    scale[scale == 0] = 1.0

    # [0,1] 正規化
    points_01 = (points - min_xyz) / scale

    # 量子化
    max_coord = (1 << quant_bits) - 1
    scaled_points = np.floor(points_01 * max_coord).astype(np.int32)

    # デバッグ出力
    print("min:", scaled_points.min(axis=0))
    print("max:", scaled_points.max(axis=0))
    print("unique x:", len(np.unique(scaled_points[:, 0])))
    print("unique y:", len(np.unique(scaled_points[:, 1])))
    print("unique z:", len(np.unique(scaled_points[:, 2])))
    print(f"[INFO] Scaling for PCC: quant_bits={quant_bits}, max_coord={max_coord}")

    return scaled_points


# ===============================
# PCC 用スケーリング
# ===============================
scaled_points = scale_pointcloud_for_pcc(points, quant_bits=10)

# Open3D PointCloud に戻す
pcd_scaled = o3d.geometry.PointCloud()
pcd_scaled.points = o3d.utility.Vector3dVector(
    scaled_points.astype(np.float64)
)

# ===============================
# 出力ディレクトリ作成
# ===============================
output_dir = os.path.dirname(output_ply_path)
if output_dir != "":
    os.makedirs(output_dir, exist_ok=True)

# ===============================
# PLY 保存
# ===============================
o3d.io.write_point_cloud(output_ply_path, pcd_scaled)
print(f"[INFO] Saved scaled point cloud to: {output_ply_path}")
