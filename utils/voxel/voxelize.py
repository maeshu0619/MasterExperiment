import numpy as np
import open3d as o3d

# ---------- 乱数点群を生成 ----------
def create_random_point_cloud(n_points=10000, space=30.0):
    points = np.random.uniform(low=0, high=space, size=(n_points, 3))
    colors = np.random.uniform(low=0, high=1, size=(n_points, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# ---------- 点群生成 ----------
cloud = create_random_point_cloud()

# ---------- Voxelグリッドでダウンサンプリング ----------
leaf_size = 3.0
filtered_cloud = cloud.voxel_down_sample(voxel_size=leaf_size)

print("before:", np.asarray(cloud.points).shape[0])
print("after :", np.asarray(filtered_cloud.points).shape[0])

# ---------- 可視化のための左右シフトと色変更 ----------
cloud_shifted = cloud.translate([-20.0, 0, 0])
cloud_colors = np.asarray(cloud_shifted.colors)
cloud_colors[:, 0] = 0.0  # 赤を0に（青＋緑 = シアン系）

filtered_shifted = filtered_cloud.translate([20.0, 0, 0])
filtered_colors = np.asarray(filtered_shifted.colors)
filtered_colors[:, 2] = 0.0  # 青を0に（赤＋緑 = 黄色系）

# ---------- 可視化 ----------
o3d.visualization.draw_geometries([cloud_shifted, filtered_shifted])
