import numpy as np

def kitti_bin_to_ply(bin_path, ply_path):
    # KITTI LiDARは float32 × 4 (x, y, z, intensity)
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3]

    n = xyz.shape[0]

    # PLYヘッダ（ASCII形式）
    header = f"""ply
                format ascii 1.0
                element vertex {n}
                property float x
                property float y
                property float z
                end_header
                """

    with open(ply_path, "w") as f:
        f.write(header)
        for x, y, z in xyz:
            f.write(f"{x} {y} {z}\n")

if __name__ == "__main__":
    # bin_path = "./Dataset/KITTI/training/velodyne/000000.bin"
    # ply_path = "./Dataset/KITTI/training/PLY/000000.ply"
    # bin_path = "../NeRF/SCNeRF/my_scene/sparse/0/points3D.bin"
    bin_path = "../NeRF/nerfstudio/data/colmap/sparse/0/points3D.bin"
    ply_path = "../Dataset/SCNeRF/point3D.ply"
    kitti_bin_to_ply(bin_path, ply_path)
