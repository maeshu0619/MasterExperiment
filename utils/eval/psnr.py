import os
import numpy as np
import open3d as o3d


def load_point_cloud_auto(path: str) -> np.ndarray:
    """
    .ply または .obj を自動で点群として読み込む
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File not found: {os.path.abspath(path)}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".ply":
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float32)

    elif ext == ".obj":
        # .obj は三角形メッシュとして読み込む
        mesh = o3d.io.read_triangle_mesh(path)
        if mesh.is_empty():
            raise ValueError(f"[ERROR] Failed to load mesh from {path}")
        pts = np.asarray(mesh.vertices, dtype=np.float32)

    else:
        raise ValueError(f"[ERROR] Unsupported file extension: {ext}")

    if pts.size == 0:
        raise ValueError(f"[ERROR] No points loaded from {path}")

    print(f"[INFO] Loaded {path} ({len(pts)} points)")
    return pts


def match_points(gt_pts: np.ndarray, dec_pts: np.ndarray) -> np.ndarray:
    """
    点数が異なる場合に最近傍対応を作成する
    """
    if len(gt_pts) == len(dec_pts):
        return dec_pts

    print(f"[WARN] Point count mismatch: GT={len(gt_pts)} vs Decoded={len(dec_pts)}")
    if len(dec_pts) == 0:
        raise ValueError("[ERROR] Cannot match points because decoded data is empty.")

    pcd_dec = o3d.geometry.PointCloud()
    pcd_dec.points = o3d.utility.Vector3dVector(dec_pts)
    kdtree = o3d.geometry.KDTreeFlann(pcd_dec)

    matched = []
    for p in gt_pts:
        _, idx, _ = kdtree.search_knn_vector_3d(p, 1)
        matched.append(dec_pts[idx[0]])
    matched = np.array(matched, dtype=np.float32)
    print(f"[INFO] Matched {len(matched)} nearest points")
    return matched


def compute_rmse_psnr(gt_pts: np.ndarray, dec_pts: np.ndarray):
    mse = np.mean(np.square(gt_pts - dec_pts))
    rmse = np.sqrt(mse)
    max_val = np.max(np.abs(gt_pts))
    psnr = 10 * np.log10((max_val ** 2) / mse)
    return rmse, psnr


if __name__ == "__main__":
    num = 20
    draco_qp = 8

    gt_path = f"../Dataset/Draco/longdress/ds-{num}/longdress_vox10_1051.ply"
    # gt_path = "../Dataset/longdress/gt/longdress_vox10_1051.ply"

    # decoded_path = f"../Compress/Voxel/RENO/data/decode/gt.ply"
    decoded_path = f"../Compress/Voxel/RENO/data/decode/ds-{num}.ply"

    # decoded_path = f"../Compress/Octree/OctAttention/decode/gt.ply"
    # decoded_path = f"../Compress/Octree/OctAttention/decode/ds-{num}.ply"
    # decoded_path = f"../Compress/Octree/Draco/build/decode/{draco_qp}/gt.ply"
    # decoded_path = f"../Compress/Octree/Draco/build/decode/{draco_qp}/ds-{num}.ply"

    gt_pts = load_point_cloud_auto(gt_path)
    dec_pts = load_point_cloud_auto(decoded_path)
    dec_pts = match_points(gt_pts, dec_pts)

    rmse, psnr = compute_rmse_psnr(gt_pts, dec_pts)
    print("\n===== Quality Comparison =====")
    print(f"RMSE = {rmse:.6f}")
    print(f"PSNR = {psnr:.3f} dB")
    print(f"Points: GT={len(gt_pts)}, Decoded={len(dec_pts)}")
