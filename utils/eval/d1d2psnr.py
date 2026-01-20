import numpy as np
import open3d as o3d

# =========================
# 共通のユーティリティ
# =========================

def load_points_from_ply(path):
    """
    PLYファイルから点群座標を読み込む関数
    出力は (N, 3) のnumpy配列
    """
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    return pts

def compute_peak_from_bbox(points):
    """
    バウンディングボックスの対角長をpeak値として計算する
    PSNRの分子側のピーク値に用いる
    """
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    diag = np.linalg.norm(maxs - mins)
    return diag

# =========================
# D1 PSNR（点対点）の計算
# =========================

def compute_d1_psnr_from_arrays(ref_points, rec_points):
    """
    ref_points: 元の点群 (N_ref, 3)
    rec_points: 復元後の点群 (N_rec, 3)
    戻り値: D1 PSNR の値（float）
    """
    # KDTreeを作成（Open3DのKDTreeを使用）
    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(ref_points)
    rec_pcd = o3d.geometry.PointCloud()
    rec_pcd.points = o3d.utility.Vector3dVector(rec_points)

    ref_kd = o3d.geometry.KDTreeFlann(ref_pcd)
    rec_kd = o3d.geometry.KDTreeFlann(rec_pcd)

    # 復元→元（forward）
    fwd_sq_dists = []
    for p in rec_points:
        # k=1の最近傍探索
        _, idx, dist_sq = ref_kd.search_knn_vector_3d(p, 1)
        fwd_sq_dists.append(dist_sq[0])
    fwd_sq_dists = np.array(fwd_sq_dists)

    # 元→復元（backward）
    bwd_sq_dists = []
    for p in ref_points:
        _, idx, dist_sq = rec_kd.search_knn_vector_3d(p, 1)
        bwd_sq_dists.append(dist_sq[0])
    bwd_sq_dists = np.array(bwd_sq_dists)

    # 対称D1: forward / backward の平均
    mse = 0.5 * (fwd_sq_dists.mean() + bwd_sq_dists.mean())

    # peak は元点群のバウンディングボックス対角長
    peak = compute_peak_from_bbox(ref_points)

    if mse == 0:
        return float("inf")

    d1_psnr = 10.0 * np.log10((peak * peak) / mse)
    return d1_psnr

def compute_d1_psnr_from_ply(ref_path, rec_path):
    """
    PLYファイルパスから直接 D1 PSNR を計算するラッパー関数
    """
    ref_points = load_points_from_ply(ref_path)
    rec_points = load_points_from_ply(rec_path)
    return compute_d1_psnr_from_arrays(ref_points, rec_points)

# =========================
# D2 PSNR（点対平面）の計算
# =========================

def estimate_normals(points, k=16):
    """
    点群から法線ベクトルを推定する関数
    Open3DのPCAベース推定を利用する
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # 近傍点数kを用いて法線推定
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
    )
    normals = np.asarray(pcd.normals, dtype=np.float64)
    return normals

def compute_d2_psnr_from_arrays(ref_points, rec_points, k_normal=16):
    """
    ref_points: 元の点群 (N_ref, 3)
    rec_points: 復元後の点群 (N_rec, 3)
    戻り値: D2 PSNR の値（float）
    """
    # 元点群に対して法線を推定
    ref_normals = estimate_normals(ref_points, k=k_normal)

    # KDTreeを作成
    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(ref_points)
    rec_pcd = o3d.geometry.PointCloud()
    rec_pcd.points = o3d.utility.Vector3dVector(rec_points)

    ref_kd = o3d.geometry.KDTreeFlann(ref_pcd)
    rec_kd = o3d.geometry.KDTreeFlann(rec_pcd)

    # 復元→元（forward）: 点対平面距離
    fwd_sq_dists = []
    for p in rec_points:
        _, idx, _ = ref_kd.search_knn_vector_3d(p, 1)
        j = idx[0]
        q = ref_points[j]
        n = ref_normals[j]
        # 点pから平面(q, n)への距離
        diff = p - q
        dist = np.dot(diff, n)
        fwd_sq_dists.append(dist * dist)
    fwd_sq_dists = np.array(fwd_sq_dists)

    # 元→復元（backward）: 元点群側でも法線が必要
    rec_normals = estimate_normals(rec_points, k=k_normal)
    bwd_sq_dists = []
    for p in ref_points:
        _, idx, _ = rec_kd.search_knn_vector_3d(p, 1)
        j = idx[0]
        q = rec_points[j]
        n = rec_normals[j]
        diff = p - q
        dist = np.dot(diff, n)
        bwd_sq_dists.append(dist * dist)
    bwd_sq_dists = np.array(bwd_sq_dists)

    mse = 0.5 * (fwd_sq_dists.mean() + bwd_sq_dists.mean())

    peak = compute_peak_from_bbox(ref_points)
    if mse == 0:
        return float("inf")

    d2_psnr = 10.0 * np.log10((peak * peak) / mse)
    return d2_psnr

def compute_d2_psnr_from_ply(ref_path, rec_path, k_normal=16):
    """
    PLYファイルパスから直接 D2 PSNR を計算するラッパー関数
    """
    ref_points = load_points_from_ply(ref_path)
    rec_points = load_points_from_ply(rec_path)
    return compute_d2_psnr_from_arrays(ref_points, rec_points, k_normal=k_normal)

# =========================
# まとめて出力するヘルパー
# =========================

def print_d1_d2_psnr(ref_path, rec_path):
    """
    元点群ファイルと復元点群ファイルから
    D1 / D2 PSNR をまとめて計算してprintする関数
    """
    ref_points = load_points_from_ply(ref_path)
    rec_points = load_points_from_ply(rec_path)

    d1 = compute_d1_psnr_from_arrays(ref_points, rec_points)
    d2 = compute_d2_psnr_from_arrays(ref_points, rec_points)

    # print(f"D1 PSNR: {d1:.4f} dB")
    # print(f"D2 PSNR: {d2:.4f} dB")

    # print(f"D1 & D2 PSNR")
    print(f"{d1:.4f}")
    print(f"{d2:.4f}")

    return [d1, d2]

NU = True
NU2 = True
# NU = False

Data = ["LongDress", "KITTI", "Ford"]

Data = ["LongDress"]
method = ["OctAttention", "Draco", "RENO"]
RepKPU = True
NU_num = 4

for i in range(len(Data)):
    for j in range(len(method)):
        print(f"\n=== {Data[i]} - {method[j]} ===")
        
        if NU == True:
            if RepKPU == True:
                matbin = ["ds-5"]
                nonuniform = f"nonuniform{NU_num}"
                for l in range(len(matbin)):
                    ref = f"../Dataset/ground/{Data[i]}/gt.ply"
                    rec = f"../Dataset/decoded/{Data[i]}/{method[j]}/RepKPU/{nonuniform}/{matbin[l]}.ply"
                    print(f"\n=== {Data[i]} - {method[j]} - {matbin[l]} ===")
                    print_d1_d2_psnr(ref, rec)
            else:
                matbin = ["ds-2", "ds-4", "ds-5", "ds-8", "ds-10", "ds-16", "ds-20"]
                nonuniform = f"nonuniform{NU_num}"
            
        else:
            matbin = ["gt", "ds-2", "ds-4", "ds-5", "ds-8", "ds-10", "ds-16", "ds-20"]
            for l in range(len(matbin)):
                ref = f"../Dataset/ground/{Data[i]}/{matbin[l]}.ply"
                rec = f"../Dataset/decoded/{Data[i]}/{method[j]}/{matbin[l]}.ply"
                print(f"\n=== {Data[i]} - {method[j]} - {matbin[l]} ===")
                print_d1_d2_psnr(ref, rec)

"""
if NU == True:
    if NU2 == True:
        matbin = ["ds-1", "ds-2", "ds-3", "ds-4", "ds-5"]
        # matbin = ["ds-1"]
        nonuniform = "nonuniform2"
    else:
        matbin = ["ds-2", "ds-4", "ds-5", "ds-8", "ds-10", "ds-16", "ds-20"]
        nonuniform = "nonuniform"
        

    data = [[[]]]
    for i in range(len(Data)):
        for j in range(len(method)):
            print(f"\n=== {Data[i]} - {method[j]} ===")
            for l in range(len(matbin)):
                ref = f"../Dataset/{nonuniform}/{Data[i]}/{matbin[l]}.ply"
                rec = f"../Dataset/decoded/{Data[i]}/{method[j]}/{nonuniform}/{matbin[l]}.ply"
                data[i][j].append(print_d1_d2_psnr(ref, rec))
                print(f"{data[i][j][l][0]}", end="	")
            print("")
            for l in range(len(matbin)):
                print(f"{data[i][j][l][1]}", end="	")
            print("")
            data.append([[]])

            # for l in range(len(matbin)):
            #     # if i == 0 & j == 0 & l == 0:
            #     #     ref = f"../Dataset/ground/{Data[i]}/gt.ply"
            #     #     rec = f"../Dataset/ground/{Data[i]}/gt.ply"
            #     #     print(f"\n=== ground - gt ===")
            #     #     print_d1_d2_psnr(ref, rec)

            #     ref = f"../Dataset/{nonuniform}/{Data[i]}/{matbin[l]}.ply"
            #     rec = f"../Dataset/decoded/{Data[i]}/{method[j]}/{nonuniform}/{matbin[l]}.ply"
            #     print(f"\n=== {Data[i]} - {method[j]} - {matbin[l]} ===")
            #     print_d1_d2_psnr(ref, rec)
else:
    matbin = ["gt", "ds-2", "ds-4", "ds-5", "ds-8", "ds-10", "ds-16", "ds-20"]
    for i in range(len(Data)):
        for j in range(len(method)):
            for l in range(len(matbin)):
                ref = f"../Dataset/ground/{Data[i]}/{matbin[l]}.ply"
                rec = f"../Dataset/decoded/{Data[i]}/{method[j]}/{matbin[l]}.ply"
                print(f"\n=== {Data[i]} - {method[j]} - {matbin[l]} ===")
                print_d1_d2_psnr(ref, rec)

"""
        
# dataname = "LongDress"
# matbin = "gt"
# ref = f"../Dataset/ground/LongDress/{matbin}.ply"
# rec = f"../Dataset/decoded/{dataname}/OctAttention/{matbin}.ply"
# print_d1_d2_psnr(ref, rec)

# dataname = "LongDress"
# matbin = "ds-16"
# ref = f"../Dataset/nonuniform/LongDress/{matbin}.ply"
# ref = f"../Dataset/ground/LongDress/gt.ply"
# rec = f"../Dataset/decoded/{dataname}/OctAttention/nonuniform/{matbin}.ply"
# print_d1_d2_psnr(ref, rec)

dataname = "LongDress"
matbin = "gt"
ref = f"../Dataset/ground/LongDress/ds-5.ply"
rec = f"../Dataset/RepKPU/uniform/output/ds-10.ply"
print_d1_d2_psnr(ref, rec)