import os
import numpy as np
import scipy.io as sio

def ford_mat_to_ply(mat_path: str, ply_path: str):
    data = sio.loadmat(mat_path)

    scan = data["SCAN"][0,0]
    XYZ = scan["XYZ"]

    # XYZ が (3, N) の range image 型の場合
    if XYZ.ndim == 2 and XYZ.shape[0] == 3:
        pts = XYZ.T  # → (N, 3)
        print(f"✔ Range image 型点群を検出: {pts.shape[0]} 点")
    else:
        raise RuntimeError("予想外のXYZ構造（3×N形式以外）です")

    n = pts.shape[0]

    # 出力フォルダ作成
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)

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
        for x, y, z in pts:
            f.write(f"{x} {y} {z}\n")

    print("✔ 保存:", ply_path)


if __name__ == "__main__":
    mat_path = "./Dataset/Ford/SCANS/Scan1000.mat"
    ply_path = "./Dataset/Ford/PLY/Scan1000.ply"

    ford_mat_to_ply(mat_path, ply_path)
