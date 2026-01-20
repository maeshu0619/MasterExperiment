import os
import numpy as np
from plyfile import PlyData, PlyElement
import open3d as o3d

FILENAME = "PartAnnotation"
in_file = f"../Dataset/ground/{FILENAME}/gt.pts"
out_file = f"../Dataset/ground/{FILENAME}/gt.ply"

def convert_pts_to_ply(input_pts, output_ply):
    if not os.path.exists(input_pts):
        print(f"[ERROR] 指定されたファイルが存在しません: {input_pts}")
        return
    try:
        # PTSファイルを読み込み（各行: x y z）
        with open(input_pts, 'r') as f:
            lines = f.readlines()

        # 数値データに変換
        vertices = np.array([list(map(float, line.strip().split())) for line in lines])

        print(f"[INFO] 変換中: {input_pts}, 頂点数: {len(vertices)}")

        # 点群をOpen3Dオブジェクトに変換
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        
        vertex = np.asarray(pcd.points)

        x = vertex[:, 0].astype(np.float32)
        y = vertex[:, 1].astype(np.float32)
        z = vertex[:, 2].astype(np.float32)

        new_props = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        values = [(x[i], y[i], z[i]) for i in range(len(x))]
        vertex_new = np.array(values, dtype=new_props)

        # if 'red' in vertex.dtype.names:
        #     new_props += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        #     for i in range(len(x)):
        #         values[i] += (vertex['red'][i], vertex['green'][i], vertex['blue'][i])

        # values = []

        ply_element = PlyElement.describe(vertex_new, 'vertex')
        PlyData([ply_element], text=True).write(output_ply)

        print(f"[INFO] PLYファイルを保存しました: {output_ply}")

    except Exception as e:
        print(f"[ERROR] 変換中にエラーが発生しました: {e}")

# 変換実行
convert_pts_to_ply(in_file, out_file)

