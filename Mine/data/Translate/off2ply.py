import os
import numpy as np
from plyfile import PlyData, PlyElement
import open3d as o3d

FILENAME = "ModelNet40"
in_file = f"../Dataset/ground/{FILENAME}/gt.off"
out_file = f"../Dataset/ground/{FILENAME}/gt.ply"

def convert_off_to_ply(input_off, output_ply):
    if not os.path.exists(input_off):
        print(f"[ERROR] 指定されたファイルが存在しません: {input_off}")
        return
    try:
        # OFFファイルを読み込み
        with open(input_off, 'r') as f:
            lines = f.readlines()
            
        # BOM除去
        header_line = lines[0].strip().replace('\ufeff', '')

        # OFF判定 + 頂点数・面数取得
        if header_line == "OFF":
            # 通常形式
            n_vertices, n_faces, _ = map(int, lines[1].strip().split())
            vertex_start_index = 2
        elif header_line.startswith("OFF"):
            # 省略形式
            parts = header_line[3:].strip().split()
            n_vertices, n_faces, _ = map(int, parts)
            vertex_start_index = 1
        else:
            print(f"[ERROR] {input_off} はOFF形式ではありません (先頭行: {header_line})")
            return

        # 頂点座標の読み込み
        vertices = np.array([list(map(float, line.strip().split())) for line in lines[vertex_start_index:vertex_start_index + n_vertices]])

        print(f"[INFO] 変換中: {input_off}, 頂点数: {n_vertices}")

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
        # for i in range(len(x)):
        #     row = [x[i], y[i], z[i]]
        #     if 'red' in vertex.dtype.names:
        #         row += [vertex['red'][i], vertex['green'][i], vertex['blue'][i]]
        #     values.append(tuple(row))

        vertex_new = np.array(values, dtype=new_props)

        # 新しいPLY構造を作成（ASCII形式で保存）
        el = PlyElement.describe(vertex_new, 'vertex')
        PlyData([el], text=True).write(output_ply)
        print(f"[OK] Converted: {output_ply}")
    except Exception as e:
        print(f"[ERROR] {input_off}: {e}")

convert_off_to_ply(in_file, out_file)