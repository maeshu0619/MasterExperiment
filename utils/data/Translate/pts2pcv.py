import os
import numpy as np
import open3d as o3d

from file_road import get_files, get_single_file_name

# 入力と出力ファイル
input_file_path = r"..\..\Dataset\PartAnnotation\04379243\points"
input_files = get_files(input_file_path)
output_path = r"F:\Experiments\MasterEx\pcv-dataset\PartAnnotation\04379243"

#data_file_path = r"F:\Dataset\PartAnnotation"
#data_files = get_files(data_file_path)

def convert_pts_to_pcd(input_files, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)  # 出力フォルダがなければ作成
    
    for file in input_files:
        if file.lower().endswith('.pts'):  # .ptsファイルのみ処理
            # PTSファイルを読み込み（各行: x y z）
            with open(file, 'r') as f:
                lines = f.readlines()

            # 数値データに変換
            vertices = np.array([list(map(float, line.strip().split())) for line in lines])

            print(f"変換中: {file}, 頂点数: {len(vertices)}")

            # 点群をOpen3Dオブジェクトに変換
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)

            # 出力ファイル名を決定
            base_name = get_single_file_name(file, with_extension=False)  # 拡張子なし
            output_pcd_file = os.path.join(output_path, base_name + '.pcd')

            # PCDをASCII形式で保存
            o3d.io.write_point_cloud(output_pcd_file, pcd, write_ascii=True)
            print(f"PCDファイルを保存しました: {output_pcd_file}")
        else:
            print(f"スキップ: {file} はPTS形式ではありません")

# 変換実行
convert_pts_to_pcd(input_files, output_path)


"""
# OFFファイル読み込み
with open(input_off, 'r') as f:
    lines = f.readlines()

if lines[0].strip() != "OFF":
    raise ValueError("OFF形式のファイルではありません")

# 頂点数と面数
n_vertices, n_faces, _ = map(int, lines[1].strip().split())

# 頂点座標の読み込み
vertices = np.array([list(map(float, line.strip().split())) for line in lines[2:2 + n_vertices]])

print(f"頂点数: {n_vertices}, 読み込んだ点数: {vertices.shape}")

# 点群をOpen3Dオブジェクトに変換
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vertices)

# PCDをASCII形式で保存
o3d.io.write_point_cloud(output_pcd, pcd, write_ascii=True)
print(f"PCDファイルをASCIIで保存しました: {output_pcd}")

# 可視化
o3d.visualization.draw_geometries([pcd])
"""