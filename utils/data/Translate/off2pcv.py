import os
import numpy as np
import open3d as o3d

from file_road import get_files, get_single_file_name

"""
# 入力と出力ファイル
input_off = r"F:\Experiments\MasterEx\demo\AncientTurtl_aligned.off"
output_pcd = r"F:\Experiments\MasterEx\demo\gt.ply"
"""

# 入力と出力ファイル
input_file_path = r"..\..\Dataset\ModelNet40\bathtub\test"
input_files = get_files(input_file_path)   
output_file = r"F:\Experiments\MasterEx\pcv-dataset\ModelNet40\bathtub\test"

def convert_off_to_pcd(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)  # 出力フォルダがなければ作成

    for file in input_files:
        if file.lower().endswith('.off'):  # 拡張子が.offの場合のみ処理
            # OFFファイルを読み込み
            with open(file, 'r') as f:
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
                print(f"スキップ: {file} はOFF形式ではありません (先頭行: {header_line})")
                continue

            # 頂点数と面数を取得
            #n_vertices, n_faces, _ = map(int, lines[1].strip().split())

            # 頂点座標の読み込み
            vertices = np.array([list(map(float, line.strip().split())) for line in lines[2:2 + n_vertices]])

            print(f"変換中: {file}, 頂点数: {n_vertices}")

            # 点群をOpen3Dオブジェクトに変換
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)

            # 出力ファイル名を決定
            base_name = get_single_file_name(file, with_extension=False)  # 拡張子なし
            output_pcd_file = os.path.join(output_path, base_name + '.pcd')

            # PCDをASCII形式で保存
            o3d.io.write_point_cloud(output_pcd_file, pcd, write_ascii=True)
            print(f"PCDファイルを保存しました: {output_pcd_file}")
    
# 変換実行
convert_off_to_pcd(input_files, output_file)