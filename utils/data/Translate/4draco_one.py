import os
import numpy as np
from plyfile import PlyData, PlyElement

# ===== 設定 =====
input_dir = "./Dataset/Ford/PLY"
output_dir = "./Dataset/Ford/PLY/Draco"
target_file = "Scan1000.ply"   # ← ここを変換したいファイル名に変更
# =================

os.makedirs(output_dir, exist_ok=True)

input_path = os.path.join(input_dir, target_file)
output_path = os.path.join(output_dir, target_file)

if not os.path.exists(input_path):
    print(f"[ERROR] 指定されたファイルが存在しません: {input_path}")
else:
    try:
        ply = PlyData.read(input_path)
        vertex = ply['vertex'].data

        # 座標をfloat32に変換
        x = np.array(vertex['x'], dtype=np.float32)
        y = np.array(vertex['y'], dtype=np.float32)
        z = np.array(vertex['z'], dtype=np.float32)

        # その他の属性（色など）も保持
        new_props = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        if 'red' in vertex.dtype.names:
            new_props += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

        values = []
        for i in range(len(x)):
            row = [x[i], y[i], z[i]]
            if 'red' in vertex.dtype.names:
                row += [vertex['red'][i], vertex['green'][i], vertex['blue'][i]]
            values.append(tuple(row))

        vertex_new = np.array(values, dtype=new_props)

        # 新しいPLY構造を作成（ASCII形式で保存）
        el = PlyElement.describe(vertex_new, 'vertex')
        PlyData([el], text=True).write(output_path)

        print(f"[OK] Converted: {target_file}")

    except Exception as e:
        print(f"[ERROR] {target_file}: {e}")
