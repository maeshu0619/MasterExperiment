import os
import numpy as np

def xyz_to_ply(root_dir):
    """
    指定ディレクトリ以下のすべての .xyz ファイルを .ply (ASCII形式) に変換する関数。
    階層構造を保ったまま同じ場所に .ply ファイルを生成する。

    Parameters
    ----------
    root_dir : str
        .xyz ファイルが含まれるルートディレクトリ

    Returns
    -------
    converted_files : list of str
        変換された .ply ファイルのパス一覧
    """

    converted_files = []

    # --- ディレクトリを再帰的に探索 ---
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith('.xyz'):
                xyz_path = os.path.join(dirpath, file)
                # ply_path = os.path.splitext(xyz_path)[0] + '.ply'
                ply_path = './Dataset/PLY/' + os.path.relpath(os.path.splitext(xyz_path)[0], './RepKPU/data') + '.ply'
                if not os.path.exists(os.path.dirname(ply_path)):
                    os.makedirs(os.path.dirname(ply_path), exist_ok=True)

                try:
                    # --- XYZファイル読み込み ---
                    points = np.loadtxt(xyz_path, dtype=np.float32)

                    if points.ndim != 2 or points.shape[1] < 3:
                        print(f"[Skip] {xyz_path} は Nx3 の形式ではありません。")
                        continue

                    n_points = points.shape[0]

                    # --- PLYヘッダ作成 ---
                    header = [
                        "ply",
                        "format ascii 1.0",
                        f"element vertex {n_points}",
                        "property float x",
                        "property float y",
                        "property float z",
                        "end_header"
                    ]

                    # --- 書き込み ---
                    with open(ply_path, 'w') as f:
                        f.write("\n".join(header) + "\n")
                        np.savetxt(f, points, fmt="%.6f %.6f %.6f")

                    converted_files.append(ply_path)
                    print(f"[OK] {xyz_path} → {ply_path}")

                except Exception as e:
                    print(f"[Error] {xyz_path} の変換に失敗しました: {e}")

    if not converted_files:
        print(f"[Info] {root_dir} 以下に .xyz ファイルが見つかりませんでした。")
    else:
        print(f"\n[Done] 合計 {len(converted_files)} 件の .xyz を .ply に変換しました。")

    return converted_files


xyz_to_ply("./RepKPU/data/PU-GAN/test/pugan_4x/partial_downsampled")