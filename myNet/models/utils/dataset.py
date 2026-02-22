# dataset.py
import os
import numpy as np
import open3d as o3d
import torch

# def load_ply(path):
#     pcd = o3d.io.read_point_cloud(path)
#     points = np.asarray(pcd.points, dtype=np.float32) 
#     return points

def load_ply(path, return_color=True):
    pcd = o3d.io.read_point_cloud(path)

    # 座標 (N,3)
    xyz = np.asarray(pcd.points, dtype=np.float32)

    if not return_color:
        return xyz

    # 色 (N,3) があるなら取得、なければゼロ埋め
    if pcd.has_colors():
        rgb = np.asarray(pcd.colors, dtype=np.float32)  # 通常 0〜1
    else:
        rgb = np.zeros_like(xyz, dtype=np.float32)

    pts = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
    return pts


class PlyDirDataset(torch.utils.data.Dataset):
    """
    - ディレクトリが渡された場合：
        その直下の .ply をすべて扱う
    - 単一 .ply ファイルが渡された場合：
        そのファイルのみを扱う
    """
    def __init__(self, args, path):
        max_files = args.max_files
        # 単一 ply ファイルの場合
        if os.path.isfile(path):
            if not path.endswith(".ply"):
                raise ValueError(f"Input file is not a .ply file: {path}")
            self.files = [path]

        # ディレクトリの場合
        elif os.path.isdir(path):
            self.files = [
                os.path.join(path, f)
                for f in sorted(os.listdir(path))
                if f.endswith(".ply")
            ]
            if len(self.files) == 0:
                raise ValueError(f"No .ply files found in directory: {path}")
            self.files = self.files[:max_files]

        else:
            raise ValueError(f"Invalid path: {path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        points = load_ply(path)              # (N, 3)
        points = torch.from_numpy(points).float()
        return points

def collect_seq_dirs(root):
    seq_dirs = []
    for dataset in os.listdir(root):
        d1 = os.path.join(root, dataset)
        for seq in os.listdir(d1):
            d2 = os.path.join(d1, seq)
            if os.path.isdir(d2):
                seq_dirs.append(d2)
    return sorted(seq_dirs)

def collect_seq_dirs2(root, dataset_name=None):
    """
    root:
        ../data/train/video_noised など
    dataset_name:
        "UVG" や "CWI" を指定
        None の場合は従来どおり全datasetを対象
    """
    seq_dirs = []

    if dataset_name is not None:
        # 指定された dataset のみ
        d1 = os.path.join(root, dataset_name)
        if not os.path.isdir(d1):
            raise ValueError(f"Dataset not found: {d1}")

        for seq in sorted(os.listdir(d1)):
            d2 = os.path.join(d1, seq)
            if os.path.isdir(d2):
                seq_dirs.append(d2)

    else:
        # 従来どおり全 dataset
        for dataset in sorted(os.listdir(root)):
            d1 = os.path.join(root, dataset)
            if not os.path.isdir(d1):
                continue
            for seq in sorted(os.listdir(d1)):
                d2 = os.path.join(d1, seq)
                if os.path.isdir(d2):
                    seq_dirs.append(d2)

    return seq_dirs
