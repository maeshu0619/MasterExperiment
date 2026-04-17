# dataset.py
import os
import numpy as np
import open3d as o3d
import torch

_PLY_CACHE = {}

# def load_ply(path):
#     pcd = o3d.io.read_point_cloud(path)
#     points = np.asarray(pcd.points, dtype=np.float32) 
#     return points

def load_ply(path, return_color=True):
    pcd = o3d.io.read_point_cloud(path)

    # 座標 (N,3)
    xyz = np.asarray(pcd.points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[0] == 0 or xyz.shape[1] != 3:
        raise ValueError(f"Empty or invalid point cloud: {path} (shape={xyz.shape})")

    if not return_color:
        return xyz

    # 色 (N,3) があるなら取得、なければゼロ埋め
    if pcd.has_colors():
        rgb = np.asarray(pcd.colors, dtype=np.float32)  # 通常 0〜1
    else:
        rgb = np.zeros_like(xyz, dtype=np.float32)

    pts = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
    return pts


def clear_ply_cache():
    _PLY_CACHE.clear()


class PlyDirDataset(torch.utils.data.Dataset):
    """
    - ディレクトリが渡された場合：
        その直下の .ply をすべて扱う
    - 単一 .ply ファイルが渡された場合：
        そのファイルのみを扱う
    """
    def __init__(self, args, path):
        self.use_cache = bool(getattr(args, "dataset_cache", True))
        if args.trainORtest == "train":
            max_files = args.max_files
        else:
            max_files = args.max_files_test
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
        if self.use_cache:
            cached = _PLY_CACHE.get(path)
            if cached is not None:
                return cached

        points = load_ply(path)
        points = torch.from_numpy(points)
        if self.use_cache:
            _PLY_CACHE[path] = points
        return points

def collect_seq_dirs(root):
    seq_dirs = []
    with os.scandir(root) as root_entries:
        for dataset_entry in root_entries:
            if not dataset_entry.is_dir():
                continue
            with os.scandir(dataset_entry.path) as seq_entries:
                for seq_entry in seq_entries:
                    if seq_entry.is_dir():
                        seq_dirs.append(seq_entry.path)
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

        with os.scandir(d1) as seq_entries:
            for seq_entry in sorted(seq_entries, key=lambda entry: entry.name):
                if seq_entry.is_dir():
                    seq_dirs.append(seq_entry.path)

    else:
        # 従来どおり全 dataset
        with os.scandir(root) as root_entries:
            for dataset_entry in sorted(root_entries, key=lambda entry: entry.name):
                if not dataset_entry.is_dir():
                    continue
                with os.scandir(dataset_entry.path) as seq_entries:
                    for seq_entry in sorted(seq_entries, key=lambda entry: entry.name):
                        if seq_entry.is_dir():
                            seq_dirs.append(seq_entry.path)

    return seq_dirs
