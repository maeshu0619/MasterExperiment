import numpy as np

def add_noise_to_random_voxel(
    points: np.ndarray,
    voxel_size: float,
    noise_std: float,
    seed: int = 42
) -> np.ndarray:
    """
    点群をボクセル分割し、ランダムな1〜3個のボクセル内の点にノイズを加える。

    Parameters:
        points: np.ndarray (N,3) - 入力点群
        voxel_size: float - ボクセルサイズ
        noise_std: float - ノイズの標準偏差（例：0.01）
        seed: int - 乱数シード（再現性のため）

    Returns:
        np.ndarray - ノイズを加えた点群（(N,3)）
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("pointsは(N,3)の配列である必要がある")

    rng = np.random.default_rng(seed)
    pmin = np.min(points, axis=0)
    voxel_indices = np.floor((points - pmin) / voxel_size).astype(np.int32)

    # 各occupied voxelに属する点のインデックスを収集
    voxel_to_indices = {}
    for i, idx in enumerate(voxel_indices):
        key = tuple(idx)
        voxel_to_indices.setdefault(key, []).append(i)

    occupied_keys = list(voxel_to_indices.keys())
    if len(occupied_keys) == 0:
        raise ValueError("occupied voxelが見つかりませんでした")

    select_k = min(3, len(occupied_keys))
    selected_keys = rng.choice(occupied_keys, size=select_k, replace=False)

    # ノイズを加える
    points_new = points.copy()
    for key in selected_keys:
        indices = voxel_to_indices[key]
        noise = rng.normal(loc=0.0, scale=noise_std, size=(len(indices), 3))
        points_new[indices] += noise

    return points_new
