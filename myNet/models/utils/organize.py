import torch

def prune_close_points_global(
    points: torch.Tensor,
    target_num: int,
    min_dist: float
):
    """
    全点群に対して，近すぎる点を間引き，
    最終的に target_num 点まで削減する

    Args:
        points: (N, C) torch.Tensor, C>=3
        target_num: int, 最終的に残したい点数
        min_dist: float, 最小許容距離（実座標 or 正規化座標）

    Returns:
        pruned_points: (M, C) torch.Tensor, M<=target_num
    """
    device = points.device
    N, C = points.shape
    xyz = points[:, :3]

    kept_xyz = []
    kept_idx = []

    # --- 距離制約による pruning ---
    for i in range(N):
        p = xyz[i]

        if len(kept_xyz) == 0:
            kept_xyz.append(p)
            kept_idx.append(i)
            continue

        ref = torch.stack(kept_xyz, dim=0)  # (M, 3)
        dist = torch.norm(ref - p, dim=1)

        if torch.all(dist >= min_dist):
            kept_xyz.append(p)
            kept_idx.append(i)

    kept_idx = torch.tensor(kept_idx, device=device)
    pruned = points[kept_idx]  # (M, C)

    # --- 点数が多すぎる場合は FPS で target_num に揃える ---
    if pruned.shape[0] > target_num:
        xyz_pruned = pruned[:, :3].unsqueeze(0)  # (1, M, 3) or (1, 3, M)
        idx = FPS(xyz_pruned, target_num).squeeze(0)  # (target_num,)
        pruned = pruned[idx]

    return pruned