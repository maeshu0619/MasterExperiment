import torch


def three_nn(xyz1: torch.Tensor, xyz2: torch.Tensor):
    """
    xyz1 : [B, 3, M]  unknown points
    xyz2 : [B, 3, N]  known points

    return:
        dist : [B, M, 3]
        idx  : [B, M, 3]
    """

    # [B, M, 3] -> [B, M, 1, 3]
    xyz1_expand = xyz1.transpose(1, 2).unsqueeze(2)
    # [B, N, 3] -> [B, 1, N, 3]
    xyz2_expand = xyz2.transpose(1, 2).unsqueeze(1)

    # pairwise squared distance: [B, M, N]
    dist2 = torch.sum((xyz1_expand - xyz2_expand) ** 2, dim=-1)

    # top-3 nearest neighbors
    dist2, idx = torch.topk(dist2, k=3, dim=-1, largest=False, sorted=False)

    # sqrt to match original TF behavior
    dist = torch.sqrt(dist2)

    return dist, idx

# utils_pointnet.py の three_nn を書き換える

def three_nn_fp(xyz1, xyz2):
    """
    xyz1: [B, 3, M]  (query)
    xyz2: [B, 3, N]  (source)
    """

    B, _, M = xyz1.shape
    _, _, N = xyz2.shape

    xyz1 = xyz1.transpose(1, 2)  # [B, M, 3]
    xyz2 = xyz2.transpose(1, 2)  # [B, N, 3]

    # 距離行列計算（展開しない）
    dist = torch.cdist(xyz1, xyz2, p=2)  # [B, M, N]

    # 上位3近傍取得
    dist, idx = torch.topk(dist, 3, dim=-1, largest=False, sorted=False)

    return dist, idx

def three_nn_chunked(xyz1, xyz2, chunk=2048):

    B, _, M = xyz1.shape
    xyz1 = xyz1.transpose(1, 2)
    xyz2 = xyz2.transpose(1, 2)

    dists = []
    idxs = []

    for i in range(0, M, chunk):
        xyz1_part = xyz1[:, i:i+chunk, :]
        dist_part = torch.cdist(xyz1_part, xyz2)
        dist_k, idx_k = torch.topk(dist_part, 3, dim=-1, largest=False)
        dists.append(dist_k)
        idxs.append(idx_k)

    return torch.cat(dists, 1), torch.cat(idxs, 1)

def three_interpolate(points: torch.Tensor,
                      idx: torch.Tensor,
                      weight: torch.Tensor):
    """
    points : [B, C, N]
    idx    : [B, M, 3]
    weight : [B, M, 3]

    return:
        interpolated_points : [B, C, M]
    """

    # --- 形状取得 ---
    B, C, N = points.shape
    _, M, K = idx.shape   # K = 3

    # --- safety ---
    if idx.dtype != torch.long:
        idx = idx.long()

    # [B, C, N] -> [B, N, C]
    points_trans = points.transpose(1, 2).contiguous()

    # idx: [B, M, 3] -> [B, M*3]
    idx_flat = idx.reshape(B, M * K)

    # points_trans : [B, N, C]
    # idx_flat     : [B, M*3]
    gathered = torch.gather(
        points_trans,
        dim=1,
        index=idx_flat.unsqueeze(-1).expand(-1, -1, C)
    )  # [B, M*3, C]

    # [B, M*3, C] -> [B, M, 3, C]
    gathered = gathered.view(B, M, K, C)

    # weight: [B, M, 3] -> [B, M, 3, 1]
    weight_expand = weight.unsqueeze(-1)

    # weighted sum over K=3
    interpolated = torch.sum(gathered * weight_expand, dim=2)  # [B, M, C]

    # [B, M, C] -> [B, C, M]
    return interpolated.transpose(1, 2).contiguous()