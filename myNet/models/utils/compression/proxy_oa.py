import torch

def _safe_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(torch.clamp(x, -60.0, 60.0))


def _normalize_weight_shape(w: torch.Tensor, B: int, N: int, dtype: torch.dtype, dev: torch.device):
    """
    重みを [B, N] に正規化する
    使える shape:
      - [B, N]
      - [B, 1, N]
    それ以外は None を返す
    """
    if w is None:
        return None

    if w.dim() == 3:
        # [B,1,N] を想定
        if w.size(1) != 1:
            return None
        w = w.squeeze(1)

    if w.dim() != 2:
        return None

    if w.size(0) != B or w.size(1) != N:
        return None

    return w.clamp(0.0, 1.0).to(device=dev, dtype=dtype)


def proxy_octattention_like_octree_loss(pts, args, final_w=None):
    """
    pts    : [B,3,N]
    """
    assert pts.dim() == 3 and pts.size(1) == 3, "pts は [B,3,N] を想定"
    B, _, N = pts.shape
    dev = pts.device
    dtype = pts.dtype

    level = getattr(args, "proxy_level", 6)
    tau_split = getattr(args, "proxy_tau_split", 0.05)
    t_exist = getattr(args, "proxy_t_exist", 0.2)
    tau_exist = getattr(args, "proxy_tau_exist", 0.1)
    alpha_occ = getattr(args, "proxy_alpha_occ", 2.0)

    # ---------------------------------------------------------
    # 座標を voxel 格子へ写像
    # ---------------------------------------------------------
    xyz = pts.permute(0, 2, 1)  # [B,N,3]

    xyz_floor = torch.floor(xyz)
    mins = xyz_floor.amin(dim=1, keepdim=True)   # [B,1,3]
    xyz_shifted = xyz - mins                     # [B,N,3]

    G = 2 ** max(level - 1, 0)
    maxv = xyz_shifted.amax(dim=1, keepdim=True)
    scale = (maxv + 1e-6) / float(G)
    uvw = xyz_shifted / scale

    ijk = torch.floor(uvw).to(torch.long)
    ijk = torch.clamp(ijk, 0, G - 1)

    local = uvw - ijk.to(dtype)
    local = torch.clamp(local, 0.0, 1.0)

    i = ijk[..., 0]
    j = ijk[..., 1]
    k = ijk[..., 2]

    parent_id = i + G * (j + G * k)
    num_parent = G ** 3

    x = local[..., 0]
    y = local[..., 1]
    z = local[..., 2]

    # ---------------------------------------------------------
    # 8 子ノードへの soft occupancy
    # ---------------------------------------------------------
    px1 = _safe_sigmoid((x - 0.5) / max(tau_split, 1e-6))
    py1 = _safe_sigmoid((y - 0.5) / max(tau_split, 1e-6))
    pz1 = _safe_sigmoid((z - 0.5) / max(tau_split, 1e-6))
    px0, py0, pz0 = 1 - px1, 1 - py1, 1 - pz1

    p = torch.stack([
        px0 * py0 * pz0,  # 000
        px1 * py0 * pz0,  # 100
        px0 * py1 * pz0,  # 010
        px1 * py1 * pz0,  # 110
        px0 * py0 * pz1,  # 001
        px1 * py0 * pz1,  # 101
        px0 * py1 * pz1,  # 011
        px1 * py1 * pz1,  # 111
    ], dim=-1)  # [B,N,8]

    p = p / (p.sum(dim=-1, keepdim=True) + 1e-12)
    if final_w is None:
        final_w = torch.ones((B, N), device=dev, dtype=dtype)
    else:
        final_w = _normalize_weight_shape(final_w, B, N, dtype, dev)
        if final_w is None:
            final_w = torch.ones((B, N), device=dev, dtype=dtype)

    p = p * final_w.unsqueeze(-1)

    p = p * final_w.unsqueeze(-1)  # [B,N,8]

    # ---------------------------------------------------------
    # 親ノードごとの occupancy 集計
    # ---------------------------------------------------------
    counts = torch.zeros((B, num_parent, 8), device=dev, dtype=dtype)
    idx = parent_id.unsqueeze(-1).expand(-1, -1, 8)
    counts.scatter_add_(dim=1, index=idx, src=p)

    occ = 1.0 - torch.exp(-alpha_occ * counts)  # [B,num_parent,8]
    pc = occ.sum(dim=-1)                        # [B,num_parent]

    exist = _safe_sigmoid((pc - t_exist) / max(tau_exist, 1e-6))
    nodes_proxy = exist.sum(dim=-1).mean()

    q = occ / (pc.unsqueeze(-1) + 1e-12)

    # 単一子ノードっぽさ
    single_per_parent = (q * q).sum(dim=-1)
    L_single = (single_per_parent * exist).sum() / (exist.sum() + 1e-12)

    # エントロピー
    entropy_per_parent = -(q * torch.log2(q + 1e-9)).sum(dim=-1)
    L_entropy = (entropy_per_parent * exist).sum() / (exist.sum() + 1e-12)

    loss_bit = (
        args.com_node * nodes_proxy
        + args.com_sin * L_single
        + args.com_ent * L_entropy
    )

    stats = {
        "bit": loss_bit.detach(),
        "bpp": (loss_bit / float(N)).detach(),
        "bpn": (loss_bit / (nodes_proxy + 1e-12)).detach(),
        "single": L_single.detach(),
        "node": nodes_proxy.detach(),
        "entropy": L_entropy.detach(),
        "exist_mean": exist.mean().detach(),
        "pc_mean": pc.mean().detach(),
    }

    return loss_bit, stats