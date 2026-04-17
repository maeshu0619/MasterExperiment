import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


_REPO_ROOT = Path(__file__).resolve().parents[3]
_OA_DIR = _REPO_ROOT / "compress" / "octree" / "OctAttention"
if str(_OA_DIR) not in sys.path:
    sys.path.append(str(_OA_DIR))


@dataclass
class ProxyOctreeConfig:
    max_depth: int = 12
    qs: float = 2.0
    bptt: int = 1024
    eps: float = 1e-12
    lambda_entropy: float = 1.0
    lambda_node_count: float = 1.0
    lambda_single_child: float = 1.0
    round_tau: float = 0.08
    mass_to_occ_gain: float = 1.0
    min_mass: float = 1e-9
    teacher_chunk_size: int = 2048
    checkpoint_path: Optional[str] = None
    ctx_dim: int = 5
    teacher_device: str = "auto"


class _OctAttentionTeacherModel(nn.Module):
    def __init__(self, max_octree_level: int = 12):
        super().__init__()
        from attentionModel import TransformerLayer, TransformerModule

        ntokens = 255
        ninp = 4 * (128 + 4 + 6)
        nhid = 300
        nlayers = 3
        nhead = 4
        dropout = 0.0

        self.pos_encoder = _PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerModule(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntokens, 128)
        self.encoder1 = nn.Embedding(max_octree_level + 1, 6)
        self.encoder2 = nn.Embedding(9, 4)
        self.decoder0 = nn.Linear(ninp, ninp)
        self.decoder1 = nn.Linear(ninp, ntokens)
        self.act = nn.ReLU()
        self.ninp = ninp
        self.max_octree_level = max_octree_level

    def forward(self, src, src_mask, data_feat=None):
        bptt = src.shape[0]
        batch = src.shape[1]

        oct_code = src[:, :, :, 0]
        level = src[:, :, :, 1]
        octant = src[:, :, :, 2]

        level = level - torch.clamp(level[:, :, -1:] - 10, min=0)
        level = level.clamp_(0, self.max_octree_level)

        a_oct = self.encoder(oct_code.long())
        a_level = self.encoder1(level.long())
        a_octant = self.encoder2(octant.long())
        feat = torch.cat((a_oct, a_level, a_octant), dim=3)
        feat = feat.reshape((bptt, batch, -1)) * math.sqrt(self.ninp)

        output = self.transformer_encoder(feat, src_mask)
        output = self.decoder1(self.act(self.decoder0(output)))
        return output


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SoftOctreeRateProxy(nn.Module):
    """
    OctAttention の実圧縮経路をできるだけ忠実に模倣する微分可能 proxy。

    1. hard 側:
       - qs で量子化
       - 重複除去
       - GenOctree / GenKparentSeq で実 OctAttention と同じ系列を生成
       - 学習済み OctAttention で各ノードの 255-way 分布を取得

    2. soft 側:
       - 量子化を sigmoid で滑らかに近似
       - leaf voxel への確率質量を集計
       - 各親ノードの 8 子 occupancy を soft に推定
       - 上記 teacher 分布に対する期待ビット数を計算
    """

    def __init__(self, cfg: ProxyOctreeConfig):
        super().__init__()
        self.cfg = cfg

        child_bits = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=torch.long,
        )
        code_ids = torch.arange(1, 256, dtype=torch.long)
        code_bits = ((code_ids.unsqueeze(1) >> torch.arange(8, dtype=torch.long)) & 1).to(torch.float32)

        self.register_buffer("child_bits", child_bits, persistent=False)
        self.register_buffer("code_bits", code_bits, persistent=False)

        self.__dict__["_oa_model"] = None
        self._oa_mask_cache: Dict[Tuple[torch.device, int], torch.Tensor] = {}
        self._gen_octree = None
        self._gen_kparent_seq = None

    def build_point_context(
        self,
        pts_xyz: torch.Tensor,
        ctx_level: int,
        final_w: Optional[torch.Tensor] = None,
        qs_override: Optional[torch.Tensor] = None,
    ):
        if pts_xyz.ndim != 3 or pts_xyz.shape[1] != 3:
            raise ValueError("pts_xyz must have shape [B, 3, N]")

        if pts_xyz.dtype in (torch.float16, torch.bfloat16):
            pts_xyz = pts_xyz.to(torch.float32)

        B, _, N = pts_xyz.shape
        device = pts_xyz.device
        dtype = pts_xyz.dtype
        point_w = self._normalize_point_weights(final_w, B, N, device, dtype)
        qs_values = self._normalize_qs_override(qs_override, B, device, dtype)

        ctx_dim = max(int(getattr(self.cfg, "ctx_dim", 5)), 5)
        ctx = pts_xyz.new_zeros((B, ctx_dim, N))
        for b in range(B):
            ctx[b] = self._build_point_context_single(
                pts_xyz=pts_xyz[b],
                point_w=point_w[b],
                ctx_level=int(ctx_level),
                qs_value=float(qs_values[b].item()),
            )
        return ctx

    def forward(
        self,
        gen_xyz: torch.Tensor,
        final_w: Optional[torch.Tensor] = None,
    ):
        if gen_xyz.ndim != 3 or gen_xyz.shape[1] != 3:
            raise ValueError("gen_xyz must have shape [B, 3, N]")

        if gen_xyz.dtype in (torch.float16, torch.bfloat16):
            gen_xyz = gen_xyz.to(torch.float32)

        B, _, N = gen_xyz.shape
        device = gen_xyz.device
        dtype = gen_xyz.dtype
        point_w = self._normalize_point_weights(final_w, B, N, device, dtype)

        rate_total = gen_xyz.new_zeros(())
        node_count = gen_xyz.new_zeros(())
        single_child_count = gen_xyz.new_zeros(())
        hard_rate_total = gen_xyz.new_zeros(())
        hard_node_count = gen_xyz.new_zeros(())
        hard_single_child_count = gen_xyz.new_zeros(())
        soft_rate_total = gen_xyz.new_zeros(())
        soft_node_count = gen_xyz.new_zeros(())
        soft_single_child_count = gen_xyz.new_zeros(())

        for b in range(B):
            pts_b = gen_xyz[b]
            w_b = point_w[b]
            bits_b, node_b, single_b, hard_bits_b, hard_node_b, hard_single_b, soft_bits_b, soft_node_b, soft_single_b = self._prepare_single_octattention_eval(pts_b, w_b)
            rate_total = rate_total + bits_b
            node_count = node_count + node_b
            single_child_count = single_child_count + single_b
            hard_rate_total = hard_rate_total + hard_bits_b
            hard_node_count = hard_node_count + hard_node_b
            hard_single_child_count = hard_single_child_count + hard_single_b
            soft_rate_total = soft_rate_total + soft_bits_b
            soft_node_count = soft_node_count + soft_node_b
            soft_single_child_count = soft_single_child_count + soft_single_b

        rate_total = rate_total / float(B)
        node_count = node_count / float(B)
        single_child_count = single_child_count / float(B)
        hard_rate_total = hard_rate_total / float(B)
        hard_node_count = hard_node_count / float(B)
        hard_single_child_count = hard_single_child_count / float(B)
        soft_rate_total = soft_rate_total / float(B)
        soft_node_count = soft_node_count / float(B)
        soft_single_child_count = soft_single_child_count / float(B)

        stats = {
            "bit": rate_total.detach(),
            "bpp": (rate_total / max(float(N), 1.0)).detach(),
            "bpn": (rate_total / (node_count + self.cfg.eps)).detach(),
            "single": single_child_count.detach(),
            "node": node_count.detach(),
            "entropy": rate_total.detach(),
            "hard_bit": hard_rate_total.detach(),
            "hard_single": hard_single_child_count.detach(),
            "hard_node": hard_node_count.detach(),
            "soft_bit": soft_rate_total.detach(),
            "soft_single": soft_single_child_count.detach(),
            "soft_node": soft_node_count.detach(),
        }

        return (
            {
                "rate_total": rate_total,
                "rate_entropy": rate_total,
                "rate_node_count": node_count,
                "rate_single_child": single_child_count,
                "soft_node_count": node_count,
                "soft_single_child_count": single_child_count,
                "hard_rate_entropy": hard_rate_total,
                "hard_node_count": hard_node_count,
                "hard_single_child_count": hard_single_child_count,
                "soft_rate_entropy_raw": soft_rate_total,
                "soft_node_count_raw": soft_node_count,
                "soft_single_child_count_raw": soft_single_child_count,
            },
            rate_total,
            stats,
        )

    def forward_ste_hard_pair(
        self,
        gen_xyz: torch.Tensor,
        final_w: torch.Tensor,
    ):
        """
        Compute hard-valued OctAttention terms and the weighted STE surrogate
        with one hard octree sequence / teacher pass per batch item.
        """
        if gen_xyz.ndim != 3 or gen_xyz.shape[1] != 3:
            raise ValueError("gen_xyz must have shape [B, 3, N]")

        if gen_xyz.dtype in (torch.float16, torch.bfloat16):
            gen_xyz = gen_xyz.to(torch.float32)

        B, _, N = gen_xyz.shape
        device = gen_xyz.device
        dtype = gen_xyz.dtype
        hard_w = self._normalize_point_weights(None, B, N, device, dtype)
        ste_w = self._normalize_point_weights(final_w, B, N, device, dtype)

        hard_rate_total = gen_xyz.new_zeros(())
        hard_node_count = gen_xyz.new_zeros(())
        hard_single_child_count = gen_xyz.new_zeros(())
        forward_rate_total = gen_xyz.new_zeros(())
        forward_node_count = gen_xyz.new_zeros(())
        forward_single_child_count = gen_xyz.new_zeros(())
        surrogate_rate_total = gen_xyz.new_zeros(())
        surrogate_node_count = gen_xyz.new_zeros(())
        surrogate_single_child_count = gen_xyz.new_zeros(())

        for b in range(B):
            prepared = self._prepare_single_hard_octattention_eval(gen_xyz[b], hard_w[b])
            if prepared is None:
                continue
            safe_pts, valid, offset_np, max_level, oct_seq_np, teacher_log2, hard_bits, hard_node, hard_single = prepared

            soft_bits, soft_node, soft_single = self._soft_terms_from_prepared(
                safe_pts=safe_pts,
                point_w=hard_w[b],
                valid=valid,
                offset_np=offset_np,
                oct_seq_np=oct_seq_np,
                max_level=max_level,
                teacher_log2=teacher_log2,
            )
            ste_bits, ste_node, ste_single = self._soft_terms_from_prepared(
                safe_pts=safe_pts.detach(),
                point_w=ste_w[b],
                valid=valid,
                offset_np=offset_np,
                oct_seq_np=oct_seq_np,
                max_level=max_level,
                teacher_log2=teacher_log2,
            )

            hard_rate_total = hard_rate_total + hard_bits.detach()
            hard_node_count = hard_node_count + hard_node.detach()
            hard_single_child_count = hard_single_child_count + hard_single.detach()
            forward_rate_total = forward_rate_total + hard_bits + (soft_bits - soft_bits.detach())
            forward_node_count = forward_node_count + hard_node + (soft_node - soft_node.detach())
            forward_single_child_count = forward_single_child_count + hard_single + (soft_single - soft_single.detach())
            surrogate_rate_total = surrogate_rate_total + ste_bits
            surrogate_node_count = surrogate_node_count + ste_node
            surrogate_single_child_count = surrogate_single_child_count + ste_single

        hard_rate_total = hard_rate_total / float(B)
        hard_node_count = hard_node_count / float(B)
        hard_single_child_count = hard_single_child_count / float(B)
        forward_rate_total = forward_rate_total / float(B)
        forward_node_count = forward_node_count / float(B)
        forward_single_child_count = forward_single_child_count / float(B)
        surrogate_rate_total = surrogate_rate_total / float(B)
        surrogate_node_count = surrogate_node_count / float(B)
        surrogate_single_child_count = surrogate_single_child_count / float(B)

        stats = {
            "bit": forward_rate_total.detach(),
            "bpp": (forward_rate_total / max(float(N), 1.0)).detach(),
            "bpn": (forward_rate_total / (forward_node_count + self.cfg.eps)).detach(),
            "single": forward_single_child_count.detach(),
            "node": forward_node_count.detach(),
            "entropy": forward_rate_total.detach(),
            "hard_bit": hard_rate_total.detach(),
            "hard_single": hard_single_child_count.detach(),
            "hard_node": hard_node_count.detach(),
            "soft_bit": surrogate_rate_total.detach(),
            "soft_single": surrogate_single_child_count.detach(),
            "soft_node": surrogate_node_count.detach(),
        }

        out_forward = {
            "rate_total": forward_rate_total,
            "rate_entropy": forward_rate_total,
            "rate_node_count": forward_node_count,
            "rate_single_child": forward_single_child_count,
            "soft_node_count": forward_node_count,
            "soft_single_child_count": forward_single_child_count,
            "hard_rate_entropy": hard_rate_total,
            "hard_node_count": hard_node_count,
            "hard_single_child_count": hard_single_child_count,
        }
        out_surrogate = {
            "rate_total": surrogate_rate_total,
            "rate_entropy": surrogate_rate_total,
            "rate_node_count": surrogate_node_count,
            "rate_single_child": surrogate_single_child_count,
            "soft_node_count": surrogate_node_count,
            "soft_single_child_count": surrogate_single_child_count,
        }
        return out_forward, out_surrogate, stats

    def forward_hard_only(self, gen_xyz: torch.Tensor):
        if gen_xyz.ndim != 3 or gen_xyz.shape[1] != 3:
            raise ValueError("gen_xyz must have shape [B, 3, N]")
        if gen_xyz.dtype in (torch.float16, torch.bfloat16):
            gen_xyz = gen_xyz.to(torch.float32)

        B, _, N = gen_xyz.shape
        device = gen_xyz.device
        dtype = gen_xyz.dtype
        point_w = self._normalize_point_weights(None, B, N, device, dtype)

        rate_total = gen_xyz.new_zeros(())
        node_count = gen_xyz.new_zeros(())
        single_child_count = gen_xyz.new_zeros(())

        for b in range(B):
            prepared = self._prepare_single_hard_octattention_eval(gen_xyz[b], point_w[b])
            if prepared is None:
                continue
            _, _, _, _, _, _, hard_bits, hard_node, hard_single = prepared
            rate_total = rate_total + hard_bits
            node_count = node_count + hard_node
            single_child_count = single_child_count + hard_single

        rate_total = rate_total / float(B)
        node_count = node_count / float(B)
        single_child_count = single_child_count / float(B)

        stats = {
            "bit": rate_total.detach(),
            "bpp": (rate_total / max(float(N), 1.0)).detach(),
            "bpn": (rate_total / (node_count + self.cfg.eps)).detach(),
            "single": single_child_count.detach(),
            "node": node_count.detach(),
            "entropy": rate_total.detach(),
            "hard_bit": rate_total.detach(),
            "hard_single": single_child_count.detach(),
            "hard_node": node_count.detach(),
            "soft_bit": rate_total.detach(),
            "soft_single": single_child_count.detach(),
            "soft_node": node_count.detach(),
        }

        out = {
            "rate_total": rate_total,
            "rate_entropy": rate_total,
            "rate_node_count": node_count,
            "rate_single_child": single_child_count,
            "soft_node_count": node_count,
            "soft_single_child_count": single_child_count,
            "hard_rate_entropy": rate_total,
            "hard_node_count": node_count,
            "hard_single_child_count": single_child_count,
        }
        return out, rate_total, stats

    def _prepare_single_octattention_eval(self, pts_xyz: torch.Tensor, point_w: torch.Tensor, qs_value: Optional[float] = None):
        prepared = self._prepare_single_hard_octattention_eval(pts_xyz, point_w, qs_value=qs_value)
        if prepared is None:
            zero = pts_xyz.new_zeros(())
            return zero, zero, zero, zero, zero, zero, zero, zero, zero
        safe_pts, valid, offset_np, max_level, oct_seq_np, teacher_log2, hard_bits, hard_node, hard_single = prepared
        soft_bits, soft_node, soft_single = self._soft_terms_from_prepared(
            safe_pts=safe_pts,
            point_w=point_w,
            valid=valid,
            offset_np=offset_np,
            oct_seq_np=oct_seq_np,
            max_level=max_level,
            teacher_log2=teacher_log2,
            qs_value=qs_value,
        )
        bits = hard_bits + (soft_bits - soft_bits.detach())
        node = hard_node + (soft_node - soft_node.detach())
        single = hard_single + (soft_single - soft_single.detach())
        return bits, node, single, hard_bits.detach(), hard_node.detach(), hard_single.detach(), soft_bits, soft_node, soft_single

    def _prepare_single_hard_octattention_eval(self, pts_xyz: torch.Tensor, point_w: torch.Tensor, qs_value: Optional[float] = None):
        finite_pts = torch.isfinite(pts_xyz).all(dim=0)
        finite_w = torch.isfinite(point_w)
        valid = finite_pts & finite_w

        if not valid.any():
            return None

        safe_pts = torch.nan_to_num(pts_xyz, nan=0.0, posinf=0.0, neginf=0.0)
        oct_seq_np, offset_np, max_level = self._build_hard_octattention_sequence(
            safe_pts,
            valid,
            qs_value=qs_value,
        )
        if max_level < 1:
            return None
        teacher_log2 = self._predict_teacher_log2_probs(oct_seq_np, safe_pts.device)
        hard_bits = self._hard_code_bits(oct_seq_np, teacher_log2)
        hard_node = safe_pts.new_tensor(float(oct_seq_np.shape[0]))
        hard_single = safe_pts.new_tensor(float(self._hard_single_child_count(oct_seq_np)))
        return safe_pts, valid, offset_np, max_level, oct_seq_np, teacher_log2, hard_bits, hard_node, hard_single

    def _safe_soft_weights(self, point_w: torch.Tensor, valid: torch.Tensor):
        safe_w = torch.nan_to_num(point_w, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        safe_w = safe_w * valid.to(dtype=safe_w.dtype)
        if torch.count_nonzero(safe_w > 0).item() == 0:
            safe_w = valid.to(dtype=safe_w.dtype)
        return safe_w

    def _soft_terms_from_prepared(
        self,
        safe_pts: torch.Tensor,
        point_w: torch.Tensor,
        valid: torch.Tensor,
        offset_np: np.ndarray,
        oct_seq_np: np.ndarray,
        max_level: int,
        teacher_log2: torch.Tensor,
        qs_value: Optional[float] = None,
    ):
        safe_w = self._safe_soft_weights(point_w, valid)
        child_occ, row_exist = self._build_soft_child_occupancy(
            pts_xyz=safe_pts,
            point_w=safe_w,
            offset_np=offset_np,
            oct_seq_np=oct_seq_np,
            max_level=max_level,
            qs_value=float(self.cfg.qs) if qs_value is None else float(qs_value),
        )

        soft_bits = self._expected_code_bits(child_occ, teacher_log2)
        soft_single = self._soft_single_child_count(child_occ)
        soft_node = row_exist.sum()
        return soft_bits, soft_node, soft_single

    def _build_point_context_single(
        self,
        pts_xyz: torch.Tensor,
        point_w: torch.Tensor,
        ctx_level: int,
        qs_value: float,
    ):
        num_points = pts_xyz.shape[1]
        ctx_dim = max(int(getattr(self.cfg, "ctx_dim", 5)), 5)
        ctx = pts_xyz.new_zeros((ctx_dim, num_points))

        finite_pts = torch.isfinite(pts_xyz).all(dim=0)
        finite_w = torch.isfinite(point_w)
        valid = finite_pts & finite_w
        if not valid.any():
            return ctx

        safe_pts = torch.nan_to_num(pts_xyz, nan=0.0, posinf=0.0, neginf=0.0)
        safe_w = torch.nan_to_num(point_w, nan=0.0, posinf=0.0, neginf=0.0).clamp_(0.0, 1.0)
        safe_w = safe_w * valid.to(dtype=safe_w.dtype)
        if torch.count_nonzero(safe_w > 0).item() == 0:
            safe_w = valid.to(dtype=safe_w.dtype)

        offset_np, max_level = self._prepare_soft_octree_grid(
            safe_pts,
            valid_mask=valid,
            qs_value=qs_value,
        )
        if max_level < 1:
            return ctx

        depth = max(1, min(int(ctx_level), int(max_level)))
        depth_maps = self._build_soft_depth_maps(safe_pts, safe_w, offset_np, max_level, qs_value=qs_value)
        child_occ = self._build_pointwise_child_occupancy(
            pts_xyz=safe_pts,
            offset_np=offset_np,
            max_level=max_level,
            depth=depth,
            qs_value=qs_value,
        )

        map_keys, map_occ = depth_maps[depth]
        child_occ = self._lookup_occ(map_keys, map_occ, child_occ)

        row_exist = 1.0 - torch.prod(1.0 - child_occ, dim=1)
        mean_occ = child_occ.mean(dim=1)
        single_prob = self._soft_single_child_prob(child_occ)

        child_coords = self._build_point_depth_coords(
            pts_xyz=safe_pts,
            offset_np=offset_np,
            max_level=max_level,
            depth=depth,
            qs_value=qs_value,
        )
        child_bits = child_coords & 1
        child_id = (
            (child_bits[:, 0] << 2) |
            (child_bits[:, 1] << 1) |
            child_bits[:, 2]
        ).long()
        self_occ = child_occ.gather(1, child_id.unsqueeze(1)).squeeze(1)

        sibling_occ = (child_occ.sum(dim=1) - self_occ) / 7.0

        grid = 1 << depth
        offsets = torch.tensor(
            [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
            device=pts_xyz.device,
            dtype=torch.long,
        )
        neighbor_coords = child_coords.unsqueeze(1) + offsets.unsqueeze(0)
        neighbor_coords = neighbor_coords.clamp_(min=0, max=grid - 1)
        neighbor_keys = self._coords_to_keys(neighbor_coords.reshape(-1, 3), grid).view(num_points, -1)
        neighbor_occ = self._lookup_occ(map_keys, map_occ, neighbor_keys)
        neighbor_occ_mean = ((neighbor_occ.sum(dim=1) - self_occ) / 26.0).clamp_(0.0, 1.0)
        child_id_norm = child_id.to(dtype=pts_xyz.dtype) / 7.0

        eps = self._effective_eps(child_occ.dtype)
        occ = child_occ.clamp(min=eps, max=1.0 - eps)
        bit_entropy = -(
            occ * (torch.log(occ) / math.log(2.0)) +
            (1.0 - occ) * (torch.log1p(-occ) / math.log(2.0))
        ).mean(dim=1)

        # 損失 proxy と同じ soft occupancy から、各点が参照すべき圧縮文脈を作る。
        # 先頭5次元は既存互換、6次元以降は周辺・兄弟ノード種類の観測を足す。
        feat = torch.stack(
            [
                row_exist,
                mean_occ,
                single_prob,
                self_occ,
                bit_entropy,
                sibling_occ.clamp(0.0, 1.0),
                neighbor_occ_mean,
                child_id_norm,
            ],
            dim=0,
        )
        feat_dim = min(ctx.shape[0], feat.shape[0])
        ctx[:feat_dim, valid] = feat[:feat_dim, valid]
        return ctx

    def _prepare_soft_octree_grid(
        self,
        pts_xyz: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        qs_value: Optional[float] = None,
    ):
        if valid_mask is not None:
            if valid_mask.ndim != 1 or valid_mask.shape[0] != pts_xyz.shape[1]:
                raise ValueError("valid_mask must have shape [N]")
            pts_xyz = pts_xyz[:, valid_mask]

        if pts_xyz.numel() == 0:
            return np.zeros((3,), dtype=np.float64), 0

        pts_np = pts_xyz.detach().transpose(0, 1).contiguous().cpu().numpy().astype(np.float64, copy=False)
        offset_np = pts_np.min(axis=0)
        qs = max(float(self.cfg.eps), float(self.cfg.qs) if qs_value is None else float(qs_value))
        qpts = np.round((pts_np - offset_np[None, :]) / qs).astype(np.int64)
        qpts = np.unique(qpts, axis=0)
        if qpts.shape[0] < 2:
            return offset_np, 0

        max_coord = int(qpts.max())
        max_level = int(math.ceil(math.log2(max(max_coord + 1, 1))))
        max_level = max(max_level, 1)
        if self.cfg.max_depth > 0:
            max_level = min(max_level, int(self.cfg.max_depth))
        return offset_np, max_level

    def _normalize_point_weights(self, final_w, B, N, device, dtype):
        if final_w is None:
            return torch.ones((B, N), device=device, dtype=dtype)
        if final_w.ndim == 3:
            if final_w.shape[1] != 1:
                raise ValueError("final_w must have shape [B,1,N] when 3D")
            final_w = final_w.squeeze(1)
        if final_w.ndim != 2 or final_w.shape != (B, N):
            raise ValueError("final_w must be None, [B,N], or [B,1,N]")
        return final_w.to(device=device, dtype=dtype).clamp(0.0, 1.0)

    def _normalize_qs_override(self, qs_override, B, device, dtype):
        if qs_override is None:
            return torch.full((B,), float(self.cfg.qs), device=device, dtype=dtype)
        if torch.is_tensor(qs_override):
            qs_override = qs_override.to(device=device, dtype=dtype).reshape(-1)
            if qs_override.numel() == 1:
                qs_override = qs_override.repeat(B)
            if qs_override.numel() != B:
                raise ValueError(f"qs_override must broadcast to batch size {B}, got shape {tuple(qs_override.shape)}")
            return qs_override.clamp_min(float(self.cfg.eps))
        return torch.full((B,), max(float(qs_override), float(self.cfg.eps)), device=device, dtype=dtype)

    @staticmethod
    def _child_octant_ids(coords: np.ndarray) -> np.ndarray:
        bits = coords & 1
        return bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]

    @staticmethod
    def _make_child_groups(child_coords: np.ndarray):
        parent_coords = child_coords >> 1
        child_ids = SoftOctreeRateProxy._child_octant_ids(child_coords)
        groups = {}
        for idx, parent in enumerate(parent_coords):
            key = (int(parent[0]), int(parent[1]), int(parent[2]))
            groups.setdefault(key, []).append((int(child_ids[idx]), child_coords[idx]))
        return groups

    @staticmethod
    def _build_octattention_sequence_from_qpts(qpts: np.ndarray):
        if qpts.ndim != 2 or qpts.shape[1] != 3:
            raise ValueError(f"qpts must have shape [N, 3], got {qpts.shape}")
        if qpts.shape[0] < 2:
            return np.zeros((0, 1, 6), dtype=np.int64), 0

        qpts = np.asarray(qpts, dtype=np.int64)
        max_coord = int(qpts.max())
        max_level = int(math.ceil(math.log2(max(max_coord + 1, 1))))
        max_level = max(max_level, 1)

        node_codes = []
        node_levels = []
        node_octants = []
        node_positions = []
        node_parent_rows = []

        current_coords = np.zeros((1, 3), dtype=np.int64)
        current_parent_rows = np.zeros((1,), dtype=np.int64)
        current_octants = np.ones((1,), dtype=np.int64)

        for depth in range(max_level):
            child_depth = depth + 1
            shift = max_level - child_depth
            child_coords = qpts >> shift if shift > 0 else qpts
            child_coords = np.unique(child_coords, axis=0)
            child_groups = SoftOctreeRateProxy._make_child_groups(child_coords)

            row_start = len(node_codes)
            current_rows = np.arange(row_start, row_start + current_coords.shape[0], dtype=np.int64)
            next_coords = []
            next_parent_rows = []
            next_octants = []

            for idx, coord in enumerate(current_coords):
                key = (int(coord[0]), int(coord[1]), int(coord[2]))
                children = child_groups.get(key)
                if not children:
                    continue
                children.sort(key=lambda item: item[0])
                code = 0
                for child_octant, _ in children:
                    code |= 1 << child_octant

                node_codes.append(code)
                node_levels.append(depth + 1)
                node_octants.append(int(current_octants[idx]))
                node_positions.append((coord << (max_level - depth)).astype(np.int64, copy=False))
                node_parent_rows.append(int(current_parent_rows[idx]))

                if depth < max_level - 1:
                    parent_row = int(current_rows[idx])
                    for child_octant, child_coord in children:
                        next_coords.append(child_coord)
                        next_parent_rows.append(parent_row)
                        next_octants.append(child_octant + 1)

            if depth < max_level - 1:
                current_coords = np.asarray(next_coords, dtype=np.int64).reshape(-1, 3)
                current_parent_rows = np.asarray(next_parent_rows, dtype=np.int64)
                current_octants = np.asarray(next_octants, dtype=np.int64)

        node_num = len(node_codes)
        seq = np.ones((node_num, 4), dtype=np.int64) * 255
        level_octant = np.zeros((node_num, 4, 2), dtype=np.int64)
        pos = np.zeros((node_num, 4, 3), dtype=np.int64)

        for row in range(node_num):
            parent_row = node_parent_rows[row]
            seq[row, -1] = node_codes[row]
            level_octant[row, -1] = (node_levels[row], node_octants[row])
            pos[row, -1] = node_positions[row]
            seq[row, :-1] = seq[parent_row, 1:]
            level_octant[row, :-1] = level_octant[parent_row, 1:]
            pos[row, :-1] = pos[parent_row, 1:]

        oct_seq_np = np.concatenate(
            (
                np.expand_dims(seq, axis=2),
                level_octant,
                pos,
            ),
            axis=2,
        )
        return oct_seq_np, max_level

    def _build_hard_octattention_sequence(
        self,
        pts_xyz: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        qs_value: Optional[float] = None,
    ):
        if valid_mask is not None:
            if valid_mask.ndim != 1 or valid_mask.shape[0] != pts_xyz.shape[1]:
                raise ValueError("valid_mask must have shape [N]")
            pts_xyz = pts_xyz[:, valid_mask]

        if pts_xyz.numel() == 0:
            pts_np = np.zeros((1, 3), dtype=np.float64)
        else:
            pts_np = pts_xyz.detach().transpose(0, 1).contiguous().cpu().numpy().astype(np.float64, copy=False)

        offset_np = pts_np.min(axis=0)
        qs = max(float(self.cfg.eps), float(self.cfg.qs) if qs_value is None else float(qs_value))
        qpts = np.round((pts_np - offset_np[None, :]) / qs).astype(np.int64)
        qpts = np.unique(qpts, axis=0)
        if qpts.shape[0] < 2:
            empty_seq = np.zeros((0, 1, 6), dtype=np.int64)
            return empty_seq, offset_np, 0

        oct_seq_np, max_level = self._build_octattention_sequence_from_qpts(qpts)
        if self.cfg.max_depth > 0 and max_level > self.cfg.max_depth:
            keep = oct_seq_np[:, -1, 1] <= self.cfg.max_depth
            oct_seq_np = oct_seq_np[keep]
            max_level = self.cfg.max_depth

        return oct_seq_np, offset_np, max_level

    def _build_point_depth_coords(
        self,
        pts_xyz: torch.Tensor,
        offset_np: np.ndarray,
        max_level: int,
        depth: int,
        qs_value: float,
    ):
        pts = pts_xyz.transpose(0, 1).contiguous()
        offset = torch.from_numpy(offset_np).to(device=pts_xyz.device, dtype=pts_xyz.dtype)
        q = torch.round((pts - offset.unsqueeze(0)) / float(max(qs_value, self.cfg.eps))).to(torch.long)
        if max_level > depth:
            q = torch.bitwise_right_shift(q, max_level - depth)

        grid = 1 << depth
        return q.clamp_(min=0, max=grid - 1)

    def _build_pointwise_child_occupancy(
        self,
        pts_xyz: torch.Tensor,
        offset_np: np.ndarray,
        max_level: int,
        depth: int,
        qs_value: float,
    ):
        coords = self._build_point_depth_coords(
            pts_xyz=pts_xyz,
            offset_np=offset_np,
            max_level=max_level,
            depth=depth,
            qs_value=qs_value,
        )
        if depth == 1:
            parent_idx = coords.new_zeros(coords.shape)
        else:
            parent_idx = torch.bitwise_right_shift(coords, 1)

        child_idx = (parent_idx.unsqueeze(1) << 1) + self.child_bits.to(device=coords.device).unsqueeze(0)
        grid = 1 << depth
        return self._coords_to_keys(child_idx.reshape(-1, 3), grid).view(-1, 8)

    def _build_soft_child_occupancy(
        self,
        pts_xyz: torch.Tensor,
        point_w: torch.Tensor,
        offset_np: np.ndarray,
        oct_seq_np: np.ndarray,
        max_level: int,
        qs_value: float,
    ):
        depth_maps = self._build_soft_depth_maps(
            pts_xyz,
            point_w,
            offset_np,
            max_level,
            qs_value=qs_value,
        )

        levels = torch.from_numpy(oct_seq_np[:, -1, 1].astype(np.int64, copy=False)).to(pts_xyz.device)
        positions = torch.from_numpy(oct_seq_np[:, -1, 3:6].astype(np.int64, copy=False)).to(pts_xyz.device)

        num_nodes = levels.numel()
        child_occ = pts_xyz.new_zeros((num_nodes, 8))
        row_exist = pts_xyz.new_zeros((num_nodes,))

        for level in torch.unique(levels, sorted=True):
            mask = levels == level
            idx = mask.nonzero(as_tuple=False).squeeze(1)
            pos = positions[idx]
            child_depth = int(level.item())

            if child_depth < 1 or child_depth > max_level:
                continue

            if child_depth == 1:
                parent_idx = pos.new_zeros((idx.numel(), 3))
            else:
                shift = max_level - (child_depth - 1)
                parent_idx = torch.bitwise_right_shift(pos, shift)

            child_idx = (parent_idx.unsqueeze(1) << 1) + self.child_bits.to(device=pos.device).unsqueeze(0)
            child_keys = self._coords_to_keys(child_idx.reshape(-1, 3), 1 << child_depth).view(-1, 8)

            map_keys, map_occ = depth_maps[child_depth]
            child_occ_level = self._lookup_occ(map_keys, map_occ, child_keys)

            child_occ[idx] = child_occ_level
            row_exist[idx] = 1.0 - torch.prod(1.0 - child_occ_level, dim=1)

        return child_occ, row_exist

    def _build_soft_depth_maps(
        self,
        pts_xyz: torch.Tensor,
        point_w: torch.Tensor,
        offset_np: np.ndarray,
        max_level: int,
        qs_value: float,
    ):
        device = pts_xyz.device
        dtype = pts_xyz.dtype

        pts = pts_xyz.transpose(0, 1).contiguous()
        offset = torch.from_numpy(offset_np).to(device=device, dtype=dtype)
        q = (pts - offset.unsqueeze(0)) / float(max(qs_value, self.cfg.eps))

        base = torch.floor(q)
        frac = q - base
        base = base.to(torch.long)

        tau = max(float(self.cfg.round_tau), 1e-6)
        w_hi = torch.sigmoid((frac - 0.5) / tau)
        w_lo = 1.0 - w_hi

        leaf_grid = 1 << max_level
        all_keys = []
        all_mass = []

        for corner in self.child_bits.to(device=device):
            cx, cy, cz = int(corner[0].item()), int(corner[1].item()), int(corner[2].item())
            wx = w_hi[:, 0] if cx == 1 else w_lo[:, 0]
            wy = w_hi[:, 1] if cy == 1 else w_lo[:, 1]
            wz = w_hi[:, 2] if cz == 1 else w_lo[:, 2]

            mass = (wx * wy * wz * point_w).reshape(-1)
            coords = base + corner.view(1, 3)
            coords = coords.clamp_(min=0, max=leaf_grid - 1)

            valid = mass > float(self.cfg.min_mass)
            if valid.any():
                all_keys.append(self._coords_to_keys(coords[valid], leaf_grid))
                all_mass.append(mass[valid])

        if not all_keys:
            empty_key = torch.empty((0,), device=device, dtype=torch.long)
            empty_occ = torch.empty((0,), device=device, dtype=dtype)
            return {depth: (empty_key, empty_occ) for depth in range(1, max_level + 1)}

        leaf_keys = torch.cat(all_keys, dim=0)
        leaf_mass = torch.cat(all_mass, dim=0)

        uniq_leaf_keys, inverse = torch.unique(leaf_keys, sorted=True, return_inverse=True)
        uniq_leaf_mass = torch.zeros(uniq_leaf_keys.shape[0], device=device, dtype=dtype)
        uniq_leaf_mass.index_add_(0, inverse, leaf_mass)
        leaf_coords = self._keys_to_coords(uniq_leaf_keys, leaf_grid)

        depth_maps = {}
        for depth in range(1, max_level + 1):
            shift = max_level - depth
            if shift > 0:
                depth_coords = torch.bitwise_right_shift(leaf_coords, shift)
            else:
                depth_coords = leaf_coords

            grid = 1 << depth
            depth_keys = self._coords_to_keys(depth_coords, grid)
            uniq_depth_keys, inverse = torch.unique(depth_keys, sorted=True, return_inverse=True)
            depth_mass = torch.zeros(uniq_depth_keys.shape[0], device=device, dtype=dtype)
            depth_mass.index_add_(0, inverse, uniq_leaf_mass)
            depth_occ = 1.0 - torch.exp(-float(self.cfg.mass_to_occ_gain) * depth_mass.clamp_min(0.0))
            depth_maps[depth] = (uniq_depth_keys, depth_occ.clamp_(0.0, 1.0))

        return depth_maps

    def _resolve_teacher_device(self, output_device: torch.device):
        mode = str(getattr(self.cfg, "teacher_device", "auto")).strip().lower()
        if mode in {"", "auto", "same", "training"}:
            return output_device
        if mode == "balanced":
            return torch.device("cpu") if output_device.type == "cuda" else output_device
        if mode == "cpu":
            return torch.device("cpu")
        if mode.startswith("cuda"):
            if torch.cuda.is_available():
                if mode == "cuda" and output_device.type == "cuda":
                    return output_device
                return torch.device(mode)
            return torch.device("cpu")
        return output_device

    def _predict_teacher_log2_probs(self, oct_seq_np: np.ndarray, device: torch.device):
        teacher_device = self._resolve_teacher_device(device)
        model = self._lazy_init_octattention_backend(device=teacher_device)

        oct_len = int(oct_seq_np.shape[0])
        if oct_len <= 0:
            return torch.empty((0, 255), device=device, dtype=torch.float32)

        bptt = max(1, min(int(self.cfg.bptt), oct_len))
        data_id, padded = self._batchify_oct_seq(oct_seq_np, bptt)
        data_id = data_id.to(device=teacher_device)
        padded = padded.to(device=teacher_device)
        src_mask = self._get_causal_mask(device=teacher_device, seq_len=bptt)

        segments = []
        with torch.inference_mode():
            for start in range(0, padded.shape[0] - bptt, bptt):
                x = padded[start:start + bptt]
                target_id = data_id[start + 1:start + bptt + 1].reshape(-1)
                valid = target_id >= 0
                if not valid.any():
                    continue

                logits = model(x, src_mask, []).reshape(-1, 255)
                log2_prob = torch.log_softmax(logits, dim=1) / math.log(2.0)
                segments.append(log2_prob[valid])

        teacher_log2 = torch.cat(segments, dim=0)[:oct_len].detach()
        return teacher_log2.to(device=device, non_blocking=True)

    def _expected_code_bits(self, child_occ: torch.Tensor, teacher_log2: torch.Tensor):
        if child_occ.numel() == 0:
            return child_occ.new_zeros(())

        eps = self._effective_eps(child_occ.dtype)
        code_bits = self.code_bits.to(device=child_occ.device, dtype=child_occ.dtype)
        total_bits = child_occ.new_zeros(())
        chunk = max(int(self.cfg.teacher_chunk_size), 1)

        for start in range(0, child_occ.shape[0], chunk):
            end = min(start + chunk, child_occ.shape[0])
            occ = child_occ[start:end].clamp(min=eps, max=1.0 - eps)
            not_occ = 1.0 - occ
            raw_code_prob = torch.ones(
                (occ.shape[0], code_bits.shape[0]),
                device=child_occ.device,
                dtype=child_occ.dtype,
            )
            for bit_idx in range(8):
                bit = code_bits[:, bit_idx].unsqueeze(0)
                raw_code_prob = raw_code_prob * (
                    occ[:, bit_idx:bit_idx + 1] * bit
                    + not_occ[:, bit_idx:bit_idx + 1] * (1.0 - bit)
                )
            total_bits = total_bits - (raw_code_prob * teacher_log2[start:end].to(dtype=child_occ.dtype)).sum()

        return total_bits

    def _hard_code_bits(self, oct_seq_np: np.ndarray, teacher_log2: torch.Tensor):
        if oct_seq_np.shape[0] == 0:
            return teacher_log2.new_zeros(())
        target = torch.from_numpy(oct_seq_np[:, -1, 0].astype(np.int64, copy=False) - 1).to(
            device=teacher_log2.device,
            dtype=torch.long,
        )
        prob = torch.exp2(teacher_log2).gather(1, target.unsqueeze(1)).squeeze(1)
        return -torch.log2(prob + 1e-7).sum()

    @staticmethod
    def _hard_single_child_count(oct_seq_np: np.ndarray) -> int:
        if oct_seq_np.shape[0] == 0:
            return 0
        codes = oct_seq_np[:, -1, 0].astype(np.uint16, copy=False)
        pop = (
            (codes & 1)
            + ((codes >> 1) & 1)
            + ((codes >> 2) & 1)
            + ((codes >> 3) & 1)
            + ((codes >> 4) & 1)
            + ((codes >> 5) & 1)
            + ((codes >> 6) & 1)
            + ((codes >> 7) & 1)
        )
        return int((pop == 1).sum())

    def _effective_eps(self, dtype: torch.dtype):
        finfo = torch.finfo(dtype)
        return max(float(self.cfg.eps), float(finfo.eps) * 16.0, 1e-6)

    def _soft_single_child_count(self, child_occ: torch.Tensor):
        if child_occ.numel() == 0:
            return child_occ.new_zeros(())
        return self._soft_single_child_prob(child_occ).sum()

    def _soft_single_child_prob(self, child_occ: torch.Tensor):
        if child_occ.numel() == 0:
            return child_occ.new_zeros((child_occ.shape[0],))
        one_minus = (1.0 - child_occ).clamp_(0.0, 1.0)
        single = child_occ.new_zeros((child_occ.shape[0],))
        for k in range(8):
            prod = torch.ones_like(single)
            for j in range(8):
                if j == k:
                    continue
                prod = prod * one_minus[:, j]
            single = single + child_occ[:, k] * prod
        return single

    def _batchify_oct_seq(self, oct_seq_np: np.ndarray, bptt: int):
        seq = oct_seq_np.copy()
        seq[:-1, 0:-1, :] = seq[1:, 0:-1, :]
        seq[:-1, -1, 1:3] = seq[1:, -1, 1:3]
        seq[:, :, 0] = seq[:, :, 0] - 1

        oct_len = seq.shape[0]
        pad_len = bptt
        padded = np.zeros((bptt + oct_len + pad_len, *seq.shape[1:]), dtype=seq.dtype)
        padded[bptt:bptt + oct_len] = seq

        data_id = np.full((bptt + oct_len + pad_len,), -1, dtype=np.int64)
        data_id[bptt:bptt + oct_len] = np.arange(oct_len, dtype=np.int64)

        return (
            torch.from_numpy(data_id).long().unsqueeze(1),
            torch.from_numpy(padded).long().unsqueeze(1),
        )

    def _get_causal_mask(self, device: torch.device, seq_len: int):
        key = (device, seq_len)
        mask = self._oa_mask_cache.get(key)
        if mask is None:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)
            self._oa_mask_cache[key] = mask
        return mask

    def _lazy_init_octree_backend(self):
        if self._gen_octree is None or self._gen_kparent_seq is None:
            from Octree import GenOctree, GenKparentSeq
            self._gen_octree = GenOctree
            self._gen_kparent_seq = GenKparentSeq

    def _lazy_init_octattention_backend(self, device: Optional[torch.device] = None):
        self._lazy_init_octree_backend()
        if self._oa_model is None:
            ckpt_path = self.cfg.checkpoint_path
            if ckpt_path is None:
                ckpt_path = str(_OA_DIR / "modelsave" / "obj" / "encoder_epoch_00800093.pth")

            model = _OctAttentionTeacherModel(max_octree_level=12)
            save_dict = torch.load(ckpt_path, map_location="cpu")
            state_dict = save_dict["encoder"] if "encoder" in save_dict else save_dict
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

            self.__dict__["_oa_model"] = model

        if device is not None:
            self.__dict__["_oa_model"] = self.__dict__["_oa_model"].to(device)
        return self._oa_model

    @staticmethod
    def _coords_to_keys(coords: torch.Tensor, grid: int):
        coords = coords.to(torch.long)
        grid_t = torch.as_tensor(grid, device=coords.device, dtype=torch.long)
        return coords[:, 0] + grid_t * (coords[:, 1] + grid_t * coords[:, 2])

    @staticmethod
    def _keys_to_coords(keys: torch.Tensor, grid: int):
        grid_t = torch.as_tensor(grid, device=keys.device, dtype=torch.long)
        xy = grid_t * grid_t
        z = torch.div(keys, xy, rounding_mode="floor")
        rem = keys - z * xy
        y = torch.div(rem, grid_t, rounding_mode="floor")
        x = rem - y * grid_t
        return torch.stack([x, y, z], dim=1)

    @staticmethod
    def _lookup_occ(keys: torch.Tensor, values: torch.Tensor, query: torch.Tensor):
        if keys.numel() == 0:
            return values.new_zeros(query.shape)

        flat_query = query.reshape(-1)
        idx = torch.searchsorted(keys, flat_query)
        idx_clamped = idx.clamp(max=max(keys.numel() - 1, 0))
        hit = (idx < keys.numel()) & (keys[idx_clamped] == flat_query)

        out = values.new_zeros(flat_query.shape)
        if hit.any():
            out[hit] = values[idx_clamped[hit]]
        return out.view_as(query)
