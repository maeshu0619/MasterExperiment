import torch
import torch.nn as nn
from ..utils.utils_pointnet import three_nn_fp, three_nn_chunked, three_interpolate


class FeaturePropagationExtended(nn.Module):
    """
    拡張 Feature Propagation Module

    出力:
        pts_all     : [B, 3, N+M]
        F_l_prime   : [B, C_l, N+M]   Analyzer 用（局所）
        F_prime     : [B, C_out, N+M] 下流用（統合）
    """

    def __init__(self, cfgs, writer):
        super().__init__()
        self.eps = 1e-10
        self.writer = writer

        self.c_l = cfgs.local_feat_dim
        self.c_f = cfgs.fused_feat_dim
        # self.c_g = cfgs.global_feat_dim

        in_channels = self.c_l + self.c_f

        layers = []
        last_c = in_channels
        for c in cfgs.fp_mlp_channels:
            layers.append(nn.Conv1d(last_c, c, 1))
            layers.append(nn.BatchNorm1d(c))
            layers.append(nn.ReLU(inplace=True))
            last_c = c

        self.mlp = nn.Sequential(*layers)

    def forward(self, pts_xyz, new_pts, pts_atr, F_l, F_f, add_idx):
        """
        pts_xyz     : [B, 3, N]
        new_pts : [B, 3, M]
        pts_atr : [B, C_a, N]
        F_l     : [B, C_l, N]
        F_f     : [B, C_f, N]
        F_g     : [B, C_g]
        """

        if new_pts.size(-1) == 0:
            B = pts_xyz.size(0)
            device = pts_xyz.device
            F_l_new = torch.empty((B, self.c_l, 0), device=device, dtype=F_l.dtype)
            C_out = self.mlp[0].out_channels if len(self.mlp) > 0 else (self.c_l + self.c_f)
            F_prime = torch.empty((B, C_out, 0), device=device, dtype=F_l.dtype)
            attr_new = torch.empty((B, pts_atr.size(1), 0), device=device, dtype=pts_atr.dtype)
            return F_l_new, F_prime, attr_new

        B, _, M = new_pts.shape
        _, _, N = pts_xyz.shape
        device = pts_xyz.device

        k = 16  # 近傍制限（必要なら調整）

        # -------------------------------------------------
        # 1. parent点取得
        # -------------------------------------------------
        parent_pts = torch.gather(
            pts_xyz,
            2,
            add_idx.unsqueeze(1).expand(-1, 3, -1)
        )  # [B,3,M]

        # -------------------------------------------------
        # 2. parentと全点の距離（ここは M×N だが Mは追加点のみ）
        # -------------------------------------------------
        dist_parent = torch.cdist(
            parent_pts.transpose(1,2),   # [B,M,3]
            pts_xyz.transpose(1,2)       # [B,N,3]
        )  # [B,M,N]

        _, neighbor_idx = torch.topk(
            dist_parent,
            k,
            dim=-1,
            largest=False,
            sorted=False
        )  # [B,M,k]

        # -------------------------------------------------
        # 3. そのk点の座標取得
        # -------------------------------------------------
        pts_xyz_t = pts_xyz.transpose(1,2)  # [B,N,3]

        neighbor_pts = torch.gather(
            pts_xyz_t,
            1,
            neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )  # [B,M,k,3]

        # -------------------------------------------------
        # 4. new pointとの距離計算
        # -------------------------------------------------
        new_pts_t = new_pts.transpose(1,2)  # [B,M,3]

        dist_local = torch.norm(
            new_pts_t.unsqueeze(2) - neighbor_pts,
            dim=-1
        )  # [B,M,k]

        dist3, idx3 = torch.topk(
            dist_local,
            3,
            dim=-1,
            largest=False,
            sorted=False
        )  # [B,M,3]

        # -------------------------------------------------
        # 5. global indexに変換
        # -------------------------------------------------
        idx_global = torch.gather(
            neighbor_idx,
            2,
            idx3
        )  # [B,M,3]

        dist3 = torch.clamp(dist3, min=self.eps)

        weight = 1.0 / dist3
        weight = weight / torch.sum(weight, dim=2, keepdim=True)

        # -------------------------------------------------
        # 6. 補間
        # -------------------------------------------------
        F_l_new = three_interpolate(F_l, idx_global, weight)
        F_f_new = three_interpolate(F_f, idx_global, weight)
        attr_new = three_interpolate(pts_atr, idx_global, weight)

        x = torch.cat([F_l_new, F_f_new], dim=1)
        F_prime = self.mlp(x)

        return F_l_new, F_prime, attr_new