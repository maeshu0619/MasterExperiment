import torch
import torch.nn as nn


class HeuristicFeaturePropagation(nn.Module):
    """
    追加点への特徴伝播をヒューリスティックに行うモジュール

    方針:
        AddModule が返す add_idx を利用し、
        追加点 new_pts の生成元（親点）の特徴をそのままコピーする。

    入力:
        pts_xyz : [B, 3, N]  (未使用だがインタフェース維持のため受け取る)
        new_pts : [B, 3, M]
        pts_atr : [B, Ca, N]
        F_l     : [B, C_l, N]
        F_f     : [B, C_f, N]
        add_idx : [B, M]     追加点の生成元インデックス

    出力:
        F_l_new  : [B, C_l, M]
        F_prime  : [B, C_out, M]  (元FPと同様にMLPで統合特徴を作る)
        attr_new : [B, Ca, M]
    """

    def __init__(self, cfgs, writer):
        super().__init__()
        self.writer = writer

        self.c_l = cfgs.local_feat_dim
        self.c_f = cfgs.fused_feat_dim

        in_channels = self.c_l + self.c_f

        layers = []
        last_c = in_channels
        for c in cfgs.fp_mlp_channels:
            layers.append(nn.Conv1d(last_c, c, 1))
            # layers.append(nn.BatchNorm1d(c))
            layers.append(nn.ReLU(inplace=True))
            last_c = c

        self.mlp = nn.Sequential(*layers)

    def forward(self, pts_xyz, new_pts, pts_atr, F_l, F_f, add_idx):
        # 追加点が0の場合の安全策
        if new_pts.size(-1) == 0:
            B = F_l.size(0)
            device = F_l.device
            F_l_new = torch.empty((B, self.c_l, 0), device=device, dtype=F_l.dtype)
            C_out = self.mlp[0].out_channels if len(self.mlp) > 0 else (self.c_l + self.c_f)
            F_prime = torch.empty((B, C_out, 0), device=device, dtype=F_l.dtype)
            attr_new = torch.empty((B, pts_atr.size(1), 0), device=device, dtype=pts_atr.dtype)
            return F_l_new, F_prime, attr_new

        # add_idx の型と形状を保証
        if add_idx.dtype != torch.long:
            add_idx = add_idx.long()  # [B, M]

        B, M = add_idx.shape

        # 親点の局所特徴をコピー
        F_l_new = torch.gather(
            F_l,
            dim=2,
            index=add_idx.unsqueeze(1).expand(-1, F_l.size(1), -1)
        )  # [B, C_l, M]

        # 親点の融合特徴をコピー
        F_f_new = torch.gather(
            F_f,
            dim=2,
            index=add_idx.unsqueeze(1).expand(-1, F_f.size(1), -1)
        )  # [B, C_f, M]

        # 親点の属性もコピー
        attr_new = torch.gather(
            pts_atr,
            dim=2,
            index=add_idx.unsqueeze(1).expand(-1, pts_atr.size(1), -1)
        )  # [B, Ca, M]

        # 統合特徴（元FPと同じ形で下流に渡す）
        x = torch.cat([F_l_new, F_f_new], dim=1)  # [B, C_l+C_f, M]
        F_prime = self.mlp(x) if len(self.mlp) > 0 else x

        return F_l_new, F_prime, attr_new