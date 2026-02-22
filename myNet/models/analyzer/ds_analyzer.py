import torch
import torch.nn as nn
from ..utils.utils_repkpu import get_knn_pts, index_points


"""
Density & Structure Analyzer Module

入力:
    pts      : [B, 3, N]        点座標
    F    : [B, C, N]        Encoderの局所特徴（F_local）

出力:
    density_score   : [B, 1, N] 密度スコア d
    structure_score : [B, 1, N] 構造スコア s
"""

class DensityStructureAnalyzer(nn.Module):
    def __init__(self, cfgs, writer):
        super().__init__()
        self.k = cfgs.k
        self.writer = writer

        # ===== MLP for density score =====
        self.density_mlp = nn.Sequential(
            nn.Conv1d(cfgs.encoder_dim * 3 + 3, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, 1)
        )

        # ===== MLP for structure score =====
        self.structure_mlp = nn.Sequential(
            nn.Conv1d(cfgs.encoder_dim * 3 + 3, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, 1)
        )

    def forward(self, pts, F):
        """
        pts   : [B, 3, N]
        F : [B, 3C, N]  (feat1 + feat2 + feat3)
        """

        # ===== kNN取得 =====
        knn_pts, knn_idx = get_knn_pts(self.k, pts, pts, return_idx=True)
        # knn_pts: [B, 3, N, k]

        # ===== 距離計算 =====
        center = pts.unsqueeze(-1)                       # [B, 3, N, 1]
        diff = knn_pts - center                          # [B, 3, N, k]
        dist = torch.norm(diff, dim=1)                   # [B, N, k]

        # ===== 近傍統計量 =====
        dist_mean = dist.mean(dim=-1, keepdim=True)      # [B, N, 1]
        dist_var  = dist.var(dim=-1, keepdim=True)       # [B, N, 1]
        knn_count = torch.full_like(dist_mean, self.k)   # [B, N, 1]

        # [B, 3, N]
        geom_stats = torch.cat([
            dist_mean.permute(0, 2, 1),
            dist_var.permute(0, 2, 1),
            knn_count.permute(0, 2, 1)
        ], dim=1)

        # ===== 特徴結合 =====
        x = torch.cat([F, geom_stats], dim=1)

        # ===== スコア算出 =====
        density_score   = self.density_mlp(x)
        structure_score = self.structure_mlp(x)

        return density_score, structure_score
