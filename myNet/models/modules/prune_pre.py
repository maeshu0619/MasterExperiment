import torch
import torch.nn as nn

class PruningModule(nn.Module):
    """
    Pruning Module (Early Prune, hard version)

    入力点群から MLP により不要点を削除し，
    ・削除後点群（Tensor, batch 内で点数統一）
    ・残存点インデックス（keep_idx）
    を返す
    """

    def __init__(self, cfgs, writer, prune_thresh=0.5):
        super().__init__()

        self.prune_thresh = prune_thresh
        self.writer = writer

        in_dim = cfgs.fp_mlp_channels[-1] + 2  # F' + d' + s'
        hidden = cfgs.prune_hidden_dim
        F_dim = cfgs.fp_mlp_channels[-1]

        self.local_mlp = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden // 2, 1, 1),
            nn.Sigmoid()
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(F_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden//2, 1)
        )

        # --- 10%削除に初期化 ---
        with torch.no_grad():
            self.global_mlp[-1].bias.fill_(-2.197)

    def forward(self, pts, Fl, Ff, d, s):

        # --- 1. 局所スコア（どこ） ---
        x = torch.cat([Ff, d, s], dim=1)   # [B, C+2, N]
        p = self.local_mlp(x).squeeze(1)   # [B, N]

        B, _, N = pts.shape

        # --- 2. グローバル削除率（いくつ） ---
        g = torch.mean(Ff, dim=2)          # [B, C]
        r = torch.sigmoid(self.global_mlp(g)).squeeze(1)

        pts_kept_list = []
        keep_idx_list = []

        for b in range(B):
            keep_num = (1 - r[b]) * N
            keep_num = torch.clamp(keep_num, min=1, max=N)
            keep_num = int(keep_num.item())

            sorted_idx = torch.argsort(p[b], descending=True)
            keep_idx = sorted_idx[:keep_num]

            pts_kept_list.append(pts[b, :, keep_idx])
            keep_idx_list.append(keep_idx)

        K_min = min(x.shape[1] for x in pts_kept_list)

        pts_pruned = torch.stack(
            [x[:, :K_min] for x in pts_kept_list],
            dim=0
        )

        keep_idx = torch.stack(
            [idx[:K_min] for idx in keep_idx_list],
            dim=0
        )

        return pts_pruned, keep_idx
