import torch
import torch.nn as nn
from .resblock import ResnetBlockConv1d


class PruningModule(nn.Module):
    """
    削除領域 + 削除点数を内部で学習するPrune
    """
    def __init__(self, cfgs, writer):
        super().__init__()
        self.cfgs = cfgs
        self.writer = writer

        in_dim = cfgs.fp_mlp_channels[-1] + 2 + cfgs.octree_ctx_dim
        hidden = cfgs.prune_hidden_dim
        num_blocks = getattr(cfgs, "prune_num_blocks", 3)

        self.tau = getattr(cfgs, "prune_tau", 0.5)
        self.target_ratio = getattr(cfgs, "prune_target_keep_ratio", 0.97)
        self.prun_cnt = cfgs.prun_cnt
        self.prun_out = cfgs.prun_out

        self.conv_in = nn.Conv1d(in_dim, hidden, 1)
        self.blocks = nn.ModuleList(
            [ResnetBlockConv1d(in_dim, hidden) for _ in range(num_blocks)]
        )
        self.bn_out = nn.BatchNorm1d(hidden)
        self.act_out = nn.ReLU()
        self.conv_out = nn.Conv1d(hidden, 1, 1)

    def sample_binary_concrete(self, logit):
        eps = 1e-10
        u = torch.rand_like(logit).clamp_(eps, 1 - eps)
        g = torch.log(u) - torch.log(1 - u)
        y = torch.sigmoid((logit + g) / self.tau)

        y_hard = (y > 0.5).float()
        mask = y_hard.detach() - y.detach() + y
        prob = torch.sigmoid(logit)

        return mask, prob

    def forward(self, pts, Ff, d, s, o):

        B, _, N = pts.shape
        c = torch.cat([Ff, d, s, o], dim=1)

        net = self.conv_in(c)
        for block in self.blocks:
            net = block(net, c)

        logit = self.conv_out(self.act_out(self.bn_out(net))).squeeze(1)
        keep_prob = torch.sigmoid(logit)  # [B, N]

        mask_soft, keep_prob_st_soft = self.sample_binary_concrete(logit)
        # mask_hard = None
        keep_for_loss_soft = keep_prob_st_soft
        # keep_for_loss_hard = keep_prob

        # ---- 削除点数制御 ----
        mean_ratio_soft = keep_for_loss_soft.mean(dim=1)
        # mean_ratio_hard = keep_for_loss_hard.mean(dim=1)
        delta = (mean_ratio_soft - self.target_ratio).abs()
        L_cnt = torch.log1p(128 * delta).mean()

        # ---- 外れ点抑制 ----
        score = d.squeeze(1)  # [B,N]
        eps = 1e-6

        high_is_inlier = getattr(self.cfgs, "prune_d_high_is_inlier", True)
        r = (1.0 / (score + eps)) if high_is_inlier else score
        r = r / (r.mean(dim=1, keepdim=True) + eps) # 正規化

        # robust重み（外れほど0に近づく）
        c = float(getattr(self.cfgs, "prune_robust_c", 2.0))
        w_inlier = 1.0 / (1.0 + (r / c) ** 2)  # [B,N]
        w_inlier = w_inlier.detach()
        L_out = torch.nn.functional.mse_loss(keep_for_loss_soft, w_inlier)
        
        # Training
        loss_prun = self.prun_cnt * L_cnt + self.prun_out * L_out
        keep_idx_soft = torch.arange(N, device=pts.device).unsqueeze(0).repeat(B, 1)
        pts_soft = pts

        # Testing        
        # ---- 点の選択：推論は必ず topk で固定点数を残す ----
        pts_kept_list = []
        keep_idx_list = []

        # 残す点数を固定（入力Nに対して target_ratio を満たす）
        K_keep = max(1, int(round(N * float(self.target_ratio))))

        for b in range(B):
            idx = torch.topk(keep_prob[b], k=K_keep).indices
            idx = torch.sort(idx).values
            pts_kept_list.append(pts[b, :, idx])
            keep_idx_list.append(idx)

        pts_hard = torch.stack(pts_kept_list, dim=0)
        keep_idx_hard = torch.stack(keep_idx_list, dim=0)

        keep_w = keep_for_loss_soft.unsqueeze(1)

        self.writer.write(
            f"L_prun  :{loss_prun:.4f}->L_cnt:{L_cnt:.4f}, L_out:{L_out:.4f}, KeepRatio:{mean_ratio_soft.mean().item():.4f}"
        )

        keep_w_full = keep_for_loss_soft.unsqueeze(1)                  # (B,1,N)
        keep_w_hard = torch.gather(keep_w_full, 2, keep_idx_hard.unsqueeze(1))  # (B,1,K_keep)

        return pts_soft, pts_hard, keep_w_hard, keep_for_loss_soft, keep_idx_hard, loss_prun