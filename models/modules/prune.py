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
        self.high_is_inlier = getattr(self.cfgs, "prune_d_high_is_inlier", True)
        self.c_robust = float(getattr(self.cfgs, "prune_robust_c", 2.0))

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

    def forward(self, pts, Ff, D, S, O):
        """==================== SetUp ===================="""
        B, _, N = pts.shape
        c = torch.cat([Ff, D, S, O], dim=1)

        net = self.conv_in(c)
        for block in self.blocks:
            net = block(net, c)

        logit = self.conv_out(self.act_out(self.bn_out(net))).squeeze(1)   # (B,N)
        keep_prob = torch.sigmoid(logit)                                   # (B,N)

        """==================== Hard ===================="""
        K_keep = max(1, int(round(N * float(self.target_ratio))))

        keep_idx_hard = torch.topk(keep_prob, k=K_keep, dim=1, largest=True).indices  # (B,K_keep)
        keep_idx_hard = torch.sort(keep_idx_hard, dim=1).values                        # (B,K_keep)

        hard_mask = torch.zeros_like(keep_prob)                                         # (B,N)
        hard_mask.scatter_(1, keep_idx_hard, 1.0)                                       # (B,N)

        hard_ratio = hard_mask.mean(dim=1)                                              # (B,)

        idx_expand = keep_idx_hard.unsqueeze(1).expand(-1, 3, -1)                       # (B,3,K_keep)
        pts_hard = torch.gather(pts, 2, idx_expand)                                     # (B,3,K_keep)

        """==================== Soft ===================="""
        thr = torch.gather(keep_prob, 1, keep_idx_hard[:, -1:].detach())                # (B,1)

        tau_match = float(getattr(self.cfgs, "prune_soft_match_tau", 0.05))
        tau_match = max(tau_match, 1e-6)

        soft_mask_raw = torch.sigmoid((keep_prob - thr) / tau_match)                    # (B,N)

        soft_mean_det = soft_mask_raw.mean(dim=1, keepdim=True).detach()                # (B,1)
        scale = hard_ratio.unsqueeze(1) / (soft_mean_det + 1e-12)                       # (B,1)
        soft_mask = (soft_mask_raw * scale).clamp(0.0, 1.0)                             # (B,N)

        """==================== STE ===================="""
        mask_st = hard_mask - soft_mask.detach() + soft_mask                            # (B,N)
        keep_w_full = mask_st.unsqueeze(1)                                              # (B,1,N)

        # Hardで残した点に対応する重みだけを返す
        keep_w_hard = torch.gather(keep_w_full, 2, keep_idx_hard.unsqueeze(1))                                                                               # (B,1,K_keep)

        """==================== Calculate Loss ===================="""
        target_ratio = torch.full_like(hard_ratio, float(self.target_ratio))            # (B,)
        mean_keep_ratio_pred = soft_mask_raw.mean(dim=1)                                # (B,)

        delta_cnt = (mean_keep_ratio_pred - target_ratio).abs()
        L_cnt = torch.log1p(128 * delta_cnt).mean()

        score = D.squeeze(1)  # (B,N)
        eps = 1e-6

        r = (1.0 / (score + eps)) if self.high_is_inlier else score
        r = r / (r.mean(dim=1, keepdim=True) + eps)

        w_inlier = 1.0 / (1.0 + (r / self.c_robust) ** 2)   # (B,N)
        w_inlier = w_inlier.detach()

        L_out = torch.nn.functional.mse_loss(mask_st, w_inlier)

        loss_prun = self.prun_cnt * L_cnt + self.prun_out * L_out

        if self.writer is not None and hasattr(self.writer, "write"):
            self.writer.write(
                f"L_prun  :{loss_prun:.4f}->"
                f"L_cnt:{L_cnt:.4f}, L_out:{L_out:.4f}, "
                f"KeepRatio(pred_raw):{mean_keep_ratio_pred.mean().item():.6f}, "
                f"KeepRatio(soft):{soft_mask.mean(dim=1).mean().item():.6f}, "
                f"KeepRatio(hard):{hard_ratio.mean().item():.6f}"
            )

        return pts_hard, keep_w_hard, keep_w_full, keep_idx_hard, loss_prun, L_cnt, L_out