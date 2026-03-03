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

        in_dim = cfgs.fp_mlp_channels[-1] + 2
        hidden = cfgs.prune_hidden_dim
        num_blocks = getattr(cfgs, "prune_num_blocks", 3)

        self.tau = getattr(cfgs, "prune_tau", 0.5)
        self.target_ratio = getattr(cfgs, "prune_target_keep_ratio", 0.93)
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

    def forward(self, pts, Fl, Ff, d, s):

        B, _, N = pts.shape
        c = torch.cat([Ff, d, s], dim=1)

        net = self.conv_in(c)
        for block in self.blocks:
            net = block(net, c)

        logit = self.conv_out(self.act_out(self.bn_out(net))).squeeze(1)
        keep_prob = torch.sigmoid(logit)  # [B, N]

        # 学習はConcrete、推論はtopk（乱数なし）にする
        if self.cfgs.trainORtest == "train":
            mask, keep_prob_st = self.sample_binary_concrete(logit)
            # 損失計算はST側の確率でやる（勾配を通す）
            keep_for_loss = keep_prob_st
        else:
            mask = None
            keep_for_loss = keep_prob

        # ---- 削除点数制御（後で変更2も反映）----
        mean_ratio = keep_for_loss.mean(dim=1)
        L_cnt = (abs(mean_ratio - self.target_ratio)).mean()

        # ---- 外れ点抑制（後で変更3も反映）----
        density = d.squeeze(1)
        d_norm = density / (density.mean(dim=1, keepdim=True) + 1e-6)

        # ※ここは変更3で差し替えるので一旦そのまま
        L_out = (keep_for_loss * d_norm).mean()

        prune_loss = 0.0
        if self.cfgs.trainORtest == "train":
            prune_loss = self.prun_cnt * L_cnt + self.prun_out * L_out
            self.writer.write(
                f"L_prun  :{prune_loss:.4f}->L_cnt:{L_cnt:.4f}, L_out:{L_out:.4f}, KeepRatio:{mean_ratio.mean().item():.4f}"
            )

        # ---- 点の選択：推論は必ず topk で固定点数を残す ----
        pts_kept_list = []
        keep_idx_list = []

        # 残す点数を固定（入力Nに対して target_ratio を満たす）
        K_keep = max(1, int(round(N * float(self.target_ratio))))

        for b in range(B):
            if self.cfgs.trainORtest == "train":
                # 学習中はConcreteのhard maskを尊重（ただし空ならtopkで救済）
                valid = mask[b] > 0.5
                idx = torch.where(valid)[0]
                if idx.numel() == 0:
                    idx = torch.topk(keep_prob[b], k=1).indices
                # 学習でも点数がバラつくので、最終的に topk で揃える（塊落ち抑制）
                if idx.numel() > K_keep:
                    idx = idx[torch.topk(keep_prob[b, idx], k=K_keep).indices]
                elif idx.numel() < K_keep:
                    idx = torch.topk(keep_prob[b], k=K_keep).indices
            else:
                # 推論は乱数なしで必ずK_keep
                idx = torch.topk(keep_prob[b], k=K_keep).indices

            pts_kept_list.append(pts[b, :, idx])
            keep_idx_list.append(idx)

        K_min = min(x.shape[1] for x in pts_kept_list)
        pts_pruned = torch.stack([x[:, :K_min] for x in pts_kept_list], dim=0)
        keep_idx = torch.stack([idx[:K_min] for idx in keep_idx_list], dim=0)

        return pts_pruned, keep_idx, prune_loss
