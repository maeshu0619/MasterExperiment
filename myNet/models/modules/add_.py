import torch
import torch.nn as nn


class AddModule(nn.Module):
    """
    追加領域 + 追加点数 + 追加ベクトルを同時に学習するAdd
    - TopKを使わない（非微分を排除）
    - Binary Concrete で追加マスクを生成
    - 返り値に L_add を含める
    """

    def __init__(self, cfgs, writer):
        super().__init__()
        self.cfgs = cfgs
        self.writer = writer

        in_dim = 3 + cfgs.local_feat_dim + 2
        hidden_dim = cfgs.add_hidden_dim

        # Gumbel-Sigmoid温度
        self.tau = float(getattr(cfgs, "add_tau", 0.5))

        # 目標追加率
        self.target_add_ratio = float(getattr(cfgs, "target_add_ratio", 0.05))

        # 損失重み
        self.add_cnt = cfgs.add_cnt
        self.add_fit = cfgs.add_fit
        self.add_rep = cfgs.add_rep

        # fit正規化用（RepKPUのconv_radius相当）
        self.conv_radius = float(getattr(cfgs, "add_conv_radius", 1.0))

        # rep閾値（RepKPUのrepulse_extent相当、conv_radiusで正規化後の距離閾値）
        self.repulse_extent = float(getattr(cfgs, "add_repulse_extent", 0.5))

        # rep計算の最大点数（O(M^2)を避ける）
        self.rep_max_points = int(getattr(cfgs, "add_rep_max_points", 2048))

        # max追加点数（安全上限）
        self.max_ratio = float(getattr(cfgs, "max_add_ratio", 0.05))

        # 共有MLP
        self.mlp_shared = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
        )

        # 追加確率（logit）
        self.mlp_add_logit = nn.Conv1d(hidden_dim, 1, 1)

        # 方向（未正規化）
        self.mlp_dir = nn.Conv1d(hidden_dim, 3, 1)

        # 距離（スカラー）
        self.mlp_mag = nn.Conv1d(hidden_dim, 1, 1)

        # デバッグ用保持
        self.last_add_prob = None
        self.last_add_mask = None
        self.last_add_loss = None

    def _sample_binary_concrete(self, logit: torch.Tensor):
        """
        Binary Concrete (Gumbel-Sigmoid)
        logit: (B,N)
        return:
          mask: (B,N) forwardはほぼ0/1、backwardは連続
          prob: (B,N) sigmoid(logit)
        """
        eps = 1e-10
        u = torch.rand_like(logit).clamp_(eps, 1.0 - eps)
        g = torch.log(u) - torch.log(1.0 - u)
        y = torch.sigmoid((logit + g) / self.tau)           # 連続
        y_hard = (y > 0.5).float()                          # 離散
        mask = y_hard.detach() - y.detach() + y             # STE
        prob = torch.sigmoid(logit)
        return mask, prob

    def _compute_L_fit(self, add_prob: torch.Tensor, offset: torch.Tensor):
        """
        L_fit: 追加点が元点から離れすぎない（= offsetが大きすぎない）
        add_prob: (B,N)
        offset  : (B,3,N)
        """
        off_norm = offset.norm(dim=1)  # (B,N)
        # conv_radiusで正規化して二乗、0に寄せる
        val = (off_norm / (self.conv_radius + 1e-8)) ** 2
        # add_probで重み付け（追加しない点のoffsetには関心がない）
        return (add_prob * val).mean()

    def _compute_L_rep(self, new_pts: torch.Tensor):
        """
        L_rep: 追加点同士の過密を抑制（近すぎたら罰）
        new_pts: (B,3,K)
        """
        B, _, K = new_pts.shape
        if K <= 1:
            return new_pts.new_tensor(0.0)

        # O(K^2)回避のためサブサンプル
        M = min(K, self.rep_max_points)
        if M < K:
            idx = torch.randperm(K, device=new_pts.device)[:M]
            p = new_pts[:, :, idx]  # (B,3,M)
        else:
            p = new_pts  # (B,3,M)

        # (B,M,3)
        p = p.permute(0, 2, 1).contiguous()

        # 正規化（RepKPU同様 conv_radius で）
        p = p / (self.conv_radius + 1e-8)

        # 距離行列 (B,M,M)
        dist = torch.cdist(p, p)

        # 対角を無視
        eye = torch.eye(dist.shape[1], device=dist.device).unsqueeze(0)
        dist = dist + eye * 1e6

        # 閾値より近い分だけペナルティ（RepKPUの clamp_max(dist - extent, max=0)^2 と同型）
        rep = torch.clamp_max(dist - self.repulse_extent, max=0.0) ** 2

        # 平均
        return rep.mean()

    def forward(self, pts, feats, density_score, structure_score):
        """
        pts             : (B,3,N)
        feats           : (B,C_l,N)
        density_score   : (B,1,N)
        structure_score : (B,1,N)

        return:
          pts_add  : (B,3,N+Kmin)
          new_pts  : (B,3,Kmin)
          add_prob : (B,1,N)
          add_idx  : (B,Kmin)
          L_add : scalar
        """
        B, _, N = pts.shape

        max_add_points = max(1, int(self.max_ratio * N))

        x = torch.cat([pts, feats, density_score, structure_score], dim=1)
        h = self.mlp_shared(x)

        add_logit = self.mlp_add_logit(h).squeeze(1)  # (B,N)
        add_mask, add_prob = self._sample_binary_concrete(add_logit)  # (B,N), (B,N)

        # 方向と距離からoffset生成
        dir_raw = torch.tanh(self.mlp_dir(h))                 # (B,3,N)
        dir_vec = dir_raw / (dir_raw.norm(dim=1, keepdim=True) + 1e-8)
        mag = torch.sigmoid(self.mlp_mag(h)).squeeze(1)       # (B,N)

        max_offset = float(getattr(self.cfgs, "max_offset", 1.0))
        offset = dir_vec * (mag.unsqueeze(1) * max_offset)    # (B,3,N)

        new_pts_list = []
        add_idx_list = []

        for b in range(B):
            valid = add_mask[b] > 0.5
            idx = torch.where(valid)[0]

            # 上限制限（安全のため）
            if idx.numel() > max_add_points:
                idx = torch.topk(add_prob[b], k=max_add_points, largest=True).indices

            # 全ゼロ崩壊対策（最低1点）
            if idx.numel() == 0:
                idx = torch.topk(add_prob[b], k=1, largest=True).indices

            pts_sel = pts[b, :, idx]                 # (3,K)
            off_sel = offset[b, :, idx]              # (3,K)
            new_pts = pts_sel + off_sel              # (3,K)

            new_pts_list.append(new_pts)
            add_idx_list.append(idx)

        K_min = min(x.shape[1] for x in new_pts_list)
        new_pts = torch.stack([x[:, :K_min] for x in new_pts_list], dim=0)  # (B,3,Kmin)
        add_idx = torch.stack([idx[:K_min] for idx in add_idx_list], dim=0) # (B,Kmin)

        # ========= 追加: L_fit / L_rep =========
        L_fit = self._compute_L_fit(add_prob, offset)
        L_rep = self._compute_L_rep(new_pts)

        L_add = 0.0
        if self.cfgs.trainORtest == "train":
            # ========= L_add（既存） =========
            mean_add_ratio = add_prob.mean(dim=1)  # (B,)
            L_cnt = ((mean_add_ratio - self.target_add_ratio) ** 2).mean()

            # ds = density_score.squeeze(1)  # (B,N)
            # ds_norm = ds / (ds.mean(dim=1, keepdim=True) + 1e-6)
            # L_where = -(add_prob * ds_norm).mean()

            # off_norm = offset.norm(dim=1)  # (B,N)
            # L_off = (add_prob * (off_norm / (max_offset + 1e-8))).mean()

            L_add = (
                self.add_cnt * L_cnt
                # + self.lambda_where * L_where
                # + self.lambda_off * L_off
                # + self.add_fit * L_fit
                # + self.add_rep * L_rep
                + self.add_fit * L_fit
                + self.add_rep * L_rep
            )

            if self.writer is not None and hasattr(self.writer, "write"):
                self.writer.write(f"L_add   :{L_add:.4f}>L_cnt:{L_cnt:.4f}, L_fit:{L_fit:.4f}, L_rep:{L_rep:.4f}, AddRatio:{mean_add_ratio.item():.4f}")

        self.last_add_prob = add_prob
        self.last_add_mask = add_mask
        self.last_add_loss = L_add

        pts_add = torch.cat([pts, new_pts], dim=2)  # (B,3,N+Kmin)
        return pts_add, new_pts, add_prob.unsqueeze(1), add_idx, L_add