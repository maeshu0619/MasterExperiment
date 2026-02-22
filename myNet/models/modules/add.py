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

        # 目標追加率（例: 0.05 なら Nの5%を追加）
        self.target_add_ratio = float(getattr(cfgs, "target_add_ratio", 0.05))

        # 損失重み
        self.add_cnt = cfgs.add_cnt
        self.add_fit = cfgs.add_fit
        self.add_rep = cfgs.add_rep

        # 追加: RepKPU由来の正則化（fit / rep）
        self.lambda_fit = float(getattr(cfgs, "lambda_add_fit", 100))
        self.lambda_rep = float(getattr(cfgs, "lambda_add_rep", 1000))

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

    def _compute_L_fit(self, pts: torch.Tensor, new_pts: torch.Tensor):
        """
        reg.py の fit_loss と同型の目的（最近傍距離→(dist/conv_r)^2→L1）を、
        (c,N) や (K,N) の巨大行列を作らずに実現する版。

        pts    : (B,3,N)
        new_pts: (B,3,K)
        """
        B, _, N = pts.shape
        _, _, K = new_pts.shape
        if K <= 0:
            return pts.new_tensor(0.0)

        _l1 = nn.L1Loss()
        conv_r = float(self.conv_radius) + 1e-8
        conv_r2 = conv_r * conv_r

        # chunk サイズ（cfg で調整可能）
        # q_chunk: クエリ側（追加点）を何点ずつ処理するか
        # r_chunk: 参照側（元点群）を何点ずつ処理するか
        q_chunk = int(getattr(self.cfgs, "add_fit_q_chunk", 256))
        r_chunk = int(getattr(self.cfgs, "add_fit_ref_chunk", 4096))

        loss_accum = pts.new_tensor(0.0)
        count = 0

        # (B,N,3)
        p_ref = pts.permute(0, 2, 1).contiguous()

        for b in range(B):
            # q: (K,3), ref: (N,3)
            q = new_pts[b].permute(1, 0).contiguous()
            ref = p_ref[b]

            # 計算安定性と速度のため、距離計算だけ float32 で行う
            # （amp を使っていてもここは安全にしたい）
            q32 = q.float()
            ref32 = ref.float()

            # ref の二乗ノルムはブロックごとに計算（全N分を保持しない）
            for qs in range(0, K, q_chunk):
                qe = q32[qs:qs + q_chunk]  # (c,3)
                c = qe.shape[0]

                # クエリ側ノルム (c,1)
                qn = (qe * qe).sum(dim=1, keepdim=True)

                # 現在の最小二乗距離を保持 (c,)
                min_sq = torch.full((c,), float("inf"), device=pts.device, dtype=torch.float32)

                # 参照点をブロック分割して min を更新
                for rs in range(0, N, r_chunk):
                    rb = ref32[rs:rs + r_chunk]  # (m,3)

                    # 参照側ノルム (1,m)
                    rn = (rb * rb).sum(dim=1, keepdim=True).t()

                    # dist_sq = qn + rn - 2*qe@rb^T  -> (c,m)
                    # (c,m) は作るが m を小さくして OOM を回避
                    dist_sq = qn + rn - 2.0 * (qe @ rb.t())
                    dist_sq = torch.clamp(dist_sq, min=0.0)

                    # クエリ点ごとに最小更新
                    blk_min, _ = dist_sq.min(dim=1)  # (c,)
                    min_sq = torch.minimum(min_sq, blk_min)

                # (dist/conv_r)^2 = min_sq / conv_r^2
                val = min_sq / conv_r2  # (c,)
                loss_q = _l1(val, torch.zeros_like(val))
                loss_accum = loss_accum + loss_q
                count += 1

        return loss_accum / float(count)

    # def _compute_L_fit(self, pts: torch.Tensor, new_pts: torch.Tensor):
    #     """
    #     reg.py の fit_loss と同型の目的（距離→min→二乗→L1）を、
    #     (B,K,N) の巨大行列を保持せず chunk で最小距離だけ計算して実現する版。

    #     pts    : (B,3,N)
    #     new_pts: (B,3,K)
    #     """
    #     B, _, N = pts.shape
    #     _, _, K = new_pts.shape
    #     if K <= 0:
    #         return pts.new_tensor(0.0)

    #     _l1 = nn.L1Loss()
    #     conv_r = (self.conv_radius + 1e-8)

    #     loss_accum = 0.0
    #     count = 0

    #     # (B,N,3)
    #     p_ref = pts.permute(0, 2, 1).contiguous()

    #     # chunk サイズ（必要ならcfgで調整可能にしてよい）
    #     chunk = int(getattr(self.cfgs, "add_fit_chunk", 1024))

    #     for b in range(B):
    #         # (K,3)
    #         q = new_pts[b].permute(1, 0).contiguous()
    #         ref = p_ref[b]  # (N,3)

    #         # 各クエリ点の最近傍距離を chunk で求める
    #         mins = []
    #         for s in range(0, K, chunk):
    #             qe = q[s:s+chunk]  # (c,3)
    #             # (c,N) を一時的に作るが c を小さくして爆発を防ぐ
    #             d = torch.cdist(qe, ref)                 # (c,N)
    #             md, _ = torch.min(d, dim=1)              # (c,)
    #             mins.append(md)
    #         min_dist = torch.cat(mins, dim=0)            # (K,)

    #         # reg.py と同様に (dist/conv_radius)^2 を 0 に寄せる（L1）
    #         val = (min_dist / conv_r) ** 2               # (K,)
    #         loss_b = _l1(val, torch.zeros_like(val))
    #         loss_accum = loss_accum + loss_b
    #         count += 1

    #     return loss_accum / float(count)

    def _compute_L_rep(self, new_pts: torch.Tensor):
        """
        reg.py の rep_loss と理論上同型にする版

        reg.py:
        norm_kp_pos = deform_kp_pos / conv_radius
        for i:
            distances = ||other - i||
            rep_loss_i = sum(clamp_max(distances - extent, 0)^2)
            loss += L1(rep_loss_i, 0) / num_kernel_points

        new_pts: (B,3,K)
        """
        B, _, K = new_pts.shape
        if K <= 1:
            return new_pts.new_tensor(0.0)

        # O(K^2)回避のためサブサンプル（既存方針は維持）
        M = min(K, self.rep_max_points)
        if M < K:
            idx = torch.randperm(K, device=new_pts.device)[:M]
            p = new_pts[:, :, idx]  # (B,3,M)
        else:
            p = new_pts  # (B,3,M)

        # 正規化（reg.py と同様）
        p = p / (self.conv_radius + 1e-8)  # (B,3,M)
        p = p.permute(0, 2, 1).contiguous()  # (B,M,3)

        # 距離行列 (B,M,M)
        dist = torch.cdist(p, p)

        # 自分自身との距離を無効化
        eye = torch.eye(dist.shape[1], device=dist.device).unsqueeze(0)  # (1,M,M)
        dist = dist + eye * 1e6

        _l1 = nn.L1Loss()
        loss = 0.0

        # reg.py と同じ集約（iごとに「他点へのペナルティのsum」→ L1(.,0) → 平均）
        for i in range(dist.shape[1]):
            distances = dist[:, i, :]  # (B,M) selfは1e6
            rep_i = torch.sum(torch.clamp_max(distances - self.repulse_extent, max=0.0) ** 2, dim=1)  # (B,)
            loss = loss + _l1(rep_i, torch.zeros_like(rep_i)) / float(dist.shape[1])

        return loss

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
            min_add = max(1, int(self.target_add_ratio * N))
            if idx.numel() < min_add:
                idx = torch.topk(add_prob[b], k=min_add, largest=True).indices

            pts_sel = pts[b, :, idx]                 # (3,K)
            off_sel = offset[b, :, idx]              # (3,K)
            new_pts = pts_sel + off_sel              # (3,K)

            new_pts_list.append(new_pts)
            add_idx_list.append(idx)

        K_min = min(x.shape[1] for x in new_pts_list)
        new_pts = torch.stack([x[:, :K_min] for x in new_pts_list], dim=0)  # (B,3,Kmin)
        add_idx = torch.stack([idx[:K_min] for idx in add_idx_list], dim=0) # (B,Kmin)
        
        L_fit = self._compute_L_fit(pts, new_pts)
        L_rep = self._compute_L_rep(new_pts)

        L_add = 0.0
        if self.cfgs.trainORtest == "train":
            mean_add_ratio = add_prob.mean(dim=1)  # (B,)
            # L_cnt = ((mean_add_ratio - self.target_add_ratio) ** 2).mean()
            
            delta = (mean_add_ratio - self.target_add_ratio).abs()
            L_cnt = torch.log1p(128 * delta).mean()

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
                self.writer.write(f"L_add   :{L_add:.4f}->L_cnt:{L_cnt:.4f}, L_fit:{L_fit:.4f}, L_rep:{L_rep:.4f}, AddRatio:{mean_add_ratio.item():.4f}")


        self.last_add_prob = add_prob
        self.last_add_mask = add_mask
        self.last_add_loss = L_add

        pts_add = torch.cat([pts, new_pts], dim=2)  # (B,3,N+Kmin)
        return pts_add, new_pts, add_prob.unsqueeze(1), add_idx, L_add