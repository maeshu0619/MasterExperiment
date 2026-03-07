import torch
import torch.nn as nn
import torch.nn.functional as F


class AddModule(nn.Module):
    """
    追加領域 + 追加点数 + 追加ベクトルを同時に学習するAdd
    - TopKを使わない（非微分を排除）
    - Binary Concrete で追加マスクを生成
    - 返り値に loss_add を含める
    """

    def __init__(self, cfgs, writer):
        super().__init__()
        self.cfgs = cfgs
        self.writer = writer

        in_dim = 3 + cfgs.local_feat_dim + 2 + cfgs.octree_ctx_dim
        hidden_dim = cfgs.add_hidden_dim

        # Gumbel-Sigmoid温度
        self.tau = float(getattr(cfgs, "add_tau", 0.5))

        # 目標追加率
        self.target_add_ratio = self.cfgs.target_add_ratio

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

    # def _compute_L_fit(self, add_prob: torch.Tensor, offset: torch.Tensor):
    #     """
    #     L_fit: 追加点が元点から離れすぎない（= offsetが大きすぎない）
    #     add_prob: (B,N)
    #     offset  : (B,3,N)
    #     """
    #     off_norm = offset.norm(dim=1)  # (B,N)
    #     # conv_radiusで正規化して二乗、0に寄せる
    #     val = (off_norm / (self.conv_radius + 1e-8)) ** 2
    #     # add_probで重み付け（追加しない点のoffsetには関心がない）
    #     return (add_prob * val).mean()

    def _compute_L_fit(self, pts: torch.Tensor, new_pts: torch.Tensor, add_prob: torch.Tensor = None):
        B, _, N = pts.shape
        _, _, K = new_pts.shape
        if K <= 0:
            return pts.new_tensor(0.0)

        conv_r = float(self.conv_radius) + 1e-8
        conv_r2 = conv_r * conv_r

        q_chunk = int(getattr(self.cfgs, "add_fit_q_chunk", 256))
        r_chunk = int(getattr(self.cfgs, "add_fit_ref_chunk", 4096))
        tau = float(getattr(self.cfgs, "add_fit_tau", 0.0))

        ref_all = pts.permute(0, 2, 1).contiguous()      # (B,N,3)
        q_all = new_pts.permute(0, 2, 1).contiguous()    # (B,K,3)

        if add_prob is not None:
            if add_prob.dim() == 3 and add_prob.shape[1] == 1:
                add_prob = add_prob.squeeze(1)
            elif add_prob.dim() == 2:
                pass
            else:
                raise ValueError("add_prob の形は (B,K) か (B,1,K) を想定している")

        N_ref_max = int(getattr(self.cfgs, "add_fit_ref_max", 0))

        loss_accum = pts.new_tensor(0.0)

        for b in range(B):
            ref = ref_all[b].float()  # (N,3)
            q = q_all[b].float()      # (K,3)

            # --- ここで ref を間引く（bごと） ---
            if N_ref_max > 0 and ref.shape[0] > N_ref_max:
                ridx = torch.randint(0, ref.shape[0], (N_ref_max,), device=ref.device)
                ref = ref.index_select(0, ridx)
                N_b = ref.shape[0]
            else:
                N_b = ref.shape[0]
            # -----------------------------------

            w = add_prob[b].float() if add_prob is not None else None

            sum_loss = torch.zeros((), device=pts.device, dtype=torch.float32)
            sum_w = torch.zeros((), device=pts.device, dtype=torch.float32)

            for qs in range(0, K, q_chunk):
                qe = q[qs:qs + q_chunk]  # (c,3)
                c = qe.shape[0]

                qn = (qe * qe).sum(dim=1, keepdim=True)  # (c,1)
                min_sq = torch.full((c,), float("inf"), device=pts.device, dtype=torch.float32)

                for rs in range(0, N_b, r_chunk):
                    rb = ref[rs:rs + r_chunk]  # (m,3)
                    rn = (rb * rb).sum(dim=1, keepdim=True).t()  # (1,m)

                    dist_sq = qn + rn - 2.0 * (qe @ rb.t())  # (c,m)
                    dist_sq = torch.clamp(dist_sq, min=0.0)

                    blk_min, _ = dist_sq.min(dim=1)
                    min_sq = torch.minimum(min_sq, blk_min)

                if tau > 0.0:
                    d = torch.sqrt(min_sq + 1e-12) / conv_r
                    val = F.relu(d - tau) ** 2
                else:
                    val = min_sq / conv_r2

                if w is not None:
                    wc = w[qs:qs + q_chunk]
                    sum_loss = sum_loss + (wc * val).sum()
                    sum_w = sum_w + wc.sum()
                else:
                    sum_loss = sum_loss + val.sum()
                    sum_w = sum_w + float(c)

            loss_b = sum_loss / (sum_w + 1e-12)
            loss_accum = loss_accum + loss_b

        return loss_accum / float(B)

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

    def _compute_L_rep_weighted(self, new_pts: torch.Tensor, add_w: torch.Tensor):
        """
        L_rep: 追加候補点同士の過密を抑制
        new_pts: (B,3,N)
        add_w  : (B,1,N) または (B,N)
        """
        B, _, N = new_pts.shape
        if N <= 1:
            return new_pts.new_tensor(0.0)

        if add_w.dim() == 3:
            add_w = add_w.squeeze(1)  # (B,N)

        M = min(N, self.rep_max_points)
        if M < N:
            idx = torch.randperm(N, device=new_pts.device)[:M]
            p = new_pts[:, :, idx]          # (B,3,M)
            w = add_w[:, idx]               # (B,M)
        else:
            p = new_pts
            w = add_w

        p = p.permute(0, 2, 1).contiguous()  # (B,M,3)
        p = p / (self.conv_radius + 1e-8)

        dist = torch.cdist(p, p)  # (B,M,M)

        eye = torch.eye(dist.shape[1], device=dist.device).unsqueeze(0)
        dist = dist + eye * 1e6

        rep = torch.clamp_max(dist - self.repulse_extent, max=0.0) ** 2  # (B,M,M)

        # 点対ごとの重み
        ww = w.unsqueeze(2) * w.unsqueeze(1)  # (B,M,M)
        rep = rep * ww

        return rep.sum() / (ww.sum() + 1e-12)

    def forward(self, pts, feats, density_score, structure_score, octree_score):
        """
        pts             : (B,3,N)
        feats           : (B,C_l,N)
        density_score   : (B,1,N)
        structure_score : (B,1,N)

        return:
        pts_add_soft : (B,3,N)
        pts_add_hard : (B,3,N+K)
        new_pts_soft : (B,3,N)
        new_pts_hard : (B,3,K)
        add_w        : (B,1,N)  # Hard選択を近似するSoft重み
        add_idx      : (B,K)
        loss_add     : scalar
        """
        B, _, N = pts.shape

        max_add_points = max(1, int(self.max_ratio * N))
        min_add = max(1, int(self.target_add_ratio * N))

        x = torch.cat([pts, feats, density_score, structure_score, octree_score], dim=1)
        h = self.mlp_shared(x)

        add_logit = self.mlp_add_logit(h).squeeze(1)                 # (B,N)
        add_mask, add_prob = self._sample_binary_concrete(add_logit) # (B,N), (B,N)

        # 方向と距離から offset を生成
        dir_raw = torch.tanh(self.mlp_dir(h))                        # (B,3,N)
        dir_vec = dir_raw / (dir_raw.norm(dim=1, keepdim=True) + 1e-8)
        mag = torch.sigmoid(self.mlp_mag(h)).squeeze(1)              # (B,N)

        max_offset = float(getattr(self.cfgs, "max_offset", 1.0))
        offset = dir_vec * (mag.unsqueeze(1) * max_offset)           # (B,3,N)

        # Soft側の追加候補点
        new_pts_soft = pts + offset                                  # (B,3,N)
        pts_add_soft = pts                                           # 学習時は点数を増やさない

        # -----------------------------
        # Hard側: 実際に追加する点を選ぶ
        # -----------------------------
        new_pts_list = []
        add_idx_list = []
        thr_list = []
        hard_ratio_list = []

        for b in range(B):
            valid = add_mask[b] > 0.5
            idx = torch.where(valid)[0]

            if idx.numel() > max_add_points:
                idx = torch.topk(add_prob[b], k=max_add_points, largest=True).indices

            if idx.numel() < min_add:
                idx = torch.topk(add_prob[b], k=min_add, largest=True).indices

            idx = torch.sort(idx).values

            pts_sel = pts[b, :, idx]
            off_sel = offset[b, :, idx]
            new_pts_b = pts_sel + off_sel

            new_pts_list.append(new_pts_b)
            add_idx_list.append(idx)

            # Hard選択集合を近似するための閾値
            # 末尾の選択点の確率を detach して使う
            thr_b = add_prob[b, idx[-1]].detach()
            thr_list.append(thr_b)

            hard_ratio_list.append(
                torch.tensor(float(idx.numel()) / float(N), device=pts.device, dtype=add_prob.dtype)
            )

        new_pts_hard = torch.stack(new_pts_list, dim=0)              # (B,3,K)
        add_idx = torch.stack(add_idx_list, dim=0)                   # (B,K)
        pts_add_hard = torch.cat([pts, new_pts_hard], dim=2)         # (B,3,N+K)

        # ---------------------------------------------------
        # Soft側: Hard選択集合を近似する連続重み soft_sel を作る
        #   - 閾値は Hard 側から detach して取得
        #   - add_prob に対してのみ勾配を流す
        # ---------------------------------------------------
        thr = torch.stack(thr_list, dim=0).unsqueeze(1)              # (B,1)
        hard_ratio = torch.stack(hard_ratio_list, dim=0)             # (B,)

        tau_match = float(getattr(self.cfgs, "add_soft_match_tau", 0.05))
        tau_match = max(tau_match, 1e-6)

        # Hard集合の指示関数 1[p >= thr] を sigmoid で近似
        soft_sel = torch.sigmoid((add_prob - thr) / tau_match)       # (B,N)

        # Soft重みの総量を Hard の実追加率に合わせる
        soft_mean_det = soft_sel.mean(dim=1, keepdim=True).detach()  # (B,1)
        scale = hard_ratio.unsqueeze(1) / (soft_mean_det + 1e-12)    # (B,1)
        soft_sel = (soft_sel * scale).clamp(0.0, 1.0)                # (B,N)

        add_w = soft_sel.unsqueeze(1)                                # (B,1,N)

        # -----------------------------
        # Soft損失: Hard近似重みで計算
        # -----------------------------
        L_fit = self._compute_L_fit(pts, new_pts_soft, add_prob=add_w)
        L_rep = self._compute_L_rep_weighted(new_pts_soft, add_w)

        # Softの総量がHardの実追加率に近づくようにする
        mean_add_ratio_soft = soft_sel.mean(dim=1)                   # (B,)
        delta = (mean_add_ratio_soft - hard_ratio.detach()).abs()
        L_cnt = torch.log1p(128 * delta).mean()

        loss_add = (
            self.add_cnt * L_cnt
            + self.add_fit * L_fit
            + self.add_rep * L_rep
        )

        if self.writer is not None and hasattr(self.writer, "write"):
            self.writer.write(
                f"L_add   :{loss_add:.4f}->"
                f"L_cnt:{L_cnt:.4f}, L_fit:{L_fit:.4f}, L_rep:{L_rep:.4f}, "
                f"AddRatio(soft):{mean_add_ratio_soft.mean().item():.6f}, "
                f"AddRatio(hard):{hard_ratio.mean().item():.6f}"
            )

        return pts_add_soft, pts_add_hard, new_pts_soft, new_pts_hard, add_w, add_idx, loss_add