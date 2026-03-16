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

        # 中間層のディメンション計算
        in_dim = 3 + cfgs.local_feat_dim + 2 + cfgs.octree_ctx_dim
        hidden_dim = cfgs.add_hidden_dim

        # 目標追加率
        self.target_add_ratio = self.cfgs.target_add_ratio

        # 損失重み
        self.add_cnt = cfgs.add_cnt
        self.add_fit = cfgs.add_fit
        self.add_rep = cfgs.add_rep

        self.tau = float(getattr(cfgs, "add_tau", 0.5)) # Gumbel-Sigmoid温度
        self.conv_radius = float(getattr(cfgs, "add_conv_radius", 1.0)) # fit正規化用（RepKPUのconv_radius相当）
        self.repulse_extent = float(getattr(cfgs, "add_repulse_extent", 0.5)) # rep閾値（RepKPUのrepulse_extent相当、conv_radiusで正規化後の距離閾値）      
        self.rep_max_points = int(getattr(cfgs, "add_rep_max_points", 2048)) # rep計算の最大点数（O(M^2)を避ける）
        self.max_ratio = float(getattr(cfgs, "max_add_ratio", 0.05)) # max追加点数（安全上限）
        self.max_offset = float(getattr(self.cfgs, "max_offset", 1.0)) # 追加点の移動距離の最大量
        self.tau_match = float(getattr(self.cfgs, "add_soft_match_tau", 0.05)) # Soft化の温度パラメータ

        # 共有MLP
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
        )
        self.mlp_logit = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 1, 1),
        ) # 追加確率（logit） # どこに
        self.mlp_dir = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 3, 1),
        ) # 方向（未正規化） # どっちに
        self.mlp_mag = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 1, 1),
        ) # 距離（スカラー） # どれくらい

    def _sample_binary_concrete(self, logit: torch.Tensor):
        """
        Binary Concrete (Gumbel-Sigmoid)
        logit: (B,N)
        return:
          mask: (B,N) forwardはほぼ0/1、backwardは連続
          prob: (B,N) sigmoid(logit)
        """
        eps = 1e-10
        # ランダム性を入れる
        u = torch.rand_like(logit).clamp_(eps, 1.0 - eps)
        g = torch.log(u) - torch.log(1.0 - u)
        y = torch.sigmoid((logit + g) / self.tau) # 連続値のサンプル
        y_hard = (y > 0.5).float() # 離散値のサンプル
        mask = y_hard.detach() - y.detach() + y # STEの算出
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
        B, _, N = new_pts.shape
        if N <= 1:
            return new_pts.new_tensor(0.0)

        if add_w.dim() == 3:
            add_w = add_w.squeeze(1)  # (B,N)

        M = min(N, self.rep_max_points)
        if M < N:
            idx = torch.randperm(N, device=new_pts.device)[:M]
            p = new_pts[:, :, idx]
            w = add_w[:, idx]
        else:
            p = new_pts
            w = add_w

        p = p.permute(0, 2, 1).contiguous()  # (B,M,3)
        p = p / (self.conv_radius + 1e-8)

        dist = torch.cdist(p, p)  # (B,M,M)
        rep = torch.clamp_max(dist - self.repulse_extent, max=0.0) ** 2

        ww = w.unsqueeze(2) * w.unsqueeze(1)  # (B,M,M)

        # 対角成分を除外
        eye = torch.eye(M, device=dist.device).unsqueeze(0)
        valid = 1.0 - eye

        rep = rep * ww * valid
        denom = (ww * valid).sum()

        return rep.sum() / (denom + 1e-12)

    # def _compute_L_rep_weighted(self, new_pts: torch.Tensor, add_w: torch.Tensor):
    #     """
    #     L_rep: 追加候補点同士の過密を抑制
    #     new_pts: (B,3,N)
    #     add_w  : (B,1,N) または (B,N)
    #     """
    #     B, _, N = new_pts.shape
    #     if N <= 1:
    #         return new_pts.new_tensor(0.0)

    #     if add_w.dim() == 3:
    #         add_w = add_w.squeeze(1)  # (B,N)

    #     M = min(N, self.rep_max_points)
    #     if M < N:
    #         idx = torch.randperm(N, device=new_pts.device)[:M]
    #         p = new_pts[:, :, idx]          # (B,3,M)
    #         w = add_w[:, idx]               # (B,M)
    #     else:
    #         p = new_pts
    #         w = add_w

    #     p = p.permute(0, 2, 1).contiguous()  # (B,M,3)
    #     p = p / (self.conv_radius + 1e-8)

    #     dist = torch.cdist(p, p)  # (B,M,M)

    #     eye = torch.eye(dist.shape[1], device=dist.device).unsqueeze(0)
    #     dist = dist + eye * 1e6

    #     rep = torch.clamp_max(dist - self.repulse_extent, max=0.0) ** 2  # (B,M,M)

    #     # 点対ごとの重み
    #     ww = w.unsqueeze(2) * w.unsqueeze(1)  # (B,M,M)
    #     rep = rep * ww

    #     return rep.sum() / (ww.sum() + 1e-12)

    def forward(self, pts, Fl, D, S, O):
        """セットアップ"""
        B, _, N = pts.shape

        # 追加点数を固定的に決める
        max_add_points = max(1, int(self.max_ratio * N))
        K = max(1, int(self.target_add_ratio * N))
        K = min(K, max_add_points)

        # 入力をチャネル方向にまとめる
        x = torch.cat([pts, Fl, D, S, O], dim=1)
        h = self.mlp(x) # 中間特徴hを算出

        # 各点の追加スコア
        add_logit = self.mlp_logit(h).squeeze(1) # (B,N) # 各点に対して1つのスコア # 確率に変換する前の生のスコア
        add_mask, add_prob = self._sample_binary_concrete(add_logit) # (B,N), (B,N) # SoftなSTEマスク、連続的な追加確率

        """点を追加するための方向ベクトルを計算"""
        dir_raw = torch.tanh(self.mlp_dir(h)) # (B,3,N) # 各点に対して3次元ベクトルを計算
        dir_vec = dir_raw / (dir_raw.norm(dim=1, keepdim=True) + 1e-8) # 方向ベクトルの大きさを1に正規化
        mag = torch.sigmoid(self.mlp_mag(h)).squeeze(1) # (B,N) # 各点について1つのスカラーを算出

        # 実際のOffsetと追加点候補の計算
        offset = dir_vec * (mag.unsqueeze(1) * self.max_offset) # (B,3,N) # 方向と大きさを掛けて最終的なオフセットを算出
        new_pts_all = pts + offset # 全追加候補点
        
        """========== Hard =========="""
        add_idx = torch.topk(add_prob, k=K, dim=1, largest=True).indices # (B,K) # 追加確率add_probが高い順にK点の点インデックスを取る
        add_idx = torch.sort(add_idx, dim=1).values # (B,K) # 昇順に並べ替え

        hard_mask = torch.zeros_like(add_prob) # (B,N) # 0で初期化したHardMask
        hard_mask.scatter_(1, add_idx, 1.0) # (B,N) # 上位K個に対してのみ存在確率を1に設定
        hard_ratio = hard_mask.mean(dim=1) # 実際のHard選択での追加率の計算

        idx_expand = add_idx.unsqueeze(1).expand(-1, 3, -1) # (B,3,K) # add_idxを(B,K)から(B,3,K)に拡張
        new_pts_hard = torch.gather(new_pts_all, 2, idx_expand) # (B,3,K) # new_pts_allの中からadd_idxの分だけ取り出す
        pts_add_hard = torch.cat([pts, new_pts_hard], dim=2) # 入力点群と追加点群の合成
    
        """========== Soft =========="""
        thr = torch.gather(add_prob, 1, add_idx[:, -1:].detach()) # (B,1) # add_idxの中で最も低い確率を閾値とする
        self.tau_match = max(self.tau_match, 1e-6) # 0除算や極端な値を避けるための下限

        soft_mask_raw = torch.sigmoid((add_prob - thr) / self.tau_match) # (B,N) # 閾値を境にTopKに近い滑らかなSoftMaskを作る
        soft_mean_det = soft_mask_raw.mean(dim=1, keepdim=True).detach() # (B,1) # SoftMaskの平均追加率を計算し、勾配を切った値として保持
        scale = hard_ratio.unsqueeze(1) / (soft_mean_det + 1e-12) # (B,1) # SoftMaskの平均がHardMaskの平均と合うようにスケール係数を作る
        soft_mask = (soft_mask_raw * scale).clamp(0.0, 1.0) # (B,N) # SoftMaskをスケーリングして、0-1に収める

        """========== STE mask =========="""
        mask_st = hard_mask - soft_mask.detach() + soft_mask # (B,N) # forwardはHardMask、backwardはSoftMaskになるSTEマスク
        add_w = mask_st.unsqueeze(1) # (B,1,N) #下流の計算のために(B,1,N)に形を整える
    
        """========== Calculate Loss =========="""
        L_fit = self._compute_L_fit(pts, new_pts_all, add_w)
        L_rep = self._compute_L_rep_weighted(new_pts_all, add_w)
        # L_fit = self._compute_L_fit(pts, new_pts_hard)
        # L_rep = self._compute_L_rep(new_pts_hard)

        mean_add_ratio_soft = soft_mask.mean(dim=1) # (B,) # スケール後SoftMaskの平均追加率
        mean_add_ratio_pred = soft_mask_raw.mean(dim=1) # (B,) # スケール前の生SoftMaskの平均追加率
        target_ratio = torch.full_like(hard_ratio, float(self.target_add_ratio)) # (B,) # 目標追加率

        delta_cnt = (mean_add_ratio_pred - target_ratio).abs() # 目標追加率と実際の追加率のずれ
        L_cnt = torch.log1p(128 * delta_cnt).mean() # 追加率のずれに対する損失（小さなずれは緩やか、大きなずれは急激にペナルティ）

        loss_add = (self.add_cnt * L_cnt + self.add_fit * L_fit + self.add_rep * L_rep) # 実際の追加損失の計算

        if self.writer is not None and hasattr(self.writer, "write"):
            self.writer.write(
                f"L_add   :{loss_add:.4f}->"
                f"L_cnt:{L_cnt:.4f}, L_fit:{L_fit:.4f}, L_rep:{L_rep:.4f}, "
                f"AddRatio(soft):{mean_add_ratio_soft.mean().item():.6f}, "
                f"AddRatio(hard):{hard_ratio.mean().item():.6f}"
            )

        return pts_add_hard, new_pts_all, new_pts_hard, add_w, add_idx, loss_add, L_cnt, L_fit, L_rep