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
        self.tau_match = float(getattr(self.cfgs, "prune_soft_match_tau", 0.05))
        self.ratio_min = float(getattr(self.cfgs, "prune_ratio_min", 0.85))
        self.ratio_max = float(getattr(self.cfgs, "prune_ratio_max", 0.999))

        self.prun_cnt = cfgs.prun_cnt
        self.prun_out = cfgs.prun_out

        self.conv_in = nn.Conv1d(in_dim, hidden, 1)
        self.blocks = nn.ModuleList(
            [ResnetBlockConv1d(in_dim, hidden) for _ in range(num_blocks)]
        )
        self.bn_out_keep = nn.BatchNorm1d(hidden) # 残す強さスコアの出力層部分
        self.act_out_keep = nn.ReLU()
        self.conv_out_keep = nn.Conv1d(hidden, 1, 1)

        self.bn_out_ratio = nn.BatchNorm1d(hidden) # 残す割合スコアの出力部分
        self.act_out_ratio = nn.ReLU()
        self.conv_out_ratio = nn.Conv1d(hidden, 1, 1)

    def sample_binary_concrete(self, logit):
        eps = 1e-10
        u = torch.rand_like(logit).clamp_(eps, 1 - eps)
        g = torch.log(u) - torch.log(1 - u)
        y = torch.sigmoid((logit + g) / self.tau)

        y_hard = (y > 0.5).float()
        mask = y_hard.detach() - y.detach() + y
        prob = torch.sigmoid(logit)

        return mask, prob

    def evaluate_mask_similarity(self, hard_mask, soft_mask, keep_prob, k=None):
        # hard_mask, soft_mask, keep_prob: (B, N)
        B, N = hard_mask.shape

        if k is None:
            # hard_mask から残存点数を推定
            k = int(hard_mask[0].sum().item())

        results = {}

        # 1. soft上位k と hard上位k の一致率
        soft_topk = torch.topk(soft_mask, k, dim=1).indices
        hard_topk = torch.topk(hard_mask, k, dim=1).indices

        match_counts = []
        jaccard_list = []

        for b in range(B):
            soft_set = set(soft_topk[b].tolist())
            hard_set = set(hard_topk[b].tolist())

            inter = len(soft_set & hard_set)
            union = len(soft_set | hard_set)

            match_counts.append(inter / k)
            jaccard_list.append(inter / (union + 1e-8))

        results["topk_match_ratio"] = sum(match_counts) / len(match_counts)
        results["jaccard"] = sum(jaccard_list) / len(jaccard_list)

        # 2. keep_prob と soft_mask の順位相関
        corr_list = []
        for b in range(B):
            rank_prob = torch.argsort(torch.argsort(keep_prob[b]))
            rank_soft = torch.argsort(torch.argsort(soft_mask[b]))

            x = rank_prob.float()
            y = rank_soft.float()

            x = x - x.mean()
            y = y - y.mean()

            denom = (x.norm() * y.norm()).item()
            if denom < 1e-12:
                corr = 0.0
            else:
                corr = torch.dot(x, y).item() / denom

            corr_list.append(corr)

        results["rank_corr"] = sum(corr_list) / len(corr_list)

        # 3. hard/soft の平均 keep率
        hard_ratio = hard_mask.mean().item()
        soft_ratio = soft_mask.mean().item()

        results["hard_ratio"] = hard_ratio
        results["soft_ratio"] = soft_ratio
        results["ratio_diff"] = abs(hard_ratio - soft_ratio)

        return results
    
    def forward(self, pts, Ff, D, S, O):
        """==================== SetUp ===================="""
        B, _, N = pts.shape # バッチサイズ、チャネル数、点数
        x = torch.cat([Ff, D, S, O], dim=1) # 入力をチャネル方向に統合

        h = self.conv_in(x) # 入力層で隠れ表現に変換
        for block in self.blocks: # ResBlockに通す
            h = block(h, x)

        # bu_outのBatchNormで特徴分布を整える # act_outのReLUで非線形変換 # conv_outで各点に対して1チャネルで出力を作る
        prun_logit = self.conv_out_keep(self.act_out_keep(self.bn_out_keep(h))).squeeze(1) # (B,N) # 各点に対する「残す強さ」の生スコア
        keep_prob = torch.sigmoid(prun_logit) # (B,N) # sigmoidに通して、各点の残存確率を(0,1)に変換

        ratio_feat = self.act_out_ratio(self.bn_out_ratio(h)) # (B, hidden, N) # 予測用ヘッドの前処理
        ratio_feat = ratio_feat.mean(dim=2, keepdim=True) # (B, hidden, 1) # 平均を取ってまとめる
        ratio_logit = self.conv_out_ratio(ratio_feat).squeeze(2).squeeze(1) # (B,) # 入力の中でどれくらいの割合をKeepするか
        ratio_raw = torch.sigmoid(ratio_logit) # (B,) # 割合スコアを0~1に変換する
        keep_ratio_pred = self.ratio_min + (self.ratio_max - self.ratio_min) * ratio_raw # (B,) # 予測割合を最小、最大値のレンジに収める

        K_keep_each = torch.round(keep_ratio_pred.detach() * N).long() # (B,) # 何点残すのかを整数変換
        K_keep_each = torch.clamp(K_keep_each, min=1, max=N) # 残す点数を1～N範囲に制限
        K_keep = int(K_keep_each.max().item()) # バッチ内で最大点数を1つ取り出す

        """==================== Hard ===================="""
        # K_keep = max(1, int(round(N * float(self.target_ratio)))) # 全点数Nのうち、何点残すのかを決定
        keep_idx_hard = torch.topk(keep_prob, k=K_keep, dim=1, largest=True).indices # (B,K_keep) # keep_probが高い順にK個の点を選ぶ
        keep_idx_hard = torch.sort(keep_idx_hard, dim=1).values # (B,K_keep) # TopKで選んだindexを昇順に並び替え
        rank = torch.arange(K_keep, device=pts.device).unsqueeze(0).expand(B, -1) # 順位番号の作成
        valid_topk_mask = rank < K_keep_each.unsqueeze(1) # 各サンプルで本当に有効な上位何個かを表すマスクを作成

        hard_mask = torch.zeros_like(keep_prob) # (B,N) # HardMaskを0で初期化
        hard_mask.scatter_(1, keep_idx_hard, valid_topk_mask.float()) # (B,N) # keep_idx_hardで指定した位置に1をセット
        hard_ratio = hard_mask.mean(dim=1) # (B,) # 実際に何割の点を残したのか算出
        idx_expand = keep_idx_hard.unsqueeze(1).expand(-1, 3, -1) # (B,3,K_keep)にkeep_idx_hardを拡張
        pts_hard = torch.gather(pts, 2, idx_expand) # (B,3,K_keep) # ptsから、Hardで選ばれた点だけを取り出す
        pts_hard = pts_hard * valid_topk_mask.unsqueeze(1).float() # 無効な位置を0にする

        # ==================== Soft ====================
        tau_match = max(self.tau_match, 1e-6)
        target_ratio = keep_ratio_pred.unsqueeze(1) # (B,1)
        thr_soft = keep_prob.mean(dim=1, keepdim=True) # 連続的な初期値

        # Newton法風に、soft_mask の平均が target_ratio に近づく閾値を連続更新する
        for _ in range(8):
            soft_tmp = torch.sigmoid((keep_prob - thr_soft) / tau_match) # (B,N)
            mean_tmp = soft_tmp.mean(dim=1, keepdim=True) # (B,1)

            # d/dthr sigmoid((x-thr)/tau) = -(1/tau) * s * (1-s)
            dmean_dthr = -(soft_tmp * (1.0 - soft_tmp) / tau_match).mean(dim=1, keepdim=True) # (B,1)
            dmean_dthr = torch.where( # ゼロ割れ防止
                dmean_dthr.abs() < 1e-8,
                torch.full_like(dmean_dthr, -1e-8),
                dmean_dthr
            )

            # f(thr) = mean_tmp - target_ratio = 0 を解く
            thr_soft = thr_soft - (mean_tmp - target_ratio) / dmean_dthr
        soft_mask = torch.sigmoid((keep_prob - thr_soft) / tau_match)        
        """
        tau_match = max(self.tau_match, 1e-6) # SoftMaskの鋭さを決める温度を基に、極端に小さすぎないようにする
        # thr_soft = torch.gather(keep_prob, 1, keep_idx_hard[:, -1:].detach()) # (B,1) # Hard境界の初期値
        last_rank = (K_keep_each - 1).clamp(min=0) # 各サンプルで最後に残す点の順位を作成
        thr_idx = keep_idx_hard.gather(1, last_rank.unsqueeze(1)) # 各サンプルでHard境界にあたる点のindexを取り出す
        thr_soft = torch.gather(keep_prob, 1, thr_idx.detach()) # keep_probを閾値として取る

        for _ in range(8): # soft_mask の平均が hard_ratio に近づくように閾値を少し補正する
            soft_tmp = torch.sigmoid((keep_prob - thr_soft) / tau_match) # (B,N) # 現在の閾値でSoftMaskの仮計算
            mean_tmp = soft_tmp.mean(dim=1, keepdim=True) # (B,1) # 仮SoftMaskの平均Keep率の計算
            thr_soft = thr_soft + (mean_tmp - hard_ratio.unsqueeze(1)).detach() # HardMaskに合うようにずらす

        soft_mask = torch.sigmoid((keep_prob - thr_soft) / tau_match) # (B,N) # 最終的なSoftMaskを作成
        """
        
        """==================== STE ===================="""
        mask_st = hard_mask - soft_mask.detach() + soft_mask # (B,N) # forwardとbackwardで分け、勾配が流れるようにする
        keep_w = mask_st.unsqueeze(1) # (B,1,N) # チャネル次元を1つ足して拡張
        keep_w_hard = torch.gather(keep_w, 2, keep_idx_hard.unsqueeze(1)) # Hardに選ばれた点に対する重みだけを抜き出す   
        keep_w_hard = keep_w_hard * valid_topk_mask.unsqueeze(1).float() # (B,1,K_keep) # 無効位置の重みを0にする

        """==================== Calculate Loss ===================="""
        L_cnt = 0.0
        """
        # L_cnt計算
        target_ratio = torch.full_like(hard_ratio, float(self.target_ratio)) # (B,)
        mean_keep_ratio = soft_mask.mean(dim=1) # (B,) # SofMaskの平均を取り、予測したKeep率を算出

        delta_cnt = (mean_keep_ratio - target_ratio).abs() # 予測Keep率と目標Keep率の差
        L_cnt = torch.log1p(128 * delta_cnt).mean() # delta_cntに近いほど小さく数値が動く
        """
        score = D.squeeze(1)  # (B,N) # 密度スコアを（B,N）に整形
        eps = 1e-6
        r = (1.0 / (score + eps)) if self.high_is_inlier else score # スコアが高いほどinlierとする
        r = r / (r.mean(dim=1, keepdim=True) + eps) # スコアを正規化
        w_inlier = 1.0 / (1.0 + (r / self.c_robust) ** 2) # (B,N) # rが大きい点ほど重みが小さくなるロバスト重み
        w_inlier = w_inlier.detach() # この損失からD側へ勾配が流れないようにする # 密度スコアをこの損失で学習しないように

        L_out = torch.nn.functional.mse_loss(mask_st, w_inlier) # 外れ点を消しているか否かの損失計算
        loss_prun = L_out # 削除損失計算
        # loss_prun = self.prun_cnt * L_cnt + self.prun_out * L_out # 削除損失計算

        if self.writer is not None and hasattr(self.writer, "write"): # ログ
            self.writer.write(
                f"L_prun  :{loss_prun:.4f}->"
                f"L_cnt:{L_cnt:.4f}, L_out:{L_out:.4f}, "
                f"SoftRatio:{soft_mask.mean(dim=1).mean().item():.6f}, "
                f"HardRatio:{hard_ratio.mean().item():.6f}"
            )        
        # print(f"SoftMask->max:{soft_mask.max().item():.4f}, mean:{soft_mask.mean().item():.4f}, min:{soft_mask.min().item():.4f}")
        # print(f"HardMask->max:{hard_mask.max().item():.4f}, mean:{hard_mask.mean().item():.4f}, min:{hard_mask.min().item():.4f}")
        # print(self.evaluate_mask_similarity(hard_mask, soft_mask, keep_prob, k=K_keep))

        return pts_hard, keep_w, keep_idx_hard, loss_prun, L_cnt, L_out