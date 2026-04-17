import torch
import torch.nn as nn
import torch.nn.functional as F
from .resblock import ResnetBlockConv1d

import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
sys.path.append(ROOT_DIR)

from utils.segmentation.utils_unoutdet import *


class PruningModule(nn.Module):
    """
    削除領域 + 削除点数を内部で学習するPrune
    """
    def __init__(self, args, writer):
        super().__init__()
        self.args = args
        self.writer = writer

        in_dim = args.fp_mlp_channels[-1] + 3 + args.octree_ctx_dim
        hidden = args.prune_hidden_dim

        self.debug_tensors = {}

        num_blocks = getattr(args, "prune_num_blocks", 3)
        self.tau = getattr(args, "prune_tau", 0.5)
        self.target_ratio = getattr(args, "prune_target_keep_ratio", 0.97)
        self.high_is_inlier = getattr(self.args, "prune_d_high_is_inlier", True)
        self.c_robust = float(getattr(self.args, "prune_robust_c", 2.0))
        self.tau_match = float(getattr(self.args, "prune_soft_match_tau", 0.05))
        self.ratio_min = float(getattr(self.args, "prune_ratio_min", 0.85))
        self.ratio_max = float(getattr(self.args, "prune_ratio_max", 0.999))
        self.hard_thr = float(getattr(self.args, "prune_hard_thr", 0.5))

        self.prun_cnt = args.prun_cnt
        self.prun_out = args.prun_out
        self.use_label_count = bool(getattr(args, "prune_use_label_count", True))

        self.conv_in = nn.Conv1d(in_dim, hidden, 1)
        self.blocks = nn.ModuleList(
            [ResnetBlockConv1d(in_dim, hidden) for _ in range(num_blocks)]
        )
        self.bn_out_keep = nn.BatchNorm1d(hidden) # 残す強さスコアの出力層部分
        self.act_out_keep = nn.ReLU()
        self.conv_out_keep = nn.Conv1d(hidden, 1, 1)

        self.bn_out_ratio = nn.BatchNorm1d(hidden) # 残す割合スコアの出力層部分
        self.act_out_ratio = nn.ReLU()
        self.conv_out_ratio = nn.Conv1d(hidden, 1, 1)
        self.last_policy_log_prob = None

    @staticmethod
    def _bernoulli_mask_log_prob(logit, hard_mask):
        log_p_keep = torch.nn.functional.logsigmoid(logit)
        log_p_drop = torch.nn.functional.logsigmoid(-logit)
        return (hard_mask.detach() * log_p_keep + (1.0 - hard_mask.detach()) * log_p_drop).mean()

    @staticmethod
    def _balanced_binary_weights(target_pos):
        B, N = target_pos.shape
        total = target_pos.new_full((B, 1), float(N))
        pos = target_pos.sum(dim=1, keepdim=True)
        neg = total - pos

        has_pos = pos > 0
        has_neg = neg > 0
        both = has_pos & has_neg

        pos_w = torch.where(
            has_pos,
            torch.where(both, 0.5 * total / pos.clamp_min(1.0), total / pos.clamp_min(1.0)),
            torch.zeros_like(pos),
        )
        neg_w = torch.where(
            has_neg,
            torch.where(both, 0.5 * total / neg.clamp_min(1.0), total / neg.clamp_min(1.0)),
            torch.zeros_like(neg),
        )

        weights = target_pos * pos_w + (1.0 - target_pos) * neg_w
        return weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-12)

    def _outlier_label_loss(self, prun_logit, target_out):
        # prun_logit は「残す」logitなので、外れ点検出では符号を反転して
        # 1=outlier を直接 foreground として扱う。
        drop_logit = -prun_logit
        weights = self._balanced_binary_weights(target_out)
        L_out_bce = F.binary_cross_entropy_with_logits(
            drop_logit,
            target_out,
            weight=weights,
            reduction="sum",
        ) / weights.sum().clamp_min(1e-12)
        L_out_lovasz = lovasz_hinge(drop_logit, target_out)
        return L_out_bce + L_out_lovasz

    def _solve_soft_threshold(self, keep_prob, keep_ratio_pred):
        tau_match = max(self.tau_match, 1e-6)
        with torch.no_grad():
            target_ratio = keep_ratio_pred.unsqueeze(1)
            thr_soft = keep_prob.mean(dim=1, keepdim=True)
            for _ in range(8):
                soft_tmp = torch.sigmoid((keep_prob - thr_soft) / tau_match)
                mean_tmp = soft_tmp.mean(dim=1, keepdim=True)
                dmean_dthr = -(soft_tmp * (1.0 - soft_tmp) / tau_match).mean(dim=1, keepdim=True)
                dmean_dthr = torch.where(
                    dmean_dthr.abs() < 1e-8,
                    torch.full_like(dmean_dthr, -1e-8),
                    dmean_dthr
                )
                thr_soft = thr_soft - (mean_tmp - target_ratio) / dmean_dthr
        return thr_soft

    def _build_hard_selection(self, pts, keep_prob, k_each):
        B, _, N = pts.shape
        max_keep = int(k_each.max().item())
        keep_idx_hard = torch.topk(
            keep_prob,
            k=max_keep,
            dim=1,
            largest=True,
            sorted=True,
        ).indices
        valid_topk_mask = (
            torch.arange(max_keep, device=pts.device).unsqueeze(0) < k_each.unsqueeze(1)
        )
        hard_mask = torch.zeros_like(keep_prob)
        hard_mask.scatter_(1, keep_idx_hard, valid_topk_mask.to(keep_prob.dtype))

        pts_prun_hard = torch.gather(
            pts,
            2,
            keep_idx_hard.unsqueeze(1).expand(-1, pts.size(1), -1),
        )
        pts_prun_hard = pts_prun_hard * valid_topk_mask.unsqueeze(1).to(pts.dtype)
        return pts_prun_hard, keep_idx_hard, hard_mask, valid_topk_mask

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
    
    def forward(self, pts, Ff, Den, Str, Oct, Out, OutLabel):
        if self.args.trainORtest == "train":
            """==================== SetUp ===================="""
            B, _, N = pts.shape # バッチサイズ、チャネル数、点数
            target_out = OutLabel.squeeze(1).to(dtype=pts.dtype).detach() # [B,N], 1=outlier(消す)
            target_keep = 1.0 - target_out # [B,N], 1=inlier(残す)
            target_keep_ratio_raw = target_keep.mean(dim=1) # [B]
            target_keep_ratio = target_keep_ratio_raw.clamp(
                min=self.ratio_min,
                max=self.ratio_max,
            )
            x = torch.cat([Ff, Den, Str, Oct, Out], dim=1) # 入力をチャネル方向に統合
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

            ratio_for_hard = target_keep_ratio if self.use_label_count else keep_ratio_pred.detach()
            K_keep_each = torch.round(ratio_for_hard.detach() * N).long() # (B,) # 何点残すのかを整数変換
            K_keep_each = torch.clamp(K_keep_each, min=1, max=N) # 残す点数を1～N範囲に制限

            """==================== Hard ===================="""
            pts_prun_hard, keep_idx_hard, hard_mask, valid_topk_mask = self._build_hard_selection(
                pts,
                keep_prob,
                K_keep_each,
            )
            hard_ratio = hard_mask.mean(dim=1) # [B] # 全点のうち何割残ったかを計算
            self.last_policy_log_prob = self._bernoulli_mask_log_prob(prun_logit, hard_mask)

            # ==================== Soft ====================
            # Hard出力はtop-kのまま、backwardではkeep_probとkeep_ratio_predの両方へ
            # 圧縮lossが戻るようにする。従来のno_grad threshold解法ではratio headが
            # 内部Pruning lossに強く支配され、圧縮効率の悪化を戻しにくかった。
            tau_match = max(self.tau_match, 1e-6)
            kth_pos = (K_keep_each - 1).clamp(min=0).unsqueeze(1)
            kth_idx = torch.gather(keep_idx_hard, 1, kth_pos)
            thr_soft = torch.gather(keep_prob, 1, kth_idx.detach())
            soft_mask_raw = torch.sigmoid((keep_prob - thr_soft) / tau_match)
            soft_mean_det = soft_mask_raw.mean(dim=1, keepdim=True).detach()
            soft_mask = (
                soft_mask_raw * (keep_ratio_pred.unsqueeze(1) / (soft_mean_det + 1e-12))
            ).clamp(0.0, 1.0)

            """==================== STE ===================="""
            mask_st = hard_mask - soft_mask.detach() + soft_mask # (B,N) # forwardとbackwardで分け、勾配が流れるようにする
            keep_w = mask_st.unsqueeze(1) # (B,1,N) # チャネル次元を1つ足して拡張
            keep_w_hard = torch.gather(keep_w, 2, keep_idx_hard.unsqueeze(1)) # Hardに選ばれた点に対する重みだけを抜き出す   
            keep_w_hard = keep_w_hard * valid_topk_mask.unsqueeze(1).to(keep_w.dtype) # (B,1,K_keep) # 無効位置の重みを0にする

            """==================== Calculate Loss ===================="""
            L_out = self._outlier_label_loss(prun_logit, target_out) # 外れ点をforegroundとして見る損失
            L_ratio = F.l1_loss(
                keep_ratio_pred,
                target_keep_ratio,
            )
            loss_prun = L_out + self.prun_cnt * L_ratio # 削除損失計算

            out_count = target_out.sum(dim=1).clamp_min(1.0)
            kept_out_ratio = (hard_mask * target_out).sum(dim=1) / out_count
            dropped_out_ratio = 1.0 - kept_out_ratio

            if (
                getattr(self.args, "verbose_step_logs", False)
                and getattr(self.args, "_log_this_step", True)
                and self.writer is not None
                and hasattr(self.writer, "write")
            ): # ログ
                self.writer.write(
                    f"L_prun  :{loss_prun:.4f}->"
                    f"L_out:{L_out:.4f}, L_ratio:{L_ratio:.4f}, "
                    f"LabelKeep:{target_keep_ratio_raw.mean().item():.6f}, "
                    f"PrunRatio(soft):{soft_mask.mean(dim=1).mean().item():.6f}, "
                    f"PrunRatio(hard):{hard_ratio.mean().item():.6f}, "
                    f"OutDrop(hard):{dropped_out_ratio.mean().item():.6f}"
                )

            # ===== デバッグ用に内部テンソルを保持 =====
            if getattr(self.args, "retain_debug_tensors", False):
                self.debug_tensors = {
                    "prun_logit": prun_logit,
                    "keep_prob": keep_prob,
                    "keep_ratio_pred": keep_ratio_pred,
                    "hard_mask": hard_mask,
                    "soft_mask": soft_mask,
                    "keep_w_full": keep_w,
                    "target_out": target_out,
                }

                for name, tensor in self.debug_tensors.items():
                    if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                        tensor.retain_grad()
            else:
                self.debug_tensors = {}

            return pts_prun_hard, keep_w, keep_idx_hard, valid_topk_mask, loss_prun, L_out
        else:
            """==================== SetUp ===================="""
            B, _, N = pts.shape # バッチサイズ、チャネル数、点数
            target_keep_ratio = None
            if OutLabel is not None and self.use_label_count:
                target_out = OutLabel.squeeze(1).to(dtype=pts.dtype).detach()
                target_keep_ratio = (1.0 - target_out).mean(dim=1).clamp(
                    min=self.ratio_min,
                    max=self.ratio_max,
                )
            x = torch.cat([Ff, Den, Str, Oct, Out], dim=1) # 入力をチャネル方向に統合
            h = self.conv_in(x) # 入力層で隠れ表現に変換
            for block in self.blocks: # ResBlockに通す
                h = block(h, x)
            prun_logit = self.conv_out_keep(self.act_out_keep(self.bn_out_keep(h))).squeeze(1) # (B,N) # 各点に対する「残す強さ」の生スコア
            keep_prob = torch.sigmoid(prun_logit) # (B,N) # sigmoidに通して、各点の残存確率を(0,1)に変換

            ratio_feat = self.act_out_ratio(self.bn_out_ratio(h)) # (B, hidden, N)
            ratio_feat = ratio_feat.mean(dim=2, keepdim=True) # (B, hidden, 1)
            ratio_logit = self.conv_out_ratio(ratio_feat).squeeze(2).squeeze(1) # (B,)
            ratio_raw = torch.sigmoid(ratio_logit) # (B,)
            keep_ratio_pred = self.ratio_min + (self.ratio_max - self.ratio_min) * ratio_raw # (B,)

            """==================== Hard ===================="""
            ratio_for_hard = target_keep_ratio if target_keep_ratio is not None else keep_ratio_pred.detach()
            K_keep_each = torch.round(ratio_for_hard.detach() * N).long() # (B,)
            K_keep_each = torch.clamp(K_keep_each, min=1, max=N)
            pts_prun_hard, keep_idx_hard, _, valid_topk_mask = self._build_hard_selection(
                pts,
                keep_prob,
                K_keep_each,
            )

            return pts_prun_hard, keep_idx_hard, valid_topk_mask
