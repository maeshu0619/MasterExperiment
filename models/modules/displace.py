import torch
import torch.nn as nn
import torch.nn.functional as F
from .resblock import ResnetBlockConv1d


class DisplacementModule(nn.Module):
    """
    各点ごとに
      - 動かす/動かさない（gate）
      - 動かす方向（dir）
      - 動かす大きさ（mag, 微小）
    を学習する Displacement
    """

    def __init__(self, cfgs, writer):
        super().__init__()
        self.cfgs = cfgs
        self.writer = writer

        # 目標移動率
        self.target_disp_ratio = float(getattr(cfgs, "target_disp_ratio", 0.05))

        # 損失重み
        self.disp_cnt = float(getattr(cfgs, "disp_cnt", 1.0))
        self.disp_fit = float(getattr(cfgs, "disp_fit", 1.0))

        feat_dim = cfgs.fp_mlp_channels[-1] + 2 + cfgs.octree_ctx_dim
        self.c_dim = 3 + feat_dim

        hidden = int(getattr(cfgs, "disp_hidden_dim", 128))
        num_blocks = int(getattr(cfgs, "disp_num_blocks", 4))

        self.num_steps = int(getattr(cfgs, "disp_num_steps", 1))
        self.step_size = float(getattr(cfgs, "disp_step_size", 1.0))
        self.step_decay = float(getattr(cfgs, "disp_step_decay", 0.95))
        self.grad_clip = float(getattr(cfgs, "disp_grad_clip", 10.0))

        # ゲート/大きさを使うか（既存実験を壊さないためのスイッチ）
        self.use_gate = bool(getattr(cfgs, "disp_use_gate", True))

        self.conv_in = nn.Conv1d(self.c_dim, hidden, 1)
        self.blocks = nn.ModuleList(
            [ResnetBlockConv1d(self.c_dim, hidden) for _ in range(num_blocks)]
        )
        self.bn_out = nn.BatchNorm1d(hidden)
        self.act_out = nn.ReLU()

        # 方向(3), 大きさ(1), ゲート(1)
        self.conv_dir = nn.Conv1d(hidden, 3, 1)
        self.conv_mag = nn.Conv1d(hidden, 1, 1)
        self.conv_gate = nn.Conv1d(hidden, 1, 1)

        # 初期状態で「ほぼ不動」に寄せる（学習を安定化）
        nn.init.zeros_(self.conv_dir.weight)
        nn.init.zeros_(self.conv_dir.bias)

        nn.init.zeros_(self.conv_mag.weight)
        nn.init.constant_(self.conv_mag.bias, -2.0)

        # 外部から損失計算に使えるように保持
        self.last_gate = None
        self.last_delta = None
        self.last_mag = None

    def _predict(self, pts, F, D, S, O):
        c = torch.cat([pts, F, D, S, O], dim=1)

        net = self.conv_in(c)
        for block in self.blocks:
            net = block(net, c)

        h = self.act_out(self.bn_out(net))

        dir_raw = torch.tanh(self.conv_dir(h))  # [-1,1]
        dir_raw = torch.clamp(dir_raw, -self.grad_clip, self.grad_clip)

        mag = torch.sigmoid(self.conv_mag(h))   # [0,1]
        gate = torch.sigmoid(self.conv_gate(h)) # [0,1]
        return dir_raw, mag, gate

    def _compute_L_disp_fit(self, delta_all: torch.Tensor, disp_w: torch.Tensor, max_disp: float):
        """
        delta_all : (B,3,N)
        disp_w    : (B,1,N) または (B,N)
        max_disp  : 変位上限
        """
        if disp_w.dim() == 3:
            disp_w = disp_w.squeeze(1)  # (B,N)

        delta_norm = torch.norm(delta_all, dim=1)  # (B,N)
        val = (delta_norm / (max_disp + 1e-8)) ** 2

        return (disp_w * val).sum() / (disp_w.sum() + 1e-12)
    
    @staticmethod
    def _normalize_dir(v):
        # v: [B,3,N]
        return v / (torch.norm(v, dim=1, keepdim=True) + 1e-8)

    @staticmethod
    def _clip_delta(delta, max_norm):
        if max_norm <= 0:
            return delta
        norm = torch.norm(delta, dim=1, keepdim=True) + 1e-8
        scale = torch.clamp(max_norm / norm, max=1.0)
        return delta * scale

    def forward(self, pts, F, D, S, O):
        pts_dis = pts
        max_disp = float(getattr(self.cfgs, "max_disp_offset", 0.01))

        loss_cnt_list = []
        loss_fit_list = []

        dis_w = None
        last_move_idx = None
        last_soft_mask_raw = None

        for step in range(max(self.num_steps, 1)):
            dir_raw, mag01, gate01 = self._predict(pts_dis, F, D, S, O)

            # 方向を単位ベクトル化
            direction = self._normalize_dir(dir_raw)

            # 最大変位以内に制限
            mag = mag01 * max_disp

            # ステップ係数
            s = self.step_size * (self.step_decay ** step)

            # ゲートを掛ける前の全候補変位
            delta_all = s * mag * direction
            delta_all = self._clip_delta(delta_all, max_disp)   # (B,3,N)

            # =====================================================
            # Hard / Soft mask を gate01 から作る
            # =====================================================
            move_prob = gate01.squeeze(1)   # (B,N)

            B, N = move_prob.shape
            K_move = max(1, int(round(N * float(self.target_disp_ratio))))

            # ===== Hard =====
            move_idx = torch.topk(move_prob, k=K_move, dim=1, largest=True).indices   # (B,K)
            move_idx = torch.sort(move_idx, dim=1).values                              # (B,K)

            hard_mask = torch.zeros_like(move_prob)                                    # (B,N)
            hard_mask.scatter_(1, move_idx, 1.0)                                       # (B,N)

            hard_ratio = hard_mask.mean(dim=1)                                         # (B,)

            # ===== Soft =====
            thr = torch.gather(move_prob, 1, move_idx[:, -1:].detach())                # (B,1)

            tau_match = float(getattr(self.cfgs, "disp_soft_match_tau", 0.05))
            tau_match = max(tau_match, 1e-6)

            soft_mask_raw = torch.sigmoid((move_prob - thr) / tau_match)               # (B,N)

            soft_mean_det = soft_mask_raw.mean(dim=1, keepdim=True).detach()           # (B,1)
            scale = hard_ratio.unsqueeze(1) / (soft_mean_det + 1e-12)                  # (B,1)
            soft_mask = (soft_mask_raw * scale).clamp(0.0, 1.0)                        # (B,N)

            # ===== STE =====
            mask_st = hard_mask - soft_mask.detach() + soft_mask                       # (B,N)
            disp_w = mask_st.unsqueeze(1)                                              # (B,1,N)

            # 実際の変位
            delta = delta_all * disp_w                                                 # (B,3,N)
            pts_dis = pts_dis + delta

            # ===== Loss =====
            target_ratio = torch.full_like(hard_ratio, float(self.target_disp_ratio))  # (B,)
            mean_move_ratio_pred = soft_mask_raw.mean(dim=1)                           # (B,)

            delta_cnt = (mean_move_ratio_pred - target_ratio).abs()
            L_cnt = torch.log1p(128 * delta_cnt).mean()

            L_fit = self._compute_L_disp_fit(delta_all, disp_w, max_disp)

            loss_cnt_list.append(L_cnt)
            loss_fit_list.append(L_fit)

            dis_w = disp_w
            last_move_idx = move_idx
            last_soft_mask_raw = soft_mask_raw

            # ログ用
            self.last_gate = gate01.detach()
            self.last_mag = mag.detach()
            self.last_delta = delta.detach()

        loss_cnt = torch.stack(loss_cnt_list).mean()
        loss_fit = torch.stack(loss_fit_list).mean()
        # loss_disp = self.disp_cnt * loss_cnt + self.disp_fit * loss_fit
        loss_disp = loss_fit

        if self.writer is not None and hasattr(self.writer, "write"):
            self.writer.write(
                f"L_disp  :{loss_disp:.4f}->"
                f"L_cnt:{loss_cnt:.4f}, L_fit:{loss_fit:.4f}, "
                f"MoveRatio(pred_raw):{last_soft_mask_raw.mean(dim=1).mean().item():.6f}, "
                f"MoveRatio(hard):{hard_ratio.mean().item():.6f}"
            )

        return pts_dis, dis_w, last_move_idx, loss_disp