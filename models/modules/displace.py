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

        feat_dim = cfgs.fp_mlp_channels[-1] + 2
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
        nn.init.constant_(self.conv_mag.bias, -6.0)   # sigmoid(-6) ≈ 0.0025

        nn.init.zeros_(self.conv_gate.weight)
        nn.init.constant_(self.conv_gate.bias, -6.0)  # sigmoid(-6) ≈ 0.0025

        # 外部から損失計算に使えるように保持
        self.last_gate = None
        self.last_delta = None
        self.last_mag = None

    def _predict(self, pts, F_prime, d_prime, s_prime):
        c = torch.cat([pts, F_prime, d_prime, s_prime], dim=1)

        net = self.conv_in(c)
        for block in self.blocks:
            net = block(net, c)

        h = self.act_out(self.bn_out(net))

        dir_raw = torch.tanh(self.conv_dir(h))  # [-1,1]
        dir_raw = torch.clamp(dir_raw, -self.grad_clip, self.grad_clip)

        mag = torch.sigmoid(self.conv_mag(h))   # [0,1]
        gate = torch.sigmoid(self.conv_gate(h)) # [0,1]
        return dir_raw, mag, gate

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

    def forward(self, pts, F_prime, d_prime, s_prime):
        pts_next = pts
        max_disp = float(getattr(self.cfgs, "max_disp_offset", 0.002))

        for step in range(max(self.num_steps, 1)):
            dir_raw, mag01, gate01 = self._predict(pts_next, F_prime, d_prime, s_prime)

            # 方向を単位ベクトル化
            direction = self._normalize_dir(dir_raw)

            # “微小”を保証：max_disp を上限にする
            mag = mag01 * max_disp

            # ステップスケジュール（既存の設計を維持）
            s = self.step_size * (self.step_decay ** step)

            if self.use_gate:
                delta = s * (gate01 * mag) * direction
            else:
                # 旧挙動に近い（ゲートなし）
                delta = s * mag * direction

            delta = self._clip_delta(delta, max_disp)
            pts_next = pts_next + delta

            # loss側で参照可能に保存（最後のstepの値）
            self.last_gate = gate01.detach()
            self.last_mag = mag.detach()
            self.last_delta = delta.detach()

        return pts_next