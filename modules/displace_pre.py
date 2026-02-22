import torch
import torch.nn as nn


class DisplacementModule(nn.Module):
    """
    Displacement Module

    各点ごとに異なる移動ベクトル delta を予測し、
    入力点群の座標を更新する

    変更点:
        - d と s も入力に含め、AddModule と同様にスコアを見ながら移動を決める
    """

    def __init__(self, cfgs, writer):
        super().__init__()
        self.cfgs = cfgs

        # 入力次元: F' + d' + s'
        in_dim = cfgs.fp_mlp_channels[-1] + 2
        hidden = cfgs.disp_hidden_dim
        self.writer = writer

        # 移動方向（点ごと）
        self.dir_mlp = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 1),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(hidden, 3, 1)
        )
        # 移動量ゲート（点ごと）
        self.gate_mlp = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 1),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(hidden, 1, 1)
        )

        nn.init.normal_(self.gate_mlp[3].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.gate_mlp[3].bias, 0.0)

    def forward(self, pts, F_prime, d_prime, s_prime):

        x = torch.cat([F_prime, d_prime, s_prime], dim=1)

        # ---------- 方向 ----------
        direction_raw = self.dir_mlp(x)
        direction = torch.tanh(direction_raw)
        direction = direction / (direction.norm(dim=1, keepdim=True) + 1e-8)

        # ---------- 距離 ----------
        mag_raw = self.gate_mlp(x)
        mag_raw = torch.clamp(mag_raw, -10, 10)
        mag = 0.5 * (torch.tanh(mag_raw) + 1.0)

        max_disp = float(getattr(self.cfgs, "max_disp_offset", 0.002))
        delta = direction * mag * max_disp

        # ---------- Debug ----------
        if self.cfgs.trainORtest == "test":
            delta_norm = torch.norm(delta, dim=1)
            print(f"\n[Displacement] max: {delta_norm.max().item():.6f}, "
                f"mean: {delta_norm.mean().item():.6f}, "
                f"min: {delta_norm.min().item():.6f}")
            print("-direction mean abs:",direction.abs().mean().item())
            print("-mag mean:",mag.mean().item())
            print("-mag_raw mean:", mag_raw.mean().item())
            print("-mag_raw min:", mag_raw.min().item())
            print("-mag_raw max:", mag_raw.max().item())

        pts_disp = pts + delta

        return pts_disp
