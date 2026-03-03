import torch
import torch.nn as nn
from typing import Optional


class ResnetBlockConv1d(nn.Module):
    """
    ScoreNet由来の 1×1Conv 残差ブロック（条件入力 c を加算注入）

    - x: (B, hidden, N)
    - c: (B, c_dim, N)

    == Layer ==
    BN
    ReLU
    Conv

    BN
    ReLU
    Conv

    Conv 条件付きバイアスの計算F

    ShrotCut identity

    OutPut x+identity+F
    """
    def __init__(self,
                 c_dim: int,
                 size_in: int,
                 size_h: Optional[int] = None,
                 size_out: Optional[int] = None):
        super().__init__()

        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.bn_0 = nn.BatchNorm1d(size_in)
        self.bn_1 = nn.BatchNorm1d(size_h)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)

        # 条件注入
        self.fc_c = nn.Conv1d(c_dim, size_out, 1)

        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)

        # 安定化のため最後の重みをゼロ初期化
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        net = self.fc_0(self.actvn(self.bn_0(x)))
        dx = self.fc_1(self.actvn(self.bn_1(net)))

        if self.shortcut is None:
            x_s = x
        else:
            x_s = self.shortcut(x)

        return x_s + dx + self.fc_c(c)