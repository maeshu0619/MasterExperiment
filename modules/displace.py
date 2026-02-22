import torch
import torch.nn as nn
from .resblock import ResnetBlockConv1d


class DisplacementModule(nn.Module):
    """
    ScoreNet + DenoiseNet 風 Displacement
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

        self.conv_in = nn.Conv1d(self.c_dim, hidden, 1)
        self.blocks = nn.ModuleList(
            [ResnetBlockConv1d(self.c_dim, hidden) for _ in range(num_blocks)]
        )
        self.bn_out = nn.BatchNorm1d(hidden)
        self.act_out = nn.ReLU()
        self.conv_out = nn.Conv1d(hidden, 3, 1)

        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def _predict_grad(self, pts, F_prime, d_prime, s_prime):
        c = torch.cat([pts, F_prime, d_prime, s_prime], dim=1)

        net = self.conv_in(c)
        for block in self.blocks:
            net = block(net, c)

        grad = self.conv_out(self.act_out(self.bn_out(net)))
        grad = torch.clamp(grad, -self.grad_clip, self.grad_clip)
        return grad

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
            grad = self._predict_grad(pts_next, F_prime, d_prime, s_prime)
            s = self.step_size * (self.step_decay ** step)

            delta = s * grad
            delta = self._clip_delta(delta, max_disp)

            pts_next = pts_next + delta

        return pts_next