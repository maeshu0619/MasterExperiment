import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _CompressionSurrogateNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, pred_clip=2.0):
        super().__init__()
        hidden_dim = max(int(hidden_dim), 16)
        self.pred_clip = float(pred_clip)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x):
        raw = self.net(x)
        if self.pred_clip > 0:
            return self.pred_clip * torch.tanh(raw / self.pred_clip)
        return raw


class SurrogateCompressionLossMixin:
    @staticmethod
    def _parse_surrogate_levels(args):
        raw = getattr(args, "compression_surrogate_levels", "4,6,8")
        if isinstance(raw, (list, tuple)):
            vals = raw
        else:
            vals = str(raw).replace(" ", "").split(",")
        levels = []
        for val in vals:
            if val == "":
                continue
            level = int(val)
            if level > 0:
                levels.append(level)
        return levels or [4, 6, 8]

    def _ensure_surrogate_device(self, device):
        first_param = next(self.compression_surrogate.parameters())
        if first_param.device != device:
            self.compression_surrogate = self.compression_surrogate.to(device)
            state = self.surrogate_optimizer.state
            for value in state.values():
                for key, item in value.items():
                    if torch.is_tensor(item):
                        value[key] = item.to(device)

    def _surrogate_target_from_actual(self, args, stats_gen, stats_ref, device):
        scale = float(getattr(args, "compression_surrogate_target_scale", 100.0))
        # Keep the teacher as a relative ratio internally. With the default
        # scale=100, the surrogate learns percent-sized values, but plotting is
        # always converted explicitly from ratio to percent below.
        values = [
            self._relative_ratio(float(stats_gen["bit"]), float(stats_ref["bit"])),
            self._relative_ratio(float(stats_gen["node"]), float(stats_ref["node"]), ref_min=1.0),
            self._relative_ratio(float(stats_gen["single"]), float(stats_ref["single"]), ref_min=1.0),
            self._relative_ratio(float(stats_gen["bpn"]), float(stats_ref["bpn"])),
        ]
        clip = float(getattr(args, "compression_surrogate_pred_clip", 2.0))
        target = scale * torch.tensor(values, device=device, dtype=torch.float32).unsqueeze(0)
        if clip > 0:
            target = target.clamp(min=-clip, max=clip)
        return target

    @staticmethod
    def _decode_keys(keys, grid):
        grid_t = torch.as_tensor(grid, device=keys.device, dtype=torch.long)
        xy = grid_t * grid_t
        z = torch.div(keys, xy, rounding_mode="floor")
        rem = keys - z * xy
        y = torch.div(rem, grid_t, rounding_mode="floor")
        x = rem - y * grid_t
        return torch.stack([x, y, z], dim=1)

    def _soft_level_stats(self, coords_norm, weights, level, args):
        grid = 2 ** int(level)
        scaled = (coords_norm * float(grid - 1)).clamp(0.0, float(grid - 1))
        base = torch.floor(scaled).to(torch.long)
        frac = scaled - base.to(scaled.dtype)

        masses = []
        keys = []
        for bx in (0, 1):
            wx = frac[0] if bx else (1.0 - frac[0])
            ix = (base[0] + bx).clamp(0, grid - 1)
            for by in (0, 1):
                wy = frac[1] if by else (1.0 - frac[1])
                iy = (base[1] + by).clamp(0, grid - 1)
                for bz in (0, 1):
                    wz = frac[2] if bz else (1.0 - frac[2])
                    iz = (base[2] + bz).clamp(0, grid - 1)
                    corner_weight = (wx * wy * wz).clamp_min(0.0)
                    masses.append(weights * corner_weight)
                    keys.append(ix + grid * (iy + grid * iz))

        mass = torch.cat(masses, dim=0)
        key = torch.cat(keys, dim=0)
        order = torch.argsort(key)
        key = key[order]
        mass = mass[order]
        unique_key, inverse = torch.unique_consecutive(key, return_inverse=True)
        voxel_mass = mass.new_zeros(unique_key.shape[0]).scatter_add(0, inverse, mass)

        gain = float(getattr(args, "compression_surrogate_occ_gain", 1.0))
        occ = 1.0 - torch.exp(-gain * voxel_mass.clamp_min(0.0))
        occ = occ.clamp(1e-6, 1.0 - 1e-6)

        node = occ.sum()
        entropy = -(occ * torch.log2(occ) + (1.0 - occ) * torch.log2(1.0 - occ)).sum()
        mass_total = voxel_mass.sum()

        if level <= 1 or unique_key.numel() == 0:
            single = node.new_zeros(())
        else:
            coords = self._decode_keys(unique_key, grid)
            parent_grid = grid // 2
            parent = torch.div(coords, 2, rounding_mode="floor")
            child_bits = coords - parent * 2
            child_id = child_bits[:, 0] * 4 + child_bits[:, 1] * 2 + child_bits[:, 2]
            parent_key = parent[:, 0] + parent_grid * (parent[:, 1] + parent_grid * parent[:, 2])
            parent_unique, parent_inv = torch.unique(parent_key, sorted=True, return_inverse=True)
            child_occ = occ.new_zeros(parent_unique.numel() * 8)
            flat_child_idx = parent_inv * 8 + child_id
            child_occ = child_occ.scatter_add(0, flat_child_idx, occ).view(-1, 8).clamp(1e-6, 1.0 - 1e-6)
            not_occ = (1.0 - child_occ).clamp(1e-6, 1.0)
            prod_not = not_occ.prod(dim=1, keepdim=True)
            single_prob = (child_occ * prod_not / not_occ).sum(dim=1)
            single = single_prob.sum()

        node_safe = node.clamp_min(1e-6)
        grid_total = float(grid ** 3)
        return torch.stack(
            [
                torch.log1p(node),
                torch.log1p(single),
                entropy / node_safe,
                torch.log1p(mass_total),
                node / max(grid_total, 1.0),
            ]
        )

    def _build_soft_compression_features(self, args, gen_xyz, gt_xyz, final_w):
        B, _, N = gen_xyz.shape
        if final_w is None:
            weights_all = gen_xyz.new_ones(B, N)
        else:
            weights_all = final_w.squeeze(1).to(device=gen_xyz.device, dtype=gen_xyz.dtype)
            if weights_all.shape[-1] > N:
                weights_all = weights_all[..., :N]
            elif weights_all.shape[-1] < N:
                pad = weights_all.new_zeros(*weights_all.shape[:-1], N - weights_all.shape[-1])
                weights_all = torch.cat([weights_all, pad], dim=-1)
            weights_all = weights_all.clamp(0.0, 1.0)

        features = []
        ref_min = gt_xyz.detach().amin(dim=2)
        ref_max = gt_xyz.detach().amax(dim=2)
        ref_span = (ref_max - ref_min).clamp_min(1e-6)

        for b in range(B):
            pts = gen_xyz[b].to(torch.float32)
            weights = weights_all[b].to(torch.float32)
            coords_norm = ((pts - ref_min[b].to(pts.dtype).unsqueeze(1)) / ref_span[b].to(pts.dtype).unsqueeze(1)).clamp(0.0, 1.0)
            w_sum = weights.sum().clamp_min(1e-6)
            w_mean = weights.mean()
            w_std = weights.std(unbiased=False)
            mean_xyz = (coords_norm * weights.unsqueeze(0)).sum(dim=1) / w_sum
            centered = coords_norm - mean_xyz.unsqueeze(1)
            std_xyz = torch.sqrt((centered.pow(2) * weights.unsqueeze(0)).sum(dim=1) / w_sum + 1e-8)
            bbox = (coords_norm.amax(dim=1) - coords_norm.amin(dim=1)).clamp_min(1e-6)
            global_feat = [
                torch.log1p(gen_xyz.new_tensor(float(N), dtype=torch.float32)),
                torch.log1p(w_sum),
                w_mean,
                w_std,
                mean_xyz[0], mean_xyz[1], mean_xyz[2],
                std_xyz[0], std_xyz[1], std_xyz[2],
                torch.log1p(bbox.prod()),
            ]
            level_feat = [
                self._soft_level_stats(coords_norm, weights, level, args)
                for level in self.surrogate_levels
            ]
            features.append(torch.cat([torch.stack(global_feat), *level_feat], dim=0))

        return torch.stack(features, dim=0).to(device=gen_xyz.device, dtype=torch.float32)

    def _set_surrogate_trainable(self, trainable):
        for param in self.compression_surrogate.parameters():
            param.requires_grad_(trainable)

    def _train_compression_surrogate(self, args, x_soft, target):
        train_steps = max(int(getattr(args, "compression_surrogate_train_steps", 2)), 0)
        if train_steps <= 0:
            return x_soft.new_zeros(())

        self._ensure_surrogate_device(x_soft.device)
        self.compression_surrogate.train()
        self._set_surrogate_trainable(True)
        x_det = x_soft.detach()
        y_det = target.detach().expand(x_det.shape[0], -1)
        last_loss = x_soft.new_zeros(())
        weights = torch.tensor(
            [
                float(getattr(args, "compression_surrogate_bit_weight", 1.0)),
                float(getattr(args, "compression_surrogate_node_weight", 0.25)),
                float(getattr(args, "compression_surrogate_single_weight", 0.25)),
                float(getattr(
                    args,
                    "compression_surrogate_bpn_weight",
                    getattr(args, "compression_surrogate_entropy_weight", 0.25),
                )),
            ],
            device=x_soft.device,
            dtype=torch.float32,
        )

        with self._compression_autocast_ctx(x_soft.device):
            for _ in range(train_steps):
                self.surrogate_optimizer.zero_grad(set_to_none=True)
                pred = self.compression_surrogate(x_det)
                loss = (F.smooth_l1_loss(pred, y_det, reduction="none") * weights.unsqueeze(0)).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.compression_surrogate.parameters(),
                    float(getattr(args, "compression_surrogate_grad_clip", 10.0)),
                )
                self.surrogate_optimizer.step()
                last_loss = loss.detach()

        self.compression_surrogate.eval()
        self._set_surrogate_trainable(False)
        self._surrogate_step += train_steps
        return last_loss

    def _get_compression_loss_surrogate(self, args, gen_xyz, gt_xyz, final_w, cache_key=None):
        cached_gt = self._get_cached_actual_gt(cache_key)
        if cached_gt is None:
            cached_gt = self._encode_actual_batch(args, gt_xyz)
            self._store_cached_actual_gt(cache_key, cached_gt)
        stats_gen = self._encode_actual_batch(args, gen_xyz)

        x_soft = self._build_soft_compression_features(args, gen_xyz, gt_xyz, final_w)
        target = self._surrogate_target_from_actual(args, stats_gen, cached_gt, gen_xyz.device)
        L_sur = self._train_compression_surrogate(args, x_soft, target)

        self._ensure_surrogate_device(gen_xyz.device)
        self.compression_surrogate.eval()
        self._set_surrogate_trainable(False)
        pred = self.compression_surrogate(x_soft)
        pred_mean = pred.mean(dim=0)
        comp_weights = torch.tensor(
            [
                float(getattr(args, "compression_surrogate_comp_bit_weight", 1.0)),
                float(getattr(args, "compression_surrogate_comp_node_weight", 0.25)),
                float(getattr(args, "compression_surrogate_comp_single_weight", 0.25)),
                float(getattr(
                    args,
                    "compression_surrogate_comp_bpn_weight",
                    getattr(args, "compression_surrogate_comp_entropy_weight", 0.25),
                )),
            ],
            device=gen_xyz.device,
            dtype=pred_mean.dtype,
        )
        target_scale = max(float(getattr(args, "compression_surrogate_target_scale", 100.0)), 1e-12)
        L_com = (
            (pred_mean * comp_weights).sum()
            / target_scale
            * float(getattr(args, "compression_surrogate_loss_scale", 1.0))
        )

        actual_bit_rel = self._relative_ratio(float(stats_gen["bit"]), float(cached_gt["bit"]))
        actual_bpp_rel = self._relative_ratio(float(stats_gen["bpp"]), float(cached_gt["bpp"]))
        pred_rel = (pred_mean / target_scale).detach()
        target_rel = (target.squeeze(0) / target_scale).detach()
        # Values saved for plots/debug are percentages, not raw ratios.
        pred_percent = 100.0 * pred_rel
        target_percent = 100.0 * target_rel
        surrogate_abs_errors = (pred_rel - target_rel).abs()
        surrogate_abs_errors_percent = 100.0 * surrogate_abs_errors
        self.last_compression_debug = {
            "metric": "surrogate_soft_to_actual_octattention",
            "total_bit": 100.0 * actual_bit_rel,
            "bpp": 100.0 * actual_bpp_rel,
            "gt_points": int(cached_gt["point_count"]),
            "gen_points": int(stats_gen["point_count"]),
            "gt_actual_bit": float(cached_gt["bit"]),
            "gen_actual_bit": float(stats_gen["bit"]),
            "surrogate_pred_bit": self._scalar(pred_percent[0]),
            "surrogate_pred_node": self._scalar(pred_percent[1]),
            "surrogate_pred_single": self._scalar(pred_percent[2]),
            "surrogate_pred_bpn": self._scalar(pred_percent[3]),
            "surrogate_target_bit": self._scalar(target_percent[0]),
            "surrogate_target_node": self._scalar(target_percent[1]),
            "surrogate_target_single": self._scalar(target_percent[2]),
            "surrogate_target_bpn": self._scalar(target_percent[3]),
            "surrogate_abs_bit_error": self._scalar(surrogate_abs_errors_percent[0]),
            "surrogate_abs_mean_error": self._scalar(surrogate_abs_errors_percent.mean()),
            "surrogate_pred_bit_ratio": self._scalar(pred_rel[0]),
            "surrogate_target_bit_ratio": self._scalar(target_rel[0]),
            "surrogate_abs_bit_error_ratio": self._scalar(surrogate_abs_errors[0]),
            "surrogate_abs_mean_error_ratio": self._scalar(surrogate_abs_errors.mean()),
            "surrogate_train_loss": self._scalar(L_sur),
        }

        if self._should_verbose_step(args):
            self.writer.write(
                f"L_com(sur):{self._scalar(L_com):.6f}->"
                f"pred%(bit,node,single,bpn):"
                f"{self._scalar(pred_percent[0]):.4f},"
                f"{self._scalar(pred_percent[1]):.4f},"
                f"{self._scalar(pred_percent[2]):.4f},"
                f"{self._scalar(pred_percent[3]):.4f}, "
                f"L_sur:{self._scalar(L_sur):.6f}, "
                f"actual_bit:{float(cached_gt['bit']):.1f}->{float(stats_gen['bit']):.1f} "
                f"({100.0 * actual_bit_rel:.4f}%)"
            )

        self._log_compression_grad_probe(args, "octattention_surrogate", L_com, gen_xyz)
        return (
            L_com,
            pred_percent[0].detach(),
            pred_percent[2].detach(),
            pred_percent[1].detach(),
            cached_gt,
            {
                "bit": float(cached_gt["bit"]),
                "bpp": float(cached_gt["bpp"]),
                "bpn": float(cached_gt["bpn"]),
                "single": float(cached_gt["single"]),
                "node": float(cached_gt["node"]),
            },
        )
