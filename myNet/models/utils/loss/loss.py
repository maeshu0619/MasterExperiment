import os
import sys
from collections import OrderedDict
from contextlib import nullcontext

import torch

from .compression import CompressionLossMixin
from .geometry import GeometryLossMixin
from .proxy import ProxyCompressionLossMixin
from .surrogate import _CompressionSurrogateNet, SurrogateCompressionLossMixin


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from models.utils.compression.proxy_octree import ProxyOctreeConfig, SoftOctreeRateProxy


class Loss(
    CompressionLossMixin,
    SurrogateCompressionLossMixin,
    ProxyCompressionLossMixin,
    GeometryLossMixin,
):
    def __init__(self, args, file_date, writer):
        self.args = args
        self.com_bit = args.com_bit
        self.com_sin = args.com_sin
        self.com_node = args.com_node

        self.lambda_p = args.lambda_p

        self.compress = args.compress
        self.file_date = file_date
        self.writer = writer
        self.bptt = args.bptt
        self.ncl = None

        self.octree_cfgs = ProxyOctreeConfig(
            max_depth=args.proxy_max_depth,
            qs=args.qs,
            bptt=int(args.bptt),
            lambda_entropy=args.proxy_lambda_entropy,
            lambda_node_count=args.proxy_lambda_node_count,
            lambda_single_child=args.proxy_lambda_single_child,
            round_tau=float(getattr(args, "proxy_round_tau", 0.12)),
            mass_to_occ_gain=float(getattr(args, "proxy_mass_to_occ_gain", 1.0)),
            teacher_device=str(getattr(args, "octattention_teacher_device", "auto")),
        )
        self.rate_proxy = SoftOctreeRateProxy(self.octree_cfgs).to(device)
        self.actual_encoder = None
        self.surrogate_levels = self._parse_surrogate_levels(args)
        self.surrogate_feature_dim = 11 + 5 * len(self.surrogate_levels)
        self.compression_surrogate = _CompressionSurrogateNet(
            in_dim=self.surrogate_feature_dim,
            hidden_dim=int(getattr(args, "compression_surrogate_hidden_dim", 128)),
            pred_clip=float(getattr(args, "compression_surrogate_pred_clip", 2.0)),
        ).to(device)
        self.surrogate_optimizer = torch.optim.Adam(
            self.compression_surrogate.parameters(),
            lr=float(getattr(args, "compression_surrogate_lr", 1e-3)),
            weight_decay=float(getattr(args, "compression_surrogate_weight_decay", 1e-5)),
        )
        for param in self.compression_surrogate.parameters():
            param.requires_grad_(False)
        self.gt_cache_enabled = bool(getattr(args, "cache_gt_loss", True))
        self.gt_cache_max_entries = max(int(getattr(args, "cache_max_entries", 64)), 0)
        self.gt_cache = OrderedDict()
        self.actual_gt_cache = OrderedDict()
        self._compression_grad_probe_count = 0
        self._surrogate_step = 0

    @staticmethod
    def _scalar(x):
        if torch.is_tensor(x):
            return float(x.detach())
        return float(x)

    @staticmethod
    def _discrete_loss_mode(args):
        return str(getattr(args, "discrete_loss_mode", "hard")).strip().lower()

    @staticmethod
    def _should_verbose_step(args):
        return bool(
            getattr(args, "verbose_step_logs", False)
            and getattr(args, "_log_this_step", True)
        )

    def _surrogate_weight(self, args):
        return float(getattr(args, "discrete_surrogate_weight", 1.0))

    def _compose_discrete_loss(self, hard_loss, surrogate_loss, args):
        """Use the hard loss value while borrowing a surrogate backward pass."""
        weight = self._surrogate_weight(args)
        if surrogate_loss is None or weight == 0.0:
            return hard_loss
        return hard_loss + weight * (surrogate_loss - surrogate_loss.detach())

    def _get_cached_gt(self, cache_key, device):
        if not self.gt_cache_enabled or not cache_key:
            return None
        cache_entry = self.gt_cache.get(cache_key)
        if cache_entry is None:
            return None
        self.gt_cache.move_to_end(cache_key)
        out = dict(cache_entry)
        if out.get("gt_inlier") is not None:
            out["gt_inlier"] = out["gt_inlier"].to(device=device, non_blocking=True)
        return out

    def _store_cached_gt(self, cache_key, cache_entry):
        if not self.gt_cache_enabled or not cache_key or self.gt_cache_max_entries <= 0:
            return
        stored = dict(cache_entry)
        if stored.get("gt_inlier") is not None:
            stored["gt_inlier"] = stored["gt_inlier"].detach().to(device="cpu")
        self.gt_cache[cache_key] = stored
        self.gt_cache.move_to_end(cache_key)
        while len(self.gt_cache) > self.gt_cache_max_entries:
            self.gt_cache.popitem(last=False)

    def _ensure_rate_proxy_device(self, device):
        if next(self.rate_proxy.buffers()).device != device:
            self.rate_proxy = self.rate_proxy.to(device)

    def _compression_autocast_ctx(self, device):
        if device.type == "cuda":
            return torch.cuda.amp.autocast(enabled=False)
        return nullcontext()

    def warmup_gt_cache(self, gt_xyz, cache_key=None):
        if not self.gt_cache_enabled or not cache_key:
            return
        device = gt_xyz.device
        self._ensure_rate_proxy_device(device)
        cached_gt = self._get_cached_gt(cache_key, device)
        if cached_gt is not None:
            return
        with self._compression_autocast_ctx(device):
            out_gt, bit_gt, stats_gt = self.rate_proxy.forward_hard_only(
                gen_xyz=gt_xyz.to(torch.float32),
            )
        cache_entry = {
            "rate_gt": self._scalar(out_gt["rate_total"]),
            "single_gt": self._scalar(out_gt["soft_single_child_count"]),
            "nodes_gt": self._scalar(out_gt["soft_node_count"]),
            "bit_gt": self._scalar(bit_gt),
            "point_count_gt": int(gt_xyz.shape[-1]),
            "stats_gt": {k: self._scalar(v) for k, v in stats_gt.items()},
        }
        self._store_cached_gt(cache_key, cache_entry)

    def get_loss(self, args, gen_pts, gt_pts, final_w, out_label, cache_key=None):
        gt_xyz = gt_pts[:, :3, :]
        gen_xyz = gen_pts[:, :3, :]
        L_com, loss_bit, loss_single, loss_nodes, cached_gt, stats_gt = self.get_compression_loss(
            args,
            gen_xyz=gen_xyz,
            gt_xyz=gt_xyz,
            final_w=final_w,
            cache_key=cache_key,
        )
        L_geom = self.get_geometry_loss(
            args,
            gen_pts=gen_pts,
            gt_pts=gt_pts,
            final_w=final_w,
            out_label=out_label,
        )

        if self._should_verbose_step(args):
            comp_debug = getattr(self, "last_compression_debug", {})
            metric = comp_debug.get("metric", self._compression_rate_metric(args))
            self.writer.write(
                f"L_com   :{self._scalar(L_com):.4f}->"
                f"L_rate({metric}):{self._scalar(loss_bit):.4f}, "
                f"L_total_bits:{float(comp_debug.get('total_bit', self._scalar(loss_bit))):.4f}, "
                f"L_bpp:{float(comp_debug.get('bpp', self._scalar(loss_bit))):.4f}, "
                f"L_single:{self._scalar(loss_single):.4f}, "
                f"L_nodes:{self._scalar(loss_nodes):.4f}, "
                f"points:{comp_debug.get('gt_points', gt_xyz.shape[-1])}->{comp_debug.get('gen_points', gen_xyz.shape[-1])}"
            )

        return L_geom, L_com, loss_bit, loss_single, loss_nodes
