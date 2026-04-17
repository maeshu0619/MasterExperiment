import os
import torch
import time
import h5py
import sys
import numpy as np
import tempfile
import shutil
from contextlib import nullcontext

import torch.nn as nn
import torch.nn.functional as F

from myNet.models.utils.pre.utils_repkpu import *
from myNet.models.utils.loss.proxy_oa import proxy_octattention_like_octree_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
sys.path.append(ROOT_DIR)

# from compress.OctAttention.compression import OA_compress
# from compress.OctAttention.encoderTool import *
# from compress.OctAttention.networkTool import *
# from compress.OctAttention.octAttention import *
# from compress.ProxyOctree.proxy_octree import OctreeProxyCompressor
from myNet.models.utils.loss.proxy_octree import *

from myNet.models.utils.pre.utils_loss import *
from myNet.models.utils.pre.utils_p2c import *

import multiprocessing as mp
import numpy as np
import os
import traceback
from collections import OrderedDict


class _QuietWriter:
    def write(self, *_args, **_kwargs):
        return None


class _OctAttentionActualEncoder:
    """
    Run the same OctAttention path as compress/octree/OctAttention/encoder.py.

    This path intentionally leaves autograd: generated points are written as a
    temporary PLY, converted to the OctAttention mat/octree format, and encoded
    by the pretrained entropy model. The returned bit count is therefore a real
    measurement, not a differentiable quantity.
    """
    def __init__(self, args, writer=None):
        self.args = args
        self.writer = writer
        self.qs = float(getattr(args, "qs", 2.0))
        self.actualcode = bool(getattr(args, "octattention_actualcode", False))
        self.tmp_root = getattr(args, "octattention_tmp_dir", "")
        self._loaded = False
        self._model = None
        self._data_prepare = None
        self._write_ply_data = None
        self._oa_bptt = int(getattr(args, "bptt", 1024))

    def _lazy_init(self):
        if self._loaded:
            return

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        oa_dir = os.path.join(repo_root, "compress", "octree", "OctAttention")
        if oa_dir not in sys.path:
            sys.path.append(oa_dir)

        from Preparedata.data import dataPrepare
        from networkTool import levelNumK
        from pt import write_ply_data
        from myNet.models.utils.loss.proxy_octree import _OctAttentionTeacherModel

        ckpt_path = getattr(self.args, "octattention_ckpt", "")
        if not ckpt_path:
            ckpt_path = os.path.join(oa_dir, "modelsave", "obj", "encoder_epoch_00800093.pth")
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.abspath(os.path.join(repo_root, ckpt_path))

        save_dic = torch.load(ckpt_path, map_location="cpu")
        state_dict = save_dic["encoder"] if "encoder" in save_dic else save_dic

        oa_model = _OctAttentionTeacherModel(max_octree_level=12)
        oa_model.load_state_dict(state_dict)
        oa_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        oa_model.to(oa_device)
        oa_model.eval()
        for param in oa_model.parameters():
            param.requires_grad_(False)

        self._model = oa_model
        self._data_prepare = dataPrepare
        self._level_num_k = levelNumK
        self._write_ply_data = write_ply_data
        self._oa_device = oa_device
        self._loaded = True

        if self.writer is not None and hasattr(self.writer, "write"):
            self.writer.write(f"OctAttention actual encoder loaded: {ckpt_path}")

    def _make_tmp_dir(self):
        root = self.tmp_root
        if not root:
            root = "/dev/shm/mynet_octattention_actual" if os.path.isdir("/dev/shm") else None
        if root:
            os.makedirs(root, exist_ok=True)
            return tempfile.mkdtemp(prefix="oa_actual_", dir=root)
        return tempfile.mkdtemp(prefix="oa_actual_")

    def encode_bits(self, pts_3n):
        self._lazy_init()
        pts_np = (
            pts_3n.detach()
            .transpose(0, 1)
            .contiguous()
            .to("cpu")
            .numpy()
            .astype(np.float32, copy=False)
        )

        tmp_dir = self._make_tmp_dir()
        try:
            ply_path = os.path.join(tmp_dir, "input.ply")
            mat_dir = os.path.join(tmp_dir, "mat")
            bin_path = os.path.join(tmp_dir, "encoded.bin")
            self._write_ply_data(ply_path, pts_np)
            mat_file, _dq_pt, _ref_pt = self._data_prepare(
                ply_path,
                saveMatDir=mat_dir,
                qs=self.qs,
                ptNamePrefix="",
                rotation=False,
            )

            import h5py as _h5py

            mat = None
            try:
                mat = _h5py.File(mat_file, "r")
                cell = mat["patchFile"]
                ref = cell[0, 0]
                data_arr = np.array(mat[ref])
                oct_data_seq = np.transpose(data_arr).astype(np.int32)[:, -self._level_num_k:, 0:6]
            finally:
                if mat is not None:
                    mat.close()

            binsz, oct_len = self._compress_oct_seq(oct_data_seq, bin_path)
            single_count = self._single_child_count(oct_data_seq)
            return {
                "bit": float(binsz),
                "bpp": float(binsz) / max(float(pts_3n.shape[-1]), 1.0),
                "bpn": float(binsz) / max(float(oct_len), 1.0),
                "single": float(single_count),
                "node": float(oct_len),
                "point_count": int(pts_3n.shape[-1]),
            }
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _single_child_count(oct_data_seq):
        oct_code = oct_data_seq[:, -1, 0].astype(np.int32)
        pop = (
            (oct_code & 1) + ((oct_code >> 1) & 1) +
            ((oct_code >> 2) & 1) + ((oct_code >> 3) & 1) +
            ((oct_code >> 4) & 1) + ((oct_code >> 5) & 1) +
            ((oct_code >> 6) & 1) + ((oct_code >> 7) & 1)
        )
        return int((pop == 1).sum())

    def _generate_square_subsequent_mask(self, size):
        return torch.triu(
            torch.full((size, size), float("-inf"), device=self._oa_device),
            diagonal=1,
        )

    @staticmethod
    def _batchify(oct_seq, bptt, oct_len):
        oct_seq = oct_seq.copy()
        oct_seq[:-1, 0:-1, :] = oct_seq[1:, 0:-1, :]
        oct_seq[:-1, -1, 1:3] = oct_seq[1:, -1, 1:3]
        oct_seq[:, :, 0] = oct_seq[:, :, 0] - 1
        pad_len = bptt
        padded = np.zeros((bptt + oct_len + pad_len, *oct_seq.shape[1:]), dtype=oct_seq.dtype)
        padded[bptt:bptt + oct_len] = oct_seq
        oct_seq_t = torch.from_numpy(padded).long()

        data_id = torch.full((bptt + oct_len + pad_len,), -1, dtype=torch.long)
        data_id[bptt:bptt + oct_len] = torch.arange(oct_len, dtype=torch.long)
        return data_id.unsqueeze(1), oct_seq_t.unsqueeze(1)

    @staticmethod
    def _estimate_bits(pro_bit, oct_seq, level_id):
        oct_values = oct_seq.astype(np.int64).reshape(-1) - 1
        level_values = np.asarray(level_id, dtype=np.int64).reshape(-1)

        prob_hit = np.take_along_axis(pro_bit[:oct_values.shape[0]], oct_values[:, None], axis=1).squeeze(1)
        bit_each = -np.log2(prob_hit + 1e-7)
        bit = float(bit_each.sum())

        level_change = np.empty(level_values.shape[0], dtype=bool)
        level_change[0] = level_values[0] != 1
        level_change[1:] = level_values[1:] != level_values[:-1]
        level_change_idx = np.flatnonzero(level_change)

        binsz_list = np.concatenate([np.cumsum(bit_each)[level_change_idx], np.array([bit])])
        oct_num_list = np.concatenate([level_change_idx + 1, np.array([oct_values.shape[0]])])
        return bit, binsz_list, oct_num_list

    def _compress_oct_seq(self, oct_data_seq, output_file):
        from networkTool import MAX_OCTREE_LEVEL

        level_id = oct_data_seq[:, -1, 1].copy()
        oct_data_seq = oct_data_seq.copy()
        if level_id.max() > MAX_OCTREE_LEVEL:
            level_id = np.minimum(level_id, MAX_OCTREE_LEVEL)

        oct_seq = oct_data_seq[:, -1:, 0].astype(int)
        oct_len = len(oct_seq)
        bptt_eff = min(self._oa_bptt, oct_len - 1)
        if bptt_eff < 32:
            raise ValueError(f"oct_len too small for OctAttention: oct_len={oct_len}")

        data_id, padded_data = self._batchify(oct_data_seq, bptt_eff, oct_len)
        pading_length = padded_data.shape[0]
        src_mask = self._generate_square_subsequent_mask(bptt_eff)
        pro_bit_chunks = [] if self.actualcode else None
        oct_values = torch.from_numpy(oct_seq.astype(np.int64).reshape(-1) - 1)
        processed = 0
        total_bits = 0.0

        with torch.inference_mode():
            for i in range(0, pading_length - bptt_eff, bptt_eff):
                inp = padded_data[i:i + bptt_eff].to(device=self._oa_device, non_blocking=True)
                node_id = data_id[i + 1:i + bptt_eff + 1].reshape(-1)
                valid_mask = node_id >= 0
                if not valid_mask.any():
                    continue
                output = self._model(inp, src_mask, [])
                output = output.reshape(-1, 255)
                prob = torch.softmax(output, dim=1)
                valid_prob = prob[valid_mask]
                valid_count = min(int(valid_prob.shape[0]), oct_len - processed)
                if valid_count <= 0:
                    break
                valid_prob = valid_prob[:valid_count]
                target = oct_values[processed:processed + valid_count].to(device=self._oa_device, non_blocking=True)
                prob_hit = valid_prob.gather(1, target.view(-1, 1)).squeeze(1)
                total_bits += float((-torch.log2(prob_hit + 1e-7)).sum().detach().cpu())
                if pro_bit_chunks is not None:
                    pro_bit_chunks.append(valid_prob.detach().cpu().numpy())
                processed += valid_count

        if processed <= 0:
            raise ValueError(f"OctAttention produced no valid probability rows: oct_len={oct_len}")
        if processed < oct_len and self.writer is not None and hasattr(self.writer, "write"):
            self.writer.write(
                f"OctAttention warning: probability rows shorter than octree length "
                f"({processed}/{oct_len}); using available rows for bit estimate."
            )
        binsz = total_bits

        if self.actualcode:
            import numpyAc
            if not pro_bit_chunks:
                raise ValueError(f"OctAttention produced no probability chunks for arithmetic coding: oct_len={oct_len}")
            pro_bit = np.vstack(pro_bit_chunks)[:oct_len]
            binsz, _binsz_list, _oct_num_list = self._estimate_bits(pro_bit, oct_seq, level_id)
            codec = numpyAc.arithmeticCoding()
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            _, binsz = codec.encode(pro_bit[:oct_len, :], oct_seq.astype(np.int16).squeeze(-1) - 1, output_file)
            del pro_bit
        del data_id, padded_data, src_mask
        if self._oa_device.type == "cuda":
            torch.cuda.empty_cache()
        return float(binsz), int(oct_len)


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

class Loss:
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

    @staticmethod
    def _compression_rate_metric(args):
        return str(getattr(args, "compression_rate_metric", "bits_per_point")).strip().lower()

    @staticmethod
    def _compression_loss_backend(args):
        return str(getattr(args, "compression_loss_backend", "proxy")).strip().lower()

    @staticmethod
    def _positive_count(count):
        return max(float(count), 1.0)

    def _metric_value(self, value, own_point_count, ref_point_count, args):
        mode = self._compression_rate_metric(args)
        if mode == "total_bits":
            return value
        if mode == "bits_per_input_point":
            return value / self._positive_count(ref_point_count)
        return value / self._positive_count(own_point_count)

    @staticmethod
    def _relative_percent(value, ref, ref_min=1e-12):
        if torch.is_tensor(value):
            ref_t = value.new_tensor(float(ref))
            denom = ref_t.abs().clamp_min(float(ref_min))
            return 100.0 * (value - ref_t) / denom
        denom = max(abs(float(ref)), float(ref_min))
        return 100.0 * (float(value) - float(ref)) / denom

    def _compression_terms_from_proxy(
        self,
        out,
        bit_ref,
        nodes_ref,
        single_ref,
        args,
        gen_point_count,
        gt_point_count,
    ):
        bit_ref_metric = self._metric_value(
            float(bit_ref),
            own_point_count=gt_point_count,
            ref_point_count=gt_point_count,
            args=args,
        )
        nodes_ref_metric = self._metric_value(
            float(nodes_ref),
            own_point_count=gt_point_count,
            ref_point_count=gt_point_count,
            args=args,
        )
        single_ref_metric = self._metric_value(
            float(single_ref),
            own_point_count=gt_point_count,
            ref_point_count=gt_point_count,
            args=args,
        )

        bit = out["rate_entropy"]
        nodes = out["soft_node_count"]
        single = out["soft_single_child_count"]

        bit_metric = self._metric_value(
            bit,
            own_point_count=gen_point_count,
            ref_point_count=gt_point_count,
            args=args,
        )
        nodes_metric = self._metric_value(
            nodes,
            own_point_count=gen_point_count,
            ref_point_count=gt_point_count,
            args=args,
        )
        single_metric = self._metric_value(
            single,
            own_point_count=gen_point_count,
            ref_point_count=gt_point_count,
            args=args,
        )

        struct_ref_min = 1.0 if self._compression_rate_metric(args) == "total_bits" else 1e-12
        L_bit = self._relative_percent(bit_metric, bit_ref_metric)
        L_nodes = self._relative_percent(nodes_metric, nodes_ref_metric, ref_min=struct_ref_min)
        L_single = self._relative_percent(single_metric, single_ref_metric, ref_min=struct_ref_min)
        L_com = (
            float(getattr(args, "proxy_lambda_entropy", 1.0)) * L_bit
            + float(getattr(args, "proxy_lambda_node_count", 1.0)) * L_nodes
            + float(getattr(args, "proxy_lambda_single_child", 1.0)) * L_single
        )
        return L_com, L_bit, L_nodes, L_single
        
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

    def _get_cached_actual_gt(self, cache_key):
        if not self.gt_cache_enabled or not cache_key:
            return None
        cache_entry = self.actual_gt_cache.get(cache_key)
        if cache_entry is None:
            return None
        self.actual_gt_cache.move_to_end(cache_key)
        return dict(cache_entry)

    def _store_cached_actual_gt(self, cache_key, cache_entry):
        if not self.gt_cache_enabled or not cache_key or self.gt_cache_max_entries <= 0:
            return
        self.actual_gt_cache[cache_key] = dict(cache_entry)
        self.actual_gt_cache.move_to_end(cache_key)
        while len(self.actual_gt_cache) > self.gt_cache_max_entries:
            self.actual_gt_cache.popitem(last=False)

    def _ensure_rate_proxy_device(self, device):
        if next(self.rate_proxy.buffers()).device != device:
            self.rate_proxy = self.rate_proxy.to(device)

    def _compression_autocast_ctx(self, device):
        if device.type == "cuda":
            return torch.cuda.amp.autocast(enabled=False)
        return nullcontext()

    def _ensure_surrogate_device(self, device):
        first_param = next(self.compression_surrogate.parameters())
        if first_param.device != device:
            self.compression_surrogate = self.compression_surrogate.to(device)
            state = self.surrogate_optimizer.state
            for value in state.values():
                for key, item in value.items():
                    if torch.is_tensor(item):
                        value[key] = item.to(device)

    def _get_actual_encoder(self, args):
        if self.actual_encoder is None:
            self.actual_encoder = _OctAttentionActualEncoder(args, writer=self.writer)
        return self.actual_encoder

    def _log_compression_grad_probe(self, args, label, L_com, gen_xyz):
        if not bool(getattr(args, "compression_grad_probe", True)):
            return
        every = max(int(getattr(args, "compression_grad_probe_every", 1)), 1)
        self._compression_grad_probe_count += 1
        if self._compression_grad_probe_count % every != 0:
            return

        requires_grad = bool(torch.is_tensor(L_com) and L_com.requires_grad)
        grad_fn = type(L_com.grad_fn).__name__ if requires_grad and L_com.grad_fn is not None else "None"
        grad_norm = None
        grad_ok = False
        err = None

        if requires_grad:
            try:
                grad = torch.autograd.grad(
                    L_com,
                    gen_xyz,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if grad is not None:
                    grad_norm = float(grad.detach().norm().cpu())
                    grad_ok = grad_norm > 0.0 and np.isfinite(grad_norm)
            except Exception as exc:
                err = str(exc)

        msg = (
            f"[GradCheck][L_com:{label}] "
            f"requires_grad={requires_grad}, grad_fn={grad_fn}, "
            f"grad_to_gen_xyz={'OK' if grad_ok else 'NG'}, "
            f"grad_norm={grad_norm if grad_norm is not None else 'None'}"
        )
        if err is not None:
            msg += f", err={err}"
        if self.writer is not None and hasattr(self.writer, "write"):
            self.writer.write(msg)

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

    def _encode_actual_batch(self, args, xyz):
        encoder = self._get_actual_encoder(args)
        stats_list = []
        for b in range(xyz.shape[0]):
            stats_list.append(encoder.encode_bits(xyz[b].to(torch.float32)))
        total_bit = sum(s["bit"] for s in stats_list)
        total_single = sum(s["single"] for s in stats_list)
        total_node = sum(s["node"] for s in stats_list)
        total_points = sum(s["point_count"] for s in stats_list)
        return {
            "bit": float(total_bit),
            "bpp": float(total_bit) / max(float(total_points), 1.0),
            "bpn": float(total_bit) / max(float(total_node), 1.0),
            "single": float(total_single),
            "node": float(total_node),
            "point_count": int(total_points),
            "per_batch": stats_list,
        }

    def _surrogate_target_from_actual(self, args, stats_gen, stats_ref, device):
        scale = float(getattr(args, "compression_surrogate_target_scale", 100.0))
        gt_bit = max(abs(float(stats_ref["bit"])), 1e-12)
        gt_node = max(abs(float(stats_ref["node"])), 1.0)
        gt_single = max(abs(float(stats_ref["single"])), 1.0)
        gt_bpn = max(abs(float(stats_ref["bpn"])), 1e-12)
        values = [
            (float(stats_gen["bit"]) - float(stats_ref["bit"])) / gt_bit,
            (float(stats_gen["node"]) - float(stats_ref["node"])) / gt_node,
            (float(stats_gen["single"]) - float(stats_ref["single"])) / gt_single,
            (float(stats_gen["bpn"]) - float(stats_ref["bpn"])) / gt_bpn,
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
                float(getattr(args, "compression_surrogate_entropy_weight", 0.25)),
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
                float(getattr(args, "compression_surrogate_comp_entropy_weight", 0.25)),
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

        actual_bit_rel = (float(stats_gen["bit"]) - float(cached_gt["bit"])) / max(abs(float(cached_gt["bit"])), 1e-12)
        actual_bpp_rel = (float(stats_gen["bpp"]) - float(cached_gt["bpp"])) / max(abs(float(cached_gt["bpp"])), 1e-12)
        pred_rel = (pred_mean / target_scale).detach()
        target_rel = (target.squeeze(0) / target_scale).detach()
        surrogate_abs_errors = (pred_rel - target_rel).abs()
        self.last_compression_debug = {
            "metric": "surrogate_soft_to_actual_octattention",
            "total_bit": 100.0 * actual_bit_rel,
            "bpp": 100.0 * actual_bpp_rel,
            "gt_points": int(cached_gt["point_count"]),
            "gen_points": int(stats_gen["point_count"]),
            "gt_actual_bit": float(cached_gt["bit"]),
            "gen_actual_bit": float(stats_gen["bit"]),
            "surrogate_pred_bit": self._scalar(pred_rel[0]),
            "surrogate_pred_node": self._scalar(pred_rel[1]),
            "surrogate_pred_single": self._scalar(pred_rel[2]),
            "surrogate_pred_entropy": self._scalar(pred_rel[3]),
            "surrogate_target_bit": self._scalar(target_rel[0]),
            "surrogate_abs_bit_error": self._scalar(surrogate_abs_errors[0]),
            "surrogate_abs_mean_error": self._scalar(surrogate_abs_errors.mean()),
            "surrogate_train_loss": self._scalar(L_sur),
        }

        if self._should_verbose_step(args):
            self.writer.write(
                f"L_com(sur):{self._scalar(L_com):.6f}->"
                f"pred%(bit,node,single,H):"
                f"{self._scalar(pred_mean[0]):.4f},"
                f"{self._scalar(pred_mean[1]):.4f},"
                f"{self._scalar(pred_mean[2]):.4f},"
                f"{self._scalar(pred_mean[3]):.4f}, "
                f"L_sur:{self._scalar(L_sur):.6f}, "
                f"actual_bit:{float(cached_gt['bit']):.1f}->{float(stats_gen['bit']):.1f} "
                f"({100.0 * actual_bit_rel:.4f}%)"
            )

        self._log_compression_grad_probe(args, "octattention_surrogate", L_com, gen_xyz)
        return (
            L_com,
            (pred_mean[0] / target_scale).detach(),
            (pred_mean[2] / target_scale).detach(),
            (pred_mean[1] / target_scale).detach(),
            cached_gt,
            {
                "bit": float(cached_gt["bit"]),
                "bpp": float(cached_gt["bpp"]),
                "bpn": float(cached_gt["bpn"]),
                "single": float(cached_gt["single"]),
                "node": float(cached_gt["node"]),
            },
        )

    def _get_compression_loss_actual_octattention(self, args, gen_xyz, gt_xyz, final_w, cache_key=None, use_proxy_surrogate=False):
        cached_gt = self._get_cached_actual_gt(cache_key)
        if cached_gt is None:
            cached_gt = self._encode_actual_batch(args, gt_xyz)
            self._store_cached_actual_gt(cache_key, cached_gt)

        stats_gen = self._encode_actual_batch(args, gen_xyz)
        gt_bit = float(cached_gt["bit"])
        gen_bit = float(stats_gen["bit"])
        denom = max(abs(gt_bit), 1e-12)
        loss_bit_value = (gen_bit - gt_bit) / denom

        L_com_hard = gen_xyz.new_tensor(loss_bit_value)
        L_com = L_com_hard

        proxy_debug = None
        if use_proxy_surrogate:
            proxy_L_com, proxy_loss_bit, proxy_loss_single, proxy_loss_nodes, _, _ = self._get_compression_loss_proxy(
                args,
                gen_xyz=gen_xyz,
                gt_xyz=gt_xyz,
                final_w=final_w,
                cache_key=cache_key,
                run_grad_probe=False,
            )
            L_com = L_com_hard + (proxy_L_com - proxy_L_com.detach())
            proxy_debug = {
                "L_com": self._scalar(proxy_L_com),
                "loss_bit": self._scalar(proxy_loss_bit),
                "loss_single": self._scalar(proxy_loss_single),
                "loss_nodes": self._scalar(proxy_loss_nodes),
            }

        loss_bit = gen_xyz.new_tensor(loss_bit_value)
        loss_single = gen_xyz.new_tensor(
            (float(stats_gen["single"]) - float(cached_gt["single"])) / max(abs(float(cached_gt["single"])), 1.0)
        )
        loss_nodes = gen_xyz.new_tensor(
            (float(stats_gen["node"]) - float(cached_gt["node"])) / max(abs(float(cached_gt["node"])), 1.0)
        )
        self.last_compression_debug = {
            "metric": "actual_octattention_total_bits",
            "total_bit": 100.0 * loss_bit_value,
            "bpp": 100.0 * ((float(stats_gen["bpp"]) - float(cached_gt["bpp"])) / max(abs(float(cached_gt["bpp"])), 1e-12)),
            "gt_points": int(cached_gt["point_count"]),
            "gen_points": int(stats_gen["point_count"]),
            "gt_actual_bit": gt_bit,
            "gen_actual_bit": gen_bit,
            "proxy_surrogate": proxy_debug,
        }

        if self._should_verbose_step(args):
            surrogate_msg = ""
            if proxy_debug is not None:
                surrogate_msg = (
                    f", proxy_surrogate_L:{proxy_debug['L_com']:.4f}, "
                    f"proxy_surrogate_bit:{proxy_debug['loss_bit']:.4f}"
                )
            self.writer.write(
                f"L_com(actual OA):{self._scalar(L_com):.6f}->"
                f"bit:{gt_bit:.1f}->{gen_bit:.1f}, "
                f"rel:{100.0 * loss_bit_value:.4f}%"
                f"{surrogate_msg}"
            )

        label = "octattention_actual_ste" if use_proxy_surrogate else "octattention_actual"
        self._log_compression_grad_probe(args, label, L_com, gen_xyz)

        stats_gt = {
            "bit": gt_bit,
            "bpp": float(cached_gt["bpp"]),
            "bpn": float(cached_gt["bpn"]),
            "single": float(cached_gt["single"]),
            "node": float(cached_gt["node"]),
        }
        return L_com, loss_bit, loss_single, loss_nodes, cached_gt, stats_gt

    def get_compression_loss(self, args, gen_xyz, gt_xyz, final_w, cache_key=None):
        backend = self._compression_loss_backend(args)
        if backend in {"octattention_surrogate", "surrogate", "soft_surrogate"}:
            return self._get_compression_loss_surrogate(
                args,
                gen_xyz=gen_xyz,
                gt_xyz=gt_xyz,
                final_w=final_w,
                cache_key=cache_key,
            )
        if backend in {"octattention_actual", "actual_octattention", "real_octattention"}:
            return self._get_compression_loss_actual_octattention(
                args,
                gen_xyz=gen_xyz,
                gt_xyz=gt_xyz,
                final_w=final_w,
                cache_key=cache_key,
                use_proxy_surrogate=False,
            )
        if backend in {"octattention_actual_ste", "actual_octattention_ste", "real_octattention_ste"}:
            return self._get_compression_loss_actual_octattention(
                args,
                gen_xyz=gen_xyz,
                gt_xyz=gt_xyz,
                final_w=final_w,
                cache_key=cache_key,
                use_proxy_surrogate=True,
            )
        if backend != "proxy":
            raise ValueError(
                "--compression_loss_backend must be one of: proxy, "
                "octattention_actual, octattention_actual_ste "
                f"(got {backend})"
            )
        return self._get_compression_loss_proxy(
            args,
            gen_xyz=gen_xyz,
            gt_xyz=gt_xyz,
            final_w=final_w,
            cache_key=cache_key,
            run_grad_probe=True,
        )

    def _get_compression_loss_proxy(self, args, gen_xyz, gt_xyz, final_w, cache_key=None, run_grad_probe=True):
        self._ensure_rate_proxy_device(gen_xyz.device)
        cached_gt = self._get_cached_gt(cache_key, gen_xyz.device)
        if cached_gt is None:
            self.warmup_gt_cache(gt_xyz, cache_key=cache_key)
            cached_gt = self._get_cached_gt(cache_key, gen_xyz.device)
        if cached_gt is None:
            with self._compression_autocast_ctx(gen_xyz.device):
                out_gt, bit_gt, stats_gt = self.rate_proxy.forward_hard_only(
                    gen_xyz=gt_xyz.to(torch.float32),
                )
            cached_gt = {
                "rate_gt": self._scalar(out_gt["rate_total"]),
                "single_gt": self._scalar(out_gt["soft_single_child_count"]),
                "nodes_gt": self._scalar(out_gt["soft_node_count"]),
                "bit_gt": self._scalar(bit_gt),
                "point_count_gt": int(gt_xyz.shape[-1]),
                "stats_gt": {k: self._scalar(v) for k, v in stats_gt.items()},
            }
        bit_gt = cached_gt["bit_gt"]
        stats_gt = cached_gt["stats_gt"]
        nodes_gt = cached_gt["nodes_gt"]
        single_gt = cached_gt["single_gt"]
        gt_point_count = int(cached_gt.get("point_count_gt", gt_xyz.shape[-1]))
        gen_point_count = int(gen_xyz.shape[-1])
        mode = self._discrete_loss_mode(args)
        if mode == "hard":
            final_w = None
        use_weighted_forward = mode in {"weighted_soft", "soft", "legacy"} and final_w is not None
        use_ste_hard = mode in {"ste_hard", "hard_ste"} and final_w is not None

        if use_ste_hard:
            with self._compression_autocast_ctx(gen_xyz.device):
                out_gen, out_surrogate, stats_gen = self.rate_proxy.forward_ste_hard_pair(
                    gen_xyz=gen_xyz.to(torch.float32),
                    final_w=final_w.to(torch.float32),
                )
        else:
            with self._compression_autocast_ctx(gen_xyz.device):
                out_gen, _, stats_gen = self.rate_proxy(
                    gen_xyz=gen_xyz.to(torch.float32),
                    final_w=final_w.to(torch.float32) if use_weighted_forward else None,
                )

        L_com_hard_or_weighted, _, _, _ = self._compression_terms_from_proxy(
            out_gen,
            bit_ref=bit_gt,
            nodes_ref=nodes_gt,
            single_ref=single_gt,
            args=args,
            gen_point_count=gen_point_count,
            gt_point_count=gt_point_count,
        )

        L_com = L_com_hard_or_weighted
        if use_ste_hard:
            L_com_surrogate, _, _, _ = self._compression_terms_from_proxy(
                out_surrogate,
                bit_ref=bit_gt,
                nodes_ref=nodes_gt,
                single_ref=single_gt,
                args=args,
                gen_point_count=gen_point_count,
                gt_point_count=gt_point_count,
            )
            L_com = self._compose_discrete_loss(L_com_hard_or_weighted, L_com_surrogate, args)

        if args.trainORtest == "test":
            self.writer.write(f"=== Compression Stats ===")
            self.writer.write(f"bit                         : {stats_gt['bit']} -> {stats_gen['bit']}")
            self.writer.write(f"bpp                         : {stats_gt['bpp']} -> {stats_gen['bpp']}")
            self.writer.write(f"bpn                         : {stats_gt['bpn']} -> {stats_gen['bpn']}")
            self.writer.write(f"single child node           : {stats_gt['single']} -> {stats_gen['single']}")
            self.writer.write(f"num of nodes                : {stats_gt['node']} -> {stats_gen['node']}")
            self.writer.write(f"num of points               : {gt_xyz.shape[2]} -> {gen_xyz.shape[2]}")

        rate_gt = out_gen["rate_entropy"].new_tensor(cached_gt["rate_gt"])
        single_gt_t = out_gen["soft_single_child_count"].new_tensor(single_gt)
        nodes_gt_t = out_gen["soft_node_count"].new_tensor(nodes_gt)

        rate_gen = out_gen["rate_entropy"].detach()
        single_gen = out_gen["soft_single_child_count"].detach()
        nodes_gen = out_gen["soft_node_count"].detach()

        loss_bit = self._relative_percent(
            self._metric_value(rate_gen, gen_point_count, gt_point_count, args),
            self._metric_value(float(rate_gt), gt_point_count, gt_point_count, args),
        )
        struct_ref_min = 1.0 if self._compression_rate_metric(args) == "total_bits" else 1e-12
        loss_single = self._relative_percent(
            self._metric_value(single_gen, gen_point_count, gt_point_count, args),
            self._metric_value(float(single_gt_t), gt_point_count, gt_point_count, args),
            ref_min=struct_ref_min,
        )
        loss_nodes = self._relative_percent(
            self._metric_value(nodes_gen, gen_point_count, gt_point_count, args),
            self._metric_value(float(nodes_gt_t), gt_point_count, gt_point_count, args),
            ref_min=struct_ref_min,
        )
        loss_total_bit = self._relative_percent(rate_gen, float(rate_gt))
        loss_bpp = self._relative_percent(
            rate_gen / self._positive_count(gen_point_count),
            float(rate_gt) / self._positive_count(gt_point_count),
        )
        self.last_compression_debug = {
            "metric": self._compression_rate_metric(args),
            "total_bit": self._scalar(loss_total_bit),
            "bpp": self._scalar(loss_bpp),
            "gt_points": gt_point_count,
            "gen_points": gen_point_count,
        }

        if run_grad_probe:
            self._log_compression_grad_probe(args, "proxy", L_com, gen_xyz)

        return L_com, loss_bit, loss_single, loss_nodes, cached_gt, stats_gt

    def get_geometry_loss(self, args, gen_pts, gt_pts, final_w=None, out_label=None):
        use_torch_d2 = args.trainORtest == "train"
        if gen_pts.shape[-1] == 0 or gt_pts.shape[-1] == 0:
            return gt_pts.new_zeros(())

        # ===== Geometry Loss =====
        L_geom = 0.0
        mode = self._discrete_loss_mode(args)
        if mode == "hard":
            final_w = None
        use_weighted_forward = mode in {"weighted_soft", "soft", "legacy"} and final_w is not None
        use_ste_hard = mode in {"ste_hard", "hard_ste"} and final_w is not None
        forward_w = final_w if use_weighted_forward else None
        if args.loss_type == "cd":
            if out_label is None:
                gt_inlinear = gt_pts
            else:
                gt_inlinear = remove_outlier_points_by_label(gt_pts, out_label)
            if gt_inlinear.shape[-1] == 0:
                return gt_pts.new_zeros(())
            if use_ste_hard:
                L_cd, L_cd_surrogate = chamfer_l2_loss_and_weight_surrogate(
                    gen_pts,
                    gt_inlinear,
                    final_w,
                )
                L_cd = self._compose_discrete_loss(L_cd, L_cd_surrogate, args)
            else:
                L_cd = chamfer_l2_loss(gen_pts, gt_inlinear, forward_w)
            L_geom = L_cd
            if self._should_verbose_step(args):
                self.writer.write(f"L_geom  :{self._scalar(L_geom):.4f}")
        elif args.loss_type == "cd+d2":
            if use_ste_hard:
                L_cd_hard, L_cd_surrogate = chamfer_l2_loss_and_weight_surrogate(
                    gen_pts,
                    gt_pts,
                    final_w,
                )
                L_cd = self._compose_discrete_loss(L_cd_hard, L_cd_surrogate, args)
            elif use_weighted_forward:
                L_cd_hard = chamfer_l2_loss(gen_pts, gt_pts)
                L_cd_soft = chamfer_l2_loss(gen_pts, gt_pts, final_w)
                L_cd = self.lambda_p * L_cd_hard + L_cd_soft
            else:
                L_cd_hard = chamfer_l2_loss(gen_pts, gt_pts)
                L_cd = L_cd_hard

            L_d2_hard = compute_d2_psnr(gen_pts, gt_pts, use_torch_ops=use_torch_d2)
            if use_weighted_forward:
                L_d2_soft = compute_d2_psnr(gen_pts, gt_pts, final_w=final_w, use_torch_ops=use_torch_d2)
                L_d2 = self.lambda_p * L_d2_hard + L_d2_soft
            else:
                L_d2 = L_d2_hard

            L_geom += L_cd + 0.2 * L_d2
            if self._should_verbose_step(args):
                self.writer.write(
                    f"L_geom  :{self._scalar(L_geom):.4f}->"
                    f"L_cd:{self._scalar(L_cd):.4f}, "
                    f"L_d2:{self._scalar(L_d2):.4f}"
                )

        return L_geom

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
