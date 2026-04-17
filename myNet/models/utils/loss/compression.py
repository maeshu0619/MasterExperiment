import numpy as np
import torch

from .actual_encoder import _OctAttentionActualEncoder


class CompressionLossMixin:
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
    def _relative_ratio(value, ref, ref_min=1e-12):
        if torch.is_tensor(value):
            ref_t = value.new_tensor(float(ref))
            denom = ref_t.abs().clamp_min(float(ref_min))
            return (value - ref_t) / denom
        denom = max(abs(float(ref)), float(ref_min))
        return (float(value) - float(ref)) / denom

    @staticmethod
    def _relative_percent(value, ref, ref_min=1e-12):
        return 100.0 * CompressionLossMixin._relative_ratio(value, ref, ref_min)

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

    def _get_actual_encoder(self, args):
        if self.actual_encoder is None:
            self.actual_encoder = _OctAttentionActualEncoder(args, writer=self.writer)
        return self.actual_encoder

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

    def _get_compression_loss_actual_octattention(self, args, gen_xyz, gt_xyz, final_w, cache_key=None, use_proxy_surrogate=False):
        cached_gt = self._get_cached_actual_gt(cache_key)
        if cached_gt is None:
            cached_gt = self._encode_actual_batch(args, gt_xyz)
            self._store_cached_actual_gt(cache_key, cached_gt)

        stats_gen = self._encode_actual_batch(args, gen_xyz)
        gt_bit = float(cached_gt["bit"])
        gen_bit = float(stats_gen["bit"])
        loss_bit_ratio = self._relative_ratio(gen_bit, gt_bit)
        loss_bit_percent = 100.0 * loss_bit_ratio

        L_com_hard = gen_xyz.new_tensor(loss_bit_ratio)
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

        loss_bit = gen_xyz.new_tensor(loss_bit_percent)
        loss_single = gen_xyz.new_tensor(
            self._relative_percent(float(stats_gen["single"]), float(cached_gt["single"]), ref_min=1.0)
        )
        loss_nodes = gen_xyz.new_tensor(
            self._relative_percent(float(stats_gen["node"]), float(cached_gt["node"]), ref_min=1.0)
        )
        self.last_compression_debug = {
            "metric": "actual_octattention_total_bits",
            "total_bit": loss_bit_percent,
            "bpp": self._relative_percent(float(stats_gen["bpp"]), float(cached_gt["bpp"])),
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
                f"rel:{loss_bit_percent:.4f}%"
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
                "octattention_actual, octattention_actual_ste, octattention_surrogate "
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
