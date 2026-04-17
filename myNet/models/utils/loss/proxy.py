import torch


class ProxyCompressionLossMixin:
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
