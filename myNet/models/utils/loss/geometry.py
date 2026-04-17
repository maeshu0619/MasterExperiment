from .utils_loss import (
    chamfer_l2_loss,
    chamfer_l2_loss_and_weight_surrogate,
    compute_d2_psnr,
    remove_outlier_points_by_label,
)


class GeometryLossMixin:
    def get_geometry_loss(self, args, gen_pts, gt_pts, final_w=None, out_label=None):
        use_torch_d2 = args.trainORtest == "train"
        if gen_pts.shape[-1] == 0 or gt_pts.shape[-1] == 0:
            return gt_pts.new_zeros(())

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
