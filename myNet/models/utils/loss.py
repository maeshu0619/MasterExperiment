import os
import torch
import time
import h5py
import sys
import numpy as np

from models.utils.utils_repkpu import *
from models.utils.proxy_oa import proxy_octattention_like_octree_loss

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
from models.utils.proxy_octree import *

from models.utils.utils_loss import *
from models.utils.utils_p2c import *

import multiprocessing as mp
import numpy as np
import os
import traceback

class Loss:
    def __init__(self, args, file_date, writer):
        self.com_bit = args.com_bit
        self.com_sin = args.com_sin
        self.com_node = args.com_node

        self.lambda_p = args.lambda_p

        self.compress = args.compress
        self.file_date = file_date
        self.writer = writer
        self.bptt = args.bptt
        self.ncl = ManifoldnessConstraint(support=8, neighborhood_size=32).to(device)

        self.octree_cfgs = ProxyOctreeConfig(
            max_depth=args.proxy_max_depth,
            lambda_entropy=args.proxy_lambda_entropy,
            lambda_node_count=args.proxy_lambda_node_count,
            lambda_single_child=args.proxy_lambda_single_child,
        )

        
    def get_loss(self, args, gen_pts, gt_pts, final_w):
        gt_xyz = gt_pts[:, :3, :]
        gen_xyz = gen_pts[:, :3, :]
        proxy = SoftOctreeRateProxy(self.octree_cfgs).to(gen_xyz.device)
        s = time.time()
        out_gt, bit_gt, _ = proxy(gen_xyz=gt_xyz, final_w=None)
        print(time.time()-s)
        out_gen, bit_gen_soft, _ = proxy(gen_xyz=gen_xyz, final_w=final_w)
        L_com = 100 * (bit_gen_soft - bit_gt) / bit_gt

        if args.trainORtest == "test":
            self.writer.write(f"=== Compression Stats ===")
            self.writer.write(f"bit                         : {out_gt['bit']} -> {out_gen['bit']}")
            self.writer.write(f"bpp                         : {out_gt['bpp']} -> {out_gen['bpp']}")
            self.writer.write(f"bpn                         : {out_gt['bpn']} -> {out_gen['bpn']}")
            self.writer.write(f"single child node           : {out_gt['single']} -> {out_gen['single']}")
            self.writer.write(f"num of nodes                : {out_gt['node']} -> {out_gen['node']}")
            self.writer.write(f"num of points               : {gt_pts.shape[2]} -> {gen_pts.shape[2]}")

        loss_bit = (float(out_gen["rate_total"].detach().cpu())-float(out_gt["rate_total"].detach().cpu()))/(float(out_gt["rate_total"].detach().cpu()) + 1e-12)
        loss_single = (float(out_gen["soft_single_child_count"].detach().cpu())-float(out_gt["soft_single_child_count"].detach().cpu()))/(float(out_gt["rate_total"].detach().cpu()) + 1e-12)
        loss_nodes = (float(out_gen["soft_node_count"].detach().cpu())-float(out_gt["soft_node_count"].detach().cpu()))/(float(out_gt["rate_total"].detach().cpu()) + 1e-12)

        # ===== Geometry Loss =====
        L_geom = 0.0
        if args.loss_type == "cd":
            L_cd_hard = chamfer_l2_loss(gen_pts, gt_pts)
            L_cd_soft = chamfer_l2_loss(gen_pts, gt_pts, final_w)
            # L_cd = self.lambda_p * L_cd_hard + L_cd_soft
            L_geom = L_cd_soft
            self.writer.write(f"L_geom  :{L_cd_soft:.4f}")
        elif args.loss_type == "cd+d2":
            L_cd_hard = chamfer_l2_loss(gen_pts, gt_pts)
            L_cd_soft = chamfer_l2_loss(gen_pts, gt_pts, final_w)
            L_cd = self.lambda_p * L_cd_hard + L_cd_soft

            L_d2_hard = compute_d2_psnr(gen_pts, gt_pts)
            L_d2_soft = compute_d2_psnr(gen_pts, gt_pts, final_w=final_w)
            L_d2 = self.lambda_p * L_d2_hard + L_d2_soft

            L_geom += L_cd + 0.2 * L_d2
            self.writer.write(f"L_geom  :{L_geom.item():.4f}->L_cd:{L_cd.item():.4f}, L_d2:{L_d2.item():.4f}")
        
        self.writer.write(f"L_com   :{L_com:.4f}->L_bit:{loss_bit:.4f}, L_single:{loss_single:.4f}, L_nodes:{loss_nodes:.4f}")
        
        return L_geom, L_com, loss_bit, loss_single, loss_nodes