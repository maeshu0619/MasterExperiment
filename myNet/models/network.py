import torch
import torch.nn as nn
import os
import time
import datetime
import sys
from torch.utils.checkpoint import checkpoint

from .encoder.point_trans import PointTransformer
from .analyzer.ds_analyzer import DensityStructureAnalyzer
from .modules.add_ import AddModule
from .encoder.fp_module import FeaturePropagationExtended
from .encoder.fp_h import HeuristicFeaturePropagation
from .modules.displace import DisplacementModule
from .modules.prune import PruningModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
sys.path.append(ROOT_DIR)

from cfgs.utils import cal_n_t

# 新規ネットワークのアーキテクチャ
class Network(nn.Module):
    def __init__(self, args, writer):
        super().__init__()
        self.args = args
        self.writer = writer
        self.encoder = PointTransformer(self.args)
        self.fp_module = FeaturePropagationExtended(self.args, self.writer)
        self.fp_h = HeuristicFeaturePropagation(self.args, self.writer)
        self.analyzer = DensityStructureAnalyzer(self.args, self.writer)

        self.adding_module = AddModule(self.args, self.writer)
        self.prun_module = PruningModule(self.args, self.writer)
        self.disp_module = DisplacementModule(self.args, self.writer)

        self.last_importance = None
        self.last_add_idx = None
    
    def forward(self, pts):
        """ SetUp """
        # 入力点群をxyzとattrに分解
        pts_xyz  = pts[:, :3, :]          # [B, 3, N]
        pts_attr = pts[:, 3:, :]          # [B, Ca, N]
        st, N0 = cal_n_t(pts_xyz)

        """ Encoder """
        if self.args.encoder_0grad: # Encoderの重みを固定
            with torch.no_grad():
                Fl, Ff = self.encoder(pts_xyz)
        else:
            Fl, Ff = self.encoder(pts_xyz)
            
        en = time.time()
        
        """ Analyzer """
        d, s = self.analyzer(pts_xyz, Fl)
        an1 = time.time()

        """ Pruning Section """
        if self.args.prune: # Pruning Moduleを使う
            """ Pruning Module """
            pts_prun, keep_idx, loss_prun = self.prun_module(pts_xyz, Fl, Ff, d, s)
            pr, Np = cal_n_t(pts_prun)

            """ Cal Feature and Attribute after Pruning """
            Fl_prun = torch.gather(Fl, 2, keep_idx.unsqueeze(1).expand(-1, Fl.size(1), -1))
            F_prun  = torch.gather(Ff, 2, keep_idx.unsqueeze(1).expand(-1, Ff.size(1), -1))
            pts_atr_prun = torch.gather(pts_attr, 2, keep_idx.unsqueeze(1).expand(-1, pts_attr.size(1), -1))
            fp1 = time.time()
            
            """ Analyzer """
            d, s = self.analyzer(pts_prun, Fl_prun)
            an2 = time.time()
        else: # Pruning Moduleをスキップ
            pts_prun = pts_xyz
            Fl_prun = Fl
            F_prun  = Ff
            pts_atr_prun = pts_attr
            Np = N0
            pr = time.time()
            fp1 = pr
            an2 = pr 
        
        if self.args.add: # Adding Moduleを使う
            """ Adding Module """
            pts_add, new_pts, _, add_idx, loss_add = self.adding_module(pts_prun, Fl_prun, d, s)
            K = new_pts.shape[2]
            ad, Na = cal_n_t(pts_add)
        else: # Adding Moduleをスキップ
            pts_add = pts_prun
            new_pts = pts_prun.new_empty(pts_prun.size(0), pts_prun.size(1), 0)
            K = 0
            ad, Na = cal_n_t(pts_add)

        """ FP Module """
        if K > 0: # 追加点群が存在する
            """ FP Module """
            Fl_add, F_add, new_atr = self.fp_h(pts_prun, new_pts, pts_atr_prun, Fl_prun, F_prun, add_idx)

            """ Cal Feature and Attribute after Adding"""
            Fl_add_all = torch.cat([Fl_prun, Fl_add], dim=2)
            F_add_all  = torch.cat([F_prun,  F_add],  dim=2)
            pts_attr_add_all = torch.cat([pts_atr_prun, new_atr], dim=2)
        else: # 追加点群が存在しない
            Fl_add_all = Fl_prun
            F_add_all  = F_prun
            pts_attr_add_all = pts_atr_prun
        fp2 = time.time()

        """ Analyzer """
        d, s = self.analyzer(pts_add, Fl_add_all)
        an3 = time.time()

        """ Displcaing Module """
        if self.args.disp: # Displacing Moduleを使う
            pts_disp = self.disp_module(pts_add, F_add_all, d, s)
        else: # Displacing Moduleをスキップ
            pts_disp = pts_add

        pts_out = torch.cat([pts_disp, pts_attr_add_all], dim=1)
        di, Nd = cal_n_t(pts_out)

        if self.args.trainORtest == "test":
            self.writer.write(f"=== The Time of Network ===")
            self.writer.write(f"Encoder      : {en-st}")
            self.writer.write(f"Analyzer 1   : {an1-en}")
            self.writer.write(f"Pruning      : {pr-an1}")
            self.writer.write(f"FP 1         : {fp1-pr}")
            self.writer.write(f"Analyzer 2   : {an2-fp1}")
            self.writer.write(f"Adding       : {ad-an2}")
            self.writer.write(f"FP 2         : {fp2-ad}")
            self.writer.write(f"Analyzer 3   : {an3-fp2}")
            self.writer.write(f"Displacing   : {di-an3}")
            self.writer.write(f"Total        : {di-st}\n")

            self.writer.write(f"=== The Num of Points ===")
            self.writer.write(f"Input  : {N0}")
            self.writer.write(f"Pruned : {Np}")
            self.writer.write(f"Added  : {Na}")
            self.writer.write(f"Output : {Nd}\n")

        # self.writer.write(f"=== The Num of Points ===")
        # self.writer.write(f"Input  : {N0}")
        # self.writer.write(f"Pruned : {Np}")
        # self.writer.write(f"Added  : {Na}")
        # self.writer.write(f"Output : {Nd}\n")

        return pts_out, loss_prun, loss_add