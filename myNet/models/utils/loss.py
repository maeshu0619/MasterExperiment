import os
import torch
import time
import sys
import numpy as np

from models.utils.utils_repkpu import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
sys.path.append(ROOT_DIR)

from compress.OctAttention.compression import OA_compress
from compress.OctAttention.encoderTool import *
from compress.OctAttention.networkTool import *
from compress.OctAttention.octAttention import *
from models.utils.utils_loss import *
from models.utils.utils_p2c import *

import multiprocessing as mp
import numpy as np
import os
import traceback

def _octseq_worker_main(conn):
    """
    Octree生成・matloaderなど「CPU/IO/メモリを食う処理」専用プロセス。
    ここで起きるメモリの溜まりは、このプロセスを再起動すればOSが回収する。
    """

    # 重要：このプロセスはGPUを絶対に触らない（GPUメモリ増加を防ぐ）
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # 重要：スレッド数を絞り、CPU暴騰を抑える（必要なら調整）
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    # ここで遅いimportをworker側に閉じ込める
    from compress.OctAttention.Preparedata.data import dataPrepare
    from compress.OctAttention.dataset import default_loader as matloader
    from compress.OctAttention.networkTool import levelNumK

    while True:
        msg = conn.recv()
        if msg is None:
            break

        try:
            args_dict = msg["args_dict"]
            pts_np = msg["pts_np"]          # (N,3) float32
            qs = msg["qs"]

            # RAMディスクを優先（存在しなければ通常dir）
            ram_dir = "/dev/shm/OctAttention_encoded"
            saveMatDir = ram_dir if os.path.isdir("/dev/shm") else "compress/OctAttention/encoded"
            os.makedirs(saveMatDir, exist_ok=True)

            # 重要：毎回ファイル名を固定して増殖させない
            matFile, DQpt, refPt = dataPrepare(
                SimpleNamespace(**args_dict),
                pts_np,
                saveMatDir,
                qs=qs,
                ptNamePrefix="tmp_fixed",
                rotation=False
            )

            cell, mat = matloader(matFile)

            # 読み終わったら即削除
            try:
                os.remove(matFile)
            except Exception:
                pass

            FeatDim = levelNumK
            oct_data_seq = np.transpose(mat[cell[0, 0]]).astype(np.int32)[:, -FeatDim:, 0:6]

            # 返すのはoct_data_seqのみ（巨大オブジェクトを返さない）
            conn.send({"ok": True, "oct_data_seq": oct_data_seq})
        except Exception:
            conn.send({"ok": False, "err": traceback.format_exc()})

class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Loss:
    def __init__(self, args, file_date, writer):
        self.alpha = args.a
        self.beta = args.b
        self.gamma = args.c
        self.sigma = args.d
        self.omega = args.e
        self.compress = args.compress
        self.file_date = file_date
        self.writer = writer
        self.writer.write(f"Loss parameter: {args.a}, {args.b}, {args.c}, {args.d}, {args.e}")
        self.bptt = args.bptt
        self.ncl = ManifoldnessConstraint(support=8, neighborhood_size=32).to(device)
        if self.compress == "OctAttention":
            self.oa_comp = Compression()
            self.model = build_model(device)
            # self.model = model
            self.qs = args.qs
            self._oa_p = None
            # self.model = model.to(device)
            saveDic = reload(None,'../compress/octree/OctAttention/modelsave/obj/encoder_epoch_00800093.pth')
            self.model.load_state_dict(saveDic['encoder'])
            # ---- Octree worker 起動（1回だけ）----
            ctx = mp.get_context("spawn")
            self._oct_conn_main, oct_conn_worker = ctx.Pipe(duplex=True)
            self._oct_proc = ctx.Process(target=_octseq_worker_main, args=(oct_conn_worker,), daemon=True)
            self._oct_proc.start()

            self.com_gt = 0

            # 念のため終了時に止める
            import atexit
            def _cleanup_oct_worker():
                try:
                    if hasattr(self, "_oct_conn_main") and self._oct_conn_main is not None:
                        self._oct_conn_main.send(None)
                except Exception:
                    pass
                try:
                    if hasattr(self, "_oct_proc") and self._oct_proc is not None:
                        self._oct_proc.join(timeout=1)
                except Exception:
                    pass

            atexit.register(_cleanup_oct_worker)

    # def _run_octattention_encoder(self, args, pts):
    #     with torch.no_grad(): 
    #         self.model.eval()
    #         return oa_main(
    #             args,
    #             pts=pts.to(device),
    #             model=self.model,
    #             qs=self.qs,
    #             writer=self.writer,
    #             file_date=self.file_date, 
    #             oa_comp=self.oa_comp
    #         )

    def _run_octattention_encoder(self, args, pts):
        """
        1) workerでoct_data_seqを生成（CPU/IO/リーク源を隔離）
        2) メインプロセスでcompress（GPU推論は維持）
        """
        with torch.no_grad():
            self.model.eval()

            # pts: [1,3,N] を (N,3) float32 numpy にして送る
            pts_np = (
                pts.detach()
                .squeeze(0)          # (3,N)
                .transpose(1, 0)     # (N,3)
                .contiguous()
                .to("cpu")
                .numpy()
                .astype(np.float32, copy=False)
            )

            # args は必要最小限だけdict化（pickle負担を減らす）
            args_dict = vars(args)

            self._oct_conn_main.send({
                "args_dict": args_dict,
                "pts_np": pts_np,
                "qs": self.qs
            })
            rep = self._oct_conn_main.recv()
            if not rep["ok"]:
                raise RuntimeError(rep["err"])

            oct_data_seq = rep["oct_data_seq"]

            # ここからは従来通りGPU側で圧縮推論
            binsz, oct_len, _, _, _ = self.oa_comp.compress(args, oct_data_seq, self.model, writer=None)

            # single_ratio はoct_data_seqから計算して良いが、重いなら省略
            single_ratio = 0.0
            try:
                from compress.OctAttention.eval import single_child_by_level
                single_ratio = single_child_by_level(oct_data_seq, writer=None)
            except Exception:
                pass

            ptNum = pts.shape[2]
            bit = binsz
            bpp = bit / ptNum
            bpn = bit / oct_len if oct_len > 0 else 0.0
            return [bit, bpp, bpn, single_ratio, oct_len]
        
    def get_loss(self, args, gen_pts, gt_pts, same = None):
        if args.compress == "OctAttention":
            time_loss = time.time()

            if not self._oct_proc.is_alive():
                print("Oct worker is dead")

            if same == None:
                com = self._run_octattention_encoder(args, gen_pts)
                while len(com) < 5:
                    com.append(None)
                com_gt = self._run_octattention_encoder(args, gt_pts)
                while len(com_gt) < 5:
                    com_gt.append(None)
            else:
                if same == 0:
                    com = self._run_octattention_encoder(args, gen_pts)
                    while len(com) < 5:
                        com.append(None)
                    com_gt = self._run_octattention_encoder(args, gt_pts)
                    while len(com_gt) < 5:
                        com_gt.append(None)
                    self.com_gt = com_gt
                else:
                    com = self._run_octattention_encoder(args, gen_pts)
                    while len(com) < 5:
                        com.append(None)
                    com_gt = self.com_gt
                    while len(com_gt) < 5:
                        com_gt.append(None)

            # ===== Geometry Loss =====
            loss_geom = 0.0
            if args.loss_type == "cd":
                loss_geom = chamfer_l2_loss(gen_pts, gt_pts)
            elif args.loss_type == "p2p":
                gt_normals = estimate_normals_open3d(gt_pts, k=16)
                loss_geom = point2plane_loss(
                    gen_pts, gt_pts,
                    gt_normals=gt_normals, k=16
                )
            elif args.loss_type == "psnr":
                loss_geom = 1-psnr_loss(gen_pts, gt_pts)
            elif args.loss_type == "cd+p2p":
                loss_cd = chamfer_l2_loss(gen_pts, gt_pts)
                gt_normals = estimate_normals_open3d(gt_pts, k=16)
                loss_p2p = point2plane_loss(
                    gen_pts, gt_pts,
                    gt_normals=gt_normals, k=16
                )
                loss_geom = loss_cd + 0.1 * loss_p2p
            elif args.loss_type == "cd+nc":
                loss_cd = chamfer_l2_loss(gen_pts, gt_pts)
                y = time.time()
                loss_nc = self.ncl(gen_pts)
                z = time.time()
                loss_geom = 0.1 * loss_cd + 100 * loss_nc
                self.writer.write(f"Loss_nc:{loss_nc.item()}, Loss_cd:{loss_cd}, Total:{loss_geom.item()}")
            elif args.loss_type == "p2p+ncl":
                gt_normals = estimate_normals_open3d(gt_pts, k=16)
                loss_p2p = point2plane_loss(gen_pts, gt_pts, gt_normals)
                loss_ncl = normal_consistency_loss(gen_pts, gt_pts, gt_normals)

                loss_geom = loss_p2p + 0.1 * loss_ncl
            else:
                raise ValueError(f"Unknown loss_type: {args.loss_type}")
            
            if args.trainORtest == "test":
                self.writer.write(f"=== Compression Stats ===")
                self.writer.write(f"bit                         : {com_gt[0]} -> {com[0]}")
                self.writer.write(f"bpp                         : {com_gt[1]} -> {com[1]}")
                self.writer.write(f"bpn                         : {com_gt[2]} -> {com[2]}")
                self.writer.write(f"ratio of single child node  : {com_gt[3]} -> {com[3]}")
                self.writer.write(f"num of nodes                : {com_gt[4]} -> {com[4]}")
                self.writer.write(f"num of points               : {gt_pts.shape[2]} -> {gen_pts.shape[2]}")

            loss_bit = (com[0]-com_gt[0])/com_gt[0]
            loss_single = (com[3]-com_gt[3])/com_gt[3]
            loss_nodes = (com[4]-com_gt[4])/com_gt[4]

            loss_num = abs(gen_pts.shape[2]-gt_pts.shape[2])/gt_pts.shape[2]
                    
            if loss_bit > 0: # 圧縮効率が悪化した場合ペナルティを大きく
                beta = self.beta * 5
            else:
                beta = self.beta
            Loss = self.alpha*loss_geom + beta*loss_bit + self.gamma*loss_single + self.sigma*loss_nodes
            self.writer.write(f"Loss:{Loss.item()}->L_geom:{loss_geom.item()}, L_bit:{loss_bit}, L_single:{loss_single}, L_nodes:{loss_nodes}")
            # self.writer.write(f"=== Loss Function ===")

            return Loss, loss_geom, loss_bit, (gen_pts.shape[2]-gt_pts.shape[2])/gt_pts.shape[2]
        else:
            return 0.0