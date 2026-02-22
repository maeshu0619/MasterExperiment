import os
import torch
import torch.optim as optim
import open3d as o3d
import sys
import argparse

import time
import datetime

from glob import glob
from models.utils.utils_repkpu import *
from models.utils.utils_repkpu import *
from einops import repeat
from models.utils.expand import *
from models.utils.dataset import PlyDirDataset
from record.write import Writing
from models.network import Network
from models.utils.loss import Loss
from models.utils.args import parse_pugan_args
from torch.utils.data import DataLoader


# 点群映像のみを使ったトレーニングか否か
Flag_video = 2
if Flag_video == 1:
    dir_input = "video_scaled"
elif Flag_video == 2:
    dir_input = "video_noised"
print(f"dir_input: {dir_input}")

def test(model, loss, args, writer):
    """==========================================================="""
    """セットアップ"""
    """==========================================================="""
    model.eval()

    writer.write(f"model: {args.ckpt}")
    writer.write(f"input dir: {args.input_dir_test}")
    
    print(f"Pruning Module used: {args.prune}")
    writer.write(f"Pruning Module used: {args.prune}")

    dataset = PlyDirDataset(args, args.input_dir_test)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)


    with torch.no_grad():
        # Lossの平均計算用の配列定義
        L_his = []
        L_geom_his = []
        L_bit_his = []
        L_num_his = []
        for step, pts in enumerate(loader):
            st_step = time.time()
            writer.write(f"＊＊＊ Step {step + 1} ＊＊＊")
            print(f"*** step {step + 1}/{len(loader)} ***")
            input_pcd = pts[0].unsqueeze(0)
            if not args.cpu:
                input_pcd = input_pcd.cuda()
            input_pcd = rearrange(input_pcd, 'b n c -> b c n')

            # patch 抽出
            pcd_pts_num = input_pcd.shape[-1]
            if args.split2patch:
                patch_pts_num = args.num_points
                sample_num = int(pcd_pts_num / patch_pts_num * args.patch_rate)
                seed = FPS(input_pcd, sample_num)
                patches = extract_knn_patch(patch_pts_num, input_pcd, seed)
                patches, centroid, furthest_distance = normalize_point_cloud(patches)  # (B_patch, 3, K)

                gen_patches_list = []
                pb = args.patch_batch_size
                B_patch = patches.shape[0]

                st_model = time.time()
                
                for i in range(0, B_patch, pb):
                    patch_chunk = patches[i:i+pb]  # (pb, 3, K)
                    gen_chunk = model.forward(patch_chunk)  # (pb, 3, K) を想定
                    gen_patches_list.append(gen_chunk)

                en_model = time.time()

                gen_patches = torch.cat(gen_patches_list, dim=0)   # (B_patch, 3, K)

            else:
                num = 1
                patch_pts_num = int(pcd_pts_num / num)
                sample_num = num
                if num != 1:
                    seed = FPS(input_pcd, sample_num)
                    patches = extract_knn_patch(patch_pts_num, input_pcd, seed)
                    patches, centroid, furthest_distance = normalize_point_cloud(patches)  # (B_patch, 3, K)
                else:
                    patches, centroid, furthest_distance = normalize_point_cloud(input_pcd)  # (1, 3, N)

                st_model = time.time()
                gen_patches, L_prun, L_add = model.forward(patches)
                en_model = time.time()

            st_fp = time.time()

            # 元スケールに戻す
            gen_patches = centroid + gen_patches * furthest_distance

            # 1点群に統合
            gen_pts = rearrange(gen_patches, 'b c n -> 1 c (b n)').contiguous()

            pts_xyz = gen_pts[:, :3, :]  # [1, 3, N]
            input_xyz = input_pcd[:, :3, :]  # [1, 3, N]

            # ---------- Loss計算 ----------
            L_geom, L_com, loss_bit, loss_num = loss.get_loss(args, pts_xyz, input_xyz)
            L = args.w_geom*L_geom + args.w_com*L_com + args.w_prun*L_prun + args.w_add*L_add
            L_his.append(L.item())
            L_geom_his.append(L_geom.item())
            L_bit_his.append(loss_bit.item())
            L_num_his.append(loss_num)
        
            out = gen_pts.squeeze(0).transpose(0, 1).detach().cpu().numpy()
            xyz = out[:, :3]
            rgb = out[:, 3:6]

            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

            en_fp = time.time()

            save_dir = args.save_ply_dir
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{step:04d}_{args.loss_type}.ply")
            print(f"save path: {save_dir}/{step:04d}_{args.loss_type}.ply")

            en_step = time.time()
            with open(save_path, "w") as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {xyz.shape[0]}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")

                data = np.concatenate([xyz, rgb], axis=1)
                np.savetxt(f, data, fmt="%f %f %f %d %d %d")

            print(f"model time: {en_model - st_model}")
            writer.write(f"{step+1} Step time: {en_step - st_step}")
            writer.write(f"  - Model time: {en_model - st_model}")
            writer.write(f"  - Forward time: {en_fp - en_model}")
            writer.write(f"  - Sum time: {en_fp - st_fp}")
            writer.write(f"  - Step time: {en_step - st_step}")
            writer.write(f"Saved PLY to: {save_path}\n")

        L_avg = np.mean(L_his)
        L_geom_avg = np.mean(L_geom_his)
        L_bit_avg = np.mean(L_bit_his)
        L_num_avg = np.mean(L_num_his)
        
        L_max = np.max(L_his)
        L_geom_max = np.max(L_geom_his)
        L_bit_max = np.max(L_bit_his)
        L_num_max = np.max(L_num_his)

        L_min = np.min(L_his)
        L_geom_min = np.min(L_geom_his)
        L_bit_min = np.min(L_bit_his)
        L_num_min = np.min(L_num_his)

        writer.write(f"=== Average ===")
        writer.write(f"Loss         : {L_avg}")
        writer.write(f"Loss Geom    : {L_geom_avg}")
        writer.write(f"Loss Bit     : {L_bit_avg}")
        writer.write(f"Loss Num     : {L_num_avg}\n")
        
        writer.write(f"=== MAX ===")
        writer.write(f"Loss         : {L_max}")
        writer.write(f"Loss Geom    : {L_geom_max}")
        writer.write(f"Loss Bit     : {L_bit_max}")
        writer.write(f"Loss Num     : {L_num_max}\n")

        writer.write("=== Min ===")
        writer.write(f"Loss         : {L_min}")
        writer.write(f"Loss Geom    : {L_geom_min}")
        writer.write(f"Loss Bit     : {L_bit_min}")
        writer.write(f"Loss Num     : {L_num_min}\n")

if __name__ == '__main__':
    """=== セットアップ ==="""
    # テストInfoのセットアップ
    file_day = datetime.datetime.now().strftime('%Y-%m-%d')
    file_time = datetime.datetime.now().strftime('%H-%M-%S')

    parser = argparse.ArgumentParser(description='Testing Arguments')
    parser.add_argument('--trainORtest', default="test", type=str, help='date')
    args = parse_pugan_args(parser, file_day, file_time, dir_input)
    
    # ログのセットアップ
    writer = Writing(file_day, file_time, filename="MyNetwork_test", dataname=dir_input)
    writer.write(f"Date of Testing: {file_day}-{file_time}")
    writer.write(f"Loss Type: {args.loss_type}")


    # モデルのセットアップ
    print(f"Model Setting: {datetime.datetime.now()}")
    model = Network(args, writer)
    if args.cpu == False:
        print(f"Using GPU for testing")
        model = model.cuda()
    print(f"Model Setted: {datetime.datetime.now()}\n")
    model.load_state_dict(torch.load(args.ckpt), strict=False)
    
    # 損失計算（推論時にどうなるのか計算するため）
    loss = Loss(args, file_day+"-"+file_time, writer)

    # テスト開始
    st = time.time()
    print(f"=== Start Testing ===")
    writer.write(f"=== Start Testing ===")
    test(model, loss, args, writer)
    en = time.time()

    FinishDate = datetime.datetime.now().strftime('%Y/%m/%d - %H:%M:%S')

    # テスト時間の記録
    print(f"Testing time: {en - st}")
    print(f"Date of finishing testing: {FinishDate}")
    writer.write(f"Testing time: {en - st}")
    writer.write(f"Date of finishing testing: {FinishDate}")