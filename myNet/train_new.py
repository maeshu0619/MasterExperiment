import os
import torch
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
import open3d as o3d
import sys
import argparse

import time
import datetime

from glob import glob
from models.utils.utils_repkpu import *
from einops import repeat
from models.utils.expand import *
from models.utils.dataset import *
from record.write import Writing
from models.network import Network
from models.utils.loss import Loss
from models.utils.loss import *
from record.plot import plot_loss_curve

from cfgs.utils import str2bool
from models.utils.args import parse_pugan_args

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

# 点群映像のみを使ったトレーニングか否か
Flag_video = 2
if Flag_video == 1:
    dir_input = "video_scaled"
elif Flag_video == 2:
    dir_input = "video_noised"
elif Flag_video == 3:
    dir_input = "video_outlier"

Flag_dir = 1
if Flag_dir == 1:
    data_input = "UVG"
print(f"dir_input: {dir_input}, data_input: {data_input}")

# def _normalize_point_cloud(pc):
#     # b, n, 3
#     centroid = torch.mean(pc, dim=1, keepdim = True) # b, 1, 3
#     pc = pc - centroid # b, n, 3
#     furthest_distance = torch.max(torch.sqrt(torch.sum(pc**2, dim=-1, keepdim=True)), dim=1, keepdim=True)[0] # b, 1, 1
#     pc = pc / furthest_distance
#     return pc


def train(model, args, loss, file_date, writer):
    """==========================================================="""
    """セットアップ"""
    """==========================================================="""
    set_seed(args.seed)
    best_loss = float('inf')

    # ===== Loss histogram (fixed size, safe) =====
    epoch_loss_history = []
    epoch_loss_geom_history = []
    epoch_loss_bit_history = []
    epoch_loss_num_history = []
    epi_loss_history = []
    epi_loss_geom_history = []
    epi_loss_bit_history = []
    epi_loss_num_history = []
    loss_bins = [-1e9, -1.0, -0.5, 0.0, 0.5, 1.0, 1e9]
    loss_hist = [0 for _ in range(len(loss_bins) - 1)]

    if Flag_dir == 1:
        seq_dirs = collect_seq_dirs2(f"../data/train/{dir_input}", dataset_name=data_input)
    else:
        seq_dirs = collect_seq_dirs(f"../data/train/{dir_input}")
    num_seq = len(seq_dirs)

    writer.write(f"Total seq directories: {num_seq}")
    start = time.time()

    # モデル保存先ファイルのセットアップ
    output_dir = os.path.join(args.out_path+"_"+args.loss_type)
    ckpt_dir = os.path.join(output_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    

    # 最適化パラメータのセットアップ
    # deform（Displacement）の重み更新は他よりも緩やかにする
    # deform（Displacement）はそのまま
    if args.deform:
        deform_params = [p for n, p in model.named_parameters()
                        if ('disp_module' in n) and p.requires_grad]

        # encoder を除外し、かつ requires_grad=True のものだけ
        other_params = [p for n, p in model.named_parameters()
                        if ('disp_module' not in n) and ('encoder' not in n) and p.requires_grad]
    else:
        deform_params = []
        other_params = [p for n, p in model.named_parameters()
                        if ('encoder' not in n) and p.requires_grad]

    # ===== 確認ログ =====
    num_enc_trainable = sum(p.requires_grad for p in model.encoder.parameters())
    writer.write(f"Trainable encoder params: {num_enc_trainable} (should be 0)")
    print(f"Trainable encoder params: {num_enc_trainable} (should be 0)")


    assert args.optim in ['adam', 'sgd']
    if args.optim == 'adam':
        optimizer = optim.Adam([{'params': other_params},
                                {'params': deform_params, 'lr': args.lr*0.1}], 
                                lr=args.lr, weight_decay=args.weight_decay)
    else:
        args.lr = args.lr * 100
        optimizer = optim.SGD([{'params': other_params},
                                {'params': deform_params, 'lr': args.lr * 0.1}], lr=args.lr)
    
    # スケジュール（学習が進むほどOptimiserの動きを緩やかにする）
    # scheduler_steplr = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.05**(1/150))
    scheduler_steplr = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)

    """==========================================================="""
    """トレーニング"""
    """==========================================================="""
    writer.write('■■■■ Begin Training ■■■■')
    best_cd = 10000

    for episode in range(args.episodes):
        writer.write(f"◆◆◆ Episode {episode + 1} / {args.episodes} ◆◆◆\n")
        model.train()
        epi_loss = 0.0
        epi_loss_geom = 0.0
        epi_loss_bit = 0.0
        epi_loss_num = 0.0
        for epoch, seq_dir in enumerate(seq_dirs):
            writer.write(f"●●● Epoch {epoch + 1}/{num_seq} : {seq_dir} ●●●\n")
            dataset = PlyDirDataset(args, seq_dir)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,   # 重要：巨大点群なので 0 推奨
                pin_memory=False
            )
        
            epoch_loss = 0.0
            epoch_loss_geom = 0.0
            epoch_loss_bit = 0.0
            epoch_loss_num = 0.0

            for step, pts in enumerate(loader):
                for same in range(3):
                    st_step = time.time()
                    writer.write(f"＊＊＊ Step {step + 1} ＊＊＊")
                    if step == 0:
                        writer.write(f"Dataset in this Epoch: {seq_dir}")

                    # pts: [1, N, 3]
                    input_pcd = pts[0].unsqueeze(0).cuda()                 # (1, N, 3)
                    input_pcd = rearrange(input_pcd, 'b n c -> b c n').contiguous()  # (1, 3, N)

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

                        optimizer.zero_grad()

                        st_model = time.time()
                        if args.parallel:
                            # writer.write(f"Using parallel model for all patches")
                            gen_patches = model.forward(patches)
                        else:
                            for i in range(0, B_patch, pb):
                                patch_chunk = patches[i:i+pb]  # (pb, 3, K)

                                # forward（encoder含めて学習するなら no_grad は付けない）
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
                        gen_patches = model.forward(patches)
                        en_model = time.time()


                    # 元スケールに戻す
                    gen_patches = centroid + gen_patches * furthest_distance

                    # 1点群に統合
                    gen_pts = rearrange(gen_patches, 'b c n -> 1 c (b n)').contiguous()
                    
                    pts_xyz = gen_pts[:, :3, :]  # [1, 3, N]
                    input_xyz = input_pcd[:, :3, :]  # [1, 3, N]

                    # ---------- Loss計算と最適化 ----------
                    L, loss_geom, loss_bit, loss_num = loss.get_loss(args, pts_xyz, input_xyz, same)
                    L.backward()
                    optimizer.step()

                    # ログ（lossは training で計算した平均の合計）
                    epoch_loss += L.item()
                    epoch_loss_geom += loss_geom.item()
                    epoch_loss_bit += loss_bit.item()
                    epoch_loss_num += loss_num
                    
                    en_step = time.time()
                    
                    # mem(f"Epi{episode + 1}/Epo{epoch + 1}/Step{step + 1}:{en_step-st_step} s | ")
                    print(f"Epi{episode + 1}/Epo{epoch + 1}/Step{step + 1}/Cnt{same+1}: {en_step-st_step} s | Bit Loss: {loss_bit}")

            # lr scheduler
            scheduler_steplr.step()

            # ログの記録
            interval = time.time() - start

            avg_epoch_loss = epoch_loss / len(loader)
            avg_epoch_loss_geom = epoch_loss_geom / len(loader)
            avg_epoch_loss_bit = epoch_loss_bit / len(loader)
            avg_epoch_loss_num = epoch_loss_num / len(loader)
            epi_loss += avg_epoch_loss
            epi_loss_geom += avg_epoch_loss_geom
            epi_loss_bit += avg_epoch_loss_bit
            epi_loss_num += avg_epoch_loss_num
            epoch_loss_history.append(avg_epoch_loss)
            epoch_loss_geom_history.append(avg_epoch_loss_geom)
            epoch_loss_bit_history.append(avg_epoch_loss_bit)
            epoch_loss_num_history.append(avg_epoch_loss_num)


            plot_loss_curve(
                loss_history=epoch_loss_history,
                save_dir=f"../log/{args.date}/MyNetwork_train/",
                filename=f"epo_{args.time}_{args.loss_type}.png",
                title=f"Loss Curve ({args.loss_type})"
            )
            plot_loss_curve(
                loss_history=epoch_loss_geom_history,
                save_dir=f"../log/{args.date}/MyNetwork_train/",
                filename=f"epo_{args.time}_{args.loss_type}_geom.png",
                title=f"Geometry Loss Curve ({args.loss_type})"
            )
            plot_loss_curve(
                loss_history=epoch_loss_bit_history,
                save_dir=f"../log/{args.date}/MyNetwork_train/",
                filename=f"epo_{args.time}_{args.loss_type}_bit.png",
                title=f"Bit Loss Curve ({args.loss_type})"
            )
            plot_loss_curve(
                loss_history=epoch_loss_num_history,
                save_dir=f"../log/{args.date}/MyNetwork_train/",
                filename=f"epo_{args.time}_{args.loss_type}_num.png",
                title=f"Num Loss Curve ({args.loss_type})"
            )

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                if args.prune:
                    model_name = f'{data_input}_{args.loss_type}.pth'
                else:
                    model_name = f'{data_input}_{args.loss_type}_nonPrune.pth'
                model_path = os.path.join(ckpt_dir, model_name)
                torch.save(model.state_dict(), model_path)
                writer.write(
                    f"New best model at epoch {epoch+1}, "
                    f"avg_epoch_loss={best_loss:.6f}\n"
                )

        avg_epi_loss = epi_loss / num_seq
        avg_epi_loss_geom = epi_loss_geom / num_seq
        avg_epi_loss_bit = epi_loss_bit / num_seq
        avg_epi_loss_num = epi_loss_num / num_seq
        epi_loss_history.append(avg_epi_loss)
        epi_loss_geom_history.append(avg_epi_loss_geom)
        epi_loss_bit_history.append(avg_epi_loss_bit)
        epi_loss_num_history.append(avg_epi_loss_num)
        
        plot_loss_curve(
            loss_history=epi_loss_history,
            save_dir=f"../log/{args.date}/MyNetwork_train/",
            filename=f"epi_{args.time}_{args.loss_type}.png",
            title=f"Loss Curve ({args.loss_type})", 
            xl = "Episode"
        )
        plot_loss_curve(
            loss_history=epi_loss_geom_history,
            save_dir=f"../log/{args.date}/MyNetwork_train/",
            filename=f"epi_{args.time}_{args.loss_type}_geom.png",
            title=f"Geometry Loss Curve ({args.loss_type})", 
            xl = "Episode"
        )
        plot_loss_curve(
            loss_history=epi_loss_bit_history,
            save_dir=f"../log/{args.date}/MyNetwork_train/",
            filename=f"epi_{args.time}_{args.loss_type}_bit.png",
            title=f"Bit Loss Curve ({args.loss_type})", 
            xl = "Episode"
        )
        plot_loss_curve(
            loss_history=epi_loss_num_history,
            save_dir=f"../log/{args.date}/MyNetwork_train/",
            filename=f"epi_{args.time}_{args.loss_type}_num.png",
            title=f"Num Loss Curve ({args.loss_type})", 
            xl = "Episode"
        )

if __name__ == '__main__':
    """=== セットアップ ==="""
    # トレーニングInfoのセットアップ
    file_day = datetime.datetime.now().strftime('%Y-%m-%d')
    file_time = datetime.datetime.now().strftime('%H-%M-%S')

    parser = argparse.ArgumentParser(description='Training Arguments')
    parser.add_argument('--trainORtest', default="train", type=str, help='date')
    args = parse_pugan_args(parser, file_day, file_time, dir_input)

    
    # ログのセットアップ
    writer = Writing(file_day, file_time, filename="MyNetwork_train", dataname=dir_input)
    writer.write(f"Date of Training: {file_day}-{file_time}")
    writer.write(f"Loss Type: {args.loss_type}")

    print(f"Pruning Module used: {args.prune}")
    writer.write(f"Pruning Module used: {args.prune}")

    if args.split2patch:
        writer.write(f"Model Input is Patch")
    else:
        writer.write(f"Model Input is Whole Point Cloud")

    # モデルのセットアップ
    model = Network(args, writer)
    # ===== RepKPU Encoder の重みをロード =====
    ckpt = torch.load("pretrained/repkpu/ckpt-best.pth", map_location="cpu")
    encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in ckpt.items()
        if k.startswith("encoder.")
    }
    model.encoder.load_state_dict(encoder_state, strict=False)
    for p in model.encoder.parameters():
        p.requires_grad = False

    incompatible = model.encoder.load_state_dict(
        encoder_state, strict=False
    )

    if args.cpu == False:
        print(f"Using GPU for training")
        model = model.cuda()
        
    loss = Loss(args, file_day+"-"+file_time, writer)
    
    # トレーニング開始
    st = time.time()
    print(f"=== Start Training ===")
    writer.write(f"=== Start Training ===")
    train(model, args, loss, file_day+"-"+file_time, writer)
    en = time.time()

    FinishDate = datetime.datetime.now().strftime('%Y/%m/%d - %H:%M:%S')

    # トレーニング時間の記録
    print(f"Training time: {en - st}")
    print(f"Date of finisshing training: {FinishDate}")
    writer.write(f"Training time: {en - st}")
    writer.write(f"Date of finisshing training: {FinishDate}")