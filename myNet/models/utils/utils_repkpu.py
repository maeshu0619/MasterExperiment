import torch
import math
from einops import rearrange
from models.pointops.functions import pointops
import logging
import os
import numpy as np
import random
import psutil

from torch.autograd import grad
from einops import rearrange, repeat
from sklearn.neighbors import NearestNeighbors
from models.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
chamfer_dist = chamfer_3DDist()

def print_gpu_mem(tag):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved  = torch.cuda.memory_reserved() / 1024**2
    print(f"[{tag}] allocated={allocated:.1f}MB reserved={reserved:.1f}MB")

def print_mem(tag):
    p = psutil.Process(os.getpid())
    print(f"[MEM] {tag}: {p.memory_info().rss / 1024**3:.2f} GB")

def print_cuda_mem(tag):
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_alloc = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[CUDA] {tag}: alloc={alloc:.2f}GB reserved={reserved:.2f}GB max={max_alloc:.2f}GB")

def mem(tag):
    if not torch.cuda.is_available():
        return
    p = psutil.Process(os.getpid())
    
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_alloc = torch.cuda.max_memory_allocated() / 1024**3
    print(f"{tag:<24}| CPU: {p.memory_info().rss / 1024**3:.2f} GB, GPU: alloc={alloc:.2f}GB")


def set_seed(seed):
    """
    Pytorch・Numpy・Pythonの乱数を全て固定し、
    同じコード、重み、データを用いた場合に
    同じ結果が出力されるようにする
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def index_points(pts, idx, chunk=262144):
    """
    chunk: 1回のgatherで処理するインデックス数の上限
    """
    batch_size = idx.shape[0]
    sample_num = idx.shape[1]
    fdim = pts.shape[1]

    reshape = False
    if idx.dim() == 3:
        reshape = True
        idx = idx.reshape(batch_size, -1)

    out = []
    for start in range(0, idx.shape[1], chunk):
        part = idx[:, start:start+chunk]                 # (B, chunk)
        part_expand = part.unsqueeze(1).expand(-1, fdim, -1)
        out.append(torch.gather(pts, 2, part_expand))    # (B, C, chunk)

    res = torch.cat(out, dim=2)

    if reshape:
        res = rearrange(res, 'b c (s k) -> b c s k', s=sample_num)

    return res

def chamfer_sqrt(p1, p2):
    """
    Chamfer距離（CD）の平均を計算する
    """
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.clamp(d1, min=1e-9)
    d2 = torch.clamp(d2, min=1e-9)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


def FPS(pts, fps_pts_num):
    """
    FPS（Furthest Point Sampling）を計算
    すでに選ばれた点から最も遠い点を順番に取る
    """
    # input: (b, 3, n)

    # (b, n, 3)
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    # (b, fps_pts_num)
    sample_idx = pointops.furthestsampling(pts_trans, fps_pts_num).long()
    # (b, 3, fps_pts_num)
    sample_pts = index_points(pts, sample_idx)

    return sample_pts

def get_knn_pts(k, pts, center_pts, return_idx=False):
    # input: (b, 3, n)
    # (b, n, 3)
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    # (b, m, 3)
    center_pts_trans = rearrange(center_pts, 'b c m -> b m c').contiguous()
    # (b, m, k)
    knn_idx = pointops.knnquery_heap(k, pts_trans, center_pts_trans).long()
    # (b, 3, m, k)
    knn_pts = index_points(pts, knn_idx)

    if return_idx == False:
        return knn_pts
    else:
        return knn_pts, knn_idx
    
# def get_knn_pts(k, pts, center_pts, return_idx=False):
#     # (b, c, n) -> (b, n, c)
#     pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()

#     # (b, c, m) -> (b, m, c)
#     center_pts_trans = rearrange(center_pts, 'b c m -> b m c').contiguous()

#     knn_idx = pointops.knnquery_heap(k, pts_trans, center_pts_trans).long()
#     knn_pts = index_points(pts, knn_idx)  # 形状はindex_points実装に依存

#     if return_idx:
#         return knn_pts, knn_idx
#     else:
#         return knn_pts

# def get_knn_pts(k, pts, center_pts, radius=0.4, return_idx=False):
#     """
#     GPU版KNNの計算
#     角中心の周囲k個の最近傍点を高速に取得
#     # """
#     # # input: (b, 3, n)

#     # # (b, n, 3)
#     # pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
#     # # (b, m, 3)
#     # center_pts_trans = rearrange(center_pts, 'b c m -> b m c').contiguous()
#     # # (b, m, k)
#     # knn_idx = pointops.knnquery_heap(k, pts_trans, center_pts_trans).long()
#     # # (b, 3, m, k)
#     # knn_pts = index_points(pts, knn_idx)

#     # if return_idx == False:
#     #     return knn_pts
#     # else:
#     #     return knn_pts, knn_idx
#     pts_np = rearrange(pts.squeeze(0), 'c n -> n c').cpu().numpy()
#     centers_np = rearrange(center_pts.squeeze(0), 'c m -> m c').cpu().numpy()

#     nbrs = NearestNeighbors(radius=radius, algorithm='kd_tree')
#     nbrs.fit(pts_np)

#     all_indices = []
#     for c in centers_np:
#         idx = nbrs.radius_neighbors([c], return_distance=False)[0]

#         # 半径内に点が多すぎる → k個に制限
#         if len(idx) >= k:
#             idx = idx[:k]
#         # 少なすぎる → 通常KNNで補完
#         else:
#             knn = NearestNeighbors(n_neighbors=k)
#             knn.fit(pts_np)
#             idx = knn.kneighbors([c], return_distance=False)[0]

#         all_indices.append(idx)

#     knn_idx = torch.from_numpy(np.stack(all_indices)).long().cuda()  # (M, k)
#     knn_pts = index_points(pts, knn_idx.unsqueeze(0)).squeeze(0)     # (3, M, k)
#     knn_pts = knn_pts.permute(1, 0, 2).contiguous()                  # (M, 3, k)

#     if return_idx:
#         return knn_pts, knn_idx
#     else:
#         return knn_pts


def normalize_point_cloud(input, centroid=None, furthest_distance=None):
    """
    正規化
    これにより、原点中心・半径1以内に数値を抑えることで、
    ネットワークがスケールに依存しないようにする
    """
    # input: (b, 3, n) tensor

    if centroid is None:
        # (b, 3, 1)
        centroid = torch.mean(input, dim=-1, keepdim=True)
    # (b, 3, n)
    input = input - centroid
    if furthest_distance is None:
        # (b, 3, n) -> (b, 1, n) -> (b, 1, 1)
        furthest_distance = torch.max(torch.norm(input, p=2, dim=1, keepdim=True), dim=-1, keepdim=True)[0]
    input = input / furthest_distance

    return input, centroid, furthest_distance


def add_noise(pts, sigma, clamp):
    """
    ノイズ付与
    """
    # input: (b, 3, n)

    assert (clamp > 0)
    jittered_data = torch.clamp(sigma * torch.randn_like(pts), -1 * clamp, clamp).cuda()
    jittered_data += pts

    return jittered_data

def extract_knn_patch(k, pts, center_pts, return_idx=False):
    # (b, 3, n) → (n, 3)
    pts_trans = rearrange(pts.squeeze(0), 'c n -> n c').contiguous()
    pts_np = pts_trans.detach().cpu().numpy()

    # (b, 3, m) → (m, 3)
    center_pts_trans = rearrange(center_pts.squeeze(0), 'c m -> m c').contiguous()
    center_pts_np = center_pts_trans.detach().cpu().numpy()

    knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_search.fit(pts_np)

    # (m, k)
    knn_idx = knn_search.kneighbors(center_pts_np, return_distance=False)

    # (m, k, 3)
    patches = np.take(pts_np, knn_idx, axis=0)
    patches = torch.from_numpy(patches).float().cuda()
    patches = rearrange(patches, 'm k c -> m c k').contiguous()

    if return_idx:
        return patches, torch.from_numpy(knn_idx).long().cuda()
    else:
        return patches
        
# def extract_knn_patch(k, pts, center_pts):
#     """
#     KNNの計算
#     これを1つのパッチとみなす
#     """
#     # input : (b, 3, n)

#     # (n, 3)
#     pts_trans = rearrange(pts.squeeze(0), 'c n -> n c').contiguous()
#     pts_np = pts_trans.detach().cpu().numpy()
#     # (m, 3)
#     center_pts_trans = rearrange(center_pts.squeeze(0), 'c m -> m c').contiguous()
#     center_pts_np = center_pts_trans.detach().cpu().numpy()
#     knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
#     knn_search.fit(pts_np)
#     # (m, k)
#     knn_idx = knn_search.kneighbors(center_pts_np, return_distance=False)
#     # (m, k, 3)
#     patches = np.take(pts_np, knn_idx, axis=0)
#     patches = torch.from_numpy(patches).float().cuda()
#     # (m, 3, k)
#     patches = rearrange(patches, 'm k c -> m c k').contiguous()

#     return patches


def get_logger(name, log_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')
    # output to console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # output to log file
    log_name = name + '_log.txt'
    file_handler = logging.FileHandler(os.path.join(log_dir, log_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def reset_model_args(train_args, model_args):
    for arg in vars(train_args):
        setattr(model_args, arg, getattr(train_args, arg))

def get_cd_loss(args, coarse_pts, gt_pts):
    loss_cd = chamfer_sqrt(coarse_pts.permute(0,2,1).contiguous(), gt_pts.permute(0,2,1).contiguous()) * 1e3
    return loss_cd

