import subprocess
import os
import json
import torch
import time
import sys
import numpy as np
import hashlib
import open3d as o3d
import torch.nn as nn
import faiss

from .utils_p2c import *

# from pytorch3d.ops.points_normals import estimate_pointcloud_normals

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
sys.path.append(ROOT_DIR)

from Chamfer3D.dist_chamfer_3D import chamfer_3DDist

chamfer_dist = chamfer_3DDist()


def make_gt_key(gt_pts: torch.Tensor) -> str:
    # 全点ハッシュは重いので、先頭の一部＋shapeで簡易キーにする
    # 衝突リスクはあるが実用上は低い。厳密にやるならデータローダから frame_id を渡すのが最適。
    x = gt_pts.detach().contiguous().view(-1)
    take = min(x.numel(), 4096)  # 先頭4096要素だけ
    cpu = x[:take].cpu().numpy().tobytes()
    h = hashlib.sha1(cpu).hexdigest()
    return f"{tuple(gt_pts.shape)}_{h}"

def estimate_normals_open3d(gt_pts: torch.Tensor, k: int = 16) -> torch.Tensor:
    # Open3DのKDTreeで法線を1回で計算（C++実装で速い）
    pts = gt_pts[0].detach().transpose(0, 1).cpu().numpy().astype(np.float64, copy=False)  # (N,3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    n = np.asarray(pcd.normals).astype(np.float32, copy=False)  # (N,3)
    normals = torch.from_numpy(n).to(gt_pts.device).transpose(0, 1).unsqueeze(0).contiguous()  # [1,3,N]
    return normals

def estimate_normals_pca(gt_pts: torch.Tensor, k: int = 16) -> torch.Tensor:
    """
    gt_pts : [B, 3, N]
    return : normals [B, 3, N]
    方針  : FAISSで各点のkNNを取り、PCAで法線を算出する。cdistは禁止。
    """
    assert gt_pts.ndim == 3
    B, _, N = gt_pts.shape

    # [B, N, 3]
    xyz = gt_pts.permute(0, 2, 1).contiguous()

    # 各点のkNN（自身も含まれる可能性があるので k+1 取って先頭を捨てる）
    knn_idx = _faiss_knn_idx(xyz, xyz, k + 1)  # [B, N, k+1]
    knn_idx = knn_idx[:, :, 1:]                # [B, N, k]

    # 近傍点取得: [B, N, k, 3]
    # knn_pts = torch.gather(
    #     xyz.unsqueeze(2).expand(-1, -1, N, -1),  # これは巨大になるので不可
    # )

    # reshape を使って安全に近傍点を集める
    # xyz_flat: [B*N, 3]
    xyz_flat = xyz.reshape(B * N, 3)

    # knn_idx_flat: [B*N*k]
    base = (torch.arange(B, device=xyz.device).view(B, 1, 1) * N)  # [B,1,1]
    knn_idx_flat = (knn_idx + base).reshape(-1)                    # [B*N*k]

    knn_pts = xyz_flat[knn_idx_flat].reshape(B, N, k, 3)           # [B,N,k,3]

    centroid = knn_pts.mean(dim=2, keepdim=True)                   # [B,N,1,3]
    diff = knn_pts - centroid                                      # [B,N,k,3]
    cov = diff.transpose(-1, -2) @ diff                            # [B,N,3,3]

    eigvals, eigvecs = torch.linalg.eigh(cov)                      # [B,N,3], [B,N,3,3]
    normals = eigvecs[..., 0]                                      # 最小固有値の固有ベクトル [B,N,3]
    normals = torch.nn.functional.normalize(normals, dim=-1)

    return normals.permute(0, 2, 1).contiguous()                   # [B,3,N]

def point2plane_loss(gen_pts: torch.Tensor,
                     gt_pts: torch.Tensor,
                     gt_normals=None,
                     k: int = 16,
                     reduction: str = "mean") -> torch.Tensor:
    """
    gen_pts: [B, 3, Ng]
    gt_pts : [B, 3, Nt]
    gt_normals: [B, 3, Nt] or None
    """
    assert gen_pts.ndim == 3 and gt_pts.ndim == 3
    B, _, Ng = gen_pts.shape
    _, _, Nt = gt_pts.shape

    # print(f"[P2P] B={B} Ng={Ng} Nt={Nt} k={k}")

    gt_xyz  = gt_pts.permute(0, 2, 1).contiguous()
    gen_xyz = gen_pts.permute(0, 2, 1).contiguous()

    # ★ ここが今回の肝：外から渡された法線を使う
    if gt_normals is None:
        gt_normals = estimate_normals_open3d(gt_pts, k=k)

    normals_xyz = gt_normals.permute(0, 2, 1).contiguous()

    nn_idx = _faiss_knn_idx(gen_xyz, gt_xyz, 1).squeeze(-1)

    base = (torch.arange(B, device=gen_pts.device).view(B, 1) * Nt)
    gt_flat = gt_xyz.reshape(B * Nt, 3)
    n_flat  = normals_xyz.reshape(B * Nt, 3)
    nn_flat = (nn_idx + base).reshape(-1)

    nn_gt = gt_flat[nn_flat].reshape(B, Ng, 3)
    nn_n  = n_flat[nn_flat].reshape(B, Ng, 3)

    diff = gen_xyz - nn_gt
    dist_plane = (diff * nn_n).sum(dim=2).abs()

    if reduction == "mean":
        return dist_plane.mean()
    elif reduction == "sum":
        return dist_plane.sum()
    else:
        return dist_plane


def _faiss_knn_idx(query_xyz_bnm: torch.Tensor, ref_xyz_brm: torch.Tensor, k: int, 
                   cpu_index = None, gpu_index = None) -> torch.Tensor:
    """
    query_xyz_bnm: [B, N, 3]  最近傍を探したい点群
    ref_xyz_brm  : [B, M, 3]  参照点群（ここから最近傍を取る）
    return       : [B, N, k]  参照点群側のインデックス（int64）
    注意: FAISSはCPU numpyを要求するのでref/queryはdetach->cpuに落として検索する。
          インデックスはGPUに戻して gather に使う。
    """
    res = faiss.StandardGpuResources()
    
    B, N, _ = query_xyz_bnm.shape
    _, M, _ = ref_xyz_brm.shape
    idx_list = []

    t0 = time.time()
    for b in range(B):
        ref_np = ref_xyz_brm[b].detach().cpu().numpy().astype(np.float32, copy=False)
        qry_np = query_xyz_bnm[b].detach().cpu().numpy().astype(np.float32, copy=False)

        cpu_index = faiss.IndexFlatL2(3)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.add(ref_np)
        _, I = gpu_index.search(qry_np, k)
        idx_list.append(torch.from_numpy(I).to(ref_xyz_brm.device, dtype=torch.long))

    return torch.stack(idx_list, dim=0)  # [B, N, k]

def knn_gpu(xyz: torch.Tensor, k: int):
    """
    xyz: (B, N, 3)
    return: idx (B, N, k)
    完全最近傍（精度100%）
    GPU並列版
    """
    B, N, _ = xyz.shape

    # 距離計算（GPU）
    dist = torch.cdist(xyz, xyz)  # (B,N,N)

    # 自分自身を除外するため大きな値を入れる
    eye = torch.eye(N, device=xyz.device).unsqueeze(0)
    dist = dist + eye * 1e9

    idx = dist.topk(k=k, largest=False)[1]
    return idx

def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(_normalize_point_cloud(p1), _normalize_point_cloud(p2))
    d1 = torch.mean(d1)
    d2 = torch.mean(d2)
    return (d1 + d2)

def _normalize_point_cloud(pc):
    # b, n, 3
    centroid = torch.mean(pc, dim=1, keepdim = True) # b, 1, 3
    pc = pc - centroid # b, n, 3
    furthest_distance = torch.max(torch.sqrt(torch.sum(pc**2, dim=-1, keepdim=True)), dim=1, keepdim=True)[0] # b, 1, 1
    pc = pc / furthest_distance
    return pc

def chamfer_l2_loss(gen_pts: torch.Tensor, gt_pts: torch.Tensor) -> torch.Tensor:
    """
    gen_pts, gt_pts: [B,3,N]
    """

    # RepKPU と同様に正規化
    gen = gen_pts.transpose(1, 2).contiguous()
    gt  = gt_pts.transpose(1, 2).contiguous()

    dist1, dist2, _, _ = chamfer_dist(gen, gt)
    return dist1.mean() + dist2.mean()

def psnr_loss(gen_pts: torch.Tensor, gt_pts: torch.Tensor, peak=1023.0) -> torch.Tensor:
    """
    PSNR = 10 * log10( peak^2 / MSE )
    Loss としては -PSNR を返す
    """

    gen = gen_pts.transpose(1, 2).contiguous()
    gt  = gt_pts.transpose(1, 2).contiguous()

    dist1, _, _, _ = chamfer_dist(gen, gt)
    mse = dist1.mean()

    psnr = 10.0 * torch.log10((peak ** 2) / (mse + 1e-8))
    return -psnr

def normal_consistency_loss(gen_pts, gt_pts, gt_normals=None, k=16):
    """
    gen_pts: [B,3,N]
    gt_pts : [B,3,N]
    """
    if gt_normals is None:
        gt_normals = estimate_normals_open3d(gt_pts, k=k)

    gen_normals = estimate_normals_open3d(gen_pts, k=k)

    gen_xyz = gen_pts.permute(0,2,1)
    gt_xyz  = gt_pts.permute(0,2,1)

    nn_idx = _faiss_knn_idx(gen_xyz, gt_xyz, 1).squeeze(-1)

    B, Ng = nn_idx.shape
    Nt = gt_pts.shape[2]

    base = (torch.arange(B, device=gen_pts.device).view(B,1) * Nt)
    gt_norm_flat = gt_normals.permute(0,2,1).reshape(B*Nt,3)
    nn_flat = (nn_idx + base).reshape(-1)

    nn_gt_norm = gt_norm_flat[nn_flat].reshape(B,Ng,3)
    gen_norm = gen_normals.permute(0,2,1)

    dot = (gen_norm * nn_gt_norm).sum(dim=2).abs()

    return (1 - dot).mean()

class ManifoldnessConstraint(nn.Module):
    """
    The Normal Consistency Constraint
    """
    def __init__(self, support=8, neighborhood_size=32):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=3, eps=1e-6)
        self.support = support
        self.neighborhood_size = neighborhood_size

        # resourceは1回でOK
        self.faiss_res = faiss.StandardGpuResources()

        # indexはメンバで持っても良いが、毎回resetが必須
        self.cpu_index = faiss.IndexFlatL2(3)
        self.gpu_index = faiss.index_cpu_to_gpu(self.faiss_res, 0, self.cpu_index)

    def faiss_knn(self, query_xyz_bnm: torch.Tensor, ref_xyz_brm: torch.Tensor, k: int) -> torch.Tensor:
        """
        注意: 毎step refが変わるので、必ず reset() → add() が必要。
        resetを忘れるとindexが増殖して結果が壊れる。
        """
        B, N, _ = query_xyz_bnm.shape
        idx_list = []

        for b in range(B):
            ref_np = ref_xyz_brm[b].detach().cpu().numpy().astype(np.float32, copy=False)
            qry_np = query_xyz_bnm[b].detach().cpu().numpy().astype(np.float32, copy=False)

            # ★必須：前回の点群を捨てる
            self.gpu_index.reset()
            self.gpu_index.add(ref_np)

            _, I = self.gpu_index.search(qry_np, k)
            idx_list.append(torch.from_numpy(I).to(ref_xyz_brm.device, dtype=torch.long))

        return torch.stack(idx_list, dim=0)

    def estimate_pointcloud_normals(self, xyz: torch.Tensor, neighborhood_size: int = 32) -> torch.Tensor:
        """
        xyz: (B, N, 3) または (B, 3, N)
        return: (B, N, 3)
        """
        assert xyz.ndim == 3

        if xyz.shape[1] == 3:
            xyz = xyz.permute(0, 2, 1).contiguous()  # (B,N,3)

        B, N, C = xyz.shape
        assert C == 3

        # kNN（自分自身を含むので +1 して除外）
        knn_idx = self.faiss_knn(xyz, xyz, neighborhood_size + 1)
        knn_idx = knn_idx[:, :, 1:]

        base = torch.arange(B, device=xyz.device).view(B,1,1) * N
        idx_flat = (knn_idx + base).reshape(-1)

        xyz_flat = xyz.reshape(B*N, 3)
        neighbors = xyz_flat[idx_flat].reshape(B, N, neighborhood_size, 3)

        centroid = neighbors.mean(dim=2, keepdim=True)
        diff = neighbors - centroid
        cov = diff.transpose(-1, -2) @ diff

        eigvals, eigvecs = torch.linalg.eigh(cov)
        normals = eigvecs[..., 0]
        normals = torch.nn.functional.normalize(normals, dim=-1)
        return normals

    def forward(self, xyz):
        xyz = xyz.permute(0,2,1).contiguous()  # (B,N,3)

        normals = self.estimate_pointcloud_normals(xyz, neighborhood_size=self.neighborhood_size)
        idx = self.faiss_knn(xyz, xyz, self.support)

        B, N, k = idx.shape
        base = torch.arange(B, device=xyz.device).view(B,1,1) * N
        idx_flat = (idx + base).reshape(-1)

        normals_flat = normals.reshape(B*N, 3)
        neighborhood = normals_flat[idx_flat].reshape(B, N, k, 3)

        cos_similarity = self.cos(neighborhood[:, :, 0, :].unsqueeze(2), neighborhood)
        penalty = 1 - cos_similarity
        penalty = penalty.std(-1)
        penalty = penalty.mean(-1)
        return penalty
    
import torch
import torch.nn.functional as F

def compute_fit_loss(sq_distances, conv_radius):
    """
    Args:
        sq_distances: (B, N, K, Nkp)
            squared distances between neighbor points and kernel points
        conv_radius: float
            convolution radius for normalization
    Returns:
        scalar loss
    """
    # sqrt to get Euclidean distance
    distances = torch.sqrt(sq_distances)  # (B, N, K, Nkp)

    # nearest neighbor distance for each kernel point
    min_dist, _ = torch.min(distances, dim=-2)  # (B, N, Nkp)

    # normalize
    loss = (min_dist / conv_radius) ** 2  # (B, N, Nkp)

    # L1 to zero
    fit_loss = F.l1_loss(loss, torch.zeros_like(loss))

    return fit_loss

def compute_rep_loss(deform_kp_pos, conv_radius, repulse_extent):
    """
    Args:
        deform_kp_pos: (B, 3, N, Nkp)
            deformed kernel point positions
        conv_radius: float
        repulse_extent: float
    Returns:
        scalar loss
    """

    B, _, N, Nkp = deform_kp_pos.shape

    # normalize
    norm_kp_pos = deform_kp_pos / conv_radius
    norm_kp_pos = norm_kp_pos.permute(0, 2, 3, 1).contiguous()
    norm_kp_pos = norm_kp_pos.view(-1, Nkp, 3)  # (B*N, Nkp, 3)

    total_loss = 0.0

    for i in range(Nkp):
        # exclude itself
        other = torch.cat(
            [norm_kp_pos[:, :i, :], norm_kp_pos[:, i+1:, :]],
            dim=1
        ).detach()

        # pairwise distance
        distances = torch.sqrt(
            torch.sum(
                (other - norm_kp_pos[:, i:i+1, :]) ** 2,
                dim=2
            )
        )

        # penalize if too close
        rep = torch.clamp_max(distances - repulse_extent, max=0.0) ** 2
        rep = torch.sum(rep, dim=1)

        total_loss += F.l1_loss(rep, torch.zeros_like(rep)) / Nkp

    return total_loss


def prune_count_loss(keep_prob, target_ratio):
    """
    keep_prob: (B,N)
    """
    B, N = keep_prob.shape
    mean_ratio = keep_prob.mean(dim=1)
    loss = ((mean_ratio - target_ratio) ** 2).mean()
    return loss


def prune_outlier_loss(keep_prob, density):
    """
    外れ点抑制
    density: (B,1,N) または (B,N)
    """
    if density.dim() == 3:
        density = density.squeeze(1)

    d_norm = density / (density.mean(dim=1, keepdim=True) + 1e-6)
    loss = (keep_prob * d_norm).mean()
    return loss