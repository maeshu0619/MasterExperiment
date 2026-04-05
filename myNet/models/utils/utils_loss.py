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

def chamfer_l2_loss(gen_pts: torch.Tensor, gt_pts: torch.Tensor, final_w: torch.Tensor = None) -> torch.Tensor:
    """
    gen_pts, gt_pts: [B,3,N]
    """
    if final_w is None:
        # RepKPU と同様に正規化
        gen = gen_pts.transpose(1, 2).contiguous()
        gt  = gt_pts.transpose(1, 2).contiguous()

        dist1, dist2, _, _ = chamfer_dist(gen, gt)
        return dist1.mean() + dist2.mean()
    elif final_w is not None:
        gen = gen_pts.transpose(1, 2).contiguous()  # [B,N,3]
        gt  = gt_pts.transpose(1, 2).contiguous()   # [B,M,3]

        dist1, dist2, _, _ = chamfer_dist(gen, gt)  # dist1: [B,N], dist2: [B,M]

        if final_w.dim() == 3:
            w = final_w.squeeze(1)  # [B,N]
        else:
            w = final_w
        w = w.clamp(0.0, 1.0).to(dist1.dtype)

        # gen->gt を重み付き平均、gt->gen は従来通り平均（ここは最小実装）
        loss1 = (dist1 * w).sum() / (w.sum() + 1e-12)
        loss2 = dist2.mean()
        return loss1 + loss2

def chamfer_prune_soft_loss(
    gen_pts: torch.Tensor,
    gt_pts: torch.Tensor,
    final_w: torch.Tensor,
    detach_dist: bool = True
) -> torch.Tensor:
    """
    gen_pts: [B, 3, N_gen]   最終出力点群
    gt_pts:  [B, 3, N_gt]    入力点群 = GT
    keep_w_full: [B, 1, N_gt]  Pruning前全点に対する keep重み
    """

    gen = gen_pts.transpose(1, 2).contiguous()   # [B, N_gen, 3]
    gt  = gt_pts.transpose(1, 2).contiguous()    # [B, N_gt, 3]

    dist1, dist2, _, _ = chamfer_dist(gen, gt)   # dist2: [B, N_gt]

    if detach_dist:
        dist2 = dist2.detach()

    final_w = final_w.clamp(0.0, 1.0).to(dist2.dtype)

    # 落とした点ほど重く見る
    drop_w = 1.0 - final_w

    loss = (dist2 * drop_w).sum() / (drop_w.sum() + 1e-12)
    return loss

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

def compute_d2_psnr(
    ref: torch.Tensor,
    rec: torch.Tensor,
    k_normal: int = 16,
    peak_mode: str = "union_bbox_diag",
    final_w: torch.Tensor = None
):
    """
    ref, rec: [B,3,N]
    final_w:
        - None      : 従来計算
        - [B,1,N_ref] or [B,N_ref]
          ref->rec 側の d2 を重み付き平均する
          chamfer_l2_loss と同じ思想
    """

    assert isinstance(ref, torch.Tensor) and isinstance(rec, torch.Tensor)
    assert ref.ndim == 3 and rec.ndim == 3

    device = ref.device

    ref_xyz = ref.permute(0, 2, 1).contiguous()  # [B,N_ref,3]
    rec_xyz = rec.permute(0, 2, 1).contiguous()  # [B,N_rec,3]

    B, N_ref, _ = ref_xyz.shape
    _, N_rec, _ = rec_xyz.shape

    if N_ref == 0 or N_rec == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=device)

    ref_normals = estimate_normals_open3d(ref, k=k_normal).permute(0, 2, 1).contiguous()
    rec_normals = estimate_normals_open3d(rec, k=k_normal).permute(0, 2, 1).contiguous()

    # ref -> rec
    ref_to_rec_idx = _faiss_knn_idx(ref_xyz, rec_xyz, 1).squeeze(-1)  # [B,N_ref]
    # rec -> ref
    rec_to_ref_idx = _faiss_knn_idx(rec_xyz, ref_xyz, 1).squeeze(-1)  # [B,N_rec]

    ref_base = (torch.arange(B, device=device).view(B, 1) * N_ref)
    rec_base = (torch.arange(B, device=device).view(B, 1) * N_rec)

    ref_xyz_flat = ref_xyz.reshape(B * N_ref, 3)
    rec_xyz_flat = rec_xyz.reshape(B * N_rec, 3)
    ref_n_flat   = ref_normals.reshape(B * N_ref, 3)
    rec_n_flat   = rec_normals.reshape(B * N_rec, 3)

    # -------------------------
    # 1. ref -> rec の d2
    # -------------------------
    ref_to_rec_flat = (ref_to_rec_idx + rec_base).reshape(-1)
    nn_rec_xyz = rec_xyz_flat[ref_to_rec_flat].reshape(B, N_ref, 3)
    nn_rec_n   = rec_n_flat[ref_to_rec_flat].reshape(B, N_ref, 3)

    diff_bwd = ref_xyz - nn_rec_xyz
    proj_bwd = (diff_bwd * nn_rec_n).sum(dim=2)
    bwd_sq   = proj_bwd * proj_bwd

    if final_w is not None:
        if final_w.dim() == 3:
            w = final_w.squeeze(1)   # [B,N_ref]
        else:
            w = final_w

        w = w.clamp(0.0, 1.0).to(bwd_sq.dtype)

        if w.shape[1] != N_ref:
            raise ValueError(
                f"final_w shape mismatch: got {tuple(w.shape)}, expected second dim = {N_ref}"
            )

        bwd_mse = (bwd_sq * w).sum() / (w.sum() + 1e-12)
    else:
        bwd_mse = bwd_sq.mean()

    # -------------------------
    # 2. rec -> ref の d2
    # -------------------------
    rec_to_ref_flat = (rec_to_ref_idx + ref_base).reshape(-1)
    nn_ref_xyz = ref_xyz_flat[rec_to_ref_flat].reshape(B, N_rec, 3)
    nn_ref_n   = ref_n_flat[rec_to_ref_flat].reshape(B, N_rec, 3)

    diff_fwd = rec_xyz - nn_ref_xyz
    proj_fwd = (diff_fwd * nn_ref_n).sum(dim=2)
    fwd_sq   = proj_fwd * proj_fwd

    fwd_mse = fwd_sq.mean()

    mse = 0.5 * (bwd_mse + fwd_mse)

    if mse <= 1e-30:
        return torch.tensor(float("inf"), dtype=torch.float32, device=device)

    if peak_mode == "union_bbox_diag":
        all_pts = torch.cat([ref_xyz, rec_xyz], dim=1)
    else:
        all_pts = ref_xyz

    mins = all_pts.amin(dim=1)
    maxs = all_pts.amax(dim=1)
    peak = torch.linalg.norm(maxs - mins, dim=1)
    peak2 = (peak * peak).mean()

    psnr = 10.0 * torch.log10(peak2 / (mse + 1e-8))
    return psnr

# def compute_d2_psnr(ref, rec, k_normal=16, peak_mode="union_bbox_diag"):
#     """
#     ref, rec:
#         - torch.Tensor [B,3,N] または [1,3,N]
#         - numpy (N,3)
#         - str (PLY path)

#     戻り値:
#         torch scalar（deviceは入力Tensorに合わせる）
#     """

#     # -------------------------
#     # 1. Tensorをnumpyへ安全に変換
#     # -------------------------
#     def _to_numpy_points(x):
#         if isinstance(x, torch.Tensor):
#             # [B,3,N] -> (N,3)
#             if x.ndim == 3:
#                 x = x[0]
#             x = x.transpose(0, 1).contiguous()
#             return x.detach().cpu().numpy().astype(np.float64)
#         elif isinstance(x, str):
#             pcd = o3d.io.read_point_cloud(x)
#             return np.asarray(pcd.points, dtype=np.float64)
#         else:
#             return np.asarray(x, dtype=np.float64)

#     ref_points = _to_numpy_points(ref)
#     rec_points = _to_numpy_points(rec)

#     if ref_points.shape[0] == 0 or rec_points.shape[0] == 0:
#         return torch.tensor(0.0, device=ref.device if isinstance(ref, torch.Tensor) else "cpu")

#     # -------------------------
#     # 2. 法線推定（Open3D）
#     # -------------------------
#     def _estimate_normals_unit(points):
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points)
#         pcd.estimate_normals(
#             search_param=o3d.geometry.KDTreeSearchParamKNN(knn=int(k_normal))
#         )
#         n = np.asarray(pcd.normals, dtype=np.float64)
#         norm = np.linalg.norm(n, axis=1, keepdims=True)
#         return n / (norm + 1e-12)

#     ref_normals = _estimate_normals_unit(ref_points)
#     rec_normals = _estimate_normals_unit(rec_points)

#     # -------------------------
#     # 3. KDTree
#     # -------------------------
#     ref_pcd = o3d.geometry.PointCloud()
#     ref_pcd.points = o3d.utility.Vector3dVector(ref_points)
#     rec_pcd = o3d.geometry.PointCloud()
#     rec_pcd.points = o3d.utility.Vector3dVector(rec_points)

#     ref_kd = o3d.geometry.KDTreeFlann(ref_pcd)
#     rec_kd = o3d.geometry.KDTreeFlann(rec_pcd)

#     # forward
#     fwd_sq = []
#     for p in rec_points:
#         _, idx, _ = ref_kd.search_knn_vector_3d(p, 1)
#         j = idx[0]
#         diff = p - ref_points[j]
#         dist = np.dot(diff, ref_normals[j])
#         fwd_sq.append(dist * dist)

#     # backward
#     bwd_sq = []
#     for p in ref_points:
#         _, idx, _ = rec_kd.search_knn_vector_3d(p, 1)
#         j = idx[0]
#         diff = p - rec_points[j]
#         dist = np.dot(diff, rec_normals[j])
#         bwd_sq.append(dist * dist)

#     mse = 0.5 * (np.mean(fwd_sq) + np.mean(bwd_sq))
#     if mse <= 1e-30:
#         return torch.tensor(float("inf"), device=ref.device if isinstance(ref, torch.Tensor) else "cpu")

#     # -------------------------
#     # 4. peak計算
#     # -------------------------
#     if peak_mode == "union_bbox_diag":
#         all_pts = np.vstack([ref_points, rec_points])
#     else:
#         all_pts = ref_points

#     mins = all_pts.min(axis=0)
#     maxs = all_pts.max(axis=0)
#     peak = np.linalg.norm(maxs - mins)

#     psnr = 10.0 * np.log10((peak * peak) / mse)

#     # torch scalarとして返す
#     device = ref.device if isinstance(ref, torch.Tensor) else "cpu"
#     return torch.tensor(psnr, dtype=torch.float32, device=device)