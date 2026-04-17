import math

import numpy as np
import torch
from einops import rearrange
from sklearn.neighbors import NearestNeighbors

from models.utils.pointcloud.utils_repkpu import FPS, index_points, normalize_point_cloud


def _to_numpy_xyz(pts_xyz):
    return rearrange(pts_xyz.squeeze(0), "c n -> n c").detach().cpu().numpy().astype(np.float32, copy=False)


def _to_numpy_centers(centers):
    return rearrange(centers.squeeze(0), "c m -> m c").detach().cpu().numpy().astype(np.float32, copy=False)


def _stable_patch_indices(owned_idx, knn_idx, patch_size, num_points):
    seen = np.zeros(num_points, dtype=np.bool_)
    merged = []

    for idx in owned_idx:
        idx = int(idx)
        if not seen[idx]:
            seen[idx] = True
            merged.append(idx)

    for idx in knn_idx:
        idx = int(idx)
        if not seen[idx]:
            seen[idx] = True
            merged.append(idx)
        if len(merged) >= patch_size:
            break

    if len(merged) < patch_size:
        raise RuntimeError(
            f"Failed to build a full patch: owned={len(owned_idx)}, gathered={len(merged)}, patch_size={patch_size}"
        )

    patch_idx = np.asarray(merged[:patch_size], dtype=np.int64)
    owned_set = set(int(x) for x in owned_idx.tolist())
    owned_mask = np.asarray([int(idx) in owned_set for idx in patch_idx], dtype=np.bool_)
    if not owned_mask.any():
        raise RuntimeError("A constructed patch does not contain any owned points.")
    return patch_idx, owned_mask


def _build_patch_assignment(pts_xyz, centers, patch_size):
    pts_np = _to_numpy_xyz(pts_xyz)
    centers_np = _to_numpy_centers(centers)
    num_points = pts_np.shape[0]

    if num_points == 0:
        raise ValueError("Patch extraction received an empty point set.")

    center_nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    center_nn.fit(centers_np)
    owner_patch = center_nn.kneighbors(pts_np, return_distance=False).reshape(-1)

    owner_count = np.bincount(owner_patch, minlength=centers_np.shape[0])
    max_owned = int(owner_count.max()) if owner_count.size > 0 else 0

    knn = NearestNeighbors(n_neighbors=patch_size, algorithm="auto")
    knn.fit(pts_np)
    knn_idx = knn.kneighbors(centers_np, return_distance=False)

    patch_indices = []
    owned_masks = []
    for patch_id in range(centers_np.shape[0]):
        owned_idx = np.nonzero(owner_patch == patch_id)[0]
        if owned_idx.size == 0:
            # Empty ownership is allowed; keep a context-only patch so nearby owned patches still have overlap.
            patch_idx = knn_idx[patch_id].astype(np.int64, copy=False)
            owned_mask = np.zeros((patch_idx.shape[0],), dtype=np.bool_)
        else:
            patch_idx, owned_mask = _stable_patch_indices(
                owned_idx=owned_idx,
                knn_idx=knn_idx[patch_id],
                patch_size=patch_size,
                num_points=num_points,
            )
        patch_indices.append(patch_idx)
        owned_masks.append(owned_mask)

    patch_indices = np.stack(patch_indices, axis=0)
    owned_masks = np.stack(owned_masks, axis=0)
    return owner_patch, owner_count, max_owned, patch_indices, owned_masks


def build_patch_info(input_pcd, args):
    if input_pcd.dim() != 3 or input_pcd.shape[0] != 1:
        raise ValueError(f"Patch mode expects input_pcd with shape [1, C, N], got {tuple(input_pcd.shape)}")

    pts_xyz = input_pcd[:, :3, :]
    pts_attr = input_pcd[:, 3:, :]
    num_points = pts_xyz.shape[-1]
    patch_size = int(min(max(int(args.num_points), 1), num_points))
    patch_rate = max(float(getattr(args, "patch_rate", 1.0)), 1.0)

    retry_count = max(int(getattr(args, "patch_cover_retry", 4)), 0)
    center_count = max(1, math.ceil(num_points / patch_size * patch_rate))

    last_owner_count = None
    for _ in range(retry_count + 1):
        centers = FPS(pts_xyz, center_count)
        owner_patch, owner_count, max_owned, patch_indices_np, owned_masks_np = _build_patch_assignment(
            pts_xyz,
            centers,
            patch_size=patch_size,
        )
        last_owner_count = owner_count
        if max_owned <= patch_size:
            break
        growth = max(1, math.ceil(center_count * (max_owned / patch_size - 1.0)))
        center_count = min(num_points, center_count + growth)
    else:
        raise RuntimeError(
            f"Failed to limit owned points per patch below patch_size={patch_size}. "
            f"last_max_owned={int(last_owner_count.max()) if last_owner_count is not None else 'n/a'}"
        )

    patch_input_idx = torch.from_numpy(patch_indices_np).to(device=input_pcd.device, dtype=torch.long)
    owned_input_mask = torch.from_numpy(owned_masks_np).to(device=input_pcd.device, dtype=torch.bool)

    patch_xyz = index_points(pts_xyz, patch_input_idx.unsqueeze(0)).squeeze(0)
    patch_xyz = rearrange(patch_xyz, "c p k -> p c k").contiguous()

    if pts_attr.shape[1] > 0:
        patch_attr = index_points(pts_attr, patch_input_idx.unsqueeze(0)).squeeze(0)
        patch_attr = rearrange(patch_attr, "c p k -> p c k").contiguous()
    else:
        patch_attr = pts_attr.new_empty((patch_xyz.shape[0], 0, patch_xyz.shape[-1]))

    patch_xyz_norm, centroid, furthest_distance = normalize_point_cloud(patch_xyz)

    return {
        "patch_xyz": patch_xyz_norm,
        "patch_attr": patch_attr,
        "patch_centroid": centroid,
        "patch_scale": furthest_distance,
        "patch_input_idx": patch_input_idx,
        "owned_input_mask": owned_input_mask,
        "num_patches": patch_xyz.shape[0],
        "num_input_points": num_points,
    }


def denormalize_patch_output(gen_patch, centroid, scale):
    gen_xyz = centroid + gen_patch[:, :3, :] * scale
    if gen_patch.shape[1] <= 3:
        return gen_xyz
    return torch.cat([gen_xyz, gen_patch[:, 3:, :]], dim=1)


def patch_info_to_cpu(patch_info):
    cached = {}
    for key, value in patch_info.items():
        if torch.is_tensor(value):
            cached[key] = value.detach().cpu()
        else:
            cached[key] = value
    return cached


def patch_info_to_device(patch_info, device, dtype=None):
    moved = {}
    for key, value in patch_info.items():
        if torch.is_tensor(value):
            to_kwargs = {"device": device}
            if dtype is not None and value.is_floating_point():
                to_kwargs["dtype"] = dtype
            moved[key] = value.to(non_blocking=True, **to_kwargs)
        else:
            moved[key] = value
    return moved


def merge_patch_outputs(patch_info, patch_outputs, device, dtype):
    merged_points = []
    merged_weights = []
    full_out_label = torch.zeros(
        1,
        1,
        patch_info["num_input_points"],
        device=device,
        dtype=dtype,
    )
    label_filled = torch.zeros(patch_info["num_input_points"], device=device, dtype=torch.bool)

    for patch_entry in patch_outputs:
        if "selected_pts" in patch_entry:
            selected_pts = patch_entry.get("selected_pts")
            selected_w = patch_entry.get("selected_w")
            if selected_pts is not None and selected_pts.numel() > 0:
                merged_points.append(selected_pts)
                if selected_w is not None:
                    merged_weights.append(selected_w)
        else:
            patch_id = int(patch_entry["patch_id"])
            pts_out = patch_entry["pts_out"]
            final_w = patch_entry.get("final_w")
            patch_meta = patch_entry["patch_meta"]

            patch_input_idx = patch_info["patch_input_idx"][patch_id]
            owned_input_mask = patch_info["owned_input_mask"][patch_id]

            anchor_idx_local = patch_meta["anchor_idx_local"].clamp_(0, patch_input_idx.shape[0] - 1)
            valid_mask = patch_meta["output_valid_mask"]
            owned_output_mask = owned_input_mask.index_select(0, anchor_idx_local)
            select_mask = valid_mask & owned_output_mask

            if select_mask.any():
                merged_points.append(pts_out[:, select_mask])
                if final_w is not None:
                    merged_weights.append(final_w[:, select_mask])

        fallback_pts = patch_entry.get("fallback_pts")
        fallback_w = patch_entry.get("fallback_w")
        if fallback_pts is not None and fallback_pts.numel() > 0:
            merged_points.append(fallback_pts)
            if fallback_w is not None:
                merged_weights.append(fallback_w)

        owned_global_idx = patch_entry.get("owned_global_idx")
        owned_out_label = patch_entry.get("owned_out_label")
        if owned_global_idx is not None and owned_out_label is not None and owned_global_idx.numel() > 0:
            full_out_label[0, 0, owned_global_idx] = owned_out_label.to(dtype=dtype)
            label_filled[owned_global_idx] = True
        else:
            patch_id = int(patch_entry["patch_id"])
            patch_meta = patch_entry["patch_meta"]
            out_label = patch_meta.get("out_label")
            if out_label is not None:
                patch_input_idx = patch_info["patch_input_idx"][patch_id]
                owned_input_mask = patch_info["owned_input_mask"][patch_id]
                owned_local_idx = torch.nonzero(owned_input_mask, as_tuple=False).flatten()
                if owned_local_idx.numel() > 0:
                    global_owned_idx = patch_input_idx.index_select(0, owned_local_idx)
                    full_out_label[0, 0, global_owned_idx] = out_label.index_select(0, owned_local_idx).to(dtype=dtype)
                    label_filled[global_owned_idx] = True

    if not merged_points:
        raise RuntimeError("No patch outputs were selected during merge.")
    if not torch.all(label_filled):
        raise RuntimeError("Merged patch labels do not cover every input point.")

    merged_pts = torch.cat(merged_points, dim=1).unsqueeze(0).contiguous()
    merged_final_w = None
    if merged_weights:
        merged_final_w = torch.cat(merged_weights, dim=1).unsqueeze(0).contiguous()

    return merged_pts, merged_final_w, full_out_label
