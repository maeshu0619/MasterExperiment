# ==========================
# CSV/NPZ書き出し
# ==========================

import csv
import numpy as np
from pathlib import Path
from cal_den import VoxelDensityResult
from utility import ensure_dir
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from kde import KDEGridResult

def save_voxel_csv(prefix: Path, res: VoxelDensityResult) -> Path:
    csv_path = prefix.with_suffix(".csv")
    ensure_dir(csv_path)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cx", "cy", "cz", "count", "density", "i", "j", "k"])  # ヘッダ
        for (c, cnt, den, ijk) in zip(res.centers, res.counts, res.density, res.index_ijk):
            w.writerow([f"{c[0]:.6f}", f"{c[1]:.6f}", f"{c[2]:.6f}", int(cnt), f"{den:.8e}", int(ijk[0]), int(ijk[1]), int(ijk[2])])
    # NPZ（プログラム連携用）
    npz_path = prefix.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        centers=res.centers,
        counts=res.counts,
        density=res.density,
        index_ijk=res.index_ijk,
        voxel_size=res.voxel_size,
        origin=res.origin,
    )
    return csv_path


def save_kde_csv(prefix: Path, res: "KDEGridResult") -> Path:
    csv_path = prefix.with_suffix(".csv")
    ensure_dir(csv_path)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gx", "gy", "gz", "neighbors", "kde"])
        for (g, n, k) in zip(res.grid_points, res.neighbor_counts, res.kde_values):
            w.writerow([f"{g[0]:.6f}", f"{g[1]:.6f}", f"{g[2]:.6f}", int(n), f"{k:.8e}"])
    # NPZ
    npz_path = prefix.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        grid_points=res.grid_points,
        kde_values=res.kde_values,
        neighbor_counts=res.neighbor_counts,
        radius=res.radius,
        sigma=res.sigma,
        grid_U=res.grid_U,
        roi_min=res.roi_min,
        roi_max=res.roi_max,
    )
    return csv_path