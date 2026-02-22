#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python Translate/ply2pcd_3DGS.py --input_ply ../Dataset/GPU/ptcloud_hd00000500.ply --out_dir out_convert --pcd_ascii  --voxel 0.0 
"""

import argparse
import os
import numpy as np

import open3d as o3d
from plyfile import PlyData, PlyElement


# ---- 3DGS (spherical harmonics) helper ----
# Inria gaussian-splatting uses SH basis constants; DC term often uses C0 = 0.28209479177387814
C0 = 0.28209479177387814

def rgb01_to_sh_dc(rgb01: np.ndarray) -> np.ndarray:
    """
    Convert RGB in [0,1] to SH DC coefficient used by many 3DGS implementations:
      sh_dc = (rgb - 0.5) / C0
    """
    return (rgb01 - 0.5) / C0


def read_ply_xyz_rgb(ply_path: str):
    """
    Read ASCII/Binary PLY with vertex properties x,y,z and red,green,blue.
    Returns:
      xyz: (N,3) float32
      rgb01: (N,3) float32 in [0,1]
    """
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data

    # xyz
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)

    # color: could be uchar 0-255 or float 0-1 (handle both)
    r = v["red"].astype(np.float32)
    g = v["green"].astype(np.float32)
    b = v["blue"].astype(np.float32)
    rgb = np.vstack([r, g, b]).T

    if rgb.max() > 1.0:
        rgb01 = (rgb / 255.0).clip(0.0, 1.0).astype(np.float32)
    else:
        rgb01 = rgb.clip(0.0, 1.0).astype(np.float32)

    return xyz, rgb01


def write_pcd(xyz: np.ndarray, rgb01: np.ndarray, out_pcd: str, ascii: bool = True):
    """
    Write point cloud to .pcd via Open3D.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb01.astype(np.float64))
    o3d.io.write_point_cloud(out_pcd, pcd, write_ascii=ascii)
    return out_pcd


def write_3dgs_init_ply(
    xyz: np.ndarray,
    rgb01: np.ndarray,
    out_ply: str,
    opacity: float = 0.1,
    scale_log: float = -3.0,
):
    """
    Create a "3D Gaussian Splatting initial gaussians" PLY.

    Properties (common in Inria gaussian-splatting trained-model PLY):
      x y z
      nx ny nz
      f_dc_0 f_dc_1 f_dc_2
      opacity
      scale_0 scale_1 scale_2
      rot_0 rot_1 rot_2 rot_3

    Notes:
      - f_dc_* are SH DC coefficients. We fill from rgb.
      - normals are set to 0.
      - opacity is constant default.
      - scale_* are set in log-space constant (exp(scale) becomes actual scale in many impls).
      - rot_* is identity quaternion (1,0,0,0).
    """
    n = xyz.shape[0]
    sh_dc = rgb01_to_sh_dc(rgb01).astype(np.float32)

    # pack structured array
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    data = np.empty(n, dtype=dtype)

    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["nx"], data["ny"], data["nz"] = 0.0, 0.0, 0.0
    data["f_dc_0"], data["f_dc_1"], data["f_dc_2"] = sh_dc[:, 0], sh_dc[:, 1], sh_dc[:, 2]
    data["opacity"] = np.float32(opacity)

    # log-scales
    data["scale_0"] = np.float32(scale_log)
    data["scale_1"] = np.float32(scale_log)
    data["scale_2"] = np.float32(scale_log)

    # identity quaternion (w,x,y,z) style. Many 3DGS impls store rot_0..3 like this.
    data["rot_0"] = np.float32(1.0)
    data["rot_1"] = np.float32(0.0)
    data["rot_2"] = np.float32(0.0)
    data["rot_3"] = np.float32(0.0)

    el = PlyElement.describe(data, "vertex")
    PlyData([el], text=True).write(out_ply)
    return out_ply


def maybe_downsample_open3d(xyz: np.ndarray, rgb01: np.ndarray, voxel_size: float):
    """
    Optional voxel downsample (keeps color).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb01.astype(np.float64))
    pcd_ds = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    xyz_ds = np.asarray(pcd_ds.points).astype(np.float32)
    rgb_ds = np.asarray(pcd_ds.colors).astype(np.float32)
    return xyz_ds, rgb_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_ply", required=True, help="input .ply (xyz + rgb)")
    ap.add_argument("--out_dir", default="out_convert", help="output directory")
    ap.add_argument("--pcd_ascii", action="store_true", help="write PCD as ASCII (default: binary)")
    ap.add_argument("--voxel", type=float, default=0.0, help="voxel downsample size (0 disables)")
    ap.add_argument("--opacity", type=float, default=0.1, help="default opacity for 3DGS init ply")
    ap.add_argument("--scale_log", type=float, default=-3.0, help="default log-scale for 3DGS init ply")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    xyz, rgb01 = read_ply_xyz_rgb(args.input_ply)

    if args.voxel and args.voxel > 0:
        xyz, rgb01 = maybe_downsample_open3d(xyz, rgb01, args.voxel)

    # 1) PCD
    out_pcd = os.path.join(args.out_dir, "output.pcd")
    write_pcd(xyz, rgb01, out_pcd, ascii=bool(args.pcd_ascii))

    # 2) 3DGS init PLY
    out_gs_ply = os.path.join(args.out_dir, "gaussians_init.ply")
    write_3dgs_init_ply(
        xyz, rgb01, out_gs_ply,
        opacity=float(args.opacity),
        scale_log=float(args.scale_log),
    )

    print("Done.")
    print("PCD:", out_pcd)
    print("3DGS init PLY:", out_gs_ply)
    print("Points:", xyz.shape[0])


if __name__ == "__main__":
    main()
