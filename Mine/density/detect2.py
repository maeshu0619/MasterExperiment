#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
点群密度解析システム（Voxel単位／パッチ（RoIグリッド）×KDE）

・入力：.pcd（ASCII/BinaryはOpen3Dに任せる）
・出力：
  - Voxel密度: CSV（各ボクセルの中心座標・カウント・密度）、NPZ（辞書形式）
  - KDE密度  : CSV（各グリッド点の座標・近傍点数・KDE密度）、NPZ
  - オプションで可視化用PLY（密度を正規化して色付け）

・ターミナル上でのデバッグ例
python density/detect.py --input .\pcd-dataset\PartAnnotation\02691156\1a04e3eab45ca15dd86060f189eb133.pcd --mode voxel --voxel-size 0.05 0.05 0.05 --output-prefix .\out\voxel --export-ply

"""

from __future__ import annotations

import numpy as np
import pathlib

from IO import PointCloudIO
from cal_den import VoxelDensityCalculator
from csv_npz import save_voxel_csv, save_kde_csv
from kde import PDVKDEDensityCalculator
from CLI import parse_args, parse_roi
from utility import derive_output_prefix  # ← 追加

import open3d as o3d

def export_lowest_density_voxels(res, k: int, output_prefix) -> "pathlib.Path":
    """
    最も密度の低い順にK個のボクセルを抽出し、CSVで保存する。
    出力列: rank, i, j, k, density, count, cx, cy, cz
    ※ ボクセル番号は index_ijk の (i,j,k)
    """
    import numpy as np
    from pathlib import Path

    # 抽出と並べ替え（密度昇順、安定ソート）
    order = np.argsort(res.density, kind="stable")[:k]
    ijk    = res.index_ijk[order]
    dens   = res.density[order]
    cnts   = res.counts[order]
    ctrs   = res.centers[order]

    # 保存パス（<prefix>_lowest10.csv）
    out_csv = Path(output_prefix).with_name(Path(output_prefix).stem + f"_lowest{k}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 書き出し
    import csv
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "i", "j", "k", "density", "count", "cx", "cy", "cz"])
        for r, (v, d, c, p) in enumerate(zip(ijk, dens, cnts, ctrs), start=1):
            w.writerow([r, int(v[0]), int(v[1]), int(v[2]), f"{float(d):.8e}", int(c),
                        f"{float(p[0]):.6f}", f"{float(p[1]):.6f}", f"{float(p[2]):.6f}"])

    # ついでに標準出力にも要約を出す（任意）
    print("[Voxel] 低密度ボクセルTop{}:".format(k))
    for r, (v, d) in enumerate(zip(ijk, dens), start=1):
        print(f"  #{r:02d} -> (i,j,k)=({v[0]},{v[1]},{v[2]}), density={float(d):.8e}")

    return out_csv


def main() -> None:
    args = parse_args()

    # 出力接頭辞（--output-prefix 未指定時は入力から自動生成）
    output_prefix = args.output_prefix or derive_output_prefix(args.input, args.out_root)

    # 入力読み込み
    pts = PointCloudIO.load_points(args.input)

    # モード分岐
    if args.mode == "voxel":
        calc = VoxelDensityCalculator(tuple(args.voxel_size))
        origin = np.array(args.voxel_origin, dtype=np.float64) if args.voxel_origin is not None else None
        res = calc.compute(pts, origin=origin)
        csv_path = save_voxel_csv(output_prefix, res)
        print(f"[Voxel] CSV: {csv_path}")
        low10 = export_lowest_density_voxels(res, 10, output_prefix)
        print(f"[Voxel] 低密度Top10 CSV: {low10}")

        if args.export_ply:
            # ボクセル中心に密度を割り当ててPLY
            ply_path = output_prefix.with_suffix(".ply")
            PointCloudIO.save_points_with_scalar(res.centers, res.density, ply_path)
            print(f"[Voxel] PLY: {ply_path}")

    elif args.mode == "kde":
        roi_min, roi_max = parse_roi(args.roi, pts)
        calc = PDVKDEDensityCalculator(tuple(args.voxel_size), args.grid_U, args.radius, args.sigma)
        res = calc.compute(pts, roi=(roi_min, roi_max))
        csv_path = save_voxel_csv(output_prefix, res)
        print(f"[KDE] CSV: {csv_path}")
        low10 = export_lowest_density_voxels(res, 10, output_prefix)
        print(f"[Voxel] 低密度Top10 CSV: {low10}")

        if args.export_ply:
            # グリッド点にKDE値を割り当ててPLY
            ply_path = output_prefix.with_suffix(".ply")
            PointCloudIO.save_points_with_scalar(res.grid_points, res.kde_values, ply_path)
            print(f"[KDE] PLY: {ply_path}")

    else:  # pragma: no cover
        raise AssertionError("到達しない分岐")


if __name__ == "__main__":
    main()
