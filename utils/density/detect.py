#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
目的：指定PCDを読み込み、ボクセル密度計算やKDE密度計算を行い、結果をCSVとPLYで出力する
備考：
- ボクセル密度計算は指定ボクセルサイズでグリッド化し、各ボクセルの中心に密度を割り当てる
- KDE密度計算は指定グリッドサイズでグリッド化し、各グリッド点にKDE値を割り当てる
- 出力はCSV形式とPLY形式で保存可能

・ターミナル上でのデバッグ例
python density/detect.py --input "F:\Experiments\MasterEx\pcd-dataset\PU-GAN\complex\AncientTurtl_aligned.pcd" --mode voxel --voxel-size 0.05 0.05 0.05 --output-prefix .\out\voxel --export-ply
"""

from __future__ import annotations

import numpy as np
from IO import PointCloudIO
from cal_den import VoxelDensityCalculator
from csv_npz import save_voxel_csv
from kde import PDVKDEDensityCalculator
from CLI import parse_args, parse_roi
from utility import derive_output_prefix

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
        if args.export_ply:
            # グリッド点にKDE値を割り当ててPLY
            ply_path = output_prefix.with_suffix(".ply")
            PointCloudIO.save_points_with_scalar(res.grid_points, res.kde_values, ply_path)
            print(f"[KDE] PLY: {ply_path}")

    else:  # pragma: no cover
        raise AssertionError("到達しない分岐")

if __name__ == "__main__":
    main()