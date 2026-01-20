# ==========================
# CLI
# ==========================

import argparse
from pathlib import Path
from typing import List
from utility import aabb_of_points
import numpy as np
from typing import Tuple

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="点群密度解析システム（Voxel/KDE）")
    p.add_argument("--input", type=Path, required=True, help="入力点群ファイル（.pcd/.plyなど）")
    p.add_argument("--mode", choices=["voxel", "kde"], required=True, help="密度計算モード")
    p.add_argument("--voxel-size", nargs=3, type=float, required=True, metavar=("SX", "SY", "SZ"), help="ボクセルサイズ[m]")
    # 未指定なら detect.py 側で入力から自動生成
    p.add_argument("--output-prefix", type=Path, required=False, default=None,
                help="出力の接頭辞（未指定時は入力パスから自動生成）")

    # 既定の出力ルート（例: .\\out）
    p.add_argument("--out-root", type=Path, default=Path("out"),
                help="入力パスをこの直下にミラーして出力する（既定: out）")


    # voxel
    p.add_argument("--voxel-origin", nargs=3, type=float, default=None, metavar=("OX", "OY", "OZ"), help="ボクセル原点（未指定はAABB最小）")

    # kde
    p.add_argument("--grid-U", type=int, default=16, help="RoIグリッド解像度（U×U×U）")
    p.add_argument("--radius", type=float, default=0.3, help="Ball Query半径[m]")
    p.add_argument("--sigma", type=float, default=0.25, help="KDE帯域幅σ[m]")
    p.add_argument("--roi", nargs='+', default=["auto"], help="RoI: auto または xmin ymin zmin xmax ymax zmax")
    p.add_argument("--export-ply", action="store_true", help="可視化用PLYを書き出す")

    return p.parse_args()


def parse_roi(arg: List[str], points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """--roiの解釈"""
    if len(arg) == 1 and arg[0].lower() == "auto":
        return aabb_of_points(points)
    if len(arg) != 6:
        raise ValueError("--roi は 'auto' または 6つの数値を与えること")
    vals = np.asarray(list(map(float, arg)), dtype=np.float64)
    roi_min = vals[:3]
    roi_max = vals[3:]
    if np.any(roi_max <= roi_min):
        raise ValueError("--roi の各軸で max > min を満たす必要がある")
    return roi_min, roi_max