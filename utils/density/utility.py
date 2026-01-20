"""
ユーティリティ関数
"""

from pathlib import Path
import numpy as np
from typing import Tuple

def ensure_dir(path: Path) -> None:
    """親ディレクトリを作成する（存在すれば何もしない）"""
    path.parent.mkdir(parents=True, exist_ok=True)


def aabb_of_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """点群のAABBを返す
    戻り値: (min_xyz[3], max_xyz[3])
    """
    pmin = np.min(points, axis=0)
    pmax = np.max(points, axis=0)
    return pmin, pmax


def voxel_index(points: np.ndarray, origin: np.ndarray, voxel_size: np.ndarray) -> np.ndarray:
    """各点に対するボクセルインデックス(整数xyz)を計算する
    ・原点originはAABBのmin等を使う
    ・voxel_sizeは各軸の長さ
    """
    rel = (points - origin) / voxel_size
    idx = np.floor(rel).astype(np.int64)
    return idx


def voxel_center_from_index(index_ijk: np.ndarray, origin: np.ndarray, voxel_size: np.ndarray) -> np.ndarray:
    """ボクセルの中心座標をインデックスから求める"""
    return origin + (index_ijk.astype(np.float64) + 0.5) * voxel_size

def derive_output_prefix(input_path: Path, out_root: Path) -> Path:
    """入力パスから出力接頭辞を自動生成する
    規則：
      - 入力パスに "PartAnnotation" が含まれていれば、そこから下位の相対パスを out_root 配下にミラーする
      - 含まれなければ、ファイル名だけを out_root 直下に置く
    例：
      in:  .\\pcd-dataset\\PartAnnotation\\0269\\...\\foo.pcd
      out: .\\out\\PartAnnotation\\0269\\...\\foo.pcd
    """
    p = Path(input_path)
    parts = p.parts
    anchor = "PartAnnotation"
    if anchor in parts:
        idx = parts.index(anchor)
        sub = Path(*parts[idx:])  # "PartAnnotation/..." を保持
    else:
        sub = Path(p.name)        # アンカーが無い場合はファイル名のみ
    return Path(out_root) / sub