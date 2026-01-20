# -*- coding: utf-8 -*-
"""
メッシュ可視化ツール（Alpha Shape / Voxel Box）
- 入力: voxel.ply（点群。色は密度の正規化グレースケール）
- オプション: voxel.npz（centers, voxel_size 等が入っていれば、箱メッシュを作る）

使い方例:
  # Alpha Shape（推奨。PLYだけでOK）
  python density/visual/display.py --input F:\Experiments\MasterEx\out\voxel.ply --mode alpha --save out\voxel_alpha_mesh.ply

  # Voxel Box（NPZを併用。密度の段差を立体で強調）
  python density/visual/display.py --input F:\Experiments\MasterEx\out\voxel.ply --mode voxelbox --npz F:\Experiments\MasterEx\out\voxel.npz --box-scale 0.95 --save out\voxel_boxes.ply
python density/visual/display.py --input "F:\Experiments\MasterEx\out\PartAnnotation\02691156\1a04e3eab45ca15dd86060f189eb133.ply" --mode voxelbox --npz "F:\Experiments\MasterEx\out\PartAnnotation\02691156\1a04e3eab45ca15dd86060f189eb133.npz" --box-scale 0.95 --save out\voxel_boxes.ply
  
  python density/visual/display.py --input "F:\Experiments\MasterEx\down_data\partial_voxel_fps.pcd" --mode voxelbox --npz F:\Experiments\MasterEx\out\voxel.npz --box-scale 0.95 --save out\voxel_boxes.ply
  """
import argparse
import glob
import os
from pathlib import Path
import numpy as np
import open3d as o3d


# =========================
# 共通ユーティリティ
# =========================
def resolve_first_path(pattern: str) -> Path:
    """グロブを解決して最初のヒットを返す。見つからなければそのままPath化。"""
    hits = glob.glob(pattern)
    return Path(hits[0] if hits else pattern)


def build_kdtree(pcd: o3d.geometry.PointCloud) -> o3d.geometry.KDTreeFlann:
    """KD木を作る"""
    return o3d.geometry.KDTreeFlann(pcd)


def transfer_vertex_colors_from_pcd(mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud) -> None:
    """メッシュ各頂点に、最も近い点群の色を転写する。
    点群に色がなければグレーにする。
    """
    if len(pcd.colors) == 0:
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.7, 0.7, 0.7]]), (len(mesh.vertices), 1)))
        return
    tree = build_kdtree(pcd)
    verts = np.asarray(mesh.vertices)
    cols = np.zeros((verts.shape[0], 3), dtype=np.float64)
    # 逐次最近傍探索（頂点数は通常そこまで大きくないので実用的）
    for i, v in enumerate(verts):
        _, idx, _ = tree.search_knn_vector_3d(v, 1)
        cols[i] = np.asarray(pcd.colors)[idx[0]]
    mesh.vertex_colors = o3d.utility.Vector3dVector(cols)


def estimate_alpha_from_knn(pcd: o3d.geometry.PointCloud, k: int = 10, sample: int = 2000, factor: float = 1.5) -> float:
    """K近傍の平均距離から alpha を自動推定するヘルパ。
    - 点数が多い場合はランダムサンプルで近似
    - factor を上げると面がつながりやすくなる
    """
    n = len(pcd.points)
    if n == 0:
        raise ValueError("点群が空である")
    idxs = np.arange(n)
    if n > sample:
        rng = np.random.default_rng(0)
        idxs = rng.choice(n, size=sample, replace=False)
    tree = build_kdtree(pcd)
    dists = []
    for i in idxs:
        # k+1 とするのは自分自身(距離0)が含まれるため
        _, _, d2 = tree.search_knn_vector_3d(pcd.points[i], k + 1)
        if len(d2) > 1:
            # 自分自身を除いた最近傍距離を使う
            dd = np.sqrt(np.asarray(d2[1:], dtype=np.float64))
            dists.append(np.mean(dd))
    mean_knn = float(np.mean(dists)) if dists else 0.01
    return factor * mean_knn


# =========================
# メッシュ生成: Alpha Shape
# =========================
def mesh_from_alpha_shape(ply_path: Path, alpha: float = None, knn: int = 10, factor: float = 1.5) -> o3d.geometry.TriangleMesh:
    """Alpha Shapeで点群から三角メッシュを生成し、頂点色を転写する"""
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        raise ValueError(f"点群の読み込みに失敗: {ply_path}")

    # Alpha自動推定（未指定時）
    if alpha is None:
        alpha = estimate_alpha_from_knn(pcd, k=knn, sample=2000, factor=factor)

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

    # 簡易クリーンアップ
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    # 頂点色 = 近傍点の色（密度グレースケール）を転写
    transfer_vertex_colors_from_pcd(mesh, pcd)
    mesh.compute_vertex_normals()
    return mesh


# =========================
# メッシュ生成: Voxel Box
# =========================
def color_from_scalar_linear(s: np.ndarray) -> np.ndarray:
    """スカラー(0-1)をグレースケールRGBにマップ"""
    s = np.clip(s, 0.0, 1.0)
    return np.stack([s, s, s], axis=1)


def mesh_from_voxel_boxes(npz_path: Path, box_scale: float = 0.95, quantile_max: float = 0.98) -> o3d.geometry.TriangleMesh:
    """voxel.npz を使って各ボクセルに箱メッシュを置く。
    - box_scale: ボクセルサイズに対する縮小率（重なり防止と視認性向上）
    - quantile_max: 密度の上位分位で正規化上限を切る（極大値で全体が暗くなるのを防ぐ）
    """
    data = np.load(str(npz_path))
    centers = data["centers"]          # (M,3)
    density = data["density"]          # (M,)
    vox = data["voxel_size"]           # (3,)
    # 正規化（上位外れ値の影響を緩和）
    dmin = float(np.min(density))
    dmax = float(np.quantile(density, quantile_max))
    if dmax > dmin:
        dnorm = (density - dmin) / (dmax - dmin)
    else:
        dnorm = np.zeros_like(density)

    # 箱の実サイズ
    size = np.asarray(vox, dtype=np.float64) * float(box_scale)

    mesh_all = o3d.geometry.TriangleMesh()
    cols_all = []

    for c, t in zip(centers, dnorm):
        box = o3d.geometry.TriangleMesh.create_box(width=size[0], height=size[1], depth=size[2])
        box.compute_vertex_normals()
        # boxのローカル中心を原点→平行移動
        box.translate(np.asarray(c, dtype=np.float64) - 0.5 * size)
        # 色（全頂点同色）
        col = color_from_scalar_linear(np.array([t]))[0]
        cols_all.append(np.tile(col, (len(box.vertices), 1)))
        # 結合
        mesh_all += box

    mesh_all.vertex_colors = o3d.utility.Vector3dVector(np.vstack(cols_all))
    mesh_all.remove_duplicated_vertices()
    mesh_all.remove_degenerate_triangles()
    mesh_all.remove_duplicated_triangles()
    mesh_all.remove_non_manifold_edges()
    mesh_all.compute_vertex_normals()
    return mesh_all


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="密度メッシュ可視化（AlphaShape/Box）")
    ap.add_argument("--input", required=True, help="入力PLY（voxel.ply）")
    ap.add_argument("--mode", choices=["alpha", "voxelbox"], default="alpha", help="メッシュ生成モード")
    # Alpha Shape
    ap.add_argument("--alpha", type=float, default=None, help="Alpha値（未指定なら自動推定）")
    ap.add_argument("--knn", type=int, default=10, help="Alpha自動推定のk近傍")
    ap.add_argument("--alpha-factor", type=float, default=1.5, help="Alpha自動推定の倍率")
    # Voxel Box
    ap.add_argument("--npz", type=str, default=None, help="voxel.npz のパス（voxelboxモードで必須）")
    ap.add_argument("--box-scale", type=float, default=0.95, help="箱サイズの縮小率（0<scale<=1）")
    ap.add_argument("--qmax", type=float, default=0.98, help="正規化上限に使う分位（外れ値抑制）")
    # 出力
    ap.add_argument("--save", type=str, default=None, help="保存先（.ply 推奨）")
    return ap.parse_args()


def main():
    args = parse_args()
    ply_path = resolve_first_path(args.input)

    if args.mode == "alpha":
        mesh = mesh_from_alpha_shape(ply_path, alpha=args.alpha, knn=args.knn, factor=args.alpha_factor)
    else:
        if args.npz is None:
            raise ValueError("--mode voxelbox では --npz を指定すること")
        mesh = mesh_from_voxel_boxes(Path(args.npz), box_scale=args.box_scale, quantile_max=args.qmax)

    # 表示
    o3d.visualization.draw_geometries([mesh], window_name=f"Mesh ({args.mode})")

    # 保存
    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(out), mesh, write_triangle_uvs=False)
        print(f"保存: {out.resolve()}")


if __name__ == "__main__":
    main()
