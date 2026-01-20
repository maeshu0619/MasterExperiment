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
"""
import argparse
import glob
import os
from pathlib import Path
import numpy as np
import open3d as o3d
from __future__ import annotations

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib が見つからないため、密度マップの描画/保存をスキップする。`pip install matplotlib` を実行すること。")
    plt = None



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

    # parse_args() にオプションを追加
    ap.add_argument("--render-map", action="store_true", help="2D密度マップを生成・表示・保存する")
    ap.add_argument("--proj", choices=["xy", "xz", "yz"], default="xy", help="密度マップの投影平面")
    ap.add_argument("--bins", type=int, default=256, help="密度マップのビン数")
    ap.add_argument("--map-save", type=str, default=None, help="密度マップの保存先（未指定なら--inputと同じ場所に自動保存）")

    return ap.parse_args()

def derive_auto_paths(input_ply: Path, mode: str, proj: str) -> tuple[Path, Path]:
    """--save未指定時の自動保存パスを決める
    - メッシュ: <input_dir>/<input_stem>_{alpha|boxes}.ply
    - 密度マップ: <input_dir>/<input_stem>_density_<proj>.png
    """
    stem = input_ply.stem
    out_mesh = input_ply.with_name(f"{stem}_{'alpha' if mode=='alpha' else 'boxes'}.ply")
    out_map = input_ply.with_name(f"{stem}_density_{proj}.png")
    return out_mesh, out_map

# メッシュ生成関数群の下に追加
def render_density_map(npz_path: Path | None, ply_path: Path, proj: str = "xy", bins: int = 256) -> tuple[object, np.ndarray]:
    """2D密度マップを作る
    優先: NPZの centers(座標) と density(重み)
    代替: PLYの点群とカラー(グレースケール→重み)
    proj: 'xy' | 'xz' | 'yz'
    """
    # 投影軸の選択
    if proj == "xy":
        ax0, ax1 = 0, 1
        axes_label = ("X", "Y")
    elif proj == "xz":
        ax0, ax1 = 0, 2
        axes_label = ("X", "Z")
    elif proj == "yz":
        ax0, ax1 = 1, 2
        axes_label = ("Y", "Z")
    else:
        raise ValueError("--proj は xy/xz/yz のいずれか")

    # データ取得（優先: NPZ）
    pts2 = None
    w = None
    if npz_path is not None and Path(npz_path).exists():
        data = np.load(str(npz_path))
        centers = data["centers"] if "centers" in data else data["grid_points"]
        w = data["density"] if "density" in data else data.get("kde_values", None)
        if w is None:
            # 最低限、重み無しでカウント密度にする
            w = np.ones((centers.shape[0],), dtype=np.float64)
        pts2 = centers[:, [ax0, ax1]].astype(np.float64)
    else:
        # 代替: PLYから（色をグレーにして重み化）
        pcd = o3d.io.read_point_cloud(str(ply_path))
        if pcd.is_empty():
            raise ValueError(f"PLYの読み込みに失敗: {ply_path}")
        pts = np.asarray(pcd.points, dtype=np.float64)
        pts2 = pts[:, [ax0, ax1]]
        if len(pcd.colors) > 0:
            col = np.asarray(pcd.colors, dtype=np.float64)
            w = col.mean(axis=1)  # グレースケールを重みとみなす
        else:
            w = np.ones((pts.shape[0],), dtype=np.float64)

    # 2Dヒストグラム（重み付き）
    x, y = pts2[:, 0], pts2[:, 1]
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=w)
    H = H.T  # imshowで上向きにするため転置

    # 描画
    fig = plt.figure()
    plt.imshow(H, origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.xlabel(axes_label[0])
    plt.ylabel(axes_label[1])
    plt.title(f"Density map ({proj.upper()})")
    plt.colorbar(label="density (arbitrary units)")
    plt.tight_layout()
    return fig, H


def main():
    args = parse_args()
    ply_path = resolve_first_path(args.input)

    if args.mode == "alpha":
        mesh = mesh_from_alpha_shape(ply_path, alpha=args.alpha, knn=args.knn, factor=args.alpha_factor)
    else:
        if args.npz is None:
            raise ValueError("--mode voxelbox では --npz を指定すること")
        mesh = mesh_from_voxel_boxes(Path(args.npz), box_scale=args.box_scale, quantile_max=args.qmax)

    # ここで自動保存パスを決定
    auto_mesh_path, auto_map_path = derive_auto_paths(ply_path, args.mode, args.proj)

    # 表示（3Dメッシュ）
    o3d.visualization.draw_geometries([mesh], window_name=f"Mesh ({args.mode})")

    # メッシュ保存：--save 未指定なら入力と同じ場所に自動保存
    mesh_out = Path(args.save) if args.save else auto_mesh_path
    mesh_out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(mesh_out), mesh, write_triangle_uvs=False)
    print(f"保存: {mesh_out.resolve()}")

    # 密度マップ：要求時に生成して保存＋表示
    if args.render_map:
        if plt is None:
            print("matplotlib未導入のため密度マップをスキップする。`pip install matplotlib` を実行する。")
        else:
            fig, _ = render_density_map(Path(args.npz) if args.npz else None, ply_path, proj=args.proj, bins=args.bins)
            map_out = Path(args.map_save) if args.map_save else auto_map_path
            map_out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(map_out), dpi=150)
            print(f"密度マップ保存: {map_out.resolve()}")
            plt.show()



if __name__ == "__main__":
    main()
