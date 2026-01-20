# 低密度ボクセルTop-Kを書き出す（CSV）


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
