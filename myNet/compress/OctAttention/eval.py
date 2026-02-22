import numpy as np

def _oct_seq_stats(oct_data_seq, writer):
    # oct_data_seq: N × K × 6 (コメントより [oct,level,octant,position(xyz)] っぽい)
    level = oct_data_seq[:, -1, 1].astype(int)
    oct_code = oct_data_seq[:, -1, 0].astype(int)

    # レベルごとのノード数
    uniq_lv, cnt_lv = np.unique(level, return_counts=True)
    top = sorted(zip(uniq_lv.tolist(), cnt_lv.tolist()), key=lambda x: x[0])

    # oct_code は 1..255 の想定
    uniq_oc, cnt_oc = np.unique(oct_code, return_counts=True)
    # 出現上位10を表示
    idx = np.argsort(-cnt_oc)[:10]
    top_oc = [(int(uniq_oc[i]), int(cnt_oc[i])) for i in idx]


def _pc_stats(tag, pc, writer):
    # pc は N×3 を想定
    pc = np.asarray(pc)
    mn = pc.min(axis=0)
    mx = pc.max(axis=0)
    span = mx - mn
    writer.write(f"[{tag}] min={mn}, max={mx}, span={span}\n")
    # 量子化済みっぽい場合にユニーク数も見る
    if np.issubdtype(pc.dtype, np.integer) or np.allclose(pc, np.round(pc)):
        uq = np.unique(pc.astype(np.int64), axis=0).shape[0]
        
    

def _cross_entropy_bits(proBit, oct_seq, levelID, writer):
    # oct_seq は shape (oct_len, 1) で 1..255 が入っている想定
    sym = oct_seq.astype(np.int64).squeeze(-1)  # 1..255
    p_true = proBit[np.arange(len(sym)), sym - 1]
    ce = -np.log2(p_true + 1e-12)  # 1ノードあたりのビット
    
    # レベル別平均も見る（どの深さで悪化しているか）
    lv = levelID.astype(int)
    for L in sorted(np.unique(lv).tolist()):
        m = (lv == L)
        if m.sum() == 0:
            continue

def _cross_entropy_bits_mine(proBit, oct_seq, levelID):
    # oct_seq は shape (oct_len, 1) で 1..255 が入っている想定
    sym = oct_seq.astype(np.int64).squeeze(-1)  # 1..255
    p_true = proBit[np.arange(len(sym)), sym - 1]
    ce = -np.log2(p_true + 1e-12)  # 1ノードあたりのビット
    
    # レベル別平均も見る（どの深さで悪化しているか）
    lv = levelID.astype(int)
    for L in sorted(np.unique(lv).tolist()):
        m = (lv == L)
        if m.sum() == 0:
            continue
        nodes = m.sum()
        mean_bits=ce[m].mean()

        return nodes, mean_bits

def _popcount_8bit(x: np.ndarray) -> np.ndarray:
    """
    8bit occupancy code の立っているビット数を返す（vectorized）
    x: shape (N,), 値域 1..255 を想定
    """
    x = x.astype(np.uint16)
    # bit数カウント（0..8）
    return (
        (x & 1) + ((x >> 1) & 1) + ((x >> 2) & 1) + ((x >> 3) & 1) +
        ((x >> 4) & 1) + ((x >> 5) & 1) + ((x >> 6) & 1) + ((x >> 7) & 1)
    ).astype(np.int32)

def single_child_stats_from_oct_seq(oct_data_seq: np.ndarray, writer=None):
    """
    oct_data_seq から
    - single child node 数（popcount==1）
    - single child chain 最大長（レベル遷移からの近似）
    をログ出力する

    注意:
      oct_data_seq は「木構造」そのものではなく「列」なので、
      chain 最大長は level の増減から推定した近似値になる。
      ただし SR前後の“比較指標”としては十分有効。
    """
    # oct_code と level を取り出す（この形式は本実装に依存）
    oct_code = oct_data_seq[:, -1, 0].astype(np.int32)  # 1..255
    level = oct_data_seq[:, -1, 1].astype(np.int32)

    pop = _popcount_8bit(oct_code)
    single_mask = (pop == 1)

    single_count = int(single_mask.sum())
    total = int(len(oct_code))
    ratio = single_count / max(total, 1)

    # ---- chain 最大長の推定 ----
    # 直前ノードが single child で、次のノードが level+1 なら「潜った」とみなす
    # そうでなければ chain をリセット
    max_chain = 0
    cur_chain = 0

    for i in range(total):
        if not single_mask[i]:
            cur_chain = 0
            continue

        if i == 0:
            cur_chain = 1
        else:
            # level が +1 でつながっているときだけ継続扱い
            if single_mask[i - 1] and (level[i] == level[i - 1] + 1):
                cur_chain += 1
            else:
                cur_chain = 1

        if cur_chain > max_chain:
            max_chain = cur_chain


    return single_count, max_chain

def single_child_by_level(oct_data_seq, writer=None):
    """
    レベル別 single child 比率を出力
    """
    oct_code = oct_data_seq[:, -1, 0].astype(np.int32)
    level = oct_data_seq[:, -1, 1].astype(np.int32)

    # popcount
    pop = (
        (oct_code & 1) + ((oct_code >> 1) & 1) +
        ((oct_code >> 2) & 1) + ((oct_code >> 3) & 1) +
        ((oct_code >> 4) & 1) + ((oct_code >> 5) & 1) +
        ((oct_code >> 6) & 1) + ((oct_code >> 7) & 1)
    )

    single_mask = (pop == 1)

    for lv in sorted(np.unique(level)):
        lv_mask = (level == lv)
        total = lv_mask.sum()
        if total == 0:
            continue
        single_cnt = (single_mask & lv_mask).sum()
        ratio = single_cnt / total
        msg = f"[SINGLE][L{lv}] {single_cnt}/{total} ({ratio:.4f})"
        if writer is not None:
            writer.write(msg)
    
    return ratio

def popcount_histogram(oct_data_seq, writer=None):
    oct_code = oct_data_seq[:, -1, 0].astype(np.int32)

    pop = (
        (oct_code & 1) + ((oct_code >> 1) & 1) +
        ((oct_code >> 2) & 1) + ((oct_code >> 3) & 1) +
        ((oct_code >> 4) & 1) + ((oct_code >> 5) & 1) +
        ((oct_code >> 6) & 1) + ((oct_code >> 7) & 1)
    )

    unique, counts = np.unique(pop, return_counts=True)

    total = len(pop)
    for p, c in zip(unique, counts):
        ratio = c / total
        msg = f"[POPCOUNT] k={p}: {c} ({ratio:.4f})"
        if writer is not None:
            writer.write(msg)

