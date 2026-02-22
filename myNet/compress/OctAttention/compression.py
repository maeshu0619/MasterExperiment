import torch
import numpy as np

from .networkTool import reload,CPrintl,expName,device
from .Preparedata.data import *
from .eval import (_pc_stats, 
                  _oct_seq_stats, 
                  _cross_entropy_bits, 
                  _cross_entropy_bits_mine, 
                  single_child_stats_from_oct_seq, 
                  single_child_by_level, 
                  popcount_histogram)

def batchify(oct_seq,bptt,oct_len):
    oct_seq[:-1,0:-1,:] = oct_seq[1:,0:-1,:]
    oct_seq[:-1,-1,1:3] = oct_seq[1:,-1,1:3]  
    oct_seq[:,:,0] = oct_seq[:,:,0] - 1
    pad_len = bptt#int(np.ceil(len(oct_seq)/bptt)*bptt - len(oct_seq))
    oct_seq = torch.Tensor(np.r_[np.zeros((bptt,*oct_seq.shape[1:])),oct_seq,np.zeros((pad_len,*oct_seq.shape[1:]))])
    dataID = torch.LongTensor(np.r_[np.ones((bptt))*-1,np.arange(oct_len),np.ones((pad_len))*-1])
    return dataID.unsqueeze(1),oct_seq.unsqueeze(1)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def _compress_lightweight(oct_data_seq, model, bptt, writer):
    """
    保存・実符号化なし
    return:
        bit, single_child_ratio, num_nodes, bpp, bpn
    """

    model.eval()

    levelID = oct_data_seq[:, -1, 1].copy()
    oct_seq = oct_data_seq[:, -1:, 0].astype(int)
    oct_len = len(oct_seq)

    # --- safety ---
    if oct_len < 64:
        return 0.0, 0.0, oct_len, 0.0, 0.0

    # --- batchify ---
    dataID, padding = batchify(oct_data_seq, bptt, oct_len)
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    probs = []

    with torch.no_grad():
        for i in range(0, padding.shape[0] - bptt, bptt):
            x = padding[i:i+bptt].long().to(device)
            out = model(x, src_mask, [])
            p = torch.softmax(out.reshape(-1, 255), dim=1)
            probs.append(p.cpu().numpy())

    probs = np.vstack(probs)[:oct_len]

    # --- entropy ---
    bit = 0.0
    for i in range(oct_len):
        v = int(oct_seq[i, -1])
        bit += -np.log2(probs[i, v-1] + 1e-9)

    # --- stats ---
    num_nodes = oct_len
    bpn = bit / num_nodes
    bpp = bpn  # OctAttention は 1 node ≈ 1 point

    single_child_nodes, max_chain = single_child_stats_from_oct_seq(oct_data_seq, None)
    single_child_ratio = single_child_nodes / max(num_nodes, 1)

    return bit, single_child_ratio, num_nodes, bpp, bpn


def OA_compress(pts, model, writer, debug=False):
    """
    pts: torch.Tensor [B, 3, N]
    return:
        bit, single_child_ratio, num_nodes, bpp, bpn
    """
    if debug:
        # OctAttention は学習時に MAX_OCTREE_LEVEL=12 を使う設定がある
        # ただし実データの量子化レベルは dataPrepare 側依存
        writer.write(f"[OA] debug: pts_shape={tuple(pts.shape)}\n")

    # --- 点群 → octree ---
    oct_data_seq = dataPrepare_from_tensor(pts)

    if debug:
        _oct_seq_stats(oct_data_seq, writer)
        # popcount / single-child も見たいなら
        popcount_histogram(oct_data_seq, writer)
        single_child_by_level(oct_data_seq, writer)


    # --- lightweight compress ---
    bit, sc_ratio, num_nodes, bpp, bpn = _compress_lightweight(
        oct_data_seq,
        model,
        bptt=1024, 
        writer=writer
    )

    return bit, sc_ratio, num_nodes, bpp, bpn
