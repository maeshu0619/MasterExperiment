'''
Author: fuchy@stu.pku.edu.cn
Description: The encoder helper
FilePath: /compression/encoderTool.py
'''

#%%
import numpy as np 
import torch
import time
import os
import tempfile
from .networkTool import *
from .dataset import default_loader as matloader
from .Preparedata.data import *
from .eval import (_pc_stats, 
                  _oct_seq_stats, 
                  _cross_entropy_bits, 
                  _cross_entropy_bits_mine, 
                  single_child_stats_from_oct_seq, 
                  single_child_by_level, 
                  popcount_histogram)

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
sys.path.append(ROOT_DIR)

from models.utils.utils_repkpu import *

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def batchify(oct_seq,bptt,oct_len):
    oct_seq[:-1,0:-1,:] = oct_seq[1:,0:-1,:]
    oct_seq[:-1,-1,1:3] = oct_seq[1:,-1,1:3]  
    oct_seq[:,:,0] = oct_seq[:,:,0] - 1
    pad_len = bptt#int(np.ceil(len(oct_seq)/bptt)*bptt - len(oct_seq))
    oct_seq = torch.Tensor(np.r_[np.zeros((bptt,*oct_seq.shape[1:])),oct_seq,np.zeros((pad_len,*oct_seq.shape[1:]))])
    dataID = torch.LongTensor(np.r_[np.ones((bptt))*-1,np.arange(oct_len),np.ones((pad_len))*-1])
    return dataID.unsqueeze(1),oct_seq.unsqueeze(1)

def encodeNode(pro,octvalue):
    assert octvalue<=255 and octvalue>=1
    pre = np.argmax(pro)+1
    return -np.log2(pro[octvalue-1]+1e-07),int(octvalue==pre)

class Compression():
    def __init__(self):
        self.pro_buffer = None
    def compress(self, args, oct_data_seq, model, writer):
        model.eval()
        levelID = oct_data_seq[:,-1,1].copy()
        oct_data_seq = oct_data_seq.copy()

        if levelID.max()>MAX_OCTREE_LEVEL:
            print('**warning!!**,to clip the level>{:d}!'.format(MAX_OCTREE_LEVEL))
            
        oct_seq = oct_data_seq[:,-1:,0].astype(int)   
        oct_len = len(oct_seq)
        if oct_len == 0:
        # Octreeが1ノードも生成されていない場合
            return 0, 0, 0.0, np.array([]), np.array([])
    
        dataID,padingdata = batchify(oct_data_seq,bptt,oct_len)
        MAX_GPU_MEM_It = args.max_gpu_mem_it # you can change this according to the GPU memory size (2**12 for 24G)
        MAX_GPU_MEM = min(bptt*MAX_GPU_MEM_It,dataID.max())+2  #  bptt <= MAX_GPU_MEM -1 < min(MAX_GPU,dataID)
        if (self.pro_buffer is None) or (self.pro_buffer.shape[0] < MAX_GPU_MEM):
            self.pro_buffer = torch.zeros(
                (MAX_GPU_MEM,255),
                device=device
            )
        pro = self.pro_buffer
        pro.zero_()

        padingLength = padingdata.shape[0]
        src_mask = generate_square_subsequent_mask(bptt).to(device)
        padingdata = padingdata
        elapsed = 0
        # proBit = []
        proBit = np.empty((oct_len, 255), dtype=np.float32)
        write_ptr = 0
        offset = 0
        trange = range
        
        with torch.no_grad():
            for n,i in enumerate(trange(0, padingLength-bptt , bptt)):
                input = padingdata[i:i+bptt].long().to(device)   #input torch.Size([256, 32, 4, 3]) bptt,batch_sz,kparent,[oct,level,octant]
                nodeID = dataID[i+1:i+bptt+1].squeeze(0) - offset
                nodeID[nodeID<0] = -1
                start_time = time.time()
                output = model(input,src_mask,[])
                elapsed =elapsed+ time.time() - start_time
                output = output.reshape(-1,255)
                nodeID = nodeID.reshape(-1)
                p  = torch.softmax(output,1)
                pro[nodeID,:] = p

                if ((n % MAX_GPU_MEM_It==0 and n>0) or n == padingLength//bptt-1):
                    size = nodeID.max().item() + 1
                    chunk = pro[:size].detach().cpu().numpy()
                    proBit[write_ptr:write_ptr+size] = chunk
                    write_ptr += size
                    offset += size
                # p  = torch.softmax(output,1)
                # pro[nodeID,:] = p
                # if( (n % MAX_GPU_MEM_It==0 and n>0) or n == padingLength//bptt-1):
                #     # proBit.append(pro[:nodeID.max()+1].detach().cpu().numpy())
                #     offset = offset + nodeID.max() +1

        torch.cuda.empty_cache()
        if len(proBit) == 0:
            # 確率が1つも計算されていない（学習途中の壊れた点群など）
            return 0, oct_len, elapsed, np.array([]), np.array([])
        # proBit = np.vstack(proBit)
        #%%
    
        bit = 0
        acc = 0
        templevel = 1
        binszList = []
        octNumList = []

        # Estimate the bitrate at each level
        for i in range(oct_len):
            octvalue = int(oct_seq[i,-1])

            # proBit が不足 or 空なら、このノードはスキップ
            if i >= proBit.shape[0]:
                # writer があればログを残す
                if writer is not None:
                    writer.write(f"[OA][WARN] proBit index overflow: i={i}, proBit.shape={proBit.shape}\n")
                continue

            if proBit[i] is None or proBit[i].size == 0:
                if writer is not None:
                    writer.write(f"[OA][WARN] empty proBit at node {i}, octvalue={octvalue}\n")
                continue

            bit0, acc0 = encodeNode(proBit[i], octvalue)
            bit += bit0
            acc += acc0
            if templevel!=levelID[i]:
                templevel = levelID[i]
                binszList.append(bit)
                octNumList.append(i+1)
        binszList.append(bit)
        octNumList.append(i+1)
        binsz = bit # estimated bin size

        del pro,input,src_mask
        del output, p
        del oct_data_seq

        torch.cuda.empty_cache()

        if len(binszList)<=7:
            return binsz,oct_len,elapsed,np.array(binszList),np.array(octNumList)  
        return binsz,oct_len,elapsed ,np.array(binszList[7:]),np.array(octNumList[7:])  

def oa_main(args, pts, model, qs, writer, file_date, oa_comp):
    pts_np = (
        pts.detach()
           .squeeze(0)
           .transpose(1, 0)
           .cpu()
           .numpy()
    )

    ram_dir = "/dev/shm/OctAttention_encoded"
    saveMatDir = ram_dir if os.path.isdir("/dev/shm") else "compress/OctAttention/encoded"
    os.makedirs(saveMatDir, exist_ok=True)

    matFile, DQpt, refPt = dataPrepare(
        args, pts_np, saveMatDir,
        qs=qs, ptNamePrefix="tmp_fixed", rotation=False
    )

    cell, mat = matloader(matFile)

    try:
        os.remove(matFile)
    except Exception:
        pass

    FeatDim = levelNumK
    oct_data_seq = np.transpose(mat[cell[0,0]]).astype(int)[:,-FeatDim:,0:6]
    
    ptNum = pts.shape[2]
    single_ratio = single_child_by_level(oct_data_seq, writer=None)

    binsz, oct_len, _, _, _ = oa_comp.compress(args, oct_data_seq, model, writer)

    bit = binsz
    bpp = bit / ptNum
    bpn = bit / oct_len
    com = [bit, bpp, bpn, single_ratio, oct_len]

    del mat, cell

    return com