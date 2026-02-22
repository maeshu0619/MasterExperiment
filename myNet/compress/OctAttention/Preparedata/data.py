'''
Author: fuchy@stu.pku.edu.cn
Date: 2021-09-17 23:30:48
LastEditTime: 2021-12-02 22:18:56
LastEditors: FCY
Description: dataPrepare helper
FilePath: /compression/Preparedata/data.py
All rights reserved.
'''
from ..Octree import GenOctree,GenKparentSeq
import numpy as np
import os
import hdf5storage

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import pt as pointCloud



def dataPrepare(args, pts,saveMatDir='Data',qs=2,ptNamePrefix='',offset='min',qlevel=None,rotation=False,normalize=False):
    p = pts
    ptName = f'pointcloud_{args.loss_type}_{args.a}_{args.b}_{args.c}_{args.d}_{args.e}'
    
    refPt = p
    if normalize is True: # normalize pc to [-1,1]^3
        p = p - np.mean(p,axis=0)
        p = p/abs(p).max()
        refPt = p

    if rotation:
        refPt = refPt[:,[0,2,1]]
        refPt[:,2] = - refPt[:,2]

    if offset is 'min':
        offset = np.min(refPt,0)

    points = refPt - offset

    if qlevel is not None:
        qs = (points.max() - points.min())/(2**qlevel-1)

    pt = np.round(points/qs)
    pt,idx = np.unique(pt,axis=0,return_index=True)
    pt = pt.astype(int)
    # pointCloud.write_ply_data('pori.ply',np.hstack((pt,c)),attributeName=['reflectance'],attriType=['uint16'])
    code,Octree,QLevel = GenOctree(pt)
    DataSturct = GenKparentSeq(Octree,4)
    
    ptcloud = {'Location':refPt}
    Info = {'qs':qs,'offset':offset,'Lmax':QLevel,'name':ptName,'levelSID':np.array([Octreelevel.node[-1].nodeid for Octreelevel in Octree])}
    patchFile = {'patchFile':(np.concatenate((np.expand_dims(DataSturct['Seq'],2),DataSturct['Level'],DataSturct['Pos']),2), ptcloud, Info)}
    hdf5storage.savemat(os.path.join(saveMatDir,ptName+'.mat'), patchFile, format='7.3', oned_as='row', store_python_metadata=True)
    DQpt = (pt*qs+offset) 
    return os.path.join(saveMatDir,ptName+'.mat'),DQpt,refPt

def dataPrepare_from_tensor(pts):
    """
    pts: torch.Tensor [B, 3, N]（B=1前提）
    return:
        oct_data_seq: np.ndarray [num_nodes, *, *]
    """
    # Tensor → numpy
    pts_np = (
        pts
        .detach()          # 計算グラフから切り離す
        .squeeze(0)
        .transpose(1, 0)
        .cpu()
        .numpy()
    )


    # Octree 生成（GenOctree, GenKparentSeq を直接呼ぶ）
    pt_int = pts_np.astype(int)

    # 1セルに複数点が入っていないか（OctAttentionの実装に一致）
    uniq, cnt = np.unique(pt_int, axis=0, return_counts=True)
    # dataPrepare_from_tensor 内（GenOctree の直前）
    pt_int = np.floor(pts_np).astype(np.int32)

    # 負座標があるので、最小値でシフトして非負に揃える（Octree実装が非負前提の可能性が高い）
    pt_int = pt_int - pt_int.min(axis=0, keepdims=True)

    # 1セル=1点を保証（最初の出現だけ残す）
    uniq, idx = np.unique(pt_int, axis=0, return_index=True)
    pt_int = pt_int[idx]
    
    code, Octree, QLevel = GenOctree(pt_int)

    DataStruct = GenKparentSeq(Octree, 4)

    oct_data_seq = np.concatenate(
        (
            np.expand_dims(DataStruct['Seq'], 2),
            DataStruct['Level'],
            DataStruct['Pos']
        ),
        axis=2
    )
    return oct_data_seq
