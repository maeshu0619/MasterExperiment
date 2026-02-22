'''
Author: fuchy@stu.pku.edu.cn
Date: 2021-09-20 08:06:11
LastEditTime: 2021-09-20 23:53:24
LastEditors: fcy
Description: the training file
             see networkTool.py to set up the parameters
             will generate training log file loss.log and checkpoint in folder 'expName'
FilePath: /compression/octAttention.py
All rights reserved.
'''
import math
import torch
import torch.nn as nn
import os
import datetime
from compress.OctAttention.networkTool import *
# from torch.utils.tensorboard import SummaryWriter
from .attentionModel import TransformerLayer,TransformerModule

##########################

ntokens = 255 # the size of vocabulary
ninp = 4*(128+4+6) # embedding dimension


nhid = 300 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 3 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 4 # the number of heads in the multiheadattention models
dropout = 0 # the dropout value
batchSize = 32



class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'

        self.pos_encoder = PositionalEncoding(ninp, dropout)

        encoder_layers = TransformerLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerModule(encoder_layers, nlayers)

        self.encoder = nn.Embedding(ntoken, 128)
        self.encoder1 = nn.Embedding(MAX_OCTREE_LEVEL+1, 6)
        self.encoder2 = nn.Embedding(9, 4)

        self.ninp = ninp
        self.act = nn.ReLU()
        self.decoder0 = nn.Linear(ninp, ninp)
        self.decoder1 = nn.Linear(ninp, ntoken)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data = nn.init.xavier_normal_(self.encoder.weight.data )
        self.decoder0.bias.data.zero_()
        self.decoder0.weight.data= nn.init.xavier_normal_(self.decoder0.weight.data )
        self.decoder1.bias.data.zero_()
        self.decoder1.weight.data = nn.init.xavier_normal_(self.decoder1.weight.data )
    def forward(self, src, src_mask, dataFeat):
        bptt = src.shape[0]
        batch = src.shape[1]

        oct = src[:,:,:,0] #oct[bptt,batchsize,FeatDim(levels)] [0~254]
        level = src[:,:,:,1]  # [0~12] 0 for padding data
        octant = src[:,:,:,2] # [0~8] 0 for padding data

        # assert oct.min()>=0 and oct.max()<255
        # assert level.min()>=0 and level.max()<=12
        # assert octant.min()>=0 and octant.max()<=8
        
        level -= torch.clip(level[:,:,-1:] - 10,0,None)# the max level in traning dataset is 10        
        torch.clip_(level,0,MAX_OCTREE_LEVEL) 
        aOct = self.encoder(oct.long()) #a[bptt,batchsize,FeatDim(levels),EmbeddingDim]
        aLevel = self.encoder1(level.long())
        aOctant = self.encoder2(octant.long())

        a = torch.cat((aOct,aLevel,aOctant),3)

        a = a.reshape((bptt,batch,-1)) 
        
        # src = self.ancestor_attention(a)
        src = a.reshape((bptt,a.shape[1],self.ninp))* math.sqrt(self.ninp)

        # src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder1(self.act(self.decoder0(output)))
        return output

######################################################################
# ``PositionalEncoding`` module 
#

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

######################################################################
# Functions to generate input and target sequence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len].clone()
    target = source[i+1:i+1+seq_len,:,-1,0].reshape(-1)
    data[:,:,0:-1,:] = source[i+1:i+seq_len+1,:,0:-1,:] # this moves the feat(octant,level) of current node to lastrow,        
    data[:,:,-1,1:3] = source[i+1:i+seq_len+1,:,-1,1:3]# which will be used as known feat
    return data[:,:,-levelNumK:,:], (target).long(),[]



######################################################################
# Run the model
# -------------
#
# model = TransformerModel(ntokens, ninp, nhead, nhid, nlayers, dropout).to(device)
def build_model(device):
    m = TransformerModel(ntokens, ninp, nhead, nhid, nlayers, dropout)
    return m.to(device)