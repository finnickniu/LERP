
from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import torch.nn.utils.rnn as rnn_utils
import numpy as np
from transformers import AutoModel
import random
from tqdm import tqdm
import math
# random.seed(2020)


class event_model(nn.Module):
    def __init__(self, fusion_dim=300,output_dim=25,dropout=0.5):
        nn.Module.__init__(self)
        self.encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.fc = nn.Linear(768, fusion_dim)
        self.dropout = nn.Dropout(dropout)
        self.clss = self.classification(fusion_dim,output_dim)
        self.sigmoid = nn.Sigmoid()
    def embedding(self,x):
        batch_output = []
        embedded = []
        for b in x:
            output = self.encoder(**b).pooler_output.unsqueeze(0)
            batch_output.append(output)
        max_length = max([i.shape[1] for i in batch_output])
        padding_batch = []
        for t in batch_output:
            if t.shape[1] < max_length:
                padding = torch.zeros((1,max_length-t.shape[1],768)).to(f"cuda:{t.get_device()}")
                t = torch.cat([t,padding],dim=1)
            padding_batch.append(t)
        embedded = torch.cat(padding_batch,dim=0)
        return embedded

    def classification(self,in_channels,out_channels):
            clss = nn.Linear(in_channels,out_channels,bias=True)
            return clss
    
    def forward(self, event_token):
        event_embedding = self.embedding(event_token)
        event_embedding = event_embedding.sum(1)
        e = self.sigmoid(self.dropout(self.clss(self.dropout(self.fc(event_embedding)))))

        return e



















