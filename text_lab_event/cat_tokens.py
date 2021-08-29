

from threading import enumerate
from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import torch.nn.utils.rnn as rnn_utils
import numpy as np
# from lab_text_dataloader import TEXTDataset
from transformers import AutoTokenizer, AutoModel
import random
from tqdm import tqdm
import math

def concat_piece_tokens(token_map,output):

    pending_list=[]

    # print(list(token_map.keys())[-1],token_map[list(token_map.keys())[-1]],output.shape)
    for k in token_map.keys():
    #     print(k)
        vals = token_map[k]

        try:
            vals.remove(511)
        except:pass
        # print(k,vals[-1],output.shape)
        if not vals:
            otuput_ = output[:,k:k+1,:]
        else:
            # print(otuput_.shape)
            otuput_ = output[:,k:vals[-1]+1,:]
        otuput_ = otuput_.sum(1)/(len(vals)+1)
        # print(otuput_.shape)
        pending_list.append(otuput_.unsqueeze(1))

    output = torch.cat(pending_list,dim=1)
    return output