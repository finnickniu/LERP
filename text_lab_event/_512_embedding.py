
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


def bert_512_bert(encoder,b):
    inputs = {
            "input_ids":b["input_ids"][:,:512],
            "attention_mask":b["attention_mask"][:,:512],
            "token_type_ids":b["token_type_ids"][:,:512]}  

    outputs = encoder(**inputs)
    # print("output 0: ", outputs.last_hidden_state.shape)
    if b["input_ids"].shape[1]<=512:
        output = outputs.last_hidden_state[:,1:-1,:]
    else:
        output = outputs.last_hidden_state[:,1:,:]
    return output