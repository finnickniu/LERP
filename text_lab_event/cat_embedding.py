
from operator import invert
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
import datetime

def cat_bert_embedding(fc1,encoder,embedding):
    seq_output = []
    output_tail = []
    n = 0
    cls = torch.FloatTensor([[101]]).to(f"cuda:{embedding['input_ids'].get_device()}",dtype=torch.long)
    seq = torch.FloatTensor([[102]]).to(f"cuda:{embedding['input_ids'].get_device()}",dtype=torch.long)
    token_type = torch.FloatTensor([[0]]).to(f"cuda:{embedding['input_ids'].get_device()}",dtype=torch.long)
    att_mask = torch.FloatTensor([[1]]).to(f"cuda:{embedding['input_ids'].get_device()}",dtype=torch.long)
    # embedding = {
    #     "input_ids":torch.FloatTensor([[103]*1023]).to(f"cuda:{embedding['input_ids'].get_device()}",dtype=torch.long),
    #     "attention_mask":torch.FloatTensor([[1]*1023]).to(f"cuda:{embedding['input_ids'].get_device()}",dtype=torch.long),
    #     "token_type_ids":torch.FloatTensor([[0]*1023]).to(f"cuda:{embedding['input_ids'].get_device()}",dtype=torch.long)} 

    # print("ori: ",embedding["input_ids"].shape[-1]-2 )

    if embedding["input_ids"].shape[-1] > 512:
        for tokens in range(int(embedding["input_ids"].shape[-1]/512)+1):
            # print(tokens,embedding["input_ids"].shape[-1],int(embedding["input_ids"].shape[-1]/512))
            if tokens == 0:
                input_ids = torch.cat((embedding["input_ids"][:,:511],seq),dim = 1)
                attention_mask = torch.cat((embedding["attention_mask"][:,:511],att_mask),dim = 1)
                token_type_ids = torch.cat((embedding["token_type_ids"][:,:511],token_type),dim = 1)
    
            # elif 1 <= tokens <= embedding["input_ids"].shape[-1]/512 -1:

            elif 2 <= embedding["input_ids"].shape[-1]/512 and 0 < tokens < int(embedding["input_ids"].shape[-1]/512) :

                if tokens == 1:
                    input_ids = torch.cat((cls,torch.cat((embedding["input_ids"][:,511:1021],seq),dim = 1)),dim = 1)
                    attention_mask = torch.cat((att_mask,torch.cat((embedding["attention_mask"][:,511:1021],att_mask),dim = 1)),dim = 1)
                    token_type_ids = torch.cat((token_type,torch.cat((embedding["token_type_ids"][:,511:1021],token_type),dim = 1)),dim = 1)
                elif tokens > 1:                                                                                                                                      
                    input_ids = torch.cat((cls,torch.cat((embedding["input_ids"][:,(510*tokens)+1:(510*tokens)+511],seq),dim = 1)),dim = 1)
                    attention_mask = torch.cat((att_mask,torch.cat((embedding["attention_mask"][:,(510*tokens)+1:(510*tokens)+511],att_mask),dim = 1)),dim = 1)
                    token_type_ids = torch.cat((token_type,torch.cat((embedding["token_type_ids"][:,(510*tokens)+1:(510*tokens)+511],token_type),dim = 1)),dim = 1)

                # print("mild: ",embedding["input_ids"].shape[-1],input_ids.shape)
            # len = 1.5 range(0,2) t = 0,1
            elif tokens >= int(embedding["input_ids"].shape[-1]/512):
                if tokens == 1 :
                    if embedding["input_ids"][:,511:].shape[1] <= 511: 
                        input_ids = torch.cat((cls,embedding["input_ids"][:,511:]),dim = 1)
                        # print("last: ",embedding["input_ids"].shape[-1],input_ids.shape)
                        attention_mask = torch.cat((att_mask,embedding["attention_mask"][:,511:]),dim = 1)
                        token_type_ids = torch.cat((token_type,embedding["token_type_ids"][:,511:]),dim = 1)
                    else: 
                        input_ids = torch.cat((torch.cat((cls,embedding["input_ids"][:,511:1021]),dim = 1),seq),dim = 1)
                        attention_mask = torch.cat((torch.cat((att_mask,embedding["attention_mask"][:,511:1021]),dim = 1),att_mask),dim = 1)
                        token_type_ids = torch.cat((torch.cat((token_type,embedding["token_type_ids"][:,511:1021]),dim = 1),token_type),dim = 1)
                        # print("mild embedding: ",input_ids.shape) 

                        input_ids_tail = torch.cat((cls,embedding["input_ids"][:,1021:]),dim = 1)
                        attention_mask_tail = torch.cat((att_mask,embedding["attention_mask"][:,1021:]),dim = 1)
                        token_type_ids_tail = torch.cat((token_type,embedding["token_type_ids"][:,1021:]),dim = 1)
                        inputs_tail = {
                            "input_ids":input_ids_tail,
                            "attention_mask":attention_mask_tail,
                            "token_type_ids":token_type_ids_tail} 

                        output_tail.append(encoder(**inputs_tail).last_hidden_state[:,1:-1,:])
                        # print("tail embedding: ",input_ids_tail.shape) 


                elif tokens == 2:
                    if embedding["input_ids"][:,1021:].shape[1] <= 511: 

                        input_ids = torch.cat((cls,embedding["input_ids"][:,1021:]),dim = 1)
                        attention_mask = torch.cat((att_mask,embedding["attention_mask"][:,1021:]),dim = 1)
                        token_type_ids = torch.cat((token_type,embedding["token_type_ids"][:,1021:]),dim = 1)
                    else:
                        input_ids = torch.cat((torch.cat((cls,embedding["input_ids"][:,1021:1531]),dim = 1),seq),dim = 1)
                        attention_mask = torch.cat((torch.cat((att_mask,embedding["attention_mask"][:,1021:1531]),dim = 1),att_mask),dim = 1)
                        token_type_ids = torch.cat((torch.cat((token_type,embedding["token_type_ids"][:,1021:1531]),dim = 1),token_type),dim = 1)
                        
                        input_ids_tail = torch.cat((cls,embedding["input_ids"][:,1531:]),dim = 1)
                        attention_mask_tail = torch.cat((att_mask,embedding["attention_mask"][:,1531:]),dim = 1)
                        token_type_ids_tail = torch.cat((token_type,embedding["token_type_ids"][:,1531:]),dim = 1)
                        inputs_tail = {
                            "input_ids":input_ids_tail,
                            "attention_mask":attention_mask_tail,
                            "token_type_ids":token_type_ids_tail}
                        output_tail.append(encoder(**inputs_tail).last_hidden_state[:,1:-1,:])


                elif tokens > 2:


                    if embedding["input_ids"][:,(510*tokens)+1:].shape[1] <= 511:

                        input_ids = torch.cat((cls,embedding["input_ids"][:,(510*tokens)+1:]),dim = 1)
                        attention_mask = torch.cat((att_mask,embedding["attention_mask"][:,(510*tokens)+1:]),dim = 1)
                        token_type_ids = torch.cat((token_type,embedding["token_type_ids"][:,(510*tokens)+1:]),dim = 1)
                    else:
                        input_ids = torch.cat((torch.cat((cls,embedding["input_ids"][:,(510*tokens)+1:(510*tokens)+511]),dim = 1),seq),dim = 1)
                        attention_mask = torch.cat((torch.cat((att_mask,embedding["attention_mask"][:,(510*tokens)+1:(510*tokens)+511]),dim = 1),att_mask),dim = 1)
                        token_type_ids = torch.cat((torch.cat((token_type,embedding["token_type_ids"][:,(510*tokens)+1:(510*tokens)+511]),dim = 1),token_type),dim = 1)
                        
                        input_ids_tail = torch.cat((cls,embedding["input_ids"][:,(510*tokens)+511:]),dim = 1)
                        attention_mask_tail = torch.cat((att_mask,embedding["attention_mask"][:,(510*tokens)+511:]),dim = 1)
                        token_type_ids_tail = torch.cat((token_type,embedding["token_type_ids"][:,(510*tokens)+511:]),dim = 1)
                        inputs_tail = {
                            "input_ids":input_ids_tail,
                            "attention_mask":attention_mask_tail,
                            "token_type_ids":token_type_ids_tail} 
                        output_tail.append(encoder(**inputs_tail).last_hidden_state[:,1:-1,:])

                # print("last: ",embedding["input_ids"].shape[-1],input_ids.shape)

                
            inputs = {
                        "input_ids":input_ids,
                        "attention_mask":attention_mask,
                        "token_type_ids":token_type_ids}  
            # if  inputs["input_ids"].shape[-1] == 0:
            #     continue 
            output = encoder(**inputs).last_hidden_state[:,1:-1,:]
            # print("embedding: ",output.shape) 

            seq_output.append(output)
        

# 
            n+=1
        if output_tail:
            seq_output.append(output_tail[0])
        output = torch.cat(seq_output,dim=1)
        output = fc1(output)
        # print("embedding: ",output.shape)

    else:
        output = encoder(**embedding).last_hidden_state[:,1:-1,:]
        # print("embedding: ",output.shape)
    if embedding["input_ids"].shape[-1]-2 != output.shape[1]:
        print("ori: ",embedding["input_ids"].shape[-1]-2)
    return output