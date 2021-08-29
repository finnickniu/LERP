
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
# random.seed(2020)

class LabelWordCompatLayer(nn.Module):
    def __init__(self,fc,embedding_dim,ngram, output_dim):
        nn.Module.__init__(self)
        self.encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        assert ngram % 2 == 1, "n-gram should be odd number {2r+1}"

        self.phrase_filter = nn.Conv2d(
            # dilation= 2,
            in_channels=1,
            out_channels=1,
            padding='same',
            kernel_size=(ngram,1))
        self.phrase_extract = nn.MaxPool2d(kernel_size=(1, output_dim))
        self.mp = nn.MaxPool1d(kernel_size=10)
        self.dropout = nn.Dropout(0.3)
        self.fc = fc
        self.fc1 = nn.Linear(embedding_dim,embedding_dim)

    def scaled_attention(self,v,c,flag='text'):
        v = self.fc(v)
        c = self.fc(c)
        B, Nt, E = v.shape
        v = v / math.sqrt(E)
        g = torch.bmm(v, c.transpose(-2, -1))
        if flag =='event':
            u = torch.relu(self.phrase_filter(g.unsqueeze(1)).squeeze(1))  # [b, l, k]
            m = self.dropout(F.max_pool2d(u,kernel_size = (1,u.shape[-1])))  # [b, l, 1]

            # m = self.dropout(self.phrase_extract(g))  # [b, l, 1]

            b = torch.softmax(m, dim=1)  # [b, l, 1]

            return b,u

        u = torch.relu(self.phrase_filter(g.unsqueeze(1)).squeeze(1))  # [b, l, k]


        m = self.dropout(self.phrase_extract(u))  # [b, l, 1]
        b = torch.softmax(m, dim=1)  # [b, l, 1]
        return b,u

    def concat_piece_tokens(self,token_map,output):
    
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
    def cat_bert_embedding(self,embedding):
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

                            output_tail.append(self.encoder(**inputs_tail).last_hidden_state[:,1:-1,:])
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
                            output_tail.append(self.encoder(**inputs_tail).last_hidden_state[:,1:-1,:])


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
                            output_tail.append(self.encoder(**inputs_tail).last_hidden_state[:,1:-1,:])

                    # print("last: ",embedding["input_ids"].shape[-1],input_ids.shape)

                    
                inputs = {
                            "input_ids":input_ids,
                            "attention_mask":attention_mask,
                            "token_type_ids":token_type_ids}  
                # if  inputs["input_ids"].shape[-1] == 0:
                #     continue 
                output = self.encoder(**inputs).last_hidden_state[:,1:-1,:]
                # print("embedding: ",output.shape) 

                seq_output.append(output)
         

# 
                n+=1
            if output_tail:
                seq_output.append(output_tail[0])
            output = torch.cat(seq_output,dim=1)
            output = self.fc1(output)
            # print("embedding: ",output.shape)

        else:
            output = self.encoder(**embedding).last_hidden_state[:,1:-1,:]
            # print("embedding: ",output.shape)
        if embedding["input_ids"].shape[-1]-2 != output.shape[1]:
            print("ori: ",embedding["input_ids"].shape[-1]-2)
        return output
 
    def embedding(self,x,token_map_list,time_stamp,flag = "text"):
        batch_output = []
        embedded = []
        padding_timestamp =[]

        if flag == "text":
            n = 0
            for b in x:
                token_map = token_map_list[n]
                # print('tensor: ',b["input_ids"].shape)
      
                ################ concat piece tokens ######################

                # outputs = self._512_bert(b)
                outputs = self.cat_bert_embedding(b)
                # print("ori: ",outputs.shape)

                outputs = self.concat_piece_tokens(token_map,outputs)
                # print("cat: ",outputs.shape)
                #############################################################
                n+=1
                batch_output.append(outputs)
            max_length = max([i.shape[1] for i in batch_output])
            # print(max_length)
            padding_batch = []
            for t in batch_output:
                if t.shape[1] < max_length:
                    padding = torch.zeros((1,max_length-t.shape[1],768)).to(f"cuda:{t.get_device()}")
                    t = torch.cat([t,padding],dim=1)
                padding_batch.append(t)
            embedded = torch.cat(padding_batch,dim=0)
        elif flag == "event":
            for b in x:
                # output = self.encoder(**b).pooler_output
                # print("input: ",b["input_ids"].shape)
                output = self.encoder(**b).last_hidden_state[:,1:-1,:].sum(1)/(b["input_ids"]).shape[1]
                # print("output: ",output.shape)

                batch_output.append(output.unsqueeze(0))
            max_length = max([i.shape[1] for i in batch_output])
            padding_batch = []
            for i in range(len(batch_output)):
                t = batch_output[i]
                ts = time_stamp[i].unsqueeze(0)
                # print(t.shape,ts.shape)
                if t.shape[1] < max_length:
                    padding = torch.zeros((1,max_length-t.shape[1],768)).to(f"cuda:{t.get_device()}")
                    time_stamp_padding = torch.zeros((1,max_length-t.shape[1])).to(f"cuda:{t.get_device()}")
                    ts = torch.cat([ts,time_stamp_padding],dim=1)
                    t = torch.cat([t,padding],dim=1)
                padding_batch.append(t)
                padding_timestamp.append(ts)
                # print(ts.shape)
            embedded = torch.cat(padding_batch,dim=0)
            padding_timestamp = torch.cat(padding_timestamp,dim=0)
        else:
            ### label 做avg pooling，sum/len()
            inputs = x[0]
            output = self.encoder(**inputs).last_hidden_state[:,1:-1,:].sum(1)/(inputs["input_ids"]).shape[1]
            # output = self.encoder(**inputs).pooler_output
            # print(output.shape)
            embedded = output.repeat(len(x),1,1)
        return embedded,padding_timestamp
###########################################################

    def forward(self, token_map, text,event_token,c0,task_token,time_stamp):
        # c0 = c0[:text.shape[0],:].long() ## random c
        v = self.dropout(self.embedding(text,token_map,time_stamp,flag = "text")[0])
        e,time_stamp =  self.dropout(self.embedding(event_token,token_map,time_stamp,flag = "event")[0]),self.embedding(event_token,token_map,time_stamp,flag = "event")[1]
        c  = self.dropout(self.embedding(c0,token_map,time_stamp,flag = "label")[0])
        t  = self.dropout(self.embedding(task_token,token_map,time_stamp,flag = "task")[0])
        # batch, text tokens,25

        b,u = self.scaled_attention(v,c)
        # e_a,u_e =  self.scaled_attention(e,c,flag='event')
        e_a,u_e =  self.scaled_attention(e,c,flag='event')

        ### dynamic attention c
        l = torch.softmax(self.dropout(F.max_pool2d(u.transpose(-1,1),kernel_size = (1,u.transpose(-1,1).shape[-1]))),dim=1)
        c = self.dropout(c*l)+c
        ######
        return b,v,c,t,u,e,e_a,time_stamp


class LEAM(nn.Module):
    def __init__(self, fusion_dim,embedding_dim, output_dim, dropout, ngram):
        nn.Module.__init__(self)


        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, fusion_dim)
            )

        # self.fc = nn.Sequential(
        #     nn.Linear(embedding_dim, 256),
        #     nn.Dropout(dropout),
        #     nn.Linear(256, output_dim)
        #     )
        self.compat_model = LabelWordCompatLayer(
            fc = self.fc,
            embedding_dim = embedding_dim,
            ngram=ngram,
            output_dim=output_dim,
        )
        self.rnn_encoder = nn.GRU(input_size=1, batch_first=True,hidden_size=fusion_dim, num_layers=1, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.sigmoid = nn.Sigmoid()
        # self.c = label_embedding.to(device)
        # self.c = torch.stack([torch.FloatTensor(np.array(list(range(1,output_dim))))]*batch_size).to(device) ## random c

    def forward(self, token_map,text,event_token,label_token,task_token,time_stamp):

        weight, embed,c,t,u,event_embedding,event_weight,time_stamp = self.compat_model(token_map,text,event_token,label_token,task_token,time_stamp)
        weighted_embed_label = self.dropout(weight*embed) 
        weighted_embed_sumed_label = weighted_embed_label.sum(1)
        weighted_event = self.dropout(event_weight*event_embedding)
        weighted_event =  weighted_event.sum(1)
        return weighted_embed_sumed_label,weight,c,t,u,weighted_embed_label,weighted_event

def collate_fn(data):
    """
    定义 dataloader 的返回值
    :param data: 第0维：data，第1维：label
    :return: 序列化的data、记录实际长度的序列、以及label列表
    """
    data.sort(key=lambda x: len(x[0]), reverse=True)
 
    data_length = [sq[0].shape[0] for sq in data]

    input_x = [i[0].tolist() for i in data]
    y = [i[1] for i in data]
    text = [i[2] for i in data]
    task_token =  [i[3] for i in data]
    label_token =  [i[4] for i in data]
    data = rnn_utils.pad_sequence([torch.from_numpy(np.array(x)) for x in input_x],batch_first = True, padding_value=0)
    return data.unsqueeze(-1), data_length, torch.tensor(y, dtype=torch.float32),text,task_token,label_token
if __name__ == "__main__":
    device = torch.device("cuda:3")
    dataset = TEXTDataset('/home/comp/cssniu/RAIM/benchmark_data/all/data/train/',flag="train",all_feature=True)
    batch_size = 10
    embedding_dim = 768 
    output_dim = 25
    dropout = 0.5
    ngram = 3
    model = LEAM(embedding_dim, output_dim, dropout, ngram, batch_size).to(device)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    for (data,length,label,text,task_token,label_token) in tqdm(trainloader):
        text = [t.to(device) for t in text]
        label_token = [l.to(device) for l in label_token]
        z,weight,c,t = model(text,label_token)
        # print(pred)
        # print(pred)
    # print(output[0].shape)
















