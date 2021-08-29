
from builtins import print
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
from text_lab_event.cat_embedding import cat_bert_embedding
from text_lab_event.cat_tokens import concat_piece_tokens
from text_lab_event._512_embedding import bert_512_bert
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
            # print(g.shape)
            u = torch.relu(self.phrase_filter(g.unsqueeze(1)).squeeze(1))  # [b, l, k]
            # m = self.dropout(self.phrase_extract(u))  # [b, l, 1]
            m = self.dropout(F.max_pool2d(u,kernel_size = (1,u.shape[-1])))  # [b, l, 1]

            # m = self.dropout(self.phrase_extract(g))  # [b, l, 1]

            b = torch.softmax(m, dim=1)  # [b, l, 1]
            return b,u

        u = torch.relu(self.phrase_filter(g.unsqueeze(1)).squeeze(1))  # [b, l, k]


        m = self.dropout(self.phrase_extract(u))  # [b, l, 1]
        b = torch.softmax(m, dim=1)  # [b, l, 1]
        return b,u

   
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

                # outputs = bert_512_bert(self.encoder,b)
                outputs = cat_bert_embedding(self.fc1,self.encoder,b)
                # print("ori: ",outputs.shape)

                outputs = concat_piece_tokens(token_map,outputs)
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

    def forward(self, token_map,text,event_token,c0,task_token,time_stamp):
        v = self.dropout(self.embedding(text,token_map,time_stamp,flag = "text")[0])
        e,time_stamp =  self.dropout(self.embedding(event_token,token_map,time_stamp,flag = "event")[0]),self.embedding(event_token,token_map,time_stamp,flag = "event")[1]
        # print(v.shape,e.shape)
        e_a,u_e =  self.scaled_attention(v,e,flag='event')

        ######
        return v,e_a


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
        # self.rnn_encoder = nn.GRU(input_size=1, batch_first=True,hidden_size=fusion_dim, num_layers=1, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.sigmoid = nn.Sigmoid()
        # self.c = label_embedding.to(device)
        # self.c = torch.stack([torch.FloatTensor(np.array(list(range(1,output_dim))))]*batch_size).to(device) ## random c
        self.text_fc = nn.Sequential(
                nn.Linear(fusion_dim, output_dim)
                )
    def forward(self, token_map,text,event_token,label_token,task_token,time_stamp):

        embed,event_weight = self.compat_model(token_map,text,event_token,label_token,task_token,time_stamp)
        # print(embed.shape,event_weight.shape)
        weighted_embed_event = self.dropout(event_weight*embed)     
        weighted_embed_sumed_event = weighted_embed_event.sum(1)
        weighted_embed_sumed_event = self.dropout(self.fc(weighted_embed_sumed_event)).unsqueeze(1)
        weighted_embed_sumed_event =  self.sigmoid(self.dropout(self.text_fc(weighted_embed_sumed_event.squeeze(1))))
        return weighted_embed_sumed_event,event_weight

        # pack= rnn_utils.pack_padded_sequence(time_stamp, torch.FloatTensor(length),batch_first=True)
        # output, hidden = self.rnn_encoder(pack)
        # padded_output, others  = rnn_utils.pad_packed_sequence(output, batch_first=True)
        # padded_output = padded_output.view(padded_output.shape[0],padded_output.shape[1],2, padded_output.shape[-1]//2)
        # last_hidden = padded_output[:,-1:,0:1,:] + padded_output[:,:1,1:2,:]
        # last_hidden = last_hidden.squeeze(1)
        # time_stamp = time_stamp.unsqueeze(-1).float()
        ## 先加后乘，或者先乘后加
        # event_embedding = event_embedding + time_stamp
        # print(event_weight)
        # weighted_embed_event = self.dropout(event_weight*embed) 
        # weighted_embed_sumed_event = weighted_embed_event.sum(1)
        # weighted_embed_sumed_event = self.dropout(self.fc(weighted_embed_sumed_event)).unsqueeze(1)


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
















