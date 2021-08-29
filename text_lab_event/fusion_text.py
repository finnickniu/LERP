import torch
import torch.nn as nn
import torch.nn.functional as F
# from text_lab_event.text_bert1 import LEAM
from text_lab_event.text_bert_no_att import LEAM

from text_lab_event.channelwise_lstm import cw_lstm_model


class fusion_layer(nn.Module):
    def __init__(self,embedding_dim,fusion_dim,dropout,ngram,output_dim = 25):
        super(fusion_layer, self).__init__()
        self.text_encoder = LEAM(fusion_dim,embedding_dim, output_dim, dropout, ngram)
        self.feature_number = 3
        self.class_number = output_dim
        self.drop_out = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(self.feature_number*fusion_dim,fusion_dim),
            nn.Dropout(dropout))
        self.norm2 =nn.LayerNorm(64)
        self.dense = nn.Sequential(
        (nn.Linear(256,128,bias=True)),
        (nn.LeakyReLU()),
        (nn.Linear(128,64,bias=True)),
        (nn.LeakyReLU()))
        self.clss = self.classification_layer(1088,25)
        self.text_fc = nn.Sequential(
            nn.Linear(fusion_dim, output_dim)
            )

    def classification_layer(self,in_channels,out_channels):
        clss = nn.Linear(in_channels,out_channels,bias=True)
        return clss

#################

### transpose permute 是维度交换, view 可以用于元素压缩和展开
    def forward(self,text_x,event_token,label_token,task_token,lab_x,length,fixed_label_embedding,fixed_task_embedding,time_stamp,Fixed,Flatten,mode='fusion'):

        text_pred,weights,c,u,weighted_embed = self.text_encoder(text_x,event_token,label_token,task_token,time_stamp)

        output_all =  self.sigmoid(self.drop_out(self.text_fc(text_pred.squeeze(1))))
        c_o = c 
        c = self.text_encoder.dropout(self.text_encoder.fc(c))
        # return [output_n1,output_n2,output_n3,output_all],attention_weights,c,t,u,weights,fused_score,text_pred,c_o,g,u1
        return output_all,c,u,weights,text_pred,c_o

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x0 = torch.randn(2,17,256).to(device)
    x1 = torch.randn(2,1,256).to(device)
    dropout = 0.5
    filter_size = 3
    net = fusion_layer(filter_size,dropout)
    net = net.to(device)
    net.train()
    output = net(x0,x1)
    print(output.shape)


















