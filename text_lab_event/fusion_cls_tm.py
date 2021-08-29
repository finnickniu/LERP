import torch
import torch.nn as nn
import torch.nn.functional as F
# from text_lab_event.text_bert import LEAM
from text_lab_event.text_event import LEAM

from text_lab_event.channelwise_lstm import cw_lstm_model


class fusion_layer(nn.Module):
    def __init__(self,embedding_dim,fusion_dim,dropout,ngram,output_dim = 25):
        super(fusion_layer, self).__init__()
        self.lab_encoder = cw_lstm_model(ngram,fusion_dim)
        self.text_encoder = LEAM(fusion_dim,embedding_dim, output_dim, dropout, ngram)
        self.feature_number = 2
        self.class_number = output_dim
        self.drop_out = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(self.feature_number*fusion_dim,fusion_dim),
            nn.Dropout(dropout))
        self.norm2 =nn.LayerNorm(64)

        self.text_fc = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.Linear(128, output_dim)
            )
        self.avgpooling =  nn.AvgPool2d(kernel_size=(self.feature_number, 1))

        # self.mlp =  nn.Sequential(
        # nn.Linear(fusion_dim*2,fusion_dim*2,bias=True),
        # nn.Linear(fusion_dim*2,400,bias=True),
        # nn.Linear(400,fusion_dim,bias=True))
        self.mlp =  nn.Sequential(
        nn.Linear(embedding_dim*2,embedding_dim,bias=True))
        self.u1s = nn.Parameter(torch.randn(size=(1, fusion_dim)), requires_grad=True)
        self.u2s = nn.Parameter(torch.randn(size=(1, fusion_dim)), requires_grad=True)
        self.u3s = nn.Parameter(torch.randn(size=(1, fusion_dim)), requires_grad=True)

    def classification_layer(self,in_channels,out_channels):
        clss = nn.Linear(in_channels,out_channels,bias=True)
        return clss
    def compute_similarity(self,ui):
        a1 = torch.bmm(ui[:,:1,:], self.u1s.repeat(ui.shape[0],1,1).transpose(-1,1))
        a2 = torch.bmm(ui[:,1:2,:], self.u2s.repeat(ui.shape[0],1,1).transpose(-1,1))
        a3 = torch.bmm(ui[:,2:3,:], self.u3s.repeat(ui.shape[0],1,1).transpose(-1,1))

        return torch.cat((a1,a2,a3),1)

    def casual_attention(self,f_m):


        attention_weights =   self.drop_out(self.compute_similarity(f_m))
        # attention_weights = torch.softmax(self.drop_out((attention_weights / attention_weights.sum(1).unsqueeze(1)).sum(-1).unsqueeze(-1)),dim=1)
        attention_weights_n1 = attention_weights[:,1:,:]
        attention_weights_n2 = torch.cat((attention_weights[:,:1,:],attention_weights[:,2:3,:]),1)
        attention_weights_n3 = attention_weights[:,:2,:]

        attention_weights_n1 = torch.softmax(attention_weights_n1,dim = 1)
        attention_weights_n2 = torch.softmax(attention_weights_n2,dim = 1)
        attention_weights_n3 = torch.softmax(attention_weights_n3,dim = 1)
        attention_weights_all = torch.softmax(attention_weights,dim = 1)
        # print("score: ",attention_weights_all[0:1,:,:].squeeze())

        return attention_weights_n1,attention_weights_n2,attention_weights_n3,attention_weights_all

#################

### transpose permute 是维度交换, view 可以用于元素压缩和展开
    def forward(self,text_x,event_token,label_token,task_token,lab_x,length,fixed_label_embedding,fixed_task_embedding,time_stamp,Fixed,Flatten,mode='fusion'):

        text_pred,weights,c,t,u,event_weight,weighted_event = self.text_encoder(text_x,event_token,label_token,task_token,time_stamp)
        lab_predict,fused_score,g,u1 = self.lab_encoder(lab_x,length,c,t)

        f_x = torch.cat((lab_predict,text_pred),-1)
        # f_x = torch.cat((F.normalize(lab_predict,p=2,dim=-1),F.normalize(text_pred,p=2,dim=-1)),-1)
        f_x = self.drop_out(self.text_encoder.fc(self.drop_out(self.mlp(f_x))))
        output = self.sigmoid(self.drop_out(self.text_fc(f_x.squeeze(1))))
        c_o = c
        c = self.text_encoder.dropout(self.text_encoder.fc(c))
        # return [output_n1,output_n2,output_n3,output_all],attention_weights,c,t,u,weights,fused_score,text_pred,c_o,g,u1
        return output,c,c,t,u,weights,fused_score,text_pred,c_o,g,u1


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


















