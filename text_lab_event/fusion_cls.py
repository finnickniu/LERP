import torch
import torch.nn as nn
import torch.nn.functional as F
from text_lab_event.text_bert import LEAM
from text_lab_event.channelwise_lstm import cw_lstm_model


class fusion_layer(nn.Module):
    def __init__(self,embedding_dim,fusion_dim,dropout,ngram,output_dim = 25):
        super(fusion_layer, self).__init__()
        self.lab_encoder = cw_lstm_model(ngram,fusion_dim)
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
        self.avgpooling =  nn.AvgPool2d(kernel_size=(self.feature_number, 1))

    def classification_layer(self,in_channels,out_channels):
        clss = nn.Linear(in_channels,out_channels,bias=True)
        return clss
    

#################

### transpose permute 是维度交换, view 可以用于元素压缩和展开
    def forward(self,text_x,event_token,label_token,task_token,lab_x,length,fixed_label_embedding,fixed_task_embedding,time_stamp,Fixed,Flatten,mode='fusion'):

        text_pred,weights,c,t,u,weighted_embed,event_embedding = self.text_encoder(text_x,event_token,label_token,task_token,time_stamp)
        # text_pred,weights,c,t,u,weighted_embed,last_hidden,event_embedding = self.text_encoder(text_x,event_token,label_token,task_token,time_stamp,data_length1)

        if Fixed:
            lab_predict,fused_score,g,u1 = self.lab_encoder(lab_x,length,fixed_label_embedding,fixed_task_embedding)
        else:
            lab_predict,fused_score,g,u1 = self.lab_encoder(lab_x,length,c,t)
        # print("lab: ",lab_predict[:1,:,:])
        # print("event: ",event_embedding[:1,:,:])
        # print("text: ",text_pred[:1,:,:])
        # print(lab_predict.shape,text_pred.shape,event_embedding.shape)
        f_x = torch.cat((lab_predict,text_pred,event_embedding),-1)
        # f_x = torch.cat((F.tanh(lab_predict),F.tanh(text_pred),F.tanh(event_embedding)),-1)
        # f_x = torch.cat((F.normalize(lab_predict,p=2,dim=-1),F.normalize(text_pred,p=2,dim=-1),F.normalize(event_embedding,p=2,dim=-1)),-1)
# 
        # f_x = torch.cat((lab_predict,text_pred,last_hidden,event_embedding),1)

        if Flatten:
            # output = self.sigmoid(self.drop_out(self.text_fc(self.fc(self.flatten(f_x)))))
            output = self.sigmoid(self.drop_out(self.text_fc(self.fc(f_x.squeeze(1)))))

        else:
            output =  self.sigmoid(self.drop_out(self.text_fc(self.avgpooling(f_x).squeeze(1))))


        c_o = c
        c = self.text_encoder.dropout(self.text_encoder.fc(c))
        return output,c,t,u,weights,fused_score,text_pred,c_o,g,u1


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


















