import os
from numpy.core.fromnumeric import shape
from sklearn.utils.extmath import weighted_mode
# os.chdir("/home/comp/cssniu/RAIM/models/")
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from text_lab_event.lab_text_event_dataloader import TEXTDataset
# from text_lab_event.fusion_cls_causal import fusion_layer
from text_lab_event.fusion_cls_text_event import fusion_layer
# from text_lab_event.fusion_cls_event import fusion_layer

from lab_testing.dataloader import knowledge_dataloader 

import numpy as np
import torch.nn.utils.rnn as rnn_utils

from sklearn import metrics
import warnings
import copy
import torch.nn.functional as F
import math
from transformers import AdamW
os.environ['CUDA_VISIBLE_DEVICES']="0,1"
warnings.filterwarnings('ignore')
torch.multiprocessing.set_sharing_strategy('file_system')

### GPU 22 cs flatten fixed, GPU 22 cs flatten not fixed, GPU 22 ca flatten fixed GPU 24 ca flatten not fixed,
num_epochs = 14
BATCH_SIZE = 3
Test_batch_size = 6
ratio_granger_loss = 0
save_dir= "weights_fusion_event"
Flatten = False
Fixed  = False
strict = True
pretrained = False
# save_name = "816_text_event_512d_concat"

# save_name = "819_text_event_alld_avg_label_no_stopword"
# save_name = "819_text_event_512d_avg_label_no_stopword"
# save_name = "819_text_event_alld_avg_label_has_stopword"
# save_name = "824_text_event_twoM_alld_avg_label_no_stopword"
# save_name = "824_text_event_twoM_512d_avg_label_no_stopword"
save_name = "825_text_event_alld_avg_label_has_stopword"

weight_dir = "logs/weights_fusion_event/824_text_event_twoM_alld_avg_label_no_stopword_epoch_6_loss_0.3365_roc_0.7709.pth"
device1 = "cuda:1" if torch.cuda.is_available() else "cpu"
device1 = torch.device(device1)
device2 = "cuda:0" if torch.cuda.is_available() else "cpu"
device2 = torch.device(device2)
Best_loss = 100
Bess_acc = 0
start_epoch = 7


hyperparams = {
               'num_epochs':num_epochs,
               'embedding_dim' : 768,
               'fusion_dim':300,
               "output_dim":25,
               'ngram':3,
               'dropout' : 0.5,
               'batch_size' : BATCH_SIZE,
               'device1':device1,
               'device2':device2}

def calc_loss_c(c,criterion,model, y, device):
    """
    torch.tensor([0,1,2]) is decoded identity label vector
    """
    ## 每个class 内部自己做cross entropy， 相当于做了25次， 也就是25个batch，python cross entropy 自带softmax,也不用做onehot
    # print(c.shape)
    f2_c = model.text_fc(c)
    # f2_c = model.fc(c)
    y_c =  torch.stack([torch.range(0, y.shape[1] - 1, dtype=torch.long)]*c.shape[0]).to(f"cuda:{f2_c.get_device()}")
    return criterion(f2_c,y_c)


def fit(epoch,model,train_iterator,test_iterator,fixed_label_embedding,fixed_task_embedding,optimizer,criterion,criterion1,criterion2,hyperparams,flag = "train"):
    global Best_loss,Bess_acc,Fixed,Flatten,save_name,save_dir,ratio_granger_loss


    if flag == "train":
        device = hyperparams['device1']
        model.train()
        data_iter = train_iterator
    else:
        device = hyperparams['device2']
        model.eval()


        data_iter = test_iterator

    fixed_label_embedding = fixed_label_embedding.to(device).transpose(1,0)
    fixed_task_embedding = fixed_task_embedding.to(device).transpose(1,0)

    model.to(device)

    criterion.to(device)
    criterion1.to(device)

    loss_ls = []
    acc_ls = []
    f1_ls=[]

    for i,(data,length,label,text,event_token,task_token,label_token,time_stamp,token_map,len_text) in enumerate(tqdm(data_iter,desc=f"{flag}ing model")):
        optimizer.zero_grad()
        text_x = [t.to(device) for t in text]
        event_token = [l.to(device) for l in event_token]

        label_token = [l.to(device) for l in label_token]
        task_token = [t.to(device) for t in task_token]
        time_stamp = [torch.from_numpy(i).to(device,dtype=torch.float) for i in time_stamp]

        lab_x = data.to(device,dtype=torch.float)
        y= label.to(device,dtype=torch.float).squeeze()
        fixed_label_embedding_batch = fixed_label_embedding.repeat(lab_x.shape[0],1,1)
        fixed_task_embedding_batch = fixed_task_embedding.repeat(lab_x.shape[0],1,1)
        if flag == "train":
            
            with torch.set_grad_enabled(True):
                pred,c,text_label,weights,weighted_event,event_weight,text_event   = model(token_map,text_x,event_token,label_token,task_token,lab_x,length,fixed_label_embedding_batch,fixed_task_embedding_batch,time_stamp,Fixed,Flatten,mode='fusion')

                loss_v = criterion1(pred, y)

                # loss_c = calc_loss_c(c,criterion,model,y,device)
  
                # loss_t1 = torch.sum(-torch.log(nn.CosineSimilarity(dim=1, eps=1e-6)(weighted_event.squeeze(),text_pred.squeeze())))/3

                # loss = (1-ratio_granger_loss)*(loss_v + loss_c) + ratio_granger_loss*(loss_t1)
              
            ###################################### event  #################
                # pred = model(text_x,event_token,label_token,task_token,lab_x,length,fixed_label_embedding_batch,fixed_task_embedding_batch,time_stamp,Fixed,Flatten,mode='fusion')
                # loss = criterion1(pred, y)
            ###################################### event  #################
                # loss = loss_v + loss_c
                loss = loss_v

                loss.backward(retain_graph=True)
                optimizer.step()

        else:
            with torch.no_grad():
                # pred,c,t,u,weights,text_pred,weighted_event,c_o   = model(text_x,event_token,label_token,task_token,lab_x,length,fixed_label_embedding_batch,fixed_task_embedding_batch,time_stamp,Fixed,Flatten,mode='fusion')
                # pred1,pred2,pred3,pred = predall
                pred,c,text_label,weights,weighted_event,event_weight,text_event   = model(token_map,text_x,event_token,label_token,task_token,lab_x,length,fixed_label_embedding_batch,fixed_task_embedding_batch,time_stamp,Fixed,Flatten,mode='fusion')

                loss_v = criterion1(pred, y)
                # loss_c = calc_loss_c(c,criterion,model,y,device)
                # loss_t1 = torch.sum(-torch.log(nn.CosineSimilarity(dim=1, eps=1e-6)(weighted_event.squeeze(),text_pred.squeeze())))/3

                # loss = (1-ratio_granger_loss)*(loss_v + loss_c) + ratio_granger_loss*(loss_t1)
                # loss = loss_v + loss_c
                loss = loss_v

            ###################################### event  #################
                # pred = model(text_x,event_token,label_token,task_token,lab_x,length,fixed_label_embedding_batch,fixed_task_embedding_batch,time_stamp,Fixed,Flatten,mode='fusion')
                # loss = criterion1(pred, y)
            ###################################### event  #################
        y = np.array(y.tolist())
        pred = np.array(pred.tolist())
        try:
            pred=(pred > 0.5) 

            f1 = metrics.f1_score(y,pred,average="micro")
            acc = metrics.roc_auc_score(y,pred,average="micro")
            # print(f'loss :{float(loss.cpu().data)} acc: {acc}')
            f1_ls.append(f1)
            acc_ls.append(acc)

        except:
            pass
        loss_ls.append(float(loss.cpu().data))

    if flag == "test":
        PATH=f"/home/comp/cssniu/RAIM/logs/{save_dir}/{save_name}_epoch_{epoch}_loss_{round(np.mean(loss_ls),4)}_roc_{round(np.mean(acc_ls),4)}.pth"
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, PATH)
    print("PHASE：{} EPOCH : {} | F1 : {} | ROC ： {} | LOSS : {}".format(flag,epoch + 1,  np.mean(f1_ls),np.mean(acc_ls), np.mean(loss_ls)))
    return model

  

def engine(hyperparams,model, train_iterator, test_iterator,fixed_label_embedding,fixed_task_embedding,optimizer,criterion,criterion1,criterion2):
# def engine(scheduler,model, train_iterator, test_iterator,optimizer,criterion,criterion1):
    global start_epoch
    for epoch in range(start_epoch,hyperparams['num_epochs']):
        model = fit(epoch,model,train_iterator,test_iterator,fixed_label_embedding,fixed_task_embedding,optimizer,criterion,criterion1,criterion2,hyperparams,flag = "train")
        # try:
        model = fit(epoch,model,train_iterator,test_iterator,fixed_label_embedding,fixed_task_embedding,optimizer,criterion,criterion1,criterion2,hyperparams,flag = "test")



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
    event_token =  [i[3] for i in data]

    task_token = [i[4] for i in data]
    label_token = [i[5] for i in data]
    input_x = rnn_utils.pad_sequence([torch.from_numpy(np.array(x)) for x in input_x],batch_first = True, padding_value=0)
    time_stamp = [i[7] for i in data]
    token_map = [i[8] for i in data]
    len_text = [i[9] for i in data]

    return input_x.unsqueeze(-1), data_length, torch.tensor(y, dtype=torch.float32),text,event_token,task_token,label_token,time_stamp,token_map,len_text

if __name__ == "__main__":
   


    task_embedding,label_embedding= knowledge_dataloader.load_embeddings("/home/comp/cssniu/RAIM/embedding.pth")
    fixed_label_embedding = torch.stack(label_embedding)
    fixed_task_embedding = torch.stack(task_embedding)

    train_data =  TEXTDataset('/home/comp/cssniu/RAIM/benchmark_data/all/data/train/',flag="train",all_feature=True)

    test_data = TEXTDataset('/home/comp/cssniu/RAIM/benchmark_data/all/data/test/',flag="test",all_feature=True)

    print('len of train data:', len(train_data))             
    print('len of test data:', len(test_data)) 
    trainloader = torch.utils.data.DataLoader(train_data, drop_last=True,batch_size=hyperparams["batch_size"], shuffle =True,collate_fn=collate_fn, num_workers=12)
    testloader = torch.utils.data.DataLoader(test_data,drop_last=True, batch_size=Test_batch_size, shuffle =True,collate_fn=collate_fn, num_workers=12)
    model = fusion_layer(hyperparams["embedding_dim"],hyperparams['fusion_dim'],hyperparams["dropout"],hyperparams["ngram"])
    if pretrained:
        model.load_state_dict(torch.load(weight_dir,map_location=torch.device(device2)), strict=strict)

    optimizer = optim.Adam(model.parameters(True), lr = 1e-5)
    criterion = nn.CrossEntropyLoss()
    criterion1 = nn.BCELoss()
    criterion2 = nn.KLDivLoss(reduce = True,size_average=False)
    # criterion2 = nn.MSELoss()

    engine(hyperparams,model,trainloader,testloader,fixed_label_embedding,fixed_task_embedding, optimizer,criterion,criterion1,criterion2)
