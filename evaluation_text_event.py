import os
from numpy.core.fromnumeric import shape
# os.chdir("/home/comp/cssniu/RAIM/models/")
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from text_lab_event.lab_text_event_dataloader import TEXTDataset
from text_lab_event.fusion_cls_text_event import fusion_layer
# from text_lab_event.fusion_cls_event import fusion_layer

from lab_testing.dataloader import knowledge_dataloader 

from lab_testing.evaluation import all_metric
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
from lab_testing.dataloader import knowledge_dataloader 
import pandas as pd
import numpy as np
import torch.nn.utils.rnn as rnn_utils

from sklearn.manifold import TSNE
import warnings
import copy
import torch.nn.functional as F
import math
import seaborn as sns
import heapq
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
os.environ['CUDA_VISIBLE_DEVICES']="0"

scaler = MinMaxScaler()
### GPU 23 avg pooling not fixed; GPU 22 avg pooling fixed, GPU 22 flatten fixed, GPU24 flatten not fixed
num_epochs = 1
BATCH_SIZE = 30
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
Best_loss = 10000
Flatten = False
Fixed  = False
# weights_dir = "logs/weights_fusion/ca_flatten_fixed_61_epoch_13_loss_0.2935_acc_0.8929.pth"
# weights_dir = "logs/weights_fusion/ca_flatten_fixed_61_epoch_0_loss_0.3132_acc_0.9036.pth"

# dir = "weights_eval/"
dir = 'weights_fusion_event/'
# weights_list = sorted(os.listdir(f"logs/{dir}"))
weights_list = ["819_text_event_alld_avg_label_no_stopword_epoch_11_loss_0.3089_roc_0.7794.pth"]



hyperparams = {
               'num_epochs':num_epochs,
               'embedding_dim' : 768,
               'fusion_dim':300,
               "output_dim":25,
               'ngram':3,
               'dropout' : 0.5,
               'batch_size' : BATCH_SIZE,
               'device1':device}
label_list = ["Acute and unspecified renal failure",
        "Acute cerebrovascular disease",
        "Acute myocardial infarction",
        "Complications of surgical procedures or medical care",
        "Fluid and electrolyte disorders",
        "Gastrointestinal hemorrhage",
        "Other lower respiratory disease",
        "Other upper respiratory disease",
        "Pleurisy; pneumothorax; pulmonary collapse",
        "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
        "Respiratory failure; insufficiency; arrest (adult)",
        "Septicemia (except in labor)",
        "Shock",
        "Chronic kidney disease",
        "Chronic obstructive pulmonary disease and bronchiectasis",
        "Coronary atherosclerosis and other heart disease",
        "Diabetes mellitus without complication",
        "Disorders of lipid metabolism",
        "Essential hypertension",
        "Hypertension with complications and secondary hypertension",
        "Cardiac dysrhythmias",
        "Conduction disorders",
        "Congestive heart failure; nonhypertensive",
        "Diabetes mellitus with complications",
        "Other liver diseases",
        ]
all_feature_list = ['Capillary refill rate', 'Diastolic blood pressure',
       'Fraction inspired oxygen', 'Glascow coma scale eye opening',
       'Glascow coma scale motor response', 'Glascow coma scale total',
       'Glascow coma scale verbal response', 'Glucose', 'Heart Rate', 'Height',
       'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate',
       'Systolic blood pressure', 'Temperature', 'Weight', 'pH']

accute = ["Acute and unspecified renal failure",
        "Acute cerebrovascular disease",
        "Acute myocardial infarction",
        "Complications of surgical procedures or medical care",
        "Fluid and electrolyte disorders",
        "Gastrointestinal hemorrhage",
        "Other lower respiratory disease",
        "Other upper respiratory disease",
        "Pleurisy; pneumothorax; pulmonary collapse",
        "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
        "Respiratory failure; insufficiency; arrest (adult)",
        "Septicemia (except in labor)",
        "Shock",]
chronic =[
        "Chronic kidney disease",
        "Chronic obstructive pulmonary disease and bronchiectasis",
        "Coronary atherosclerosis and other heart disease",
        "Diabetes mellitus without complication",
        "Disorders of lipid metabolism",
        "Essential hypertension",
        "Hypertension with complications and secondary hypertension"]
mixed= [
        "Cardiac dysrhythmias",
        "Conduction disorders",
        "Congestive heart failure; nonhypertensive",
        "Diabetes mellitus with complications",
        "Other liver diseases",
        ]
tsne = TSNE(random_state=0, perplexity=10)
# sns.set(rc={'figure.figsize':(25,20)})
palette = sns.color_palette("bright", 25)

palette1 = sns.color_palette("dark", 25)

def calc_loss_c(c,criterion,model, y, device):
    """
    torch.tensor([0,1,2]) is decoded identity label vector
    """
    ## 每个class 内部自己做cross entropy， 相当于做了25次， 也就是25个batch，python cross entropy 自带softmax,也不用做onehot
    # print(c.shape)
    f2_c = model.text_fc(c)
    # f2_c = model.fc(c)
    y_c =  torch.stack([torch.range(0, y.shape[1] - 1, dtype=torch.long)]*c.shape[0]).to(device)
    return criterion(f2_c,y_c)

def is_subtoken(word):
    if word[:2] == "##":
        return True
    else:
        return False




def fit(w,pred_result,label_result,criterion,criterion1,epoch,model,testloader,fixed_label_embedding,fixed_task_embedding,hyperparams,flag = "test"):
    global Flatten,Fixed



    device = hyperparams['device1']
    model.eval()


    data_iter = testloader

    fixed_label_embedding_batch = fixed_label_embedding.to(device).transpose(1,0)
    fixed_task_embedding_batch = fixed_task_embedding.to(device).transpose(1,0)

    model.to(device)


    micro_auc_list = []
    macro_auc_list = []
    micro_f1_list = []
    macro_f1_list = []
    micro_precision_list = []
    macro_precision_list = []
    micro_recall_list = []
    macro_recall_list = []
    for i,(data,length,label,text,event_token,task_token,label_token,time_stamp,token_map,len_text) in enumerate(tqdm(data_iter,desc=f"{flag}ing model")):
        event_token = [l.to(device) for l in event_token]
        text_x = [t.to(device) for t in text]
        label_token = [l.to(device) for l in label_token]
        task_token = [t.to(device) for t in task_token]
        time_stamp = [torch.from_numpy(i).to(device,dtype=torch.float) for i in time_stamp]

        lab_x = data.to(device,dtype=torch.float)
        y= label.to(device,dtype=torch.float).squeeze()
        with torch.no_grad():
            # pred,c,t,u,weights,text_pred,weighted_event,event_weight,c_o   = model(text_x,event_token,label_token,task_token,lab_x,length,fixed_label_embedding_batch,fixed_task_embedding_batch,time_stamp,Fixed,Flatten,mode='fusion')
            pred,c,text_label,weights,weighted_event,event_weight,text_event   = model(token_map,text_x,event_token,label_token,task_token,lab_x,length,fixed_label_embedding_batch,fixed_task_embedding_batch,time_stamp,Fixed,Flatten,mode='fusion')

            # pred,weights   = model(text_x,event_token,label_token,task_token,lab_x,length,fixed_label_embedding_batch,fixed_task_embedding_batch,time_stamp,Fixed,Flatten,mode='fusion')

            y = np.array(y.tolist())
         
 
            pred = np.array(pred.tolist())
         


            micro_auc,macro_auc,micro_f1,macro_f1,micro_precision,macro_precision,micro_recall,macro_recall = all_metric(y,pred)
            if micro_auc > 0:
                micro_auc_list.append(micro_auc)
            if macro_auc > 0:
                macro_auc_list.append(macro_auc)
            if micro_f1 > 0:
                micro_f1_list.append(micro_f1)
            if macro_f1 > 0:
                macro_f1_list.append(macro_f1)
            if micro_precision > 0:
                micro_precision_list.append(micro_precision)
            if macro_precision > 0:
                macro_precision_list.append(macro_precision)
            if micro_recall > 0:
                micro_recall_list.append(micro_recall)
            if macro_recall > 0:
                macro_recall_list.append(macro_recall)
    micro_auc_mean = np.array(micro_auc_list).mean()
    macro_auc_mean = np.array(macro_auc_list).mean()

    micro_f1_mean = np.array(micro_f1_list).mean()
    macro_f1_mean = np.array(macro_f1_list).mean()

    micro_precision_mean = np.array(micro_precision_list).mean()
    macro_precision_mean = np.array(macro_precision_list).mean()

    micro_recall_mean = np.array(micro_recall_list).mean()
    macro_recall_mean = np.array(macro_recall_list).mean()
    print('weights: ',w)
    print('micro roc auc: ',micro_auc_mean)
    print('macro roc auc: ',macro_auc_mean)
    print('micro f1: ',micro_f1_mean)
    print('macro f1: ',macro_f1_mean)
    print('micro precision: ',micro_precision_mean)
    print('macro precision: ',macro_precision_mean)
    print('micro recall: ',micro_recall_mean)
    print('macro recall: ',macro_recall_mean)

    print('-' * 20) 
    f = 'note_event_819_all_nosw.txt'
    with open(f,"a") as file:

        file.write(f'weights : {w}'+"\n")
        file.write(f'micro roc auc : {micro_auc_mean}'+"\n")
        file.write(f'macro roc auc : {macro_auc_mean}'+"\n")
        file.write(f'micro f1 : {micro_f1_mean}'+"\n")
        file.write(f'macro f1 : {macro_f1_mean}'+"\n")
        file.write(f'micro precision : {micro_precision_mean}'+"\n")
        file.write(f'macro precision : {macro_precision_mean}'+"\n")
        file.write(f'micro recall : {micro_recall_mean}'+"\n")
        file.write(f'macro recall : {macro_recall_mean}'+"\n")
        file.write('-' * 20+"\n")






  

def engine(w,hyperparams,model,testloader,fixed_label_embedding,fixed_task_embedding,criterion,criterion1):
# def engine(scheduler,model, train_iterator, test_iterator,optimizer,criterion,criterion1):

    start_epoch = 0
    pred_result = []
    label_result = []
    for epoch in range(start_epoch,hyperparams['num_epochs']):
        fit(w,pred_result,label_result,criterion,criterion1,epoch,model,testloader,fixed_label_embedding,fixed_task_embedding,hyperparams,flag = "test")
        # scheduler.step()
    pred_result = np.array(pred_result)
    label_result = np.array(label_result)
    # X_tsne = TSNE(n_components=2,random_state=33).fit_transform(pred_result)
    # X_tsne = scaler.fit_transform(X_tsne)
    # pred_result = scaler.fit_transform(pred_result)

    # pred_df = pd.DataFrame(pred_result)
    # label_df = pd.DataFrame(label_result)
    # with open("data.tsv", 'w') as write_tsv:
    #     write_tsv.write(pred_df.to_csv(sep='\t', index=False,header=False))
    # with open("label.tsv", 'w') as write_tsv:
    #     write_tsv.write(label_df.to_csv(sep='\t', index=False,header=False))



    # sns.scatterplot(X_tsne[:,0].squeeze(), X_tsne[:,1].squeeze(), hue=label_result)

    # plt.savefig('ns_models/images/tsne_word_label.jpg')



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


    test_data = TEXTDataset('/home/comp/cssniu/RAIM/benchmark_data/all/data/test/',flag="test",all_feature=True)

    print('len of test data:', len(test_data)) 
    testloader = torch.utils.data.DataLoader(test_data,drop_last=True, batch_size=BATCH_SIZE, shuffle =True,collate_fn=collate_fn, num_workers=12)

    model = fusion_layer(hyperparams["embedding_dim"],hyperparams['fusion_dim'],hyperparams["dropout"],hyperparams["ngram"])
    for w in tqdm(weights_list):
        weights_dir = os.path.join(f'logs/{dir}',w)
        model.load_state_dict(torch.load(weights_dir,map_location=torch.device(device)), strict=True)
        criterion = nn.CrossEntropyLoss()
        criterion1 = nn.BCELoss()
        engine(w,hyperparams,model,testloader,fixed_label_embedding,fixed_task_embedding,criterion,criterion1)
