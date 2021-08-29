import os
from numpy.core.fromnumeric import shape
# os.chdir("/home/comp/cssniu/RAIM/models/")
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from text_lab.lab_text_dataloader import TEXTDataset
from torchtext import data 
from text_lab_event.lab_text_event_dataloader import TEXTDataset
from text_lab_event.fusion_cls_text_event import fusion_layer

from lab_testing.dataloader import knowledge_dataloader 

from lab_testing.evaluation import all_metric
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
from lab_testing.dataloader import knowledge_dataloader 

import numpy as np
from transformers import AutoTokenizer, AutoModel

import torch.nn.utils.rnn as rnn_utils
import pandas as pd
from sklearn import metrics
import re

from sklearn.manifold import TSNE
import warnings
import copy
import torch.nn.functional as F
import math
import seaborn as sns
import heapq
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
warnings.filterwarnings('ignore')

### GPU 23 avg pooling not fixed; GPU 22 avg pooling fixed, GPU 22 flatten fixed, GPU24 flatten not fixed
num_epochs = 1
BATCH_SIZE = 30
device = "cuda:3" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
Best_loss = 10000
Flatten = True
Fixed  = False
# weights_dir = "logs/weights_fusion_event/85_text_event_early_fusion_epoch_8_loss_0.3474_roc_0.7937.pth"
# weights_dir = "logs/weights_fusion_event/87_text_event_two_modality_fusion_epoch_8_loss_0.3108_roc_0.7758.pth"
weights_dir = "logs/weights_fusion_event/819_text_event_alld_avg_label_no_stopword_epoch_11_loss_0.3089_roc_0.7794.pth"
# weights_dir = "logs/weights_fusion_event/819_text_event_alld_avg_label_no_stopword_epoch_7_loss_0.3215_roc_0.7725.pth"

# weights_dir = "logs/weights_fusion_event/819_text_event_alld_avg_label_no_stopword_epoch_10_loss_0.3186_roc_0.7752.pth"

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
all_feature_list = ['Capillary refill rate', 
        'Diastolic blood pressure',
       'Fraction inspired oxygen', 
       'Glascow coma scale eye opening',
       'Glascow coma scale motor response', 
       'Glascow coma scale total',
       'Glascow coma scale verbal response', 
       'Glucose', 
       'Heart Rate', 
       'Height',
       'Mean blood pressure', 
       'Oxygen saturation', 
       'Respiratory rate',
       'Systolic blood pressure', 
       'Temperature', 
       'Weight', 
       'pH']
tsne = TSNE(random_state=0, perplexity=10)
# sns.set(rc={'figure.figsize':(20,15)})
palette = sns.color_palette("bright",25)

palette1 = sns.color_palette("dark", 3)

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

def detoeken(tokens):
    restored_text = []
    for i in range(len(tokens)):
        if not is_subtoken(tokens[i]) and (i+1)<len(tokens) and is_subtoken(tokens[i+1]):
            restored_text.append(tokens[i] + tokens[i+1][2:])
            if (i+2)<len(tokens) and is_subtoken(tokens[i+2]):
                restored_text[-1] = restored_text[-1] + tokens[i+2][2:]
        elif not is_subtoken(tokens[i]):
            restored_text.append(tokens[i])
    return restored_text

def decode_attetnion(restored_original_text,tokens,score_index):

    restored_text = []
    tokens = [tokens[i] for i in score_index]
    for i in range(len(tokens)):
        if not is_subtoken(tokens[i]) and (i+1)<len(tokens) and is_subtoken(tokens[i+1]):
            w = tokens[i] + tokens[i+1][2:]
            if w in restored_original_text:
                restored_text.append(w)

        if (i+2)<len(tokens) and is_subtoken(tokens[i+2]):
                w = restored_text[-1] + tokens[i+2][2:]
                if w in restored_original_text:
                    restored_text[-1] = w

        elif not is_subtoken(tokens[i]):
            restored_text.append(tokens[i])

    return restored_text


def fit(criterion,criterion1,model,case_study_data_list,fixed_label_embedding,fixed_task_embedding,hyperparams,event_codes,flag = "test"):
    global Flatten,Fixed



    device = hyperparams['device1']
    model.eval()



    fixed_label_embedding = fixed_label_embedding.to(device).transpose(1,0)
    fixed_task_embedding = fixed_task_embedding.to(device).transpose(1,0)

    model.to(device)
    tsne_results = []
    label_result = []
    for case_study_data in case_study_data_list:
        data,length,y,text_x,event_token,task_token,label_token,string_token,time_stamp,token_map =  case_study_data[0],case_study_data[1],case_study_data[2],case_study_data[3],case_study_data[4],case_study_data[5],case_study_data[6],case_study_data[7],case_study_data[8],case_study_data[9]
        text_x = [text_x.to(device)]
        event_token = [event_token.to(device)]

        label_token = [label_token.to(device)]
        task_token = [task_token.to(device)]
        lab_x = data.to(device,dtype=torch.float).unsqueeze(0)
        y= y.to(device,dtype=torch.float).squeeze().unsqueeze(0)
        time_stamp = time_stamp.to(device,dtype=torch.float)

        with torch.no_grad():
            # pred,weights = model(text_x,label_token,task_token)

            pred,c,text_label,weights,weighted_event,event_weight,text_event  = model([token_map],text_x,event_token,label_token,task_token,lab_x,length,fixed_label_embedding,fixed_task_embedding,time_stamp,Fixed,Flatten,mode='fusion')

            weights = weights.cpu().data.squeeze(0).tolist()
            weights = [i[0] for i in weights]

            # print('weights: ',weights.shape)
            # print('weights: ',weights[0:1,0:1].squeeze(0))
            # weights =weights.cpu().data

            # weights = weights[0:1,1:-1].squeeze(0).tolist()   

            event_weight =event_weight.cpu().data.squeeze(0).tolist() 
            event_weight = [i[0] for i in event_weight]

            # event_weight =event_weight.cpu().data
            # event_weight = event_weight[0:1,1:-1].squeeze(0).tolist()  
            # print(len(weights),len(string_token),weights) 
            # max_num_index_weights = sorted(list(map(weights.index, heapq.nlargest(int(len(weights)*1), weights))))
            # print("text: ",[string_token[i] for i in max_num_index_weights])
            avg_att_weights = [(g + h) / 2 for g, h in zip(weights, event_weight)]
            max_num_index_weights = sorted(list(map(avg_att_weights.index, heapq.nlargest(int(len(avg_att_weights)*0.8), avg_att_weights))))
            print("text: ",[string_token[i] for i in max_num_index_weights])
            y =y.cpu().data

            y_index = np.argwhere(y[0] ==1 ).squeeze()
            label = [label_list[i] for i in y_index]
            print(label)
            # g = scaler.fit_transform(g.squeeze())
            text_label = text_event.cpu().data.squeeze(0)
            text_label = scaler.fit_transform(text_label)
            print(text_label.shape)

            text_label = pd.DataFrame(text_label)
            x_axis_labels = range(25) 
            y_axis_labels = event_codes # labels for y-axis
            # res = sns.heatmap(text_label,xticklabels=x_axis_labels, yticklabels=y_axis_labels,cmap="Blues")

            res = sns.heatmap(text_label,cmap="Blues")
            # res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 8)
            # res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 8)
            # sns.heatmap(text_label,cmap="Blues")
            plt.savefig('LDAM/images/heat_map_event_label.jpg')
            plt.clf()

         
    # c = scaler.fit_transform(c.squeeze())
    # t = scaler.fit_transform(t.squeeze())
    # X_tsne2 = tsne.fit_transform(c)
    # X_tsne3 = tsne.fit_transform(t)
    # # X_tsne2 = scaler.fit_transform(X_tsne2.squeeze())
    # # X_tsne3 = scaler.fit_transform(X_tsne3.squeeze())
    # sns.scatterplot(X_tsne2[:,0].squeeze(), X_tsne2[:,1].squeeze())
    # for t in range(len(X_tsne2)):
    #     plt.text(X_tsne2[t:t+1,0].squeeze(),X_tsne2[t:t+1,1].squeeze()-0.05,range(25)[t])
    
    # sns.scatterplot(X_tsne3[:,0].squeeze(), X_tsne3[:,1].squeeze())
    # # for i in range(17):
    # #     plt.scatter(X_tsne3[:, 0], X_tsne3[:, 1],color = "orange")
    # for t in range(len(X_tsne3)):
    #     plt.text(X_tsne3[t:t+1,0].squeeze(),X_tsne3[t:t+1,1].squeeze()+0.05,range(17)[t],size = "large")
    # plt.savefig(f'LDAM/images/tsne_label_task.jpg')


def is_subtoken(word):
    if word[:2] == "##":
        return True
    else:
        return False
def detoeken(tokens):
    restored_text = []
    for i in range(len(tokens)):

        if not is_subtoken(tokens[i]):
          restored_text.append(tokens[i])
          n = 1
          while True:
            if (i+n) < len(tokens) and  is_subtoken(tokens[i+n]):
                restored_text[-1] = restored_text[-1] + tokens[i+n][2:]
            else:
                break
            n += 1
    return restored_text

def rm_stop_words(text):
    stopword = list(pd.read_csv('/home/comp/cssniu/RAIM/stopwods.csv').values.squeeze())
    tmp = text.split(" ")
    for t in stopword:
        while True:
            if t in tmp:
                tmp.remove(t)
            else:
                break
    text = ' '.join(tmp)
    # print(len(text))
    return text


def collate_fn(data):
    """
    定义 dataloader 的返回值
    :param data: 第0维：data，第1维：label
    :return: 序列化的data、记录实际长度的序列、以及label列表
    """
    data_length = [data[0].shape[0]]

    input_x = data[0].tolist()
    y =  data[1]
    text =  data[2]
    event_token = data[3]
    task_token =  data[4]
    label_token =   data[5]
    string_token =  data[6]
    time_stamp = data[7]
    toke_map = data[8]

    data = rnn_utils.pad_sequence([torch.from_numpy(np.array(x)) for x in input_x],batch_first = True, padding_value=0)
    return data.unsqueeze(-1), data_length, torch.tensor(y, dtype=torch.float32),text,event_token,task_token,label_token,string_token,torch.tensor(time_stamp, dtype=torch.float32),toke_map

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",do_lower_case=True)

    task_embedding,label_embedding= knowledge_dataloader.load_embeddings("/home/comp/cssniu/RAIM/embedding.pth")
    fixed_label_embedding = torch.stack(label_embedding)
    fixed_task_embedding = torch.stack(task_embedding)

    model = fusion_layer(hyperparams["embedding_dim"],hyperparams['fusion_dim'],hyperparams["dropout"],hyperparams["ngram"])
    model.load_state_dict(torch.load(weights_dir,map_location=torch.device(device)), strict=True)

    criterion = nn.CrossEntropyLoss()
    criterion1 = nn.BCELoss()
    best_f1 = 0
    best_f1_name =''
    # for test_name in tqdm(sorted(os.listdir('/home/comp/cssniu/RAIM/benchmark_data/all/data/test/'))):

    data_dir = os.path.join("/home/comp/cssniu/RAIM/benchmark_data/all/data/test")

    text_dir = os.path.join("/home/comp/cssniu/RAIM/benchmark_data/all/text/test")
    event_dir = os.path.join("/home/comp/cssniu/RAIM/benchmark_data/all/event1/test")
    description_df =  pd.read_csv("/home/comp/cssniu/RAIM/benchmark_data/text/task_label_def.csv")
    test_name = '53193_episode1_timeseries.csv'
    # test_name = '9501_episode1_timeseries.csv'
    test_name_list = ["44_episode1_timeseries.csv","53193_episode1_timeseries.csv","10068_episode1_timeseries.csv"]
    test_name_list = ['10068_episode1_timeseries.csv']

    # test_name_list = ['54289_episode1_timeseries.csv']
    # test_name_list = ["11061_episode1_timeseries.csv"]

    case_study_data_list = []
    for test_name in test_name_list:
        data = pd.read_csv(os.path.join(data_dir,test_name))[all_feature_list].values
        text_df = pd.read_csv(os.path.join(text_dir,test_name))
        event_df = pd.read_csv(os.path.join(event_dir,test_name))[["procedure_event","input_event_mv","input_event_cv"]].values
        time_stamp = pd.read_csv(os.path.join(event_dir,test_name))["time_stamp"].values
        temp = []
        event_codes = []
        for i in range((len(event_df))):
            e = event_df[i]
            for j in e:

                if not pd.isnull(j):
                 
                    j = j.lower()
                    j = re.sub(r'[^a-zA-Z\s]', '', j)
                    if j in event_codes: continue
                    if not pd.isnull(time_stamp[i]):
                        temp.append(time_stamp[i])
 
                    # event_token = tokenizer.tokenize(j)
                    # print(event_token)

                    event_codes.append(j)
        # event_codes = list(set(event_codes))
        # event_codes = event_codes

        print(event_codes)
        time_stamp = np.array(temp)


        text = text_df["TEXT_y"].values[0]
        # case_study_data = [data_file,lab_file]
        task = list(description_df["Description"].values[:-25])
        label = list(description_df["Description"].values[-25:])
        text = rm_stop_words(text)
        # task = rm_stop_words(list(description_df["Description"].values[:-25]))
        # label = rm_stop_words(list(description_df["Description"].values[-25:]))
        # event_codes = rm_stop_words(event_codes)
        task_token = tokenizer(task, return_tensors="pt", padding=True)
        label_token = tokenizer(label, return_tensors="pt", padding=True)
        text_token = tokenizer(text, return_tensors="pt", padding=True)
        event_token = tokenizer(event_codes,  return_tensors="pt", padding=True)
        EVENT_CODE_token = tokenizer.tokenize(event_codes,is_split_into_words=True)

        string_token = tokenizer.tokenize(text)
        token_map = {}
        t = 0
        while True:
            if t > 510 or t > len(string_token)-1:
                break
            sub_token = []
            for t1 in range(t+1,len(string_token)):
                if '#' in string_token[t1]:
                    sub_token.append(t1)
                else:break
            token_map[t] = sub_token
            if len(sub_token) > 0:
                t += len(sub_token)+1
            else:
                t += 1
        string_token = detoeken(string_token)
        # print(string_token)
        y = text_df[label_list].values

        # print([label_list[int(i)] for i in np.where(y[0] == 1)[0]])
        temp_data = [data,y,text_token,event_token,task_token,label_token,string_token,time_stamp,token_map]
        case_study_data_list.append(collate_fn(temp_data))
    fit(criterion,criterion1,model,case_study_data_list,fixed_label_embedding,fixed_task_embedding,hyperparams,event_codes,flag = "test")
