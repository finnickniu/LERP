from matplotlib.colors import cnames
from threading import enumerate
from builtins import print
import torch
import numpy as np
import os 
import pickle
import pandas as pd
from collections import deque
from scipy import stats
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel
import re
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",do_lower_case=True,TOKENIZERS_PARALLELISM=True)

class TEXTDataset(object):
    def __init__(self, data_dir,flag="train",all_feature=False):
        self.data_dir = data_dir
        self.all_feature = all_feature
        self.lab_list = sorted(os.listdir(data_dir))
        self.stopword = list(pd.read_csv('/home/comp/cssniu/RAIM/stopwods.csv').values.squeeze())
        # if flag=="train":
        #     self.lab_list = self.lab_list[:int(0.375*len(os.listdir(data_dir)))]

        self.text_dir = os.path.join("/home/comp/cssniu/RAIM/benchmark_data/all/text/",flag)
        self.event_dir = os.path.join('/home/comp/cssniu/RAIM/benchmark_data/all/event1/',flag)
        self.description_df =  pd.read_csv("/home/comp/cssniu/RAIM/benchmark_data/text/task_label_def.csv")
        

        self.all_feature_list = ['Capillary refill rate', 'Diastolic blood pressure',
       'Fraction inspired oxygen', 'Glascow coma scale eye opening',
       'Glascow coma scale motor response', 'Glascow coma scale total',
       'Glascow coma scale verbal response', 'Glucose', 'Heart Rate', 'Height',
       'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate',
       'Systolic blood pressure', 'Temperature', 'Weight', 'pH']
        self.label_list = ["Acute and unspecified renal failure",
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


    def is_subtoken(self,word):
        if word[:2] == "##":
            return True
        else:
            return False
    def detoeken(self,tokens):
        restored_text = []
        for i in range(len(tokens)):
            if not self.is_subtoken(tokens[i]):
                restored_text.append(tokens[i])
                n = 1
                while True:
                    if (i+n) < len(tokens) and  self.is_subtoken(tokens[i+n]):
                        restored_text[-1] = restored_text[-1] + tokens[i+n][2:]
                    else:
                        break
                    n += 1
        return restored_text
    def rm_stop_words(self,text):
        tmp = text.split(" ")
        for t in self.stopword:
            while True:
                if t in tmp:
                    tmp.remove(t)
                else:
                    break
        text = ' '.join(tmp)
        # print(len(text))
        return text

   
    def __getitem__(self, idx):
        if self.all_feature:
            lab_file = pd.read_csv(os.path.join(self.data_dir,self.lab_list[idx]))[self.all_feature_list]
            # lab_file = self.imputation(lab_file,self.all_feature_list)

        else:
            lab_file = pd.read_csv(os.path.join(self.data_dir,self.lab_list[idx]))[self.feature_list]
            # lab_file = self.imputation(lab_file,self.feature_list)
        # print(self.lab_list[idx])
        lab_x = lab_file.values
        label_file =  pd.read_csv(os.path.join(self.text_dir,self.lab_list[idx]))
        # text = label_file["TEXT_LONG"].values[0].split(";")
        text = label_file["TEXT_y"].values[0]

        text = re.sub(r'#|[0-9]+', '', text)
        # print('ori: ',len(text))


        if len(text) == 0:
            print(os.path.join(self.text_dir,self.lab_list[idx]))
        # print('later: ',len(text))

        # print('later: ',text)
        # print(text.split(" "))
        # print(self.stopword)
        # print("len text: ",len(text.split(" ")))
        time_stamp = []
        event_codes = []

        event_file = pd.read_csv(os.path.join(self.event_dir,self.lab_list[idx]))[["procedure_event","input_event_mv","input_event_cv"]].values
        time_stamp = pd.read_csv(os.path.join(self.event_dir,self.lab_list[idx]))["time_stamp"].values
        temp = []
        for i in range((len(event_file))):
            e = event_file[i]
            for j in e:
                if not pd.isnull(j):
                    j = j.lower()
                    j = re.sub(r'[^a-zA-Z\s]', '', j)
                    if j in event_codes: continue
                    if not pd.isnull(time_stamp[i]):
                        temp.append(time_stamp[i])
 

                    event_codes.append(j)
        # event_codes = list(set(event_codes))
        # event_codes = event_codes
        time_stamp = temp
        if not event_codes:
            event_codes.append('Nan')
        if not time_stamp:
            time_stamp.append(0)
        # print(label_file["TEXT_y"])
        task = list(self.description_df["Description"].values[:-25])
        label = list(self.description_df["Description"].values[-25:])
        text = self.rm_stop_words(text)
        # task = self.rm_stop_words(list(self.description_df["Description"].values[:-25]))
        # label = self.rm_stop_words(list(self.description_df["Description"].values[-25:]))
        # event_codes = self.rm_stop_words(event_codes)

        event_token = tokenizer(event_codes, return_tensors="pt", padding=True)

        task_token = tokenizer(task, return_tensors="pt",  max_length = 512, padding=True)

        label_token = tokenizer(label, return_tensors="pt", max_length = 512, padding=True)

        text_token = tokenizer(text, return_tensors="pt",  max_length = 512, padding=True)

        # print(text_token['attention_mask'][0])
        text_token_ = tokenizer.tokenize(text)

        # print(text_token_)
        token_map = {}
        t = 0
        while True:
            if t > 510 or t > len(text_token_)-1:
                break
            sub_token = []
            for t1 in range(t+1,len(text_token_)):
                if '#' in text_token_[t1]:
                    sub_token.append(t1)
                else:break
            token_map[t] = sub_token
            if len(sub_token) > 0:
                t += len(sub_token)+1
            else:
                t += 1
        # print("word legnth: ",text_token["input_ids"].shape,len(token_map.keys()))
        # print(list(token_map.keys())[-1]+1,len(text_token_),text_token["input_ids"].shape[-1]-2)
        # print(len(token_map.keys())+len(token_map.values()))
        # print(token_map.values())
        y = label_file[self.label_list].values
        # y = [[0,1] if i else [1,0] for i in y]
        time_stamp = np.array(time_stamp)
        # print(self.detoeken(text_token_))
        # return lab_x,y,text_token,event_token,task_token,label_token,text_token_,time_stamp,token_map,[len(self.detoeken(text_token_))]
        return lab_x,y,text_token,event_token,task_token,label_token,text_token_,time_stamp,token_map,[1]

    def __len__(self):
        return len(self.lab_list)


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

if __name__ == '__main__':
    dataset = TEXTDataset('/home/comp/cssniu/RAIM/benchmark_data/all/data/train/',flag="train",all_feature=True)
    batch_size = 1
    device = "cuda:2" if torch.cuda.is_available() else "cpu"

    # model = cw_lstm_model(output=True)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    for (data,length,label,text,event_toekn,task_token,label_token,time_stamp,token_map,token_length) in tqdm(trainloader):
        pass
        # break
        # print(token_map)
        # break
        # pred = model(data,length)
        # print(text)
        # print(pred.shape)

