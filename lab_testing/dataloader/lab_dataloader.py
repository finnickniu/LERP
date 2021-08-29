import torch
import numpy as np
import os 
import pickle
import pandas as pd
from collections import deque
from scipy import stats
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_sequence


class TEXTDataset(object):
    def __init__(self, data_dir,flag="train",all_feature=False):
        self.data_dir = data_dir
        self.all_feature = all_feature
        self.lab_list = sorted(os.listdir(data_dir))
        self.text_dir = os.path.join("/home/comp/cssniu/RAIM/benchmark_data/all/text/",flag)
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


   
    def __getitem__(self, idx):
        if self.all_feature:
            lab_file = pd.read_csv(os.path.join(self.data_dir,self.lab_list[idx]))[self.all_feature_list]
            # lab_file = self.imputation(lab_file,self.all_feature_list)

        else:
            lab_file = pd.read_csv(os.path.join(self.data_dir,self.lab_list[idx]))[self.feature_list]
            # lab_file = self.imputation(lab_file,self.feature_list)

        lab_x = lab_file.values
        label_file =  pd.read_csv(os.path.join(self.text_dir,self.lab_list[idx]))

        y = label_file[self.label_list].values
        # y = [[0,1] if i else [1,0] for i in y]
        
        return lab_x,y

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

    data = rnn_utils.pad_sequence([torch.from_numpy(np.array(x)) for x in input_x],batch_first = True, padding_value=0)
    return data.unsqueeze(-1), data_length, torch.tensor(y, dtype=torch.float32)

if __name__ == '__main__':
    dataset = TEXTDataset('/home/comp/cssniu/RAIM/benchmark_data/all/data/train/',flag="train",all_feature=True)
    batch_size = 1
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    for (data,length,label,text,task_token,label_token) in trainloader:
        print(data.shape)

        break
        # print(text)

