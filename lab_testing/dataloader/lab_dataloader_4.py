import torch
import numpy as np
import os 
import pickle
import pandas as pd
from collections import deque
from scipy import stats
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_sequence

from channel_wise_rnn import *


class LABDataset(object):
    def __init__(self, data_dir,all_feature=False):
        self.data_dir = data_dir
        self.all_feature = all_feature
        self.lab_list = sorted(os.listdir(data_dir))[:-1]
        self.label_file =  sorted(os.listdir(data_dir))[-1:]

        self.feature_list =["Glucose","Oxygen saturation","pH","Temperature","Diastolic blood pressure",
            "Heart Rate","Mean blood pressure","Respiratory rate","Systolic blood pressure"]
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

        self.imputation_dic = {"Glucose":128,
            "Oxygen saturation":98.0,"pH":7.4,
            "Temperature":36.6,"Diastolic blood pressure":59.0,
            "Heart Rate":86.0,"Mean blood pressure":77.0,
            "Respiratory rate":19,"Systolic blood pressure":118.0}
        
        self.all_imputation_dic = {
            "Capillary refill rate":0.0,
            "Diastolic blood pressure":59.0,
            "Fraction inspired oxygen": 0.21,
            "Glascow coma scale eye opening":4,
            "Glascow coma scale motor response":6,
            "Glascow coma scale total":15,
            "Glascow coma scale verbal response":5,
            "Glucose":128,
            "Heart Rate":86.0,
            "Height":170,
            "Mean blood pressure":77.0,
            "Oxygen saturation":98.0,
            "Respiratory rate":19,
            "Systolic blood pressure":118.0,
            "Temperature":36.6,
            "Weight":81.0,
            "pH":7.4,
            }
    def imputation(self,data,feature_list):
        for n in  feature_list:
            sliding_window = deque(maxlen=4)
            df_column = data[n]

            
            imputed_result = deque()
            init_imputation_value = stats.mode(df_column.values)[0][0]
            if not np.isnan(init_imputation_value):
                sliding_window = [init_imputation_value]*4
            else:
                if self.all_feature:
                    sliding_window = [self.all_imputation_dic[n]]*4
                else:
                    sliding_window = [self.imputation_dic[n]]*4
            # print(self.sliding_window)
            init_imputation_value =  stats.mode(sliding_window)[0][0]
            for i in df_column:
                if np.isnan(i) or i == 0:
                    imputed_result.append(init_imputation_value)
                else:
                    imputed_result.append(i)
                    sliding_window.append(i)

            data[n] = imputed_result

        return data

    def __getitem__(self, idx):
        if self.all_feature:
            lab_file = pd.read_csv(os.path.join(self.data_dir,self.lab_list[idx]))[self.all_feature_list]
            # lab_file = self.imputation(lab_file,self.all_feature_list)

        else:
            lab_file = pd.read_csv(os.path.join(self.data_dir,self.lab_list[idx]))[self.feature_list]
            # lab_file = self.imputation(lab_file,self.feature_list)

        lab_x = lab_file.values
        y_file = pd.read_csv(os.path.join(self.data_dir,self.label_file[0]))
        y = y_file[y_file.stay==self.lab_list[idx]][self.label_list].values[0]
        y1 = 0
        y2 = 0
        y3 = 0
        if 1 in y[:13]: y1 =1
        if 1 in y[13:20]: y2 =1
        if 1 in y[20:]: y3 =1

        # y = [[0,1] if i else [1,0] for i in y]

        return lab_x,np.array([y1,y2,y3])

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
    dataset = LABDataset('/home/comp/cssniu/RAIM/benchmark_data/train/',all_feature=True)
    batch_size = 4
    model = cw_lstm_model(output=True)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    for (data,length,label) in trainloader:
        # pred = model(data,length)
        print(label)
        # print(pred.shape)

