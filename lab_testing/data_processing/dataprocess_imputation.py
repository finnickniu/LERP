import torch
import numpy as np
import os 
import pickle
import pandas as pd
from collections import deque
from scipy import stats
import re



all_feature_list = ['Capillary refill rate', 'Diastolic blood pressure',
'Fraction inspired oxygen', 'Glascow coma scale eye opening',
'Glascow coma scale motor response', 'Glascow coma scale total',
'Glascow coma scale verbal response', 'Glucose', 'Heart Rate', 'Height',
'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate',
'Systolic blood pressure', 'Temperature', 'Weight', 'pH']

all_imputation_dic = {
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

revise_list= ['Glascow coma scale eye opening',
'Glascow coma scale motor response', 
'Glascow coma scale verbal response', ]


def imputation(data,feature_list,imputation_dic):
        for n in  feature_list:
            sliding_window = deque(maxlen=4)
            df_column = data[n]
            # if n in revise_list:
            #     # df_column = pd.DataFrame([d if  pd.isnull(d) else float(d[0]) for d in df_column])[0]
            #     temp = []
            #     for d in df_column:
            #         if not pd.isnull(d):
            #             print(d)
            #             float(d[0])
            #         temp.append(d)
            #     df_column =  pd.DataFrame(temp)[0]
            imputed_result = deque()
            init_imputation_value = stats.mode(df_column.values)[0][0]

            if not pd.isnull(init_imputation_value):
                sliding_window = [init_imputation_value]*4
            else:
                sliding_window = [imputation_dic[n]]*4
            # print(self.sliding_window)
            init_imputation_value =  stats.mode(sliding_window)[0][0]
            for i in df_column:
                if pd.isnull(i) or i == 0 or i == "None" or i =="Spontaneously":
                    imputed_result.append(init_imputation_value)
                else:
                    imputed_result.append(i)
                    sliding_window.append(i)
            # imputed_result1 = [float(d[0]) if isinstance(d,str) else d for d in imputed_result]
            data[n] = imputed_result
            if n in revise_list:
                # df_column = pd.DataFrame([d if  pd.isnull(d) else float(d[0]) for d in df_column])[0]
                temp = []
                for d in imputed_result:
                    if isinstance(d,str):
                        num =re.findall(r"\d",d)
                        if num:
                            d = float(num[0])
                        else:
                            d = float(all_imputation_dic[n])
                    temp.append(d)
                data[n] = temp       
        return data

def imputation_data(data_dir,sv_dir):
    lab_list = sorted(os.listdir(data_dir))[:-1]
    label_file =  sorted(os.listdir(data_dir))[-1:]
    for idx in range(len(lab_list)):
        print("processing: ",os.path.join(data_dir,lab_list[idx]))
        lab_file = pd.read_csv(os.path.join(data_dir,lab_list[idx]))[all_feature_list]
        lab_file = imputation(lab_file,all_feature_list,all_imputation_dic)
        # print(lab_file)
        lab_file.to_csv(os.path.join(sv_dir,lab_list[idx]),index=False)

 

if __name__ == '__main__':
    data_dir = '/home/comp/cssniu/mimic3-benchmarks/data/phenotyping/train'
    save_dir = "/home/comp/cssniu/RAIM/benchmark_data/train"
    imputation_data(data_dir,save_dir)