import torch
import numpy as np
import os 
import pickle
import pandas as pd

class VitalDataset(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_list = os.listdir(self.data_dir)

    def __getitem__(self, idx):

        data_file = open(os.path.join(self.data_dir,self.data_list[idx]), 'rb')
        data_f = pickle.load(data_file)
        data_x = data_f["x_ecg"]
        
        y = data_f["y_pyenotype"]
        y =  list(map(int, y))
        return data_x,y

    def __len__(self):
        return len(self.data_list)




if __name__ == '__main__':
    dataset = VitalDataset('/home/comp/cssniu/RAIM/pickle_data/vital_data_imputed/train')
    
    x = dataset.__getitem__(0)
    print(x)

