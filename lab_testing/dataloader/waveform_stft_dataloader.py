import torch
import numpy as np
import os 
import pickle
import pandas as pd

class WAVEFORMDataset(object):
    def __init__(self, ecg_dir):
        self.ecg_dir = ecg_dir
        self.ecg_list = os.listdir(self.ecg_dir)
 
    def __getitem__(self, idx):


        ecg_file = open(os.path.join(self.ecg_dir, self.ecg_list[idx]), 'rb')
        ecg_f = pickle.load(ecg_file)
        print(ecg_f)
        ecg_x = ecg_f["x_ecg"]
        y = ecg_f["y_pyenotype"]        
        return ecg_x, y
    def __len__(self):
        return len(self.ecg_list)




if __name__ == '__main__':
    dataset = WAVEFORMDataset('/home/comp/cssniu/RAIM/pickle_data/waveform_stft/train')
    

    x = dataset.__getitem__(0)

