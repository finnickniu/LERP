import torch
import numpy as np
import os 
import pickle
import pandas as pd
from torchvision import transforms

class TotalDataset(object):
    def __init__(self, ecg_dir, vital_dir,lab_dir):
        self.ecg_dir = ecg_dir
        self.vital_dir = vital_dir
        self.lab_dir = lab_dir
        self.inter_list = self.gen_index()
        # self.transform = transforms.Compose([transforms.ToTensor,
        #                         transforms.Normalize([0.5], [0.5])])
        # self.annotations = list(sorted(os.listdir(os.path.join(root, "ann"))))
    def gen_index(self):
        ecg_list = os.listdir(self.ecg_dir)
        vital_list = os.listdir(self.vital_dir)
        vital_list = list(map(lambda x: x.replace("n", ""),vital_list))
        lab_list = os.listdir(self.lab_dir)
        inter_list1 = sorted(list(set(ecg_list).intersection(set(vital_list))))
        inter_list2 = sorted(list(set(inter_list1).intersection(set(lab_list))))
        print("dataset length: ",len(inter_list1))
        return inter_list2

    def __getitem__(self, idx):
        ecg_file = open(os.path.join(self.ecg_dir,self.inter_list[idx]), 'rb')
        ecg_f = pickle.load(ecg_file)
        ecg_x = ecg_f["x_ecg"]
        y = ecg_f["y_pyenotype"]
        y = list(map(int, y))
        vital_name = self.inter_list[idx].replace(".pickle","n.pickle")

        vital_file = open(os.path.join(self.vital_dir,vital_name), 'rb')
        vital_f = pickle.load(vital_file)
        vital_x = vital_f["x_ecg"]

        lab_file = open(os.path.join(self.lab_dir,self.inter_list[idx]), 'rb')
        lab_f = pickle.load(lab_file)
        lab_x = lab_f["features"]

        # ecg_x = self.transform(ecg_x)
        # vital_x = self.transform(vital_x)
        # lab_x = self.transform(lab_x)

        return ecg_x,vital_x,lab_x,y
    def __len__(self):
        return len(self.inter_list)




if __name__ == '__main__':
    dataset = TotalDataset('/home/comp/cssniu/RAIM/pickle_data/waveform_stft/train','/home/comp/cssniu/RAIM/pickle_data/vital_data_imputed/train',
    '/home/comp/cssniu/RAIM/pickle_data/lab_demo_data_imputed/train')
    

    x = dataset.__getitem__(0)

