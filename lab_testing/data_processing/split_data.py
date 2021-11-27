import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from dateutil import tz
import random
import glob
import csv
import os.path
import pickle
import sys
import re
from tqdm import tqdm
import shutil
lab_train_dir = "benchmark_data/train/"
lab_test_dir = "benchmark_data/test/"
hadm_dir = "../mimic3-benchmarks1/data/root/all/"
lab_train_list = os.listdir(lab_train_dir)
lab_test_list = os.listdir(lab_test_dir)
text_dir = "benchmark_data/text/discharge_summary_sentl_all_with_label.csv"
df_text =  pd.read_csv(text_dir)
label_sv_dir = "benchmark_data/all/text"
def gen_file(data_dir,data_list,flag="train"):
    for f in tqdm(data_list):
        df = pd.read_csv(os.path.join(data_dir,f))
        pattern = re.compile(r'\d+')
        results = pattern.findall(f)
        if f == "listfile.csv":continue
        subj_id = int(results[0])
        eposide = int(results[1])-1  
        hadm_id = list(set(pd.read_csv(os.path.join(hadm_dir,str(subj_id),"diagnoses.csv"))["HADM_ID"].values))[eposide]
        text = df_text[df_text["HADM_ID"]==hadm_id]
        if len(text["TEXT_Sentence"]) != 0:
            text.to_csv(os.path.join(label_sv_dir,f),index=False)

        # if len(text["TEXT"])==0 or len(text["TEXT_LONG"])==0:continue
        # text.to_csv(os.path.join(label_sv_dir,flag,f),index=False)
        # text.to_csv(os.path.join(label_sv_dir,f),index=False)


   
# gen_file(lab_train_dir,lab_train_list,flag="train")
# gen_file(lab_test_dir,lab_test_list,flag="train")   


data_dir = "benchmark_data/all/data"
# lab_train_list = os.listdir(os.path.join(label_sv_dir,"train"))
# lab_test_list = os.listdir(os.path.join(label_sv_dir,"test"))
lab_train_list = os.listdir(label_sv_dir)
lab_test_list = os.listdir(label_sv_dir)
def gen_data(data_dir1,data_dir2,data_list,flag="train"):
    for f in tqdm(data_list):
        # shutil.copy(os.path.join(data_dir1,f),os.path.join(data_dir2,flag,f))
        try:
            shutil.copy(os.path.join(data_dir1,f),os.path.join(data_dir2,f))
        except: continue


# gen_data(lab_train_dir,data_dir,lab_train_list,flag="train")
# gen_data(lab_test_dir,data_dir,lab_test_list,flag="test")
# print(len(os.listdir("benchmark_data/all/data")))
# print(len(os.listdir("benchmark_data/all/text")))


data_list = os.listdir("benchmark_data/all/data")

training_index = sorted(random.sample(range(0,len(data_list)), int(len(os.listdir("benchmark_data/all/data"))*0.8)))
test_index = set(range(len((data_list)))) - set(training_index)
print(len(data_list))
print(len(training_index))
print(len(test_index))

text_dir = "benchmark_data/all/text"
data_dir = "benchmark_data/all/data"
def split_data(data_list,training_index,test_index):
    for tr in tqdm(training_index):
        f_name = data_list[tr]
        if ".csv" not in f_name:continue
        shutil.copy(os.path.join(text_dir,f_name),os.path.join(text_dir,"train",f_name))
        shutil.copy(os.path.join(data_dir,f_name),os.path.join(data_dir,"train",f_name))
    
    for te in tqdm(test_index):

        f_name = data_list[te]
        if ".csv" not in f_name:continue
        shutil.copy(os.path.join(text_dir,f_name),os.path.join(text_dir,"test",f_name))
        shutil.copy(os.path.join(data_dir,f_name),os.path.join(data_dir,"test",f_name))


split_data(data_list,training_index,test_index)




