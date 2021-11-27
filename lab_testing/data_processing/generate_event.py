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
import wfdb
import matplotlib.pyplot as plt
import posixpath
from collections import deque
from scipy import stats
from collections import defaultdict
import time
import re
from tqdm import tqdm

FLAG = 'test'


clinical_dir = '/home/comp/cssniu/mimiciii/mimic-iii-clinical-database-1.4'
out_dir = '/home/comp/cssniu/RAIM/pickle_data/intervention_pickle_data'
files = ['PROCEDUREEVENTS_MV.csv',
        'INPUTEVENTS_MV.csv',
        'INPUTEVENTS_CV.csv']
hadm_dir = "../mimic3-benchmarks1/data/root/all/"
pheno_dir1 = "../mimic3-benchmarks1/data/phenotyping/train/"
pheno_dir2 = "../mimic3-benchmarks1/data/phenotyping/test/"

dic_item = pd.read_csv(os.path.join(clinical_dir,"D_ITEMS.csv"),low_memory=False)
# for f in files:
#     df = pd.read_csv(os.path.join(clinical_dir,f))
#     print(df)
numeric_train_data = os.listdir(f'benchmark_data/all/data/{FLAG}/')
sv_dir = f'benchmark_data/all/event1/{FLAG}/'

procedure_mv = pd.read_csv(os.path.join(clinical_dir,files[0]),low_memory=False)
procedure_mv.sort_values("ROW_ID",inplace=True)

input_mv = pd.read_csv(os.path.join(clinical_dir,files[1]),low_memory=False)
input_mv.sort_values("ROW_ID",inplace=True)

input_cv = pd.read_csv(os.path.join(clinical_dir,files[2]),low_memory=False)
input_cv.sort_values("ROW_ID",inplace=True)
def get_event(start_time,event_array,event_df):
    result={}
    time = {}
    for t in tqdm(range(len(event_df))):
        try:
            cur_time = event_df['STARTTIME'].values[t]
        except:
            cur_time = event_df['CHARTTIME'].values[t]

        cur_hr = (datetime.strptime(cur_time, "%Y-%m-%d %H:%M:%S") - datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")).total_seconds() / 3600.0
        item_id =  event_df["ITEMID"].values[t]

        diff_hour = [abs(cur_hr-t) for t in total_hour]

        min_index = diff_hour.index(min(diff_hour))
        # print("min(diff_hour) :",start_time,cur_time,cur_hr,total_hour[min_index],min(diff_hour),item_id)

        result[min_index] = item_id
        time[min_index] = int(cur_hr)
        # event_array[min_index]=1
    return result,time


for f in numeric_train_data:
    if f == "listfile.csv":continue
    try:
        n_df = pd.read_csv(os.path.join(pheno_dir1,f))
    except:
        n_df = pd.read_csv(os.path.join(pheno_dir2,f))
    total_hour = n_df["Hours"].values
    pattern = re.compile(r'\d+')
    results = pattern.findall(f)
    subj_id = int(results[0])
    eposide = int(results[1])-1  
    hadm_id = list(pd.read_csv(os.path.join(hadm_dir,str(subj_id),"stays.csv"))["HADM_ID"].values)[eposide]
    time_df =pd.read_csv(os.path.join(hadm_dir,str(subj_id),"stays.csv"))
    start_time = time_df[time_df["HADM_ID"]==hadm_id]["INTIME"].values[0]


    p_event = procedure_mv[procedure_mv["HADM_ID"]==hadm_id]
    im_event = input_mv[input_mv["HADM_ID"]==hadm_id]
    ic_event = input_cv[input_cv["HADM_ID"]==hadm_id]
    event_array = np.array([0]*len(n_df))
    event_dic = defaultdict(list)

    event_result1={}
    event_result2={}
    event_result3={}
    cur_hr1 = {}
    cur_hr2 = {}
    cur_hr3 = {}

    if not p_event.empty:
        event_result1,cur_hr1 = get_event(start_time,event_array,p_event)
    if not im_event.empty:
        event_result2,cur_hr2 = get_event(start_time,event_array,im_event)

    if not ic_event.empty:
        event_result3,cur_hr3 = get_event(start_time,event_array,ic_event)

    feature1=[]
    feature2=[]
    feature3=[]
    feature4 = []
    for d in tqdm(range(len(n_df))):
        event1 = dic_item[dic_item['ITEMID']==event_result1.get(d)]['LABEL'].values
        event2 = dic_item[dic_item['ITEMID']==event_result2.get(d)]['LABEL'].values
        event3 = dic_item[dic_item['ITEMID']==event_result3.get(d)]['LABEL'].values
        if event1:
            feature1.append(event1[0])
        else:
            feature1.append(np.nan)
        if event2:
            feature2.append(event2[0])
        else:
            feature2.append(np.nan)
        if event3:
            feature3.append(event3[0])
        else:
            feature3.append(np.nan)
        time_stamp = np.nan
        if not pd.isna(cur_hr1.get(d)):
            time_stamp = cur_hr1.get(d)
        elif not pd.isna(cur_hr2.get(d)):
            time_stamp = cur_hr2.get(d)
        elif not pd.isna(cur_hr3.get(d)):
            time_stamp = cur_hr3.get(d)
        # if not pd.isna(time_stamp):
            # time_stamp = time.mktime(time.strptime(time_stamp, "%Y-%m-%d %H:%M:%S"))
        feature4.append(time_stamp)
    # print('-'*100)
    # print(pd.read_csv(os.path.join('benchmark_data/all/data/train/',f)))

    sv_df = pd.concat((pd.DataFrame(feature4,columns=['time_stamp']),pd.DataFrame(feature1,columns=['procedure_event']),pd.DataFrame(feature2,columns=['input_event_mv']),pd.DataFrame(feature3,columns=['input_event_cv'])),axis=1)
    sv_df.to_csv(os.path.join(sv_dir,f),index=False)


        # print(d,event_result1.get(d),event_result2.get(d),event_result3.get(d))