import torch
import numpy as np
import os 
import pickle
import pandas as pd
from collections import deque,Counter
from scipy import stats
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_sequence
import re
from transformers import BartTokenizer
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
import nltk
nltk.download('words')
from nltk.corpus import words
from random import sample
tokenizer = RegexpTokenizer('\s+', gaps= True)
def data_processing(data):
    
    return ''.join([i for i in data.lower() if not i.isdigit()])
data_dir = '/home/comp/cssniu/mimiciii/mimic-iii-clinical-database-1.4/NOTEEVENTS.csv'
sv_dir =  f'benchmark_data/all/text/'
text_csv = pd.read_csv(data_dir,low_memory=False)
## [2083180 rows x 11 columns]
# print(text_csv.loc[text_csv['CATEGORY'] == 'Discharge summary'])
## 59652 rows x 11 columns
discharge_summary_df = text_csv[text_csv['CATEGORY'] == 'Discharge summary']
report_df = discharge_summary_df[discharge_summary_df['DESCRIPTION']=="Report"]
## [55177 rows x 11 columns]
# print(report_df)
# print(len(set(report_df['SUBJECTj_df_ID'].values)))
unique_subjID = set(report_df['SUBJECT_ID'].values)
for i in tqdm(unique_subjID):
    suj_df = report_df[report_df['SUBJECT_ID']==i]
    HADM_ID = suj_df["HADM_ID"].values
    CHARTDATE = suj_df["CHARTDATE"].values
    text = suj_df['TEXT'].values
    for n,t in enumerate(text):
        breif_course = re.findall(r"Brief Hospital Course.*?[\s\S]*Medications on Admission",t)
        chief_complaint = re.findall(r"Chief Complaint:\n.*?\n\n",t)
        # print(chief_complaint)

        if  breif_course:
                # print(chief_complaint)

            if not chief_complaint:
                chief_complaint = [' '.join(sample(list(words.words()), 2))]
                # print(words.words())
# 
            breif_course = [breif_course[0].replace('Brief Hospital Course','').replace("Medications on Admission","")]
            breif_course = tokenizer.tokenize(breif_course[0])
            # print(breif_course)
            breif_course = [b.lower() for b in breif_course if b.isalpha()] 
            breif_course = [w for w in breif_course if len(w) > 1 if w not in stopwords.words('english')]
            breif_course = pd.DataFrame(breif_course,columns=['text'])
 

            breif_course.to_csv(os.path.join(sv_dir,f"{i}_{int(HADM_ID[n])}.csv"),index=False)
    


                


            # breif_course = [data_processing(breif_course[0])]
            # breif_course = pd.DataFrame(breif_course,columns=['text'])

            # print(breif_course['text'].values)


     

        #         print(".....................................................")
        
    # break

# text = report_df[report_df['SUBJECT_ID']==176]['TEXT'].values[0]

# for subt in text.split("\n\n"):
#     if "Brief Hospital Course" in subt:

#         print(subt)
