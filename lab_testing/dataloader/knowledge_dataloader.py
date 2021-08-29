import torch
import numpy as np

all_labels =  ['ECG II','ABPSys', 'ABPDias', 'NBPSys', 'NBPDias', 'PULSE', 'RESP',
       'SpO2','Glucose', 'Oxygen saturation', 'pH', 'Temperature', 'Diastolic blood pressure', 
       'Heart Rate', 'Mean blood pressure', 'Respiratory rate', 'Systolic blood pressure']


def load_embeddings(dir):
    # "/home/comp/cssniu/RAIM/embedding.pth"
    f = torch.load(dir)
    labels = f[:17]
    tasks = f[17:]
    f_labels = [i['emdedding_name'] for i in f]
    labels_features = [i['embedding'] for i in labels]
    tasks_features = [i['embedding'] for i in tasks]
    return labels_features,tasks_features

