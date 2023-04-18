import numpy as np
import torch
from torch import nn
import torch.optim as optim
import matplotlib.image as mpi
import pandas as pd

TRAINING_DATA_PATH = "/groups/CS156b/data/student_labels/train2023.csv"


def get_train_set():
    df = pd.read_csv(TRAINING_DATA_PATH)
    X = []
    y = [] #Here, y will represent the labels of the Pleural Effusion column
    X_pseudo = []

    for _, r in df.iterrows():
        path = "/groups/CS156b/data/" + r['Path']
        img = mpi.imread(path)
        if r['Pleural Effusion'] != None:
            X.append(mpi.imread(path))
            y.append(r['Pleural Effusion'].asint())
        else:
            X_pseudo.append(img)
    
    X_train = torch.Tensor(X, dtype=torch.float32)
    y_train = torch.Tensor(y, dtype=torch.float32)
    X_predict = torch.Tensor(X_pseudo, dtype=torch.float32)
    return X_train, y_train, X_predict

X_train, y_train, X_predict = get_train_set()