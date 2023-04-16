# import matplotlib.image as mpi
import pandas as pd
import numpy as np


NUM_CLASSES = 9
ID_COLUMN = "Unnamed: 0"


if __name__ == "__main__":
    df = pd.read_csv("data/train2023.csv")
    with open("data/train2023_labels.npy", "wb") as f:
        np.save(f, df.iloc[:, -NUM_CLASSES:].values)
