import matplotlib.image as mpi
import pandas as pd
import numpy as np


NUM_CLASSES = 9
ROWNUM_COLUMN = "Unnamed: 0.1"
ID_COLUMN = "Unnamed: 0"
PATH_COLUMN = "Path"
DATA_PATH = "/groups/CS156b/data/"
TRAIN_CSV_PATH = "/groups/CS156b/data/student_labels/train2023.csv"
NUM_IMAGES = 10000


if __name__ == "__main__":
    df = pd.read_csv(TRAIN_CSV_PATH)
    images = []
    for idx, row, in df.iterrows():
        img = mpi.imread(DATA_PATH + row[PATH_COLUMN])
        images.append(img)
        if idx >= NUM_IMAGES:
            break

    with open("/groups/CS156b/2023/Xray-diagnosis/data/first10kimages.npy", "wb") as f:
        np.save(f, np.array(images))
