import matplotlib.image as mpi
import pandas as pd
import numpy as np


NUM_CLASSES = 9
ROWNUM_COLUMN = "Unnamed: 0.1"
ID_COLUMN = "Unnamed: 0"
PATH_COLUMN = "Path"
DATA_PATH = "/groups/CS156b/data/"
TRAIN_CSV_PATH = "/groups/CS156b/data/student_labels/train2023.csv"
NUM_IMAGES = 100


if __name__ == "__main__":
    print("BEGIN SCRIPT")
    df = pd.read_csv(TRAIN_CSV_PATH)
    images = []
    for idx, row, in df.iterrows():
        print(f"Idx#{idx}")
        if idx >= NUM_IMAGES:
            break
        img = mpi.imread(DATA_PATH + row[PATH_COLUMN])
        images.append(img)

    images = np.array(images)
    print(f"JEFE: {images.shape}")

    with open(f"/groups/CS156b/2023/Xray-diagnosis/data/first{NUM_IMAGES}images.npy", "wb") as f:
        np.save(f, images)
