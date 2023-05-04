import cv2
import pandas as pd
import numpy as np


# TOTAL IMAGES = 178158
# SPLIT INTO THREE PART OF SIZES: 60000, 59079, 59079

ROWNUM_COLUMN = "Unnamed: 0.1"
ID_COLUMN = "Unnamed: 0"
PATH_COLUMN = "Path"
DATA_PATH = "/groups/CS156b/data/"
TRAIN_CSV_PATH = "/groups/CS156b/data/student_labels/train2023.csv"
NUM_IMAGES = 60000
START = 0
NEW_DIMS = 224


if __name__ == "__main__":
    print("BEGIN SCRIPT")
    df = pd.read_csv(TRAIN_CSV_PATH)
    NUM_IMAGES = len(df.index) - START
    print(f"NUM_IMAGES: {NUM_IMAGES}")
    assert NUM_IMAGES > 0
    images = np.zeros((NUM_IMAGES, NEW_DIMS, NEW_DIMS), dtype=np.uint8)
    for idx, row, in df.iterrows():
        print(f"Idx#{idx}")
        if idx < START:
            continue
        elif idx >= START + NUM_IMAGES:
            break
        img = cv2.imread(DATA_PATH + row[PATH_COLUMN], cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (NEW_DIMS, NEW_DIMS))
        images[idx - START] = img

    print(f"bytes1:")
    print(f"\tsize={images.size}")
    print(f"\titemsize={images.itemsize}")
    print(f"\ttotal = {images.size * images.itemsize}")
    print(f"bytes2={images.nbytes}")
    
    print(f"JEFE: {images.shape}")

    with open(f"/groups/CS156b/2023/Xray-diagnosis/data/part1_allTrain_224x224.npy", "wb") as f:
        np.save(f, images)
