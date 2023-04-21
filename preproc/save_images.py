import cv2
import pandas as pd
import numpy as np


NUM_CLASSES = 9
ROWNUM_COLUMN = "Unnamed: 0.1"
ID_COLUMN = "Unnamed: 0"
PATH_COLUMN = "Path"
DATA_PATH = "/groups/CS156b/data/"
TRAIN_CSV_PATH = "/groups/CS156b/data/student_labels/train2023.csv"
NUM_IMAGES = 3
NEW_DIMS = 256
NEW_DIMS = 256

if __name__ == "__main__":
    print("BEGIN SCRIPT")
    df = pd.read_csv(TRAIN_CSV_PATH)
    images = np.zeros((NUM_IMAGES, NEW_DIMS, NEW_DIMS), dtype=np.uint8)
    for idx, row, in df.iterrows():
        print(f"Idx#{idx}")
        if idx >= NUM_IMAGES:
            break
        img = cv2.imread(DATA_PATH + row[PATH_COLUMN], cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (NEW_DIMS, NEW_DIMS))
        images[idx] = img

    print(f"bytes1:")
    print(f"\tsize={images.size}")
    print(f"\titemsize={images.itemsize}")
    print(f"\ttotal = {images.size * images.itemsize}")
    print(f"bytes2={images.nbytes}")
    
    print(f"JEFE: {images.shape}")

    with open(f"/groups/CS156b/2023/Xray-diagnosis/data/first{NUM_IMAGES}images.npy", "wb") as f:
        np.save(f, images)
