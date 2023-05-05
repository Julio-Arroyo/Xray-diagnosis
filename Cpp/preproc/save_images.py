import cv2
import pandas as pd
import numpy as np
import argparse


# TOTAL IMAGES = 178158
# SPLIT INTO THREE PART OF SIZES: 60000, 59079, 59078

ROWNUM_COLUMN = "Unnamed: 0.1"
ID_COLUMN = "Unnamed: 0"
PATH_COLUMN = "Path"
DATA_PATH = "/groups/CS156b/data/"
TRAIN_CSV_PATH = "/groups/CS156b/2023/Xray-diagnosis/Cpp/data/train2023.csv"

pp = argparse.ArgumentParser(description='')
pp.add_argument('--start', type=int, required=True)
pp.add_argument('--num-images', type=int, required=True)
pp.add_argument('--image-dims', type=int, required=True)
pp.add_argument('--part', type=int, required=True)
args = pp.parse_args()


if __name__ == "__main__":
    df = pd.read_csv(TRAIN_CSV_PATH)
    print(f"args.start: {args.start}")
    print(f"args.num_images: {args.num_images}")
    print(f"args.image_dims: {args.image_dims}")
    print(f"args.part: {args.part}")
    images = np.zeros((args.num_images, args.image_dims, args.image_dims), dtype=np.uint8)
    for idx, row, in df.iterrows():
        if idx < args.start:
            continue
        elif idx >= args.start + args.num_images:
            break
        else:
            print(f"Idx#{idx}: {DATA_PATH + row[PATH_COLUMN]}")
            img = cv2.imread(DATA_PATH + row[PATH_COLUMN], cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (args.image_dims, args.image_dims))
            images[idx - args.start] = img

    print(f"bytes1:")
    print(f"\tsize={images.size}")
    print(f"\titemsize={images.itemsize}")
    print(f"\ttotal = {images.size * images.itemsize}")
    print(f"bytes2={images.nbytes}")
    
    print(f"JEFE: {images.shape}")

    with open(f"/groups/CS156b/2023/Xray-diagnosis/data/part{args.part}_allTrain_224x224.npy", "wb") as f:
        np.save(f, images)
