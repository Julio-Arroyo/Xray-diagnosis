import torch
import torch.nn as nn
import numpy as np


IMAGES_PREFIX = "/groups/CS156b/2023/Xray-diagnosis/data/"
LABELS_PATH = "/groups/CS156b/2023/Xray-diagnosis/data/train2023_labels.npy"
DISEASE_COL = 0  # 0-th index is no-finding
SEED = 69

class DataModule(nn.Module):
    def __init__(self, phase: str):
        super(DataModule, self).__init__()

        np.random.seed(SEED)

        inputs = np.load(IMAGES_PREFIX + "allTrain_224x224.npy")
        if not inputs.shape == (178157, 224, 224):
            print(inputs.shape)
            assert False

        print(inputs[23459])  # check that it looks reasonable

        N = inputs.shape[0]
        labels = np.load(LABELS_PATH)[:N, DISEASE_COL]  # keep only one disease

        # remove nan labels
        print("filtering for nan labels")
        NAN_indices = []
        for i in range(N):
            if (labels[i] != 1 and
                labels[i] != 0 and
                labels[i] != -1):
                NAN_indices.append(i)
        NAN_indices = np.array(NAN_indices, dtype=np.uint8)
        N = len(NAN_indices)
        print(f"There are {N} NAN labels")

        # format images into correct shape and type
        inputs = np.expand_dims(inputs[NAN_indices], axis=1).astype(np.int8)

        self.register_buffer("inputs", torch.tensor(inputs))

    def forward(self, dummy):
        return dummy


if __name__ == "__main__":
    pseudo_train_dataset = torch.jit.script(DataModule("pseudo_train"))
    torch.jit.save(pseudo_train_dataset, f"/groups/CS156b/2023/Xray-diagnosis-rosa1/Xray-diagnosis/data/NANs_for_idx{DISEASE_COL}.pt")
