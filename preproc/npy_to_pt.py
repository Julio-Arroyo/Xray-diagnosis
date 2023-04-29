import torch
import torch.nn as nn
import numpy as np


IMAGES_PATH = "/groups/CS156b/2023/Xray-diagnosis/data/first60k.npy"
LABELS_PATH = "/groups/CS156b/2023/Xray-diagnosis/data/train2023_labels.npy"
DISEASE_COL = 0  # 0-th index is no-finding


class DataModule(nn.Module):
    def __init__(self):
        super(DataModule, self).__init__()

        inputs = np.load(IMAGES_PATH)
        N = inputs.shape[0]
        labels = np.load(LABELS_PATH)[:N, DISEASE_COL]  # keep only one disease

        # remove nan labels
        print("removing nan labels")
        non_NAN_indices = []
        for i in range(N):
            if (labels[i] == 1 or
                labels[i] == 0 or
                labels[i] == -1):
                non_NAN_indices.append(i)
        non_NAN_indices = np.array(non_NAN_indices, dtype=np.uint8)
        N = len(non_NAN_indices)

        # format images into correct shape and type
        inputs = np.expand_dims(inputs[non_NAN_indices], axis=1).astype(np.int8)
        labels = labels[non_NAN_indices].astype(np.int8)

        self.register_buffer("inputs", torch.tensor(inputs))
        self.register_buffer("labels", torch.tensor(labels))

    def forward(self, dummy):
        return dummy


if __name__ == "__main__":
    dataset = torch.jit.script(DataModule())
    torch.jit.save(dataset, "/groups/CS156b/2023/Xray-diagnosis/Cpp/data/first60k.pt")
