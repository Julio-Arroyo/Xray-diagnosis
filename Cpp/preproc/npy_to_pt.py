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
        print("removing nan labels")
        non_NAN_indices = []
        for i in range(N):
            if (labels[i] == 1 or
                labels[i] == 0 or
                labels[i] == -1):
                non_NAN_indices.append(i)
        non_NAN_indices = np.array(non_NAN_indices, dtype=np.uint8)
        N = len(non_NAN_indices)
        print(f"There are {N} non-NAN labels")

        # format images into correct shape and type
        inputs = np.expand_dims(inputs[non_NAN_indices], axis=1).astype(np.int8)

        # labels has shape (N,) and each entry is 0, 1, or -1
        labels = labels[non_NAN_indices].astype(np.int8)  
        labels_multiclass = np.zeros((N, 3), dtype=np.int8)
        for i in range(N):
            # 0 -> 0, 1 -> 1, -1 -> 2
            labels_multiclass[i, labels[i]] = 1

        val_size = 0.2
        val_samples = int(val_size * N)

        # randomly split the data into training and validation sets
        indices = np.random.permutation(N)
        np.save("/groups/CS156b/2023/Xray-diagnosis/Cpp/data/entire_multiclass_first20val_last80train.npy", indices)
        if phase == 'train':
            print(f"Number of pairs train: {len(indices[val_samples:])}")
            inputs, labels_multiclass = (inputs[indices[val_samples:]],
                                         labels_multiclass[indices[val_samples:]])
        elif phase == 'val':
            print(f"Number of pairs val: {len(indices[:val_samples])}")
            inputs, labels_multiclass = (inputs[indices[:val_samples]],
                                         labels_multiclass[indices[:val_samples]])

        print(f"inputs shape: {inputs.shape}")
        print(f"labels shape: {labels_multiclass.shape}")
        self.register_buffer("inputs", torch.tensor(inputs))
        self.register_buffer("labels", torch.tensor(labels_multiclass))

    def forward(self, dummy):
        return dummy


if __name__ == "__main__":
    train_dataset = torch.jit.script(DataModule("train"))
    val_dataset = torch.jit.script(DataModule("val"))

    torch.jit.save(train_dataset, "/groups/CS156b/2023/Xray-diagnosis/Cpp/data/entire_train_multiclass.pt")
    torch.jit.save(val_dataset, "/groups/CS156b/2023/Xray-diagnosis/Cpp/data/entire_val_multiclass.pt")
