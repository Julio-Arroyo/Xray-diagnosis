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

        # inputs_part1 = np.load(IMAGES_PREFIX + "part1_allTrain_224x224.npy")
        # inputs_part2 = np.load(IMAGES_PREFIX + "part2_allTrain_224x224.npy")
        # inputs_part3 = np.load(IMAGES_PREFIX + "part3_allTrain_224x224.npy")
        assert inputs_part1.shape == (60000, 224, 224)
        assert inputs_part2.shape == (59079, 224, 224)
        assert inputs_part3.shape == (59078, 224, 224)
        inputs = np.concatenate((inputs_part1, inputs_part2, inputs_part3), axis=0, dtype=np.int8)

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

        val_size = 0.2
        val_samples = int(val_size * N)

        # randomly split the data into training and validation sets
        indices = np.random.permutation(N)
        if phase == 'train':
            print(f"Number of pairs train: {len(indices[val_samples:])}")
            inputs, labels = inputs[indices[val_samples:]], labels[indices[val_samples:]]
        elif phase == 'val':
            print(f"Number of pairs val: {len(indices[:val_samples])}")
            inputs, labels = inputs[indices[:val_samples]], labels[indices[:val_samples]]

        self.register_buffer("inputs", torch.tensor(inputs))
        self.register_buffer("labels", torch.tensor(labels))

    def forward(self, dummy):
        return dummy


if __name__ == "__main__":
    train_dataset = torch.jit.script(DataModule("train"))
    val_dataset = torch.jit.script(DataModule("val"))

    torch.jit.save(train_dataset, "/groups/CS156b/2023/Xray-diagnosis/Cpp/data/PY_first60k_train.pt")
    torch.jit.save(val_dataset, "/groups/CS156b/2023/Xray-diagnosis/Cpp/data/PY_first60k_val.pt")
