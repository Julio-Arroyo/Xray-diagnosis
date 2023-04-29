import torch
import torch.nn as nn
import numpy as np


class DataModule(nn.Module):
    def __init__(self):
        super(DataModule, self).__init__()
        X = np.load("data/first60k.npy")
        self.register_buffer('data', torch.tensor(X))
    
    def forward(self, dummy):
        return dummy


if __name__ == "__main__":
    dataset = torch.jit.script(DataModule())

    torch.jit.save(dataset, "/groups/CS156b/2023/Xray-diagnosis/Cpp/data/first60k.pt")
