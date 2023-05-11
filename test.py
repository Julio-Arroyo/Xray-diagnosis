import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models.cnn_scalar import CNN

tensor = torch.load("/groups/CS156b/2023/Xray-diagnosis-rosa1/Xray-diagnosis/data/NANs_for_idx0.pt")
