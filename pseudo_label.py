import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import gc
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from resnet import ResNet, ResidualBlock



HPC_PATH = "/groups/CS156b/2023/Xray-diagnosis/"
IMAGES_PATH = "first60k.npy"
LABELS_PATH = "data/train2023_labels.npy"
DISEASE_COL = 5 # 0-th index is no-finding



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"DEVICE: {DEVICE}")


# load the numpy array from the specified path
print("Loading input array")
inputs = np.load(IMAGES_PATH)
N = inputs.shape[0]
labels = np.load(LABELS_PATH)[:N, DISEASE_COL]  # keep only one disease

# remove nan labels
print("removing nan labels")
non_NAN_indices = []
NAN_indices = []
for i in range(N):
    if (labels[i] == 1 or
        labels[i] == 0 or
        labels[i] == -1):
        non_NAN_indices.append(i)
    else:
        NAN_indices.append(i)
non_NAN_indices = np.array(non_NAN_indices, dtype=np.uint8)
NAN_indices = np.array(NAN_indices, dtype=np.uint8)
N = len(non_NAN_indices)

# format images into correct shape and type
inputs = np.expand_dims(inputs[non_NAN_indices], axis=1).astype(np.float32)
nan_inputs = np.expand_dims(inputs[NAN_indices], axis=1).astype(np.float32)
labels = labels[non_NAN_indices].astype(np.float32)
print(labels.shape)

print(f"inputs.shape={inputs.shape}, labels.shape={labels.shape}")

val_size = 0.2
val_samples = int(val_size * N)

# randomly split the data into training and validation sets
indices = np.random.permutation(N)
train_indices, val_indices = indices[val_samples:], indices[:val_samples]

# define the training and validation datasets
print("defining dataset")
train_dataset = TensorDataset(torch.from_numpy(inputs[train_indices]).to(DEVICE),
                              torch.from_numpy(labels[train_indices]).to(DEVICE))
val_dataset = TensorDataset(torch.from_numpy(inputs[val_indices]).to(DEVICE),
                            torch.from_numpy(labels[val_indices]).to(DEVICE))
test_dataset = TensorDataset(torch.from_numpy(nan_inputs).to(DEVICE))

print("Define dataloaders")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
print(val_loader)

num_classes = 3
num_epochs = 20
batch_size = 16
learning_rate = 0.01

model = ResNet(ResidualBlock, [1, 4, 6, 3]).to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)  

# Train the model
total_step = len(train_loader)


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del images, labels, outputs
        torch.cuda.empty_cache()
        gc.collect()

    print ('Epoch [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, loss.item()))
    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
    
        print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total)) 

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        # total += labels.size(0)
        # correct += (predicted == labels).sum().item()
        # del images, labels, outputs

    print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))   
