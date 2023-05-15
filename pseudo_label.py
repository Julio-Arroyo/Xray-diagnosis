import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import gc
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models.cnn_scalar import CNN



HPC_PATH = "/groups/CS156b/2023/Xray-diagnosis/"
IMAGES_PATH = "first60k.npy"
LABELS_PATH = "data/train2023_labels.npy"
DISEASE_COL = 5 # 0-th index is no-finding



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"DEVICE: {DEVICE}")


# load the numpy array from the specified path
print("Loading input array")
inputs = np.load(HPC_PATH+IMAGES_PATH)
N = inputs.shape[0]
labels = np.load(HPC_PATH+LABELS_PATH)[:N, DISEASE_COL]  # keep only one disease

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
# nan_inputs = inputs[non_NAN_indices].astype(np.float32)
non_expanded_inputs = np.copy(inputs)
inputs = np.expand_dims(non_expanded_inputs[non_NAN_indices], axis=1).astype(np.float32)
nan_inputs = np.expand_dims(non_expanded_inputs[NAN_indices], axis=1).astype(np.float32)
all_labels = np.copy(labels) #used for predicting pseudolabels
labels = labels[non_NAN_indices].astype(np.float32)

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

num_epochs = 20
batch_size = 16
learning_rate = 0.01

model = CNN().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 10
train_losses = []
val_losses = []

print("begin training loop")
for epoch in range(num_epochs):
    model.eval()  # set the model to evaluation mode
    
    val_loss = 0.0
    
    # disable gradient computation for validation
    with torch.no_grad():
        # loop over the validation data loader
        for i, data in enumerate(val_loader, 0):
            # get the inputs and labels from the data loader
            inputs, labels = data
            
            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # add the loss to the running validation loss
            val_loss += loss.item()

    running_loss = 0.0
    
    model.train()  # set the model to training mode
    
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        
        optimizer.zero_grad()  # zero the parameter gradients
        
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    train_losses.append(running_loss)
    val_losses.append(val_loss)
            
    # print statistics every epoch
    print('[%d] train loss: %.3f, val loss: %.3f' %
          (epoch + 1, running_loss / len(train_loader), val_loss / len(val_loader)))
    # torch.save(model.state_dict(), ".pt")


print('Finished Training')
# Predict the NaN labels using the trained model
model.eval()
pseudo_labels = None
with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        inputs = data[0]
        outputs = model(inputs)
        pseudo_labels = outputs
all_labels[NAN_indices] = pseudo_labels.numpy()
non_val_indices = np.delete(np.arange(N), val_indices)
pseudo_training_inputs = np.expand_dims(non_expanded_inputs[non_val_indices], axis=1).astype(np.float32)
#remove val_indices from all_inputs
pseudo_dataset = TensorDataset(torch.from_numpy(pseudo_training_inputs).to(DEVICE),
                              torch.from_numpy(all_labels[non_val_indices].astype(np.float32)).to(DEVICE))

pseudo_loader = DataLoader(pseudo_dataset, batch_size=16, shuffle=False)


print("Created Pseudo Dataset")
pseudo_train_losses = []
pseudo_val_losses = []

print("begin training loop on pseudo data")
for epoch in range(num_epochs):
    model.eval()  # set the model to evaluation mode
    
    pseudo_val_loss = 0.0
    
    # disable gradient computation for validation
    with torch.no_grad():
        # loop over the validation data loader
        for i, data in enumerate(val_loader, 0):
            # get the inputs and labels from the data loader
            inputs, labels = data
            
            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # add the loss to the running validation loss
            pseudo_val_loss += loss.item()

    pseudo_running_loss = 0.0
    
    model.train()  # set the model to training mode
    
    for i, data in enumerate(pseudo_loader, 0):
        inputs, labels = data
        
        optimizer.zero_grad()  # zero the parameter gradients
        
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        pseudo_running_loss += loss.item()

    pseudo_train_losses.append(pseudo_running_loss)
    pseudo_val_losses.append(pseudo_val_loss)
            
    # print statistics every epoch
    print('[%d] train loss: %.3f, val loss: %.3f' %
          (epoch + 1, pseudo_running_loss / len(pseudo_loader), pseudo_val_loss / len(val_loader)))










