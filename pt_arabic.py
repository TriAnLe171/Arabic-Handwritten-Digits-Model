import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import pandas as pd
from torchvision import transforms
import numpy as np

'''Training file'''
# Read the CSV file
data_folder = pd.read_csv("D:\mlr\Arabic Handwritten Digits Dataset CSV\csvTrainImages 60k x 784.csv",header=None)
data_folder_labels = pd.read_csv("D:\mlr\Arabic Handwritten Digits Dataset CSV\csvTrainLabel 60k x 1.csv",header=None)

features = data_folder.iloc[:, :].values
labels = data_folder_labels.iloc[:, -1].values

# convert numpy arrays to pytorch tensors
X_train = torch.stack([torch.from_numpy(np.array(i)) for i in features])
y_train = torch.stack([torch.from_numpy(np.array(i)) for i in labels])
print(y_train.shape)
train_dataset = torch.utils.data.TensorDataset(X_train, y_train) #Take X_train, y_train tensors as inputs and combines them into a single dataset
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

'''Testing file'''
# Read the CSV file
data_folder_test = pd.read_csv("D:\mlr\Arabic Handwritten Digits Dataset CSV\csvTestImages 10k x 784.csv",header=None)
data_folder_test_labels=pd.read_csv("D:\mlr\Arabic Handwritten Digits Dataset CSV\csvTestLabel 10k x 1.csv",header=None)

features_test = data_folder_test.iloc[:, :].values
labels_test = data_folder_test_labels.iloc[:, -1].values

# convert numpy arrays to pytorch tensors
X_test = torch.stack([torch.from_numpy(np.array(i)) for i in features_test])
y_test = torch.stack([torch.from_numpy(np.array(i)) for i in labels_test])

test_dataset = torch.utils.data.TensorDataset(X_test, y_test) #Take X_test, y_test tensors as inputs and combines them into a single dataset
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print('download completed')

for X, y in test_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        # Convert input tensor to the same data type as weight tensors
        x = x.to(self.linear_relu_stack[0].weight.dtype)
        x = x.view(x.size(0), -1)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


#loss_fn = nn.MSELoss()
loss_fn = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters())

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)

        # y_onehot = F.one_hot(y, num_classes=10).float()
        # loss = loss_fn(pred, y_onehot)
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            # y_onehot = F.one_hot(y, num_classes=10).float()
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
print("Done!")