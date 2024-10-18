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
data_folder_train = pd.read_csv(
    "csvTrainImages 60k x 784.csv", header=None)
data_folder_labels = pd.read_csv(
    "csvTrainLabel 60k x 1.csv", header=None)
data_folder_test = pd.read_csv(
    "csvTestImages 10k x 784.csv", header=None)
data_folder_test_labels = pd.read_csv(
    "csvTestLabel 10k x 1.csv", header=None)


features_train = data_folder_train.iloc[:, :].values
labels_train = data_folder_labels.iloc[:, -1].values
features_test = data_folder_test.iloc[:, :].values
labels_test = data_folder_test_labels.iloc[:, -1].values

X_train = torch.stack([torch.from_numpy(np.array(i))
                      for i in features_train])
y_train = torch.stack([torch.from_numpy(np.array(i))
                      for i in labels_train])
print(X_train.shape)
X_test = torch.stack([torch.from_numpy(np.array(i))
                     for i in features_test])
y_test = torch.stack([torch.from_numpy(np.array(i))
                     for i in labels_test])

batch_size = 100
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print('download completed')

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Con2d(in_channels,out_channels,kernel_size,stride= 1,padding)
        # Conv2d
        # For a single image, the dimensions are (channels, height, width).
        # For a batch of images, the dimensions are (batch_size, channels, height, width)

        # 1st convolution

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        # 2nd convolution

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()

        # Linear layers
        self.linear_layer = nn.Sequential(
            # modified
            nn.Linear(self.get_Size(self.flatten), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def get_Size(self,x):
        x= self.calculate_conv_output_size()
        return x
    
    def calculate_conv_output_size(self):
        # Create a dummy input tensor
        x = torch.zeros(1, 1, 28, 28) #batch_size=1, single input channel, 28x28
        # Pass through the convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        # Calculate the output size
        output_size = x.size(1) * x.size(2) * x.size(3)
        return output_size
    
    def forward(self, x):
        #x=self.flatten(x)
        x = x.view(batch_size, 1, 28, 28)
        x = self.conv1(x.to(self.linear_layer[0].weight.dtype)) #modified -> convert to the same data type as the first linear layer 
        x = self.conv2(x)
        # print(x.shape)
        x = self.flatten(x) # modified -> after flattening, the accuracy got much higher (~98.8%)
        x = x.view(x.size(0), -1)
        logits = self.linear_layer(x)

        return logits

x=CNN().calculate_conv_output_size()
print(x)
model = CNN().to(device)
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
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 2
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
print("Done!")