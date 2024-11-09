import os

import torch
import torchvision
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt

train_path = './data/raw/handwritten_kyrgyz_letters/train'
test_path = './data/raw/handwritten_kyrgyz_letters/test'

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
])

train_data = torchvision.datasets.ImageFolder(root=train_path,transform=trans)
test_data = torchvision.datasets.ImageFolder(root=test_path,transform=trans)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class Kyrgyz(nn.Module):
    def __init__(self):
        super(Kyrgyz, self).__init__()
        self.features = nn.Sequential(
            # conv1 : 3 * 134 * 134
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # conv2 : 16 * 66 * 66
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # conv3 : 32 * 32 * 32
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        
        # Flatten : 64 * 15 * 15
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(in_features=64*15*15, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=512, out_features=36)
        )
    
    def forward(self,x):
        x = self.features(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits
    

model = Kyrgyz().to(device)
print(model)

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    top1_acc = 0.0
    top5_acc = 0.0
    size = len(dataloader.dataset)

    for batch_idx, (imgs,labels) in enumerate(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)

        pred = model(imgs)
        loss = loss_fn(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Top-1 accuracy
        predicted_1 = pred.argmax(1)
        top1_acc += (predicted_1 == labels).float().sum().item()

        # Top-5 accuracy
        _,predicted_5 = pred.topk(k=5, dim=1)
        labels_resize = labels.view(-1,1)
        top5_acc += torch.eq(predicted_5, labels_resize).float().sum().item()
        
        if batch_idx % 100 == 0:
            current = batch_idx * len(imgs)
            print(f"[{current:>5d}/{size:>5d}]")
    return total_loss/size, top1_acc/size, top5_acc/size

def test(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    total_loss = 0.0
    top1_acc = 0.0
    top5_acc = 0.0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            pred = model(imgs)
            loss = loss_fn(pred, labels)

            total_loss += loss.item()

            # Top-1 accuracy
            predicted_1 = pred.argmax(1)
            top1_acc += (predicted_1 == labels).float().sum().item()

            # Top-5 accuracy
            _,predicted_5 = pred.topk(k=5, dim=1)
            labels_resize = labels.view(-1,1)
            top5_acc += torch.eq(predicted_5, labels_resize).float().sum().item()

    return total_loss/size, top1_acc/size, top5_acc/size


batch_size = 64
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


res = {
    "train_loss": [],
    "train_top1_acc": [],
    "train_top5_acc": [],
    "val_loss": [],
    "val_top1_acc": [],
    "val_top5_acc": [],
}

epochs = 15
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss, train_top1, train_top5 = train(train_dataloader, model, loss_fn, optimizer)
    val_loss, val_top1, val_top5 = test(test_dataloader, model, loss_fn)


    res['train_loss'].append(train_loss)
    res['train_top1_acc'].append(train_top1)
    res['train_top5_acc'].append(train_top5)

    res['val_loss'].append(val_loss)
    res['val_top1_acc'].append(val_top1)
    res['val_top5_acc'].append(val_top5)
    
print('Done!')

# 
fig, ax = plt.subplots(1,3,figsize=(30,10))
x = np.arange(epochs)

ax[0].plot(x, res['train_loss'], label='Training Loss')
ax[0].plot(x, res['val_loss'], label='Validation Loss')
ax[0].legend(loc=0)
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')

ax[1].plot(x, res['train_top1_acc'], label='Training Top-1 Accuracy')
ax[1].plot(x, res['val_top1_acc'], label='Validation Top-1 Accuracy')
ax[1].legend(loc=0)
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Top-1 Accuracy')

ax[2].plot(x, res['train_top5_acc'], label='Training Top-5 Accuracy')
ax[2].plot(x, res['val_top5_acc'], label='Validation Top-5 Accuracy')
ax[2].legend(loc=0)
ax[2].set_xlabel('Epochs')
ax[2].set_ylabel('Top-5 Accuracy')

# Display the plot
print('Training Top-1 Accuracy: ',res['train_top1_acc'][epochs-1])
print('Validation Top-1 Accuracy: ',res['val_top1_acc'][epochs-1])
print('Trainging Top-5 Accuracy: ',res['train_top5_acc'][epochs-1])
print('Validation Top-5 Accuracy: ',res['train_top5_acc'][epochs-1])