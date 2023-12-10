#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Classification de MNIST avec un MLP ou un CNN
#
# Voir
#  https://github.com/pytorch/examples/blob/master/mnist/main.py
#  https://gist.github.com/xmfbit/b27cdbff68870418bdb8cefa86a2d558

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# we use GPU if available, otherwise CPU
# NB: with several GPUs, "cuda" --> "cuda:0" or "cuda:1"...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, f'({torch.cuda.get_device_name(device)})' if torch.cuda.is_available() else '')

# import datasets 
from torchvision import datasets, transforms

trans = transforms.Compose( [ transforms.ToTensor(), transforms.Normalize( (0.1307,),(0.3081,))])

train_set = datasets.MNIST( './data', train=True, transform=trans, download=True )
test_set = datasets.MNIST( './data', train=False, transform=trans, download=True )

# define data loaders
batch_size = 100
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=TODO)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=TODO)

print('total training batch number: {}'.format(TODO))
print('total testing batch number: {}'.format(TODO))

# display some images
# for an alternative see https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
def imshow(tensor, title=None):
    img = tensor.cpu().clone()
    img = img.squeeze()
    plt.imshow(img, cmap='gray')
    if title is not None:
        plt.title(title)
    plt.pause(0.5)

plt.figure()
for ii in range(10):
    imshow(train_set.TODO , title='MNIST example ({})'.format(train_set.TODO) )
plt.close()

# define MLP model
DATA_SIZE = TODO
NUM_CLASSES = TODO
NUM_HIDDEN_1 = 256 # try 512
NUM_HIDDEN_2 = 256


class RegSoftNet(nn.Module):
    def __init__(self):
        super(RegSoftNet, self).__init__()
        self.fc = TODO
    def forward(self, x):
        x = x.view(-1, DATA_SIZE) # reshape the tensor
        x = TODO
        return x

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = TODO
        self.fc2 = TODO
        self.fc3 = TODO
    def forward(self, x):
        x = x.view(-1, DATA_SIZE) # reshape the tensor 
        x = TODO
        x = TODO
        x = TODO
        return x


NUM_CONV_1=TODO
NUM_CONV_2=TODO
NUM_FC=TODO
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet,self).__init__()
        self.conv_1 = nn.Conv2d(TODO,TODO,5,1) # kernel_size = 5
        self.conv_2 = nn.Conv2d(TODO,TODO,5,1) # kernel_size = 5
        self.fc_1 = nn.Linear(TODO_H, TODO)
        self.fc_2 = nn.Linear(TODO,NUM_CLASSES)
    def forward(self,x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x,2,2)
        x = x.view(-1,TODO_H)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x
        # en utilisant loss = F.nll_loss(output, target) on peut faire
        # return F.log_softmax(x, dim=1)

# define model (choose MLP or CNN)
model = RegSoftNet()
#model = MLPNet()
#model = CNNNet()

model.to(device) # puts model on GPU / CPU

# optimization hyperparameters
optimizer = TODO
loss_fn = TODO

# main loop (train+test)
for epoch in range(10):
    # training
    model.train() # mode "train" agit sur "dropout" ou "batchnorm"
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        x, target = x.to(device), target.to(device)
        out = model(x)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        if batch_idx %100 ==0:
            print('epoch {} batch {} [{}/{}] training loss: {}'.format(epoch,batch_idx,batch_idx*len(x),
                    len(train_loader.dataset),loss.item()))
    # testing
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_loader):
            x, target = x.to(device), target.to(device)
            out = model(x)
            loss = loss_fn(out, target)
            # _, prediction = torch.max(out.data, 1)
            prediction = out.argmax(dim=1, keepdim=True) # index of the max log-probability
            correct += prediction.eq(target.view_as(prediction)).sum().item()
    taux_classif = 100. * correct / len(test_loader.dataset)
    print('Accuracy: {}/{} (tx {:.2f}%, err {:.2f}%)\n'.format(correct,
     len(test_loader.dataset), taux_classif, 100.-taux_classif))

