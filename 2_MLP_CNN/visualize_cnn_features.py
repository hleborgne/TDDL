#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Visualize CNN features from a saved model

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# we use GPU if available, otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# load data
from torchvision import datasets, transforms
trans = transforms.Compose( [ transforms.ToTensor(), transforms.Normalize( (0.1307,),(0.3081,))])
test_set = datasets.MNIST( './data', train=False, transform=trans, download=True )

# define CNN model
DATA_SIZE = 784
NUM_CLASSES = 10
NUM_CONV_1=10 # try 32
NUM_CONV_2=20 # try 64
NUM_FC=500 # try 1024

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet,self).__init__()
        self.conv_1 = nn.Conv2d(1,NUM_CONV_1,5,1) # kernel_size = 5
        self.conv_2 = nn.Conv2d(NUM_CONV_1,NUM_CONV_2,5,1) # kernel_size = 5
        # self.drop = nn.Dropout2d()
        self.fc_1 = nn.Linear(4*4*NUM_CONV_2, NUM_FC)
        self.fc_2 = nn.Linear(NUM_FC,NUM_CLASSES)
    def forward(self,x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv_2(x))
        # x = F.relu(self.drop(self.conv_2(x)))
        x = F.max_pool2d(x,2,2)
        x = x.view(-1,4*4*NUM_CONV_2)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x

model = CNNNet()
if device.type=='cuda':
    model.cuda()
model.load_state_dict(torch.load('model_cnn.pth', map_location=device))
model.eval()
# torch.no_grad()

# define hooks to register feature maps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.conv_1.register_forward_hook(get_activation('conv_1'))
model.conv_2.register_forward_hook(get_activation('conv_2'))

# loop to display image + feature maps
nos_image=0
fig_0, axarr_0 = plt.subplots()
fig_1, axarr_1 = plt.subplots(5,2)
fig_2, axarr_2 = plt.subplots(5,4)
while (nos_image>=0):
    # convert input to appropriate format
    # then compute output (--> forward pass)
    x=test_set[nos_image][0]
    x=x.unsqueeze(0)
    x=x.to(device)
    out = model(x)
    pred = out.argmax(dim=1, keepdim=True).data.cpu().numpy()[0,0]
    sco = out[0,pred].data.cpu().numpy()

    # display image
    axarr_0.imshow( (test_set.data[nos_image,:,:]).cpu() , cmap='gray')
    fig_0.suptitle('img {} (lab={} pred={} sco={})'.format(nos_image,test_set.targets[nos_image],pred,sco))

    # get featuremaps
    activ_1 = activation['conv_1'].squeeze()
    activ_2 = activation['conv_2'].squeeze()

    # display feature maps
    for idx in range(activ_1.size(0)):
        axarr_1[idx%5,int(idx/5)].imshow( (activ_1[idx]).cpu())
    fig_1.suptitle('Conv_1 feature maps', fontsize=16)

    for idx in range(activ_2.size(0)):
        axarr_2[idx%5,int(idx/5)].imshow( (activ_2[idx]).cpu())
    fig_2.suptitle('Conv_2 feature maps', fontsize=16)

    fig_0.canvas.draw()
    fig_1.canvas.draw()
    fig_2.canvas.draw()
    plt.pause(0.5)
    nos_image = int(input('Image number ?: '))
plt.close()

