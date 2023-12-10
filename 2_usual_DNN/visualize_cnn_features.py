#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Visualize CNN features from a saved model
# (see end of mnist_MLP_CNN_pytorch.py with a CNNNet)

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# we use GPU if available, otherwise CPU
# NB: with several GPUs, "cuda" --> "cuda:0" or "cuda:1"...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("### device:", device, f'({torch.cuda.get_device_name(device)})' if torch.cuda.is_available() else '')

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
model.load_state_dict(torch.load('params_model_cnn.pth', map_location=device))
model.eval()

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
print("### type an image number. Hard ones are 115, 247, 445")
print("### (...) 9729, 9770, 9792")
print("### set a negative image number to exit")
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
    axarr_0.imshow( (test_set.data[nos_image,:,:]).cpu(), cmap='gray')
    fig_0.suptitle('img {} (lab={} pred={} sco={:2.3f})'.format(nos_image,test_set.targets[nos_image],pred,sco))

    # get featuremaps
    activ_1 = activation['conv_1'].squeeze()
    activ_2 = activation['conv_2'].squeeze()

    # display feature maps
    for idx in range(activ_1.size(0)):
        axarr_1[idx%5,int(idx/5)].imshow((activ_1[idx]).cpu(), cmap='gray')
    fig_1.suptitle('Conv_1 feature maps', fontsize=16)

    for idx in range(activ_2.size(0)):
        axarr_2[idx%5,int(idx/5)].imshow((activ_2[idx]).cpu(), cmap='gray')
    fig_2.suptitle('Conv_2 feature maps', fontsize=16)

    fig_0.canvas.draw()
    fig_1.canvas.draw()
    fig_2.canvas.draw()
    plt.pause(0.5)
    nos_image = int(input('Image number ?: '))
plt.close()

# 115 9 4
# 247 2 4
# 445 0 6
# 449 5 3
# 543 7 8
# 582 2 8
# 583 7 2
# 625 4 6
# 646 6 2
# 659 7 2
# 684 2 7
# 947 9 8
# 1014 5 6
# 1112 6 4
# 1182 5 6
# 1192 4 9
# 1226 2 7
# 1242 9 4
# 1247 5 9
# 1319 3 8
# 1527 5 1
# 1530 7 8
# 1621 6 0
# 1709 3 9
# 1717 0 8
# 1901 4 9
# 1982 5 6
# 2018 2 1
# 2109 7 3
# 2130 9 4
# 2135 1 6
# 2293 0 9
# 2369 3 5
# 2414 4 9
# 2597 3 5
# 2648 0 9
# 2654 1 6
# 2771 9 4
# 2896 0 8
# 2927 2 3
# 2939 5 9
# 2953 5 3
# 3073 2 1
# 3422 0 6
# 3520 4 6
# 3558 0 5
# 3559 5 8
# 3767 2 7
# 3778 2 5
# 3808 8 7
# 3906 3 1
# 3985 4 9
# 4176 7 2
# 4224 7 9
# 4497 7 8
# 4507 9 1
# 4571 0 6
# 4639 9 8
# 4740 5 3
# 4823 4 9
# 4956 4 8
# 5201 9 4
# 5887 0 7
# 5937 3 5
# 5972 3 5
# 5997 9 5
# 6023 9 3
# 6028 3 5
# 6555 9 8
# 6571 7 9
# 6572 7 1
# 6576 1 7
# 6597 7 0
# 6651 6 0
# 6783 6 1
# 7472 7 2
# 8408 5 8
# 8527 9 4
# 9009 2 7
# 9642 7 9
# 9664 7 2
# 9679 3 6
# 9729 6 5
# 9770 0 5
# 9792 9 4
