#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# utilitaire pour calculer la taille d'un modèle (en octets)

import torch
import torch.nn as nn
import torch.nn.functional as F

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

DATA_SIZE = 784
NUM_HIDDEN_1 = 256 # try 512
NUM_HIDDEN_2 = 256
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(DATA_SIZE, NUM_HIDDEN_1)
        self.fc2 = nn.Linear(NUM_HIDDEN_1, NUM_HIDDEN_2)
        self.fc3 = nn.Linear(NUM_HIDDEN_2, NUM_CLASSES)
    def forward(self, x):
        x = x.view(-1, DATA_SIZE) # reshape the tensor 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LSTMNet(nn.Module):
    def __init__(self,in_size,hidden_size, nb_layer, nb_classes):
        super(LSTMNet,self).__init__()
        self.hidden_size = hidden_size
        self.nb_layer = nb_layer
        self.nb_classes = nb_classes
        self.lstm = nn.LSTM(in_size,hidden_size,nb_layer,batch_first=True)
        self.fc = nn.Linear(hidden_size,nb_classes)

    def forward(self,x):
        # initial states
        h0 = torch.zeros(self.nb_layer, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.nb_layer, x.size(0), self.hidden_size)

        out,_ = self.lstm(x, (h0,c0))
        out = self.fc(out[:,-1,:])
        return out

class BiLSTMNet(nn.Module):
    def __init__(self,in_size,hidden_size, nb_layer, nb_classes):
        super(BiLSTMNet,self).__init__()
        self.hidden_size = hidden_size
        self.nb_layer = nb_layer
        self.nb_classes = nb_classes
        self.lstm = nn.LSTM(in_size,hidden_size,nb_layer,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_size*2,nb_classes)  # 2 for bidirection

    def forward(self,x):
        # initial states
        h0 = torch.zeros(self.nb_layer*2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.nb_layer*2, x.size(0), self.hidden_size)

        out,_ = self.lstm(x, (h0,c0))
        out = self.fc(out[:,-1,:])
        return out

def get_model_size(model):
    # compte le nombre de parametres; convertit en taille mémoire
    param_size = 0
    for p in model.parameters():
        param_size += p.numel() * p.element_size()

    # idem pour les 'buffer', les autres espace mémoire
    # typiquement, pour stockage de moyenne et variance de batch-norm
    buffer_size = 0
    for b in model.buffers():
        buffer_size += b.numel() * b.element_size()

    return (param_size + buffer_size) / 1024**2

model = CNNNet()
print('CNN model size: {:.3f} MB'.format(get_model_size(model)))
model = MLPNet()
print('MLP model size: {:.3f} MB'.format(get_model_size(model)))
input_size = 28
hidden_size = 128
num_layers = 1
model = LSTMNet(input_size, hidden_size, num_layers, NUM_CLASSES)
print('LSTM model size: {:.3f} MB'.format(get_model_size(model)))
model = BiLSTMNet(input_size, hidden_size, num_layers, NUM_CLASSES)
print('BiLSTM model size: {:.3f} MB'.format(get_model_size(model)))

