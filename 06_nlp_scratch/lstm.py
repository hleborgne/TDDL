import torch
import torch.nn as nn

# define LSTM model
class LSTMNet(nn.Module):
    def __init__(self,in_size,hidden_size, nb_layer, nb_classes):
        super(LSTMNet,self).__init__()
        self.hidden_size = hidden_size
        self.nb_layer = nb_layer
        self.nb_classes = nb_classes
        # change 'bidirectional' to get a BiLSTM
        # batch_first=False --> input and output tensors are provided as (seq, batch, feature)
        self.lstm = nn.LSTM(in_size,hidden_size,nb_layer,batch_first=False,bidirectional=False)
        self.fc = nn.Linear(hidden_size,nb_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,x):
        # initial states; x.size(1) = batch_size avec batch_first=False
        h0 = torch.zeros(self.nb_layer, x.size(1), self.hidden_size)#.to(device)
        c0 = torch.zeros(self.nb_layer, x.size(1), self.hidden_size)#.to(device)
        out,(hn,cn) = self.lstm(x, (h0,c0)) # self.lstm(x) : zero par défaut 
        out = self.fc(out[-1,:,:]) # dernière couche cachée de la séquence avec batch_first=False
        # out = self.fc(out[:,-1,:]) # idem avec batch_first=True
        out = self.softmax(out)
        return out,hn

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


