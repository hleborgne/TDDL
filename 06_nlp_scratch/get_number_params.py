import torch

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        pp += p.nelement()
        print('\t layer size: {} --> {}'.format(p.size(), p.numel() ))
        # pp += p.numel()
    return pp

### remarque: on peut charger le modèle sans le définir explicitement 
### par une classe dans ce programme car c'est fait dans les fichiers
### rnn.py et lstm.py accessible dans ce répertoire
rnn = torch.load('char-rnn-classification.pt')
print('RNN has {} parameters'.format( get_n_params(rnn) ))
lstm = torch.load('char-lstm-classification.pt')
print('LSTM has {} parameters'.format( get_n_params(lstm) ))
