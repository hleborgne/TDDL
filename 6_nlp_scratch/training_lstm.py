import torch
from utils import *
import time
from lstm import LSTMNet
import torch.nn as nn

# log_directory for tensorboard
from torch.utils.tensorboard import SummaryWriter
tb_writer = SummaryWriter('log/lstm')

### get training data
datapath='data/names/'
train_data,all_categories = get_language_data(datapath)
n_categories = len(all_categories)

print(f'There are {n_categories} languages.\nNumber of family name per language:')
for categ in train_data.keys():
    print('   {}\t {}'.format(categ, len(train_data[categ]) ))

n_hidden = 64
num_layers = 1
model = LSTMNet(n_letters, n_hidden, num_layers, n_categories)#.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = torch.nn.NLLLoss()

current_loss = 0
n_iters = 100000
plot_every = 1000

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample(all_categories,train_data)
    model.zero_grad()

    output, _ = model(line_tensor)
    loss = criterion(output,category_tensor)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    current_loss += loss
    if iter % plot_every == 0:
        tb_writer.add_scalar('LSTM training loss', current_loss / plot_every, iter)
        current_loss = 0

### save model
torch.save(model, 'char-lstm-classification.pt')



