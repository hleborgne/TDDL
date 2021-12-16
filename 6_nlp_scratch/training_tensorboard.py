# original inspiration: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
#
# download data from https://download.pytorch.org/tutorial/data.zip
import torch
from rnn import RNN
from utils import *
import time
from torch.utils.tensorboard import SummaryWriter

# log_directory for tensorboard
tb_writer = SummaryWriter('log/nlp_scratch_exp_1')

### get training data
datapath='data/names/'
train_data,all_categories = get_language_data(datapath)
n_categories = len(all_categories)

print(f'There are {n_categories} languages.\nNumber of family name per language:')
for categ in train_data.keys():
    print('   {}\t {}'.format(categ, len(train_data[categ]) ))

### create model
n_hidden = 128
model_net = RNN(n_letters, n_hidden, n_categories)

### Tensorboard visualization of the network
# - create (any valid) input data
# - visualize the built model in tensorboeard
category, line, category_tensor, line_tensor = randomTrainingExample(all_categories,train_data)
hidden = model_net.initHidden()
tb_writer.add_graph(model_net, (line_tensor[0], hidden  ))
# tb_writer.close()

#### training
criterion = torch.nn.NLLLoss() # the RNN already has a softmax as output
learning_rate = 0.005 

def train(category_tensor, line_tensor):
    hidden = model_net.initHidden()

    model_net.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = model_net(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    for p in model_net.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

n_iters = 100000
print_every = 5000
plot_every = 1000
current_loss = 0
start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample(all_categories,train_data)
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output,all_categories)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # add the loss value to tensorboard
    if iter % plot_every == 0:
        tb_writer.add_scalar('training loss', current_loss / plot_every, iter)
        current_loss = 0

# force to write last logs then close
tb_writer.flush()
tb_writer.close()

### save model
# torch.save(model_net, 'char-rnn-classification.pt')
