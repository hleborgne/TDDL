# original inspiration: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
#
# download data from https://download.pytorch.org/tutorial/data.zip
import torch
from rnn import RNN
from utils import *
import time

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

#### training
optimizer = torch.optim.Adam(model_net.parameters(), lr = 0.001)
criterion = torch.nn.NLLLoss()
learning_rate = 0.005 

def train(category_tensor, line_tensor):
    hidden = model_net.initHidden()

    model_net.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = model_net(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # for p in model_net.parameters():
    #     p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

n_iters = 100000
print_every = 5000
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0
all_losses = []

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

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

### plot loss
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)
plt.xlabel('iterations')
plt.ylabel('training loss')
plt.show()

### save model
torch.save(model_net, 'char-rnn-adam-classification.pt')
