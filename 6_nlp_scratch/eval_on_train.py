import torch
import sys
from utils import *

# model='rnn-adam'
# model='lstm'
model='rnn'

# get list of all possible category names from trianing data
datapath='data/names/'
train_data,all_categories = get_language_data(datapath)

#initialize confusion matrix
n_categories = len(all_categories)
confusion = torch.zeros(n_categories, n_categories)

# load model
if model == 'lstm':
    rnn = torch.load('char-lstm-classification.pt')
elif model == 'rnn':
    rnn = torch.load('char-rnn-classification.pt')
elif model == 'rnn-adam':
    rnn = torch.load('char-rnn-adam-classification.pt')

# return a prediction given a name
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        if model == 'lstm':
            output, hidden = rnn(line_tensor)
        else:
            output, hidden = rnn(line_tensor[i], hidden)

    return output

print('----------------\n   Effectifs\n----------------')
for categ in train_data.keys():
    print('   {}\t {}'.format(categ, len(train_data[categ]) ))
    for name in train_data[categ]:
        output = evaluate((lineToTensor(name)))
        guess, guess_i = categoryFromOutput(output,all_categories)
        category_i = all_categories.index(categ)
        confusion[category_i][guess_i] += 1

effectif = confusion.sum(dim=0)
print('----------------\n   Scores\n----------------')
for i in range(n_categories):
    confusion[i] = confusion[i] / (1e-16+confusion[i].sum())
    print(  '   {} \t {:2.1%}'.format( all_categories[i],(confusion[i][i]).item()))
print('------')
print(  'Global (flat) \t {:2.1%}'.format(  confusion.diag().mean().item() ))
weighted_conf = confusion.diag() * (effectif / effectif.sum())
print(  'Global (wght) \t {:2.1%}'.format(  weighted_conf.sum().item() ))
print('-----------\n')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()
