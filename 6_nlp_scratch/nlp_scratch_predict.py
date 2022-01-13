import torch
import sys
from rnn import RNN
from utils import *

# get list of all possible category names from trianing data
datapath='data/names/'
_,all_categories = get_language_data(datapath)

# load model
rnn = torch.load('char-rnn-classification.pt')

# return a prediction given a name
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# predicts one name (from a string)
def predict_one(line, n_predictions=3):
    print(f'origin of name [{line}]:')
    output = evaluate((lineToTensor(line)))

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('   (%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])

    return predictions

# predicts several names (from a file)
def predict_list(filename):
    for name in readLines(filename):
        print(f'### {name}')
        predict_one(name)

if __name__ == '__main__':
    arg1 = sys.argv[1]
    if '.' in arg1:
        predict_list(arg1)
    else:
        predict_one(arg1)


# Go through a bunch of examples and record which are correctly guessed
#for i in range(n_confusion):
#    category, line, category_tensor, line_tensor = randomTrainingExample()
#    output = evaluate(line_tensor)
#    guess, guess_i = categoryFromOutput(output)
#    category_i = all_categories.index(category)
#    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
#for i in range(n_categories):
#    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(confusion.numpy())
#fig.colorbar(cax)

# Set up axes
#ax.set_xticklabels([''] + all_categories, rotation=90)
#ax.set_yticklabels([''] + all_categories)

# Force label at every tick
#ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
#plt.show()

