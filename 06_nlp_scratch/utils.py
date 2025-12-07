import unicodedata
import string
import torch

from os import listdir, path
from os.path import isfile, join

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
# def letterToTensor(letter):
#     tensor = torch.zeros(1, n_letters)
#     tensor[0][letterToIndex(letter)] = 1
#     return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def categoryFromOutput(output,all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def get_language_data(datapath):
    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []
    filenames = [f for f in listdir(datapath) if isfile(join(datapath, f))]
    # from os import walk
    # filenames = next(walk(datapath), (None, None, []))[2] 
    
    # on ordonne pour assurer une "certaine portabilité" car l'ordre de
    # ces fichiers est celui de la sortie du modèle par la suite.
    # Retrouver l'index des catégories dépend donc de l'ordre de ces fichiers
    # et 'listdir' renvoie un ordre dépendant de l'OS.
    # Ca reste assez sale quand même...
    filenames.sort()
    
    for filename in filenames:
        category = path.splitext(path.basename(datapath+filename))[0]
        all_categories.append(category)
        lines = readLines(datapath+filename)
        category_lines[category] = lines

    return category_lines,all_categories

import time
import math
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


import random
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(all_categories,train_data):
    category = randomChoice(all_categories)
    line = randomChoice(train_data[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor
