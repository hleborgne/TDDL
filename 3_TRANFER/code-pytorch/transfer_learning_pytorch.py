# -*- coding: utf-8 -*-
# Useful link and source: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# (Or more generally https://pytorch.org/tutorials/)
# Dataset from https://www.pyimagesearch.com/2019/01/14/machine-learning-in-python/
#              FlickrLogos: www.multimedia-computing.de/flickrlogos

import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split

# Récupérer un réseau pré-entraîné

print("Récupération du réseau pré-entraîné et des données")

resnet = models.resnet18(pretrained=True)

# Lire un problème cible

# means et std du dataset ImageNet utilisé pour entraîner ResNet
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# on définit les transformations à appliquer aux images du dataset
data_transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# on lit une première fois les images du dataset
#image_directory = "python-machine-learning/3scenes/" # à adapter en fonction de l'endroit où sont stockées les données
image_directory = "../data/" # à adapter en fonction de l'endroit où sont stockées les données
dataset_full = datasets.ImageFolder(image_directory, data_transforms)
loader_full = torch.utils.data.DataLoader(dataset_full, batch_size=16, shuffle=True, num_workers=4)

# on split en train, val et test à partir de la liste des images
np.random.seed(42L)
samples_train, samples_test = train_test_split(dataset_full.samples)
samples_train, samples_val = train_test_split(samples_train)

print("Nombre d'images de train : %i" % len(samples_train))
print("Nombre d'images de val : %i" % len(samples_val))
print("Nombre d'images de test : %i" % len(samples_test))


# on définit d'autres dataset pytorch à partir des listes d'images de train / val / test
dataset_train = datasets.ImageFolder(image_directory, data_transforms)
dataset_train.samples = samples_train
dataset_train.imgs = samples_train
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=4)

dataset_val = datasets.ImageFolder(image_directory, data_transforms)
dataset_val.samples = samples_val
dataset_val.imgs = samples_val

dataset_test = datasets.ImageFolder(image_directory, data_transforms)
dataset_test.samples = samples_test
dataset_test.imgs = samples_test

torch.manual_seed(42L)

# Transfert d'apprentissage

import torch.nn as nn
import torch.optim as optim

# on utilisera le GPU (beaucoup plus rapide) si disponible, sinon on utilisera le CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu") # forcer en CPU s'il y a des problèmes de mémoire GPU (+ être patient...)

# on indique qu'il est inutile de calculer les gradients des paramètres de resnet
for param in resnet.parameters():
    param.requires_grad = False

# on remplace la dernière couche fully connected à 1000 sorties (classes d'ImageNet) par une fully connected à 3 sorties (nos classes).
# par défaut, les gradients des paramètres cette couche seront bien calculés
resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=3, bias=True)
resnet.to(device) # on utilise le GPU / CPU en fonction de ce qui est disponible

resnet.train(True) # pas indispensable ici, mais bonne pratique de façon général : permet notamment d'activer / désactiver le dropout en fonction de si on entraîne ou si on teste le modèle

# on définit une loss et un optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.fc.parameters(), lr=0.001, momentum=0.9)

PRINT_LOSS = False

# fonction classique d'entraînement d'un modèle, voir TDs précédents
def train_model(model, loader, optimizer, criterion, n_epochs=10):
    for epoch in range(n_epochs): # à chaque epochs
        print("EPOCH % i" % epoch)
        for i, data in enumerate(loader): # on itère sur les minibatchs via le loader
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) # on passe les données sur CPU / GPU
            optimizer.zero_grad() # on réinitialise les gradients
            outputs = model(inputs) # on calcule l'output
            
            loss = criterion(outputs, labels) # on calcule la loss
            if PRINT_LOSS:
		print(loss.item())
            
            loss.backward() # on effectue la backprop pour calculer les gradients
            optimizer.step() # on update les gradients en fonction des paramètres

print("Apprentissage en transfer learning")
resnet.train(True) # pas indispensable ici, mais bonne pratique de façon général : permet notamment d'activer / désactiver le dropout en fonction de si on entraîne ou si on teste le modèle
torch.manual_seed(42L)
train_model(resnet, loader_train, optimizer, criterion, n_epochs=10)

# on définit une fonction d'évaluation
def evaluate(model, dataset):
    avg_loss = 0.
    avg_accuracy = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
    for data in loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        n_correct = torch.sum(preds == labels)
        
        avg_loss += loss.item()
        avg_accuracy += n_correct
        
    return avg_loss / len(dataset), float(avg_accuracy) / len(dataset)

resnet.train(False)
loss, accuracy = evaluate(resnet, dataset_test)
print("Accuracy: %.1f%%" % (100 * accuracy))

# Fine tuning

# on réinitialise resnet
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=3, bias=True)
resnet.to(device)

# cette fois on veut updater tous les paramètres
params_to_update = resnet.parameters()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# on ré-entraîne
print("Apprentissage avec fine-tuning")
resnet.train(True)
torch.manual_seed(42L)
train_model(resnet, loader_train, optimizer, criterion, n_epochs=10)

# on ré-évalue les performances
resnet.train(False)
loss, accuracy = evaluate(resnet, dataset_test)
print("Accuracy: %.1f%%" % (100 * accuracy))

