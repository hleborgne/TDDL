# -*- coding: utf-8 -*-
# Fizz Buzz in pyTorch (herve.le-borgne@cea.fr)
# see original TF code at http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/

import numpy as np
import torch

# we use GPU if available, otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device is {device}')

NUM_DIGITS = 10

# codage binaire d'un chiffre (max NUM_DIGITS bits)
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

# creation verite terrain: [number, "fizz", "buzz", "fizzbuzz"]
def fizz_buzz_encode(i):
    # TODO return 0
    # TODO return 1
    # TODO return 2
    # TODO return 3

# données d'entraînement (X) et labels (Y)
X=(torch.FloatTensor(np.stack([binary_encode(i, NUM_DIGITS) for i in range(TODO,TODO)], axis=0))).to(device)
Y=(torch.LongTensor([fizz_buzz_encode(i) for i in range(TODO, TODO)]).squeeze()).to(device)

# données de test
X_test=(torch.FloatTensor(np.stack([binary_encode(i, NUM_DIGITS) for i in range(TODO, TODO)], axis=0))).to(device)


# nombre de neurones dans la couche cachée
NUM_HIDDEN = 100

# définition du MLP à 1 couche cachée (non linearite ReLU)
model = torch.nn.Sequential(
    torch.nn.Linear(TODO, NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN, TODO)
    )
model.to(device) # puts model on GPU / CPU

# fonction de coût 
loss_fn = torch.nn.CrossEntropyLoss()

# optimiseur
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05)

# affichage attendu par l'application
def fizz_buzz(i, prediction):
    return [str(i), TODO, TODO, "fizzbuzz"][prediction]

# on lance les calculs
BATCH_SIZE = 128
raw_data_test = np.arange(1, 101) # valeurs de test
for epoch in range(10000):
    for start in range(0, len(X), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = X[start:end]
        batchY = Y[start:end]

        # prediction et calcul loss
        y_pred = model(batchX)
        loss = loss_fn(y_pred, batchY)
    
        # mettre les gradients à 0 avant la passe retour (backward)
        optimizer.zero_grad()
    
        # rétro-propagation
        loss.backward()
        optimizer.step()

    # calcul coût  (et affichage)
    loss = loss_fn( model(X), Y)
    if epoch%100 == 0:
        print('epoch {:5d}\ttraining loss {:1.4f}'.format(epoch, loss.item()))

    # visualisation des résultats en cours d'apprentissage
    # (doit être fait sur l'ensemble de validation normalement)
    if(epoch%1000==0):
        Y_test_pred = model(X_test)
        val, idx = torch.max(Y_test_pred,1)
        ii=idx.data.cpu().numpy()
        # numbers = np.arange(1, 101)
        output = np.vectorize(fizz_buzz)(raw_data_test, ii)
        print(output)

# Sortie finale (calcul lisible)
Y_test_pred = model(X_test)
val, idx = torch.max(Y_test_pred,1)
ii=idx.data.cpu().numpy()
output = np.vectorize(fizz_buzz)(raw_data_test, ii)
print("============== Final result ============")
print(output)

# Sortie finale (calcul plus compact des predictions)
Y_test_pred = model(X_test)
predictions = zip(range(1, 101), Y_test_pred.max(1)[1].data.tolist())
print("============== Final result ============")
print ([ fizz_buzz (i , x ) for (i , x ) in predictions ])
