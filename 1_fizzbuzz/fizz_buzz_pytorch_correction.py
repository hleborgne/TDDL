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
    if   i % 15 == 0: return 3
    elif i % 5  == 0: return 2
    elif i % 3  == 0: return 1
    else:             return 0

# données d'entraînement (X) et labels (Y)
X=(torch.FloatTensor(np.stack([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)], axis=0))).to(device)
Y=(torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)]).squeeze()).to(device)

# NB: np.stack(..., axis=0 ) is not strictly required, it allows to speedup the conversion to torch tensor.

# [exo 1.2] données de validation (méthode: tirage aléatoire du train initial)
# [exo 2.2] ici on peut changer la taille de l'ensemble d'apprentissage
#           attention, si train+val fait plus de 1024 échantillons, 
#           il faut modifier NUM_DIGITS aussi (11 pour 2048, etc.)
NUM_VAL=100
p = np.random.permutation(range(len(X)))
X_train, Y_train = X[p,:] , Y[p]
X_val,   Y_val   = X_train[0:NUM_VAL,:], Y_train[0:NUM_VAL]
X_train, Y_train = X_train[NUM_VAL: ,:], Y_train[NUM_VAL:]

# données de test
X_test=(torch.FloatTensor(np.stack([binary_encode(i, NUM_DIGITS) for i in range(1,101)], axis=0))).to(device)

# nombre de neurones dans la couche cachée
NUM_HIDDEN = 100 # [exo 2.2] valeur de la couche cachée

# définition du MLP à 1 couche cachée (non linearite ReLU)
model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN, 4))
model.to(device) # puts model on GPU / CPU

# fonction de coût 
loss_fn = torch.nn.CrossEntropyLoss()

# optimiseur
# [exo 2.2] valeur du pas d'apprentissage: modifier 'lr'
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05)

# affichage attendu par l'application
def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

# on lance les calculs
BATCH_SIZE = 128
for epoch in range(10000):  # [exo 2.4] nombre d'itérations
    for start in range(0, len(X_train), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = X_train[start:end]
        batchY = Y_train[start:end]

        # prediction et calcul loss
        y_pred = model(batchX)
        loss = loss_fn(y_pred, batchY)

        # mettre les gradients à 0 avant la passe retour (backward)
        optimizer.zero_grad()

        # rétro-propagation
        loss.backward()
        optimizer.step()

    # calcul coût  (et affichage)
    loss = loss_fn( model(X_train), Y_train)
    if epoch%100 == 0:
        print('epoch {:5d} training loss {:1.4f}'.format(epoch, loss.item()))

    # affichage de la performance courante
    #   sur train (1-erreur empirique)
    #   sur val  ([exo 1.2] meilleure estimation)
    # [exo 2.4] l'influence du nb d'itérations doit se faire sur val normalement
    if(epoch%1000==0):
        Y_train_pred = model(X_train)
        # Y_train_pred = Y_train_pred.cpu()
        print("train perf: {:1.2f}".format( np.mean(Y_train.data.cpu().numpy() == Y_train_pred.max(1)[1].data.cpu().numpy() )))
        Y_val_pred = model(X_val)
        print("  val perf: {:1.2f}".format( np.mean(Y_val.data.cpu().numpy() == Y_val_pred.max(1)[1].data.cpu().numpy() )))

# Sortie finale (affichage sur les données de test)
# ci-dessous un code plus long mais plus lisible
# Y_test_pred = model(X_test)
# val, idx = torch.max(Y_test_pred,1)
# ii=idx.data.numpy()
# print( np.vectorize(fizz_buzz)(np.arange(1, 101), ii) )
#
### Calcul plus compact des predictions
Y_test_pred = model(X_test)
predictions = zip(range(1, 101), list(Y_test_pred.max(1)[1].data.tolist()))
print("============== Final result ============")
print ([fizz_buzz(i, x) for (i, x) in predictions])

# [exo 1.1] Performances de test
gtY = np.array([fizz_buzz_encode(i) for i in np.arange(1, 101)])
print("test perf: {:1.2%}".format(np.mean(gtY == Y_test_pred.max(1)[1].data.cpu().numpy())))

# equivalent
# pred=Y_test_pred.max(1)[1].data.tolist()
# gt=[fizz_buzz_encode(i) for i in range(1,101)]
# print(f'{sum([i==j for i,j in zip(pred,gt)])} element are correct')
# print("test perf: {:1.2%}".format(np.mean(np.array(gt) == np.array(pred))))
