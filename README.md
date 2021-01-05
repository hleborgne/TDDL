# TPDL
Travaux dirigés de deep learning

## TD 1: Fizz Buzz
Initiation à la syntaxe et aux bases de [PyTorch](https://pytorch.org/).

Code pour [Tensorflow](https://www.tensorflow.org/) disponible mais non corrigé en TD.

## TD 2: MLP et CNN
Apprentissage de chiffres manuscrits sur [MNIST](http://yann.lecun.com/exdb/mnist/) avec un MLP et un CNN

```bash
conda create --name cs_td2 python=3.8
conda activate cs_td2
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install -c conda -forge matplotlib
```
## TD 3: transfer learning et finetuning
Transfert d'apprentissage entre ImageNet et un petit problème cible. Étude du réglage fin du réseau.
```
conda activate cs_td2
conda install -c anaconda scikit-learn
```

## TD 4: LSTM
Apprentissage de chiffres manuscrits sur [MNIST](http://yann.lecun.com/exdb/mnist/) avec un (bi) LSTM

## TD 5: GAN
Modèle génératif (GAN) sur exemples jouet
```
conda activate cs_td2
conda install -c anaconda absl-py 
```
