# TD DL
Travaux dirigés de deep learning

## TD 1: Fizz Buzz
Initiation à la syntaxe et aux bases de [PyTorch](https://pytorch.org/).

Code pour [Tensorflow](https://www.tensorflow.org/) disponible mais non corrigé en TD.

## TD 2: DNN classiques (MLP, CNN, (bi)LSTM
Apprentissage de chiffres manuscrits sur [MNIST](http://yann.lecun.com/exdb/mnist/) avec un MLP, un CNN et un (bi)LSTM

```bash
conda create --name cs_td2 python=3.8
conda activate cs_td2
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install -c conda-forge matplotlib
```
ou
```bash
conda create --name cs_td python=3.9
conda activate cs_td
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install -c conda-forge matplotlib
```

## TD 3: transfer learning et finetuning
Transfert d'apprentissage entre ImageNet et un petit problème cible. Étude du réglage fin du réseau.
```bash
conda activate cs_td2
conda install -c anaconda scikit-learn
```

## TD 4: GAN
Modèle génératif (GAN) sur exemples jouet
```bash
conda activate cs_td2
conda install -c anaconda absl-py 
```

## TD 5: NLP et Tensorboard

```bash
conda activate cs_td2
# python -m pip install -U torch - tb - profiler
pip install -U torch -tb-profiler
conda install -c conda-forge --name cs_td2 tensorboard
conda update -c conda-forge --name cs_td2 tensorboard
```
