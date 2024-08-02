# TD DL
Travaux dirigés de deep learning

## TD 1: Bases de PyTorch
Initiation à la syntaxe et aux bases de [PyTorch](https://pytorch.org/) avec:
- l'implémentation du jeu "fizz buzz" par apprentissage
- manipulation et visualisation d'un ensemble de données visuelles
- apprentissage de portes logiques par un modèle neuronal

Code pour [Tensorflow](https://www.tensorflow.org/) partiellement disponible mais non corrigé en TD.

## TD 2: DNN classiques (MLP, CNN, (bi)LSTM
Apprentissage de chiffres manuscrits sur [MNIST](http://yann.lecun.com/exdb/mnist/) avec un MLP, un CNN et un (bi)LSTM


```bash
conda create --name cs_td # python=3.11 en 2023
conda activate cs_td2
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge matplotlib
```

Si vous avez une GPU, il faut préalablement installer les drivers NVIDIA (et redémarrer votre machine). Avec e.g. ubuntu 22.04:
```
ubuntu-drivers devices # --> liste des drivers disponibles
sudo apt install nvidia-driver-535
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
Pour l'inférence avec le framework [Aidge](https://projects.eclipse.org/projects/technology.aidge) on peut installer un environnement séparé (ou ajouter Aidge à l'environnement `cs_td2`):
```
conda create --name aidge python=3.8
conda activate aidge
git clone --recursive https://gitlab.eclipse.org/eclipse/aidge/aidge.git
cd aidge && pip install .

# test
python -c "import aidge_core; import aidge_backend_cpu; print(aidge_core.Tensor.get_available_backends())"
```
Pour les mesures en transport optimal:
```bash
 pip install geomloss
```

## TD 5: NLP et Tensorboard

```bash
conda activate cs_td2
# python -m pip install -U torch - tb - profiler
pip install -U torch -tb-profiler
conda install -c conda-forge --name cs_td2 tensorboard
conda update -c conda-forge --name cs_td2 tensorboard
```
