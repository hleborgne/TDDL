# TD DL
Travaux dirigés de deep learning. Il est conseillé de mettre en place un environement virtuel avec Mamba, à installer avec [Miniforge](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh)

## TD 1: Bases de PyTorch
Initiation à la syntaxe et aux bases de [PyTorch](https://pytorch.org/) avec:
- l'implémentation du jeu "fizz buzz" par apprentissage
- manipulation et visualisation d'un ensemble de données visuelles
- apprentissage de portes logiques par un modèle neuronal

```bash
mamba create --name cs_td
mamba activate cs_td
mamba install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia # sept. 2024
mamba install matplotlib
```

Si vous avez une GPU, il faut préalablement installer les drivers NVIDIA (et redémarrer votre machine). Avec e.g. ubuntu 22.04:
```
ubuntu-drivers devices # --> liste des drivers disponibles
sudo apt install nvidia-driver-535
```

Code pour [Tensorflow](https://www.tensorflow.org/) partiellement disponible mais non corrigé en TD.

## TD 2: DNN classiques: MLP, CNN, (bi)LSTM
* Apprentissage de chiffres manuscrits sur [MNIST](http://yann.lecun.com/exdb/mnist/) avec un MLP, un CNN et un (bi)LSTM
* Visualaisation des *feature maps* d'un CNN
* Calcul de l'occupationmémoire d'un modèle

## TD 3: transfer learning et finetuning
* Transfert d'apprentissage entre ImageNet et un petit problème cible. 
* Étude du réglage fin (*fine tunig*) du réseau.

```bash
mamba activate cs_td
mamba install scikit-learn
pip install timm # huggingface models for computer vision
```

## TD 4: GAN
* Modèle génératif (GAN) sur des nuages de points 2D et 3D
* Inférence avec le framework de deep learning embarqué [Aidge](https://projects.eclipse.org/projects/technology.aidge)
```bash
mamba activate cs_td
pip install absl-py
```
Pour l'inférence avec le framework [Aidge](https://projects.eclipse.org/projects/technology.aidge) on peut installer un environnement séparé (ou ajouter Aidge à l'environnement `cs_td`):
```
mamba create --name aidge python=3.8
mamba activate aidge
git clone --recursive https://gitlab.eclipse.org/eclipse/aidge/aidge.git
cd aidge && pip install .
cd aidge/aidge_core && pip install .
cd ../aidge_backend_cpu/ && pip install .
cd ../aidge_onnx/ && pip install .

# il faut re-charger l'environnement *aidge* puis tester avec
python -c "import aidge_core; import aidge_backend_cpu; print(aidge_core.Tensor.get_available_backends())"
```
Pour les mesures en transport optimal:
```bash
 pip install geomloss
```

## TD 5: NLP et Tensorboard

```bash
mamba activate cs_td
# python -m pip install -U torch - tb - profiler
pip install -U torch -tb-profiler
mamba install --name cs_td tensorboard
# mamba update --name cs_td tensorboard
```
