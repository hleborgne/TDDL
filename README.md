# TD DL
Travaux dirigés de deep learning. Il est conseillé de mettre en place un environnement virtuel avec Mamba (à installer avec [Miniforge](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh)) ou préférentiellement [uv](https://docs.astral.sh/uv/) avec:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
NB: PyTorch [ne maintient plus](https://github.com/pytorch/pytorch/issues/138506) les packages conda depuis octobre 2024. Il reste possible d'utiliser des environnements conda/mamba et `pip`. De plus, il existe encore des [packages conda-forge de PyTorch](https://anaconda.org/conda-forge/pytorch) maintenus par la communauté.


## TD 1: Bases de PyTorch
Initiation à la syntaxe et aux bases de [PyTorch](https://pytorch.org/) avec:
- l'implémentation du jeu "fizz buzz" par apprentissage
- manipulation et visualisation d'un ensemble de données visuelles
- apprentissage de portes logiques par un modèle neuronal

```bash
uv venv --python=3.11 # python version >= 3.9; une version trop récente peut poser problème pour certains projets (pas les TD)
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Si vous avez une GPU, il faut préalablement installer les drivers NVIDIA (et redémarrer votre machine). Avec e.g. ubuntu:
```
ubuntu-drivers devices # --> liste des drivers disponibles
sudo apt install nvidia-driver-535
```
puis (ici avec CUDA 11.8; autres versions possibles sur [le site de PyTorch](https://pytorch.org/))
```
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## TD 2: DNN classiques: MLP, CNN, (bi)LSTM
* Apprentissage de chiffres manuscrits sur [MNIST](http://yann.lecun.com/exdb/mnist/) avec un MLP, un CNN et un (bi)LSTM
* Visualisation des *feature maps* d'un CNN
* Calcul de l'occupation mémoire d'un modèle

## TD 3: transfer learning et finetuning
* Transfert d'apprentissage entre ImageNet et un petit problème cible. 
* Étude du réglage fin (*fine tuning*) du réseau.

```bash
uv pip install scikit-learn timm
```
`timm` fournit des modèles de vision par ordinateur 

## TD 4: GAN
* Modèle génératif (GAN) sur des nuages de points 2D et 3D
* Inférence avec le framework de deep learning embarqué [Aidge](https://projects.eclipse.org/projects/technology.aidge)
```bash
uv pip install absl-py onnx
```
Pour l'inférence avec le framework [Aidge](https://projects.eclipse.org/projects/technology.aidge) on peut installer un environnement séparé:
```
mamba create --name aidge python=3.9
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
uv pip install geomloss
```

## TD 5: NLP et Tensorboard
* apprentissage RNN et LSTM sur des mots (lettres)
* monitoring avec tensorboard

```bash
uv pip install tensorboard torch-tb-profiler
```
