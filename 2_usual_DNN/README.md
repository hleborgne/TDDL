Train a multilayer perceptron (MLP) and a convolutional neural network (CNN) to classify handwritten digits ([MNIST](http://yann.lecun.com/exdb/mnist/)). Implementation [PyTorch](https://pytorch.org/) and [Tensorflow](https://www.tensorflow.org/) with *eager execution*.

## Requirements (example)
Originally based on Python 3.7.x. Works with Python 3.8.5
- Pytorch 1.1.0 to 1.7.0 (see warning below for older version)

You should install using [conda](https://docs.conda.io/en/latest/miniconda.html) to avoid installing CUDA and CuDNN by yourself. Some Python packages are required as well:
- matplotlib
- NumPy (installed with tf/pytorch)

**Warning** with old PyTorch (January 2020, about version 1.3.1), `torchvision` is not compatible with PILLOW 7 (__'PILLOW_VERSION' removed from 'PIL'__). You need to downgrade it to e.g. version 6.2.1. With PyTorch 1.7.0, it works directly.

To only remove warnings, you can add this in the Python program:
```
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
```
## Exercise
To work on the exercice use program `mnist_MLP_CNN_pytorch_exercice.py` that contains TODOs.

## Run the program (correction)
For PyTorch, choose the model into `mnist_MLP_CNN_pytorch.py` (line 112) then run:

```bash
python mnist_MLP_CNN_pytorch.py

```

## For those interested
For Tensorflow (2.0.0) and the MLP model:
```bash
cd code-tf2/src/
python train.py
```
For Tensorflow and the CNN model:
```bash
cd code-tf2/src/
python train.py --model cnn
```


