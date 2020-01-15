Train a multilayer perceptron (MLP) and a convolutional neural network (CNN) to classify handwritten digits ([MNIST](http://yann.lecun.com/exdb/mnist/)). Implementation [PyTorch](https://pytorch.org/) and [Tensorflow](https://www.tensorflow.org/) with *eager execution*.

## Requirements (exemple)
Based on Python 3.7.x
- Pytorch 1.1.0 to 1.3.1 (see warningbelow)
- Tensorflow 2.0.0

You should install using [conda](https://docs.conda.io/en/latest/miniconda.html) to avoid to install CUDA and CuDNN by yourself. Some python packages are required as well:
- matplotlib
- numpy (installed with tf/pytorch)

**Warning** in january 2020, `torchvision` is not compatible with PILLOW 7 (__'PILLOW_VERSION' removed from 'PIL'__). You need to downgrade it to e.g version 6.2.1.

## Run the program 
For PyTorch, choose the model into `mnist_MLP_CNN_pytorch.py` (line 102) then run:

```bash
python mnist_MLP_CNN_pytorch.py

```

For Tensorflow and the MLP model:
```bash
cd code-tf2/src/
python train.py
```
For Tensorflow and the CNN model:
```bash
cd code-tf2/src/
python train.py --model cnn
```


