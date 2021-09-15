#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# utilitaire pour calculer des tailles de feature map dans un réseau CNN.
# Surtout pour l'étendue spatiale. En pratique, il faut aussi considérer
# le nombre de feature maps i.e nombre de noyaux pour avoir la taille
# « complète » de sortie.
import numpy as np

def conv(sz_in,ker,stride,padding=0,dilatation=1.):
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    return np.floor(1+ (sz_in + 2*padding -dilatation*(ker - 1)-1)/stride)

def pool(sz_in,ker,stride,padding=0,dilatation=1.):
    # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
    return np.floor(1+ (sz_in + 2*padding -dilatation*(ker - 1)-1)/stride)

INPUT_SIZE=28

c1 = conv(INPUT_SIZE,5,1)
p1 = pool(c1,2,2)
c2 = conv(p1,5,1)
p2 = pool(c2,2,2)

out = p2

print ('input (spatial) size: {}'.format(INPUT_SIZE))
print ('output (spatial) size: {}'.format(out))