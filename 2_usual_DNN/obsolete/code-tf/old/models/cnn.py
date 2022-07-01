from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 

K = tf.keras



def build(ch):

    # input
    x = K.layers.Input(shape=[28,28,1], dtype=tf.float32)

    # convolutional part
    h = K.layers.Conv2D( # 24x24xch
        filters=ch, 
        kernel_size=5, 
        padding='valid',
        activation=tf.nn.relu)(x)

    h = K.layers.MaxPool2D()(h) # 12x12xch

    h = K.layers.Conv2D( # 8x8
        filters=ch, 
        kernel_size=5, 
        padding='valid',
        activation=tf.nn.relu)(h) 

    h = K.layers.MaxPooling2D()(h) # 4x4xch

    # fully-connected part
    h = K.layers.Flatten()(h) #1x(16xch)
    
    h = K.layers.Dropout(0.5)(h) # some regularization

    h = K.layers.Dense(units=2*ch, activation=tf.nn.relu)(h)
    
    # output
    y = K.layers.Dense(units=10)(h)

    return K.Model(inputs=x, outputs=y)
