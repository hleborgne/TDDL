from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 

K = tf.keras



def build(ch):

    # input
    x = K.layers.Input(shape=[28,28,1], dtype=tf.float32)

    # hidden layers
    h = K.layers.Flatten()(x) #1x(16xch)
    # h = K.layers.Dropout(0.5)(h) # some regularization
    h = K.layers.Dense(units=8*ch, activation=tf.nn.relu)(h)
    #h = K.layers.Dropout(0.5)(h) # some regularization
    h = K.layers.Dense(units=4*ch, activation=tf.nn.relu)(h)
    h = K.layers.Dropout(0.5)(h) # some regularization
    h = K.layers.Dense(units=2*ch, activation=tf.nn.relu)(h)
    
    # output
    y = K.layers.Dense(units=10)(h)

    return K.Model(inputs=x, outputs=y)
