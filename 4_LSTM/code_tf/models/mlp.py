from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 

K = tf.keras



class MLP(K.Model):
    
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = K.layers.Flatten()
        self.dense_1 = K.layers.Dense(units=512, activation=tf.nn.relu)
        self.dense_2 = K.layers.Dense(units=256, activation=tf.nn.relu)
        self.dense_3 = K.layers.Dense(units=128, activation=tf.nn.relu)
        self.dropout = K.layers.Dropout(0.5)
        self.outputs = K.layers.Dense(10)

    def call(self, x, training=True):
        h = self.flatten(x) 
        h = self.dense_1(h) 
        h = self.dense_2(h) 
        h = self.dense_3(h) 
        h = self.dropout(h, training=training) 
        y = self.outputs(h)
        return y