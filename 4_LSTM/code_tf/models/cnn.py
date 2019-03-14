from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 

K = tf.keras



class CNN(K.Model):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv_1  = K.layers.Conv2D(64, 5, activation=tf.nn.relu)
        self.pool_1  = K.layers.MaxPool2D()
        self.conv_2  = K.layers.Conv2D(64, 5, activation=tf.nn.relu)
        self.pool_2  = K.layers.MaxPool2D()
        self.flatten = K.layers.Flatten()
        self.dropout = K.layers.Dropout(0.5)
        self.dense   = K.layers.Dense(128, activation=tf.nn.relu)
        self.outputs = K.layers.Dense(10)
    
    def call(self, x, training=True):
        h = self.conv_1(x)
        h = self.pool_1(h)        
        h = self.conv_2(h)
        h = self.pool_2(h)        
        h = self.flatten(h)        
        h = self.dropout(h, training=training)
        h = self.dense(h)
        y = self.outputs(h)
        return y