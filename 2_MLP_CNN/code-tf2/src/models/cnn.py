import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    InputLayer, Conv2D, MaxPool2D, Flatten, Dense, Dropout)



class CNN(Model):
    def __init__(self, ch=64, name='cnn', **kwargs):
        super(CNN, self).__init__(name=name, **kwargs)
        
        self.input_layer = InputLayer((28, 28, 1),
            name='{}_input'.format(name))
        
        self.conv_layer_1 = Conv2D(8*ch, (5,5), activation=tf.nn.relu,
            name='{}_conv_1'.format(name))
        self.pool_layer_1 = MaxPool2D(
            name='{}_maxpool_1'.format(name))

        self.conv_layer_2 = Conv2D(4*ch, (5,5), activation=tf.nn.relu,
            name='{}_conv_2'.format(name))
        self.pool_layer_2 = MaxPool2D(
            name='{}_maxpool_2'.format(name))

        self.flatten = Flatten(
            name='{}_flatten'.format(name))
        self.dropout = Dropout(0.5,
            name='{}_dropout'.format(name))
        self.dense_layer = Dense(2*ch, activation=tf.nn.relu,
            name='{}_dense'.format(name))
        
        self.output_layer = Dense(10, activation=tf.nn.softmax)
    
    def call(self, inputs, training=False):
        net = self.input_layer(inputs)
        net = self.conv_layer_1(net)
        net = self.pool_layer_1(net)
        net = self.conv_layer_2(net)
        net = self.pool_layer_2(net)
        net = self.flatten(net)
        net = self.dropout(net, training=training) 
        net = self.dense_layer(net)
        net = self.output_layer(net)
        return net