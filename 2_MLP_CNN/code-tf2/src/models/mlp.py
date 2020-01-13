import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Dropout



class MLP(Model):
    def __init__(self, ch=64, name='mlp', **kwargs):
        super(MLP, self).__init__(name=name, **kwargs)
        
        self.input_layer = InputLayer((28, 28, 1),
            name='{}_input'.format(name))
        self.flatten = Flatten(
            name='{}_flatten'.format(name))
        
        self.hidden_layer_1 = Dense(8*ch, activation=tf.nn.relu,
            name='{}_dense_1'.format(name))
        self.hidden_layer_2 = Dense(4*ch, activation=tf.nn.relu,
            name='{}_dense_2'.format(name))
        self.dropout_layer = Dropout(0.5,
            name='{}_dropout'.format(name))
        self.hidden_layer_3 = Dense(2*ch, activation=tf.nn.relu,
            name='{}_dense_3'.format(name))
        
        self.output_layer = Dense(10, activation=tf.nn.softmax)
    
    def call(self, inputs, training=False):
        net = self.input_layer(inputs)
        net = self.flatten(net)
        net = self.hidden_layer_1(net)
        net = self.hidden_layer_2(net)
        net = self.dropout_layer(net, training=training) 
        net = self.hidden_layer_3(net)
        net = self.output_layer(net)
        return net