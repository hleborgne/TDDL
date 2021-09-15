import tensorflow as tf 

from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, RNN, LSTMCell, Dense, InputLayer



class LSTM(Model):

    def __init__(self, name='lstm', **kwargs):
        super(LSTM, self).__init__(name=name, **kwargs)

        self.input_layer = InputLayer((28, 28, 1),
            name='{}_input'.format(name))

        self.reshape = Reshape([28, 28],
            name='{}_reshape'.format(name)) # sequence of 28 elements of size 28
        self.rnn = RNN(input_shape=[28, 28],
            cell=LSTMCell(units=128,
                name='{}_rnn_lstm'.format(name)), 
            name='{}_rnn'.format(name))
        
        self.dense = Dense(10, activation=tf.nn.softmax,
            name='{}_output'.format(name))

    def call(self, inputs, training=True):
        net = self.input_layer(inputs)
        net = self.reshape(net)
        net = self.rnn(net, training=training)
        net = self.dense(net)
        return net
