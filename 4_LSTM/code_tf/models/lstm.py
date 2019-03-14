from __future__ import absolute_import, division, print_function

import tensorflow as tf 

K = tf.keras



class LSTM(K.Model):

    def __init__(self):
        super(LSTM, self).__init__()

        self.reshape = K.layers.Reshape([28, 28]) # sequence of 28 elements of size 28
        self.rnn = K.layers.RNN(
            cell = K.layers.LSTMCell(units=128),
            input_shape=[28, 28],
        )
        self.dense = K.layers.Dense(10)

    def call(self, x, training=True):
        h = self.reshape(x)
        h = self.rnn(h, training=training)
        y = self.dense(h)

        return y
