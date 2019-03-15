from __future__ import absolute_import, division, print_function

import os   
import numpy as np 
import tensorflow as tf

from operator import itemgetter

from absl import flags, logging, app

from data import mnist_tf2
from models.mlp import MLP
from models.cnn import CNN
from models.bilstm import BiLSTM
from models.lstm import LSTM
from models.gru import GRU

K = tf.keras # alias for keras

models = {
    'mlp': MLP,
    'cnn': CNN,
    'lstm': LSTM,
    'bilstm': BiLSTM,
    'gru': GRU,
}

logging.set_verbosity(logging.INFO)

# ======================== Define some usefull flags ===========================

# For command line users
# ----------------------

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 100, '')
flags.DEFINE_float('learning_rate', 0.001, '')
flags.DEFINE_integer('final_step', -1, 'set to `-1` to train indefinitely')
flags.DEFINE_integer('train_info_freq', 100, '')
flags.DEFINE_integer('valid_info_freq', 1000, '')
flags.DEFINE_boolean('final_test', False,'')
flags.DEFINE_string('model', 'mlp', '')

# example:
# >> python main.py --batch_size=200 --final_step=2000

# For IPython notebook users
# --------------------------

# class HPARAMS:
#     def __init__(self, batch_size=100, learning_rate=0.001, final_step=-1, 
#                  train_info_freq=10, valid_info_freq=1000, final_test=False, 
#                  model='mlp'):
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.final_step = final_step
#         self.train_info_freq = train_info_freq
#         self.valid_info_freq = valid_info_freq
#         self.final_test = final_test
#         self.model = model
# FLAGS = HPARAMS() # change hyper-parameters here

# ================================ Read data ===================================

def main(argv):

    datasets = mnist_tf2.load(FLAGS.batch_size)
    train_dataset = datasets['train']
    valid_dataset = datasets['valid']
    test_dataset = datasets['test']

    # ========================== Create computation graph ==========================

    # create model
    model = models[FLAGS.model]()
    model.build(input_shape=(FLAGS.batch_size,28,28,1))
    model.summary()

    loss_func = tf.losses.CategoricalCrossentropy()
    optimizer = tf.optimizers.Adam(FLAGS.learning_rate)
    mean_accuracy = tf.metrics.Accuracy()
    mean_loss = tf.metrics.Mean()

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            softmax = tf.nn.softmax(logits)
            loss = loss_func(tf.one_hot(labels, 10), softmax)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        mean_accuracy.update_state(predictions, labels)
        mean_loss.update_state(loss)

    @tf.function
    def evaluation_step(images, labels):
        logits = model(images, training=False)
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        mean_accuracy.update_state(predictions, labels)

    # ============================== Do the training ===============================

    train_iterator = train_dataset.__iter__()
    
    for step, example in enumerate(train_dataset):
        if step == FLAGS.final_step:
            break

        images, labels = itemgetter('image', 'label')(example)
        train_step(images, labels)

        if step % FLAGS.train_info_freq == 0:
            template = '| step: {:6d} | loss: {:7.5f} | accuracy: {:7.5f} | valid accuracy:   ---   |'
            logging.info(template.format(step, mean_loss.result(), mean_accuracy.result()))

            mean_accuracy.reset_states()
            mean_loss.reset_states()

        if step % FLAGS.valid_info_freq == 0:
            mean_accuracy.reset_states()
            for example in valid_dataset:
                images, labels = itemgetter('image', 'label')(example)
                evaluation_step(images, labels)

            template = '| step: {:6d} | loss:   ---   | accuracy:   ---   | valid accuracy: {:7.5f} |'
            logging.info(template.format(step, mean_accuracy.result()))

    if FLAGS.final_test:
        mean_accuracy.reset_states()
        for example in test_dataset:
            images, labels = itemgetter('image', 'label')(example)
            evaluation_step(images, labels)

        template = 'final accuracy: {:7.5f}'
        logging.info(template.format(mean_accuracy.result()))


if __name__ == '__main__':
    app.run(main)