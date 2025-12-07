from __future__ import absolute_import, division, print_function

import os   
import numpy as np 
import tensorflow as tf

from operator import itemgetter

from absl import flags, logging, app # prepare for tensorflow 2.0

from data import mnist
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
flags.DEFINE_integer('initial_step', 0, '')
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
#     def __init__(self, batch_size=100, learning_rate=0.001, initial_step=0, 
#                  final_step=-1, train_info_freq=10, valid_info_freq=1000, 
#                  final_test=False, model='mlp'):
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.initial_step = initial_step
#         self.final_step = final_step
#         self.train_info_freq = train_info_freq
#         self.valid_info_freq = valid_info_freq
#         self.final_test = final_test
#         self.model = model
# FLAGS = HPARAMS() # change hyper-parameters here

# ================================ Read data ===================================

def main(argv):
    # load datasets
    datasets = mnist.load(FLAGS.batch_size)
    train_iterator = datasets['train']
    valid_iterator = datasets['valid']
    test_iterator = datasets['test']

    # dataset switch (allows us to switch between train and valid set as will)
    dataset_switch = tf.placeholder(tf.string, shape=[])
    data_iterator = tf.data.Iterator.from_string_handle(
        string_handle=dataset_switch, 
        output_types=datasets['output_types'], 
        output_shapes=datasets['output_shapes']
    )

    features = data_iterator.get_next()
    images, labels = itemgetter('image', 'label')(features)

    # ========================== Create computation graph ==========================

    # create model
    model = models[FLAGS.model]() # call the constructor of the model

    # create train op
    training = tf.placeholder_with_default(False, [])
    logits = model(images, training=training)
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
    loss = tf.losses.softmax_cross_entropy(tf.one_hot(labels, 10), logits)
    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

    # Accuracy metric tensorflow 2.0 style
    def accuracy_tf_1(predictions, labels, name):
        result, update_state =  tf.metrics.accuracy(labels=labels, 
            predictions=predictions, name=name)
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=name)
        reset_state = tf.variables_initializer(var_list=running_vars)
        return result, update_state, reset_state

    # metric for training, validation and testing:
    train_acc, update_train_acc, reset_train_acc = accuracy_tf_1(
        predictions, labels, 'train_accuracy')
    valid_acc, update_valid_acc, reset_valid_acc = accuracy_tf_1(
        predictions, labels, 'valid_accuracy')
    test_acc, update_test_acc, reset_test_acc = accuracy_tf_1(
        predictions, labels, 'test_accuracy')

    # ============================== Do the training ===============================

    with tf.Session() as sess:

        # initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # create handles to choose the dataset to use
        train_switch = sess.run(train_iterator.string_handle())
        valid_switch = sess.run(valid_iterator.string_handle())
        test_switch = sess.run(test_iterator.string_handle())

        step = FLAGS.initial_step
        while step != FLAGS.final_step:
            step += 1

            sess.run([train_op, update_train_acc], feed_dict={
                dataset_switch: train_switch,
                training: True,
            })

            if step % FLAGS.train_info_freq == 0:
                loss_val, accuracy_val = sess.run([loss, train_acc], feed_dict={
                    dataset_switch: train_switch
                })
                template = '| step {:6d} | loss: {:7.5f} | accuracy: {:7.5f} | valid accuracy:   ---   |'
                logging.info(template.format(step, loss_val, accuracy_val))
                sess.run(reset_train_acc)

            if step % FLAGS.valid_info_freq == 0:

                sess.run([valid_iterator.initializer, reset_valid_acc])
                while True:
                    try:
                        sess.run(update_valid_acc, feed_dict={
                            dataset_switch: valid_switch
                        })
                    except tf.errors.OutOfRangeError:
                        break

                accuracy_val = sess.run(valid_acc)
                template = '| step {:6d} | loss:   ---   | accuracy:   ---   | valid accuracy: {:7.5f} |'
                logging.info(template.format(step, accuracy_val))

        if FLAGS.final_test:

            sess.run([test_iterator.initializer, reset_test_acc])
            while True:
                try:
                    sess.run(update_test_acc, feed_dict={
                        dataset_switch: test_switch
                    })
                except tf.errors.OutOfRangeError:
                    break

            accuracy_val = sess.run(test_acc)
            template = 'final accuracy: {:7.5f}'
            logging.info(template.format(accuracy_val))


if __name__ == '__main__':
    app.run(main)