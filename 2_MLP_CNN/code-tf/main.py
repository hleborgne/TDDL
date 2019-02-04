from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os   
import numpy as np 
import tensorflow as tf

from operator import itemgetter

from data import mnist
from models import mlp
from models import cnn

K = tf.keras # alias for keras

models = {
    'mlp': mlp,
    'cnn': cnn,
}

tf.logging.set_verbosity(tf.logging.INFO)

# ======================== Define some usefull flags ===========================

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 100, '')
flags.DEFINE_float('learning_rate', 0.001, '')
flags.DEFINE_integer('initial_step', 0, '')
flags.DEFINE_integer('final_step', -1, 'set to `-1` to train indefinitely')
flags.DEFINE_integer('info_freq', 10, '')
flags.DEFINE_bool('fine_tune', False, '')
flags.DEFINE_string('model', 'mlp', '')
flags = tf.app.flags
FLAGS = flags.FLAGS

# example:
# >> python main.py --batch_size=200 --final_step=2000

# ================================ Read data ===================================

# load datasets
datasets = mnist.load(FLAGS.batch_size)
train_iterator = datasets['train']
valid_iterator = datasets['valid']

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
model = models[FLAGS.model].build(ch=64)

# create train op
training = tf.placeholder_with_default(False, [])
logits = model(images, training=training)
predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
loss = tf.losses.softmax_cross_entropy(tf.one_hot(labels, 10), logits)
train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

# metric for training:
accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

# metrics for validation:
valid_accuracy, valid_accuracy_op = tf.metrics.accuracy(labels=labels, predictions=predictions)

# ============================== Do the training ===============================

with tf.Session() as sess:

    # initialize variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    # create handles to choose the dataset to use
    train_switch = sess.run(train_iterator.string_handle())
    valid_switch = sess.run(valid_iterator.string_handle())

    # training loop
    step = FLAGS.initial_step
    while step != FLAGS.final_step:
        step += 1

        sess.run(train_op, {
            dataset_switch: train_switch,
            training: True,    
        })

        if step % FLAGS.info_freq == 0:
            loss_val, accuracy_val = sess.run([loss, accuracy], {dataset_switch: train_switch})
            tf.logging.info(
                'step {} - loss: {:7.5f} - accuracy: {:7.5f}'.format(step, loss_val, accuracy_val))
    
    # validation
    while True: # iterate on the whole validation set
        try:
            sess.run(valid_accuracy_op, {dataset_switch: valid_switch})
        except tf.errors.OutOfRangeError:
            break
    
    valid_accuracy_val = sess.run(valid_accuracy)

    tf.logging.info('validation_accuracy: {}'.format(valid_accuracy_val))

    # you should obtain an accuracy over 99% with 2000 step and a batch size of 200 with the cnn.
    # and over 98% with 4000 steps and batch size 400 with the mlp