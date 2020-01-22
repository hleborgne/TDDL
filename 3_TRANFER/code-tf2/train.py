# -*- coding: utf-8 -*
from __future__ import absolute_import, division, print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from absl import app, flags, logging
from operator import itemgetter

from datasets import beers as target_dataset
from models.mobilenet import MobileNet

# default project structure

# project
#   +--code-tf2
#   |   +--data
#   |      +--carlsberg
#   |      |   +--images.jpg
#   |      |   +--...
#   |      +--chimay
#   |      |   +--images.jpg
#   |      |   +--...
#   |      +--...
#   |   +--datasets
#   |      +--coast_forest_highway.py
#   |   +--models
#   |      +--mobilenet.py
#   |   +--train.py
#   |   +--README_tf.md   

# NB: vous pouvez faire un lien dynamique vers les données:
#           cd project
#           ln -s /path/to/data data

# Lancer le script
# >> python3 train.py --batch_size=16 --final_step=20 --info_freq=1
#
# you can try with larger batch_size or more steps for better performances (but it's slower)

# ======================== Load a pre-trained network ==========================

def main(argv):

    np.random.seed(43) # to make the results reproductible
    tf.random.set_seed(42) # to make the results reproductible 

    # Create working directories
    experiment_dir  = os.path.join(FLAGS.output_dir,
        FLAGS.experiment_name)

    # Logging training informations
    logging.get_absl_handler().use_absl_log_file('logs', experiment_dir)

    # ======================= Read target problem data =========================

    train_dataset, valid_dataset, test_dataset = target_dataset.load(
        FLAGS.data_dir, FLAGS.batch_size)

    # ========================= Do transfer learning ===========================

    model = MobileNet(fine_tune=FLAGS.fine_tune)
    model.build(input_shape=(FLAGS.batch_size, 224, 224, 3))
    model.summary()

    # Create training operations
    loss_func = tf.losses.CategoricalCrossentropy()
    optimizer = tf.optimizers.Adam(FLAGS.learning_rate)
    train_accuracy = tf.metrics.Accuracy(name='train_accuracy')
    train_loss = tf.metrics.Mean()

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model(images)
            loss = loss_func(tf.one_hot(labels, 6), logits)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss.update_state(loss)

        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        train_accuracy.update_state(predictions, labels)

    # Create a metric to compute the accuracy on the validation set 

    valid_loss = tf.metrics.Mean()
    valid_accuracy = tf.metrics.Accuracy(name='valid_accuracy')

    @tf.function
    def valid_step(images, labels):
        logits = model(images)
        loss = loss_func(tf.one_hot(labels, 6), logits)

        valid_loss.update_state(loss)
        
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        valid_accuracy.update_state(predictions, labels)

    test_loss = tf.metrics.Mean()
    test_accuracy = tf.metrics.Accuracy(name='test_accuracy')

    @tf.function
    def test_step(images, labels):
        logits = model(images)
        loss = loss_func(tf.one_hot(labels, 6), logits)

        test_loss.update_state(loss)
        
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        test_accuracy.update_state(predictions, labels)
    
    # =========================== Train the model ==============================

    # training

    for step, example in train_dataset.enumerate(FLAGS.initial_step):
        if step == FLAGS.final_step: break

        images, labels = example['image'], example['label']

        train_step(images, labels)

        if step % FLAGS.info_freq == 0:
            template = 'step {} - loss: {:4.2f} - accuracy: {:5.2%}'
            logging.info(
                template.format(step, train_loss.result(), train_accuracy.result()))

            train_loss.reset_states()
            train_accuracy.reset_states()


        if step % FLAGS.valid_freq == 0:
            for example in valid_dataset:
                images, labels = example['image'], example['label']
                valid_step(images, labels)    

            template = '------------------------------------------- Validation: loss = {:5.2f},  accuracy {:5.2%}'
            logging.info(
                template.format(valid_loss.result(), valid_accuracy.result()))

            valid_loss.reset_states()
            valid_accuracy.reset_states()
 
    # validation

    for example in test_dataset:
        images, labels = example['image'], example['label']
        test_step(images, labels)    

    template = 'Test: loss = {:5.2f},  accuracy {:5.2%}'
    logging.info(
        template.format(test_loss.result(), test_accuracy.result()))


if __name__ == '__main__':

    FLAGS = flags.FLAGS
    flags.DEFINE_string('output_dir', os.path.join('outputs'), "")
    flags.DEFINE_string('experiment_name', 'test', "")

    flags.DEFINE_integer('batch_size', 10, '')
    flags.DEFINE_float('learning_rate', 0.001, '')

    flags.DEFINE_integer('initial_step', 0, '')
    flags.DEFINE_integer('final_step', -1, 'set to `-1` to train indefinitely')
    flags.DEFINE_integer('info_freq', 1, '')
    flags.DEFINE_integer('valid_freq', 5, '')
    flags.DEFINE_string('data_dir', 'data', '')
    flags.DEFINE_bool('fine_tune', False, '')

    app.run(main)


