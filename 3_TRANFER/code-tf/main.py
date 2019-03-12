# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from operator import itemgetter

K = tf.keras


np.random.seed(42) # to make the results reproductible
tf.random.set_random_seed(43) # to make the results reproductible 
tf.logging.set_verbosity(tf.logging.INFO)

# default project structure

# project
#   +--code-tf
#   |   +--main.py
#   |   +--README_tf.md
#   |
#   +--data
#       +--coast
#       |   +--images.jpg
#       |   +--...
#       +--forest--
#       |   +--images.jpg
#       |   +--.
#       +--highway
#           +--image.jpg
#           +--...

# NB: vous pouvez faire un lien dynamique vers les données:
#           cd project
#           ln -s /path/to/data data

# ======================== Define some usefull flags ===========================

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 10, '')
flags.DEFINE_float('learning_rate', 0.001, '')
flags.DEFINE_integer('initial_step', 0, '')
flags.DEFINE_integer('final_step', -1, 'set to `-1` to train indefinitely')
flags.DEFINE_integer('info_freq', 10, '')
flags.DEFINE_integer('info_valid_freq', 5, '')
flags.DEFINE_string('data_dir', '../data', '')
flags.DEFINE_bool('fine_tune', False, '')

# example:
# >> python main.py --batch_size=16 --final_step=20 --info_freq=1
#
# you can try with larger batch_size or more steps for better performances (but it's slower)

# ======================== Load a pre-trained network ==========================

ResNet50 = hub.Module(
    "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1",
    trainable=FLAGS.fine_tune,
    # tags=set() if not FLAGS.fine_tune else {'train'} # à décommenter à vos risques et périls ! 
    )
height, width = hub.get_expected_image_size(ResNet50)

# ========================= Read target problem data ===========================

# Get the filenames and label of our data
image_filenames = []
image_labels = []
for label, category in enumerate(['coast', 'forest', 'highway']):
    image_names = os.listdir(os.path.join(FLAGS.data_dir, category))
    image_names = sorted(image_names) # to make the results reproductibles
    image_filenames += [os.path.join(
        FLAGS.data_dir, category, image_name) for image_name in image_names]
    image_labels += [label] * len(image_names)

# Split data in three for training, validation and test
train_image_filenames, train_image_labels = [], []
valid_image_filenames, valid_image_labels = [], []
test_image_filenames, test_image_labels  = [], []

for image_filename, image_label in zip(image_filenames, image_labels):

    # 80% of data in training set, 10% in validation set and 10% in test set
    # 56.25% of data in training set, 18.75% in validation set and 25% in test set
    x = np.random.choice(['train', 'valid', 'test'], p=[0.75*0.75,0.75*0.25,0.25])

    if x == 'train':
        train_image_filenames.append(image_filename)
        train_image_labels.append(image_label)
    if x == 'valid':
        valid_image_filenames.append(image_filename)
        valid_image_labels.append(image_label)
    if x == 'test':
        test_image_filenames.append(image_filename)
        test_image_labels.append(image_label)

# Create three `tf.data.Iterator` objects

def make_iterator(filenames, labels, batch_size, shuffle_and_repeat=False):
    """function that creates a `tf.data.Iterator` object"""
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    if shuffle_and_repeat:
        dataset = dataset.apply(
            tf.data.experimental.shuffle_and_repeat(buffer_size=1000))

    def parse(filename, label):
        """function that reads the image and normalizes it"""
        image = tf.read_file(filename)
        image = tf.image.decode_jpeg(image)
        image = tf.cast(image, tf.float32)
        image = image / 256
        return {'image': image, 'label': label}

    dataset = dataset.apply(tf.data.experimental.map_and_batch(
        map_func=parse, batch_size=batch_size, num_parallel_batches=8))

    return dataset.make_one_shot_iterator()

train_iterator = make_iterator(train_image_filenames, train_image_labels,
    batch_size=FLAGS.batch_size, shuffle_and_repeat=True)
valid_iterator = make_iterator(valid_image_filenames, valid_image_labels,
    batch_size=FLAGS.batch_size)
test_iterator = make_iterator(test_image_filenames, test_image_labels,
    batch_size=FLAGS.batch_size)

# ====================== Do the actual transfer learning =======================

# Define our model with keras

inputs = K.layers.Input(shape=[height, width, 3])
feature_vector = K.layers.Lambda(ResNet50)(inputs)
outputs = K.layers.Dense(units=3)(feature_vector)

final_model = K.Model(inputs=inputs, outputs=outputs)

# Create training operations

features = train_iterator.get_next()
images, labels = itemgetter('image', 'label')(features)

logits = final_model(images)

predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

loss = tf.losses.softmax_cross_entropy(tf.one_hot(labels, 3), logits)
optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
train_op = optimizer.minimize(loss)

# Create some metrics to monitor the training

accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

# Create a metric to compute the accuracy on the validation set 

valid_features = valid_iterator.get_next()
valid_images, valid_labels = itemgetter('image', 'label')(valid_features)

valid_logits = final_model(valid_images)

valid_predictions = tf.cast(tf.argmax(valid_logits, axis=-1), tf.int32)
valid_accuracy, valid_accuracy_op = tf.metrics.accuracy(valid_labels, valid_predictions)

# Create a metric to compute the accuracy on the test set 

test_features = test_iterator.get_next()
test_images, test_labels = itemgetter('image', 'label')(test_features)

test_logits = final_model(test_images)

test_predictions = tf.cast(tf.argmax(test_logits, axis=-1), tf.int32)
test_accuracy, test_accuracy_op = tf.metrics.accuracy(test_labels, test_predictions)

# ============================= Train the model ================================

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    step = FLAGS.initial_step
    while step != FLAGS.final_step:
        step += 1

        sess.run(train_op)

        if step % FLAGS.info_freq == 0:
            loss_value, accuracy_value = sess.run([loss, accuracy])
            tf.logging.info(
                'step {} - loss: {:7.5f} - train accuracy: {:7.5f}'.format(step, loss_value, accuracy_value))
        # validation
        # FIXME reinitialisation de l'iterator???
        if step % FLAGS.info_valid_freq == 0:
            while True:
                try:
                    sess.run(valid_accuracy_op)
                except tf.errors.OutOfRangeError:
                    break

            valid_accuracy_value = sess.run(valid_accuracy)
            tf.logging.info('validation_accuracy: {}'.format(valid_accuracy_value))
    # test
    while True:
        try:
            sess.run(test_accuracy_op)
        except tf.errors.OutOfRangeError:
            break

    test_accuracy_value = sess.run(test_accuracy)
    tf.logging.info('test_accuracy: {}'.format(test_accuracy_value))