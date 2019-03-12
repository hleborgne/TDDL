# -*- coding: utf-8 -*
from __future__ import absolute_import, division, print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras as K

from absl import app, flags, logging
from operator import itemgetter

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

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 10, '')
flags.DEFINE_float('learning_rate', 0.001, '')
flags.DEFINE_integer('initial_step', 0, '')
flags.DEFINE_integer('final_step', -1, 'set to `-1` to train indefinitely')
flags.DEFINE_integer('info_freq', 1, '')
flags.DEFINE_integer('valid_freq', 5, '')
flags.DEFINE_string('data_dir', '../data', '')
flags.DEFINE_bool('fine_tune', False, '')

# example:
# >> python main.py --batch_size=16 --final_step=20 --info_freq=1
#
# you can try with larger batch_size or more steps for better performances (but it's slower)

# ======================== Load a pre-trained network ==========================

def main(argv):

    np.random.seed(43) # to make the results reproductible
    tf.random.set_seed(42) # to make the results reproductible 
    logging.set_verbosity(logging.INFO)

    # ======================= Read target problem data =========================

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

    # Create three `tf.data.Dataset` objects

    def make_iterator(filenames, labels, batch_size, shuffle_and_repeat=False):
        """function that creates a `tf.data.Iterator` object"""
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        if shuffle_and_repeat:
            dataset = dataset.apply(
                tf.data.experimental.shuffle_and_repeat(buffer_size=1000))

        def parse(filename, label):
            """function that reads the image and normalizes it"""
            image = tf.io.read_file(filename)
            image = tf.image.decode_jpeg(image)
            image = tf.image.resize(image, [224,224])
            image = tf.cast(image, tf.float32)
            image = image / 256
            return {'image': image, 'label': label}

        dataset = dataset.apply(tf.data.experimental.map_and_batch(
            map_func=parse, batch_size=batch_size, num_parallel_batches=8))

        return dataset

    train_dataset = make_iterator(train_image_filenames, train_image_labels,
        batch_size=FLAGS.batch_size, shuffle_and_repeat=True)
    valid_dataset = make_iterator(valid_image_filenames, valid_image_labels,
        batch_size=FLAGS.batch_size)
    test_dataset = make_iterator(test_image_filenames, test_image_labels,
        batch_size=FLAGS.batch_size)

    # ====================== Do the actual transfer learning ===================

    # Define our model with keras model subclassing

    class MyModel(K.Model):

        def __init__(self):
            super(MyModel, self).__init__()
            self.backbone = hub.KerasLayer( # Siiick !!!
                'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/1',
                trainable=FLAGS.fine_tune,
                output_shape=[1280],
            )
            self.dense = K.layers.Dense(units=3)
            self.softmax = K.layers.Softmax()
        
        def call(self, x):
            h = self.backbone(x)
            h = self.dense(h)
            y = self.softmax(h)
            return y

    model = MyModel()
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
            loss = loss_func(tf.one_hot(labels, 3), logits)

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
        loss = loss_func(tf.one_hot(labels, 3), logits)

        valid_loss.update_state(loss)
        
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        valid_accuracy.update_state(predictions, labels)

    test_loss = tf.metrics.Mean()
    test_accuracy = tf.metrics.Accuracy(name='test_accuracy')

    @tf.function
    def test_step(images, labels):
        logits = model(images)
        loss = loss_func(tf.one_hot(labels, 3), logits)

        test_loss.update_state(loss)
        
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        test_accuracy.update_state(predictions, labels)
    
    # =========================== Train the model ==============================

    # training

    step = FLAGS.initial_step
    train_iterator = train_dataset.__iter__()
    while step != FLAGS.final_step:
        step += 1

        example = next(train_iterator)
        images, labels = example['image'], example['label']

        train_step(images, labels)

        if step % FLAGS.info_freq == 0:
            
            template = 'step {} - loss: {:4.2f} - accuracy: {:5.2f}%'
            logging.info(
                template.format(step, train_loss.result(), train_accuracy.result()*100))

            train_loss.reset_states()
            train_accuracy.reset_states()


        if step % FLAGS.valid_freq == 0:
            
            for example in valid_dataset:
                images, labels = example['image'], example['label']
                valid_step(images, labels)    

            template = '------------------------------------------- Validation: loss = {:5.2f},  accuracy {:5.2f}%'
            logging.info(
                template.format(valid_loss.result(), valid_accuracy.result()*100))

            valid_loss.reset_states()
            valid_accuracy.reset_states()

    # validation

    for example in test_dataset:
        images, labels = example['image'], example['label']
        test_step(images, labels)    

    template = 'Test: loss = {:5.2f},  accuracy {:5.2f}%'
    logging.info(
        template.format(test_loss.result(), test_accuracy.result()*100))


if __name__ == '__main__':
    app.run(main)
