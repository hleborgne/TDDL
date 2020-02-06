from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

K = tf.keras # alias for keras



def load(batch_size):
    # Load dataset
    (x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()

    # let split the train set in two 90% for training and 10% for validation
    np.random.seed(42) # necessary to make the results reproductible
    N = len(x_train)
    x_valid = np.zeros([N%10,28,28])
    y_valid = np.zeros([N%10])
    
    I = np.random.choice(range(N), N // 10, replace=False)
    
    x_valid, y_valid = x_train[I], y_train[I]
    x_train = np.delete(x_train, I, 0)
    y_train = np.delete(y_train, I, 0)

    def make_dataset(x, y, shuffle_and_repeat=False):
        """function that creates a `tf.data.Iterator` object"""

        dataset = tf.data.Dataset.from_tensor_slices((x, y))  
        if shuffle_and_repeat:
            dataset = dataset.apply(
                tf.data.experimental.shuffle_and_repeat(buffer_size=100000))

        def parse(image, label):
            """function that normalizes the examples"""
            image = tf.reshape(image, [28,28,1]) # add channel dimension
            image = tf.cast(image, tf.float32)
            image = image / 127.5 - 1.0

            label = tf.cast(label, tf.int32)

            return {'image': image, 'label': label}

        dataset = dataset.apply(tf.data.experimental.map_and_batch(
            map_func=parse, batch_size=batch_size, num_parallel_batches=8))

        return dataset

    train_dataset = make_dataset(x_train, y_train, shuffle_and_repeat=True)
    valid_dataset = make_dataset(x_valid, y_valid)
    test_dataset = make_dataset(x_test, y_test)

    return {
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset
    }
