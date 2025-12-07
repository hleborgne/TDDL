import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# dataset from tensorflow-datasets: https://www.tensorflow.org/datasets

def load(data_dir, batch_size=1):

    # Get the filenames and label of our data
    image_filenames = []
    image_labels = []
    for label, category in enumerate(['carlsberg', 'chimay', 'corona', 'fosters', 'guiness', 'tsingtao']):
        image_names = os.listdir(os.path.join(data_dir, category))
        image_names = sorted(image_names) # to make the results reproductibles
        image_filenames += [os.path.join(
            data_dir, category, image_name) for image_name in image_names]
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

    # Create the `tf.data.Dataset` objects
    def create_dataset(filenames, labels, batch_size, shuffle_and_repeat=False):
        """function that creates a `tf.data.Iterator` object"""
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        if shuffle_and_repeat:
            dataset = dataset.repeat() # repeated to be used as long as needed
            dataset = dataset.shuffle(10000) # shuffle the dataset

        # preprocessing step
        def prepare(filename, label):
            """function that reads the image and normalizes it"""
            image = tf.io.read_file(filename)
            image = tf.image.decode_jpeg(image)
            image = tf.image.resize(image, [224,224])
            image = tf.cast(image, tf.float32)
            image = image / 256
            return {'image': image, 'label': label}

        dataset = dataset.map(prepare, num_parallel_calls=8)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(2)

        return dataset

    train_dataset = create_dataset(train_image_filenames, train_image_labels,
        batch_size=batch_size, shuffle_and_repeat=True)
    valid_dataset = create_dataset(valid_image_filenames, valid_image_labels,
        batch_size=batch_size)
    test_dataset = create_dataset(test_image_filenames, test_image_labels,
        batch_size=batch_size)
    
    return train_dataset, valid_dataset, test_dataset
    