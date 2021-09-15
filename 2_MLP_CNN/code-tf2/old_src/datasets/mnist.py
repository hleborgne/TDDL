import os
import tensorflow as tf
import tensorflow_datasets as tfds

# dataset from tensorflow-datasets: https://www.tensorflow.org/datasets

def load(batch_size=1, split='train'):

    # download the dataset
    dataset = tfds.load(
        name='mnist',
        split=split,
        data_dir=os.path.join('..', 'data', 'mnist'),
        shuffle_files=True,
        download=True)

    # preprocessing step
    def prepare(features):
        image = tf.cast(features['image'], tf.float32) / 255. * 2 - 1
        label = tf.one_hot(features['label'], 10)
        return {'image': image, 'label': label}

    dataset = dataset.map(prepare, num_parallel_calls=8)
    dataset = dataset.repeat() # repeated to be used as long as needed
    dataset = dataset.shuffle(60000) # shuffle the dataset
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)

    return dataset

if __name__ == '__main__':
    data_dir = os.path.join('..', 'data', 'mnist')
    os.makedirs(data_dir, exist_ok=True)

    builder = tfds.builder('mnist')
    builder.download_and_prepare(download_dir=os.path.join('..','data','mnist'))
