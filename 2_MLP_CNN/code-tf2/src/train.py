import os
import sys
import numpy as np
import tensorflow as tf
import random

from absl import app, flags, logging

from datasets import mnist
from models.mlp import MLP 
from models.cnn import CNN

datasets = {
    'mnist': mnist
}

models = {
    'mlp': MLP,
    'cnn': CNN
}

# Allow memory growth + graph optimizations
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.optimizer.set_jit(True)

def main(argv):
    # Create working directories
    experiment_dir  = os.path.join(FLAGS.output_dir, 
        FLAGS.experiment_name, FLAGS.model, FLAGS.dataset)
    
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    saved_model_dir = os.path.join(experiment_dir, 'saved_models')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(saved_model_dir, exist_ok=True)

    # Logging training informations
    logging.get_absl_handler().use_absl_log_file('logs', experiment_dir)

    # Load dataset, model and optimizer
    dataset = datasets[FLAGS.dataset]
    train_dataset = dataset.load(FLAGS.batch_size, split='train')
    test_dataset = dataset.load(FLAGS.batch_size, split='test')

    model = models[FLAGS.model](ch=FLAGS.width_multiplier)
    model.build(input_shape=(FLAGS.batch_size, 28, 28, 1))

    optimizer = tf.optimizers.Adam(FLAGS.learning_rate)

    # define metrics
    train_loss = tf.keras.metrics.Mean(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_accuracy')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # define training and evaluation steps

    @tf.function
    def forward(features, training=False):
        print('build eval')
        predictions = model.call(features['image'], training=training)
        loss = tf.losses.categorical_crossentropy(
                y_true=features['label'],
                y_pred=predictions)
        return loss, predictions

    @tf.function
    def train_step(features):
        print('build train')
        with tf.GradientTape() as tape:
            loss, predictions = forward(features, training=True)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Manage checkpoints
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(0), model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(
        ckpt, os.path.join(experiment_dir, 'checkpoints'), max_to_keep=1)
    
    # Restore checkpoint
    if FLAGS.restore: ckpt.restore(manager.latest_checkpoint)

# ================================ TRAINING ====================================
    
    for step, features in train_dataset.enumerate(FLAGS.initial_step):
        
        train_step(features)

        if step % FLAGS.eval_freq == 0:
            for train_features in train_dataset.take(10):
                loss, predictions = forward(train_features)
                train_accuracy(tf.math.argmax(train_features['label'], axis=-1), predictions)
                train_loss(loss)
            for test_features in test_dataset.take(10):
                loss, predictions = forward(test_features)
                test_accuracy(tf.math.argmax(test_features['label'], axis=-1), predictions)
                test_loss(loss)
    

            template = 'step: {:06d} - train loss/acc: {:3.2f}/{:2.2%} - test loss/acc: {:3.2f}/{:2.2%}'
            logging.info(template.format(step, 
                train_loss.result(), train_accuracy.result(), 
                test_loss.result(), test_accuracy.result()))
            
            # Reset the metrics for the next epoch
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()
        
        if step % FLAGS.save_freq == 0 and step != 0:
            manager.save()  
        
        ckpt.step.assign_add(1)
        if step == FLAGS.final_step + 1: break
    
    # Save model
    for name, model in framework.models.items():
        model.save_weights(os.path.join(
            saved_model_dir, '{}_{:06d}.h5'.format(name, step)))


if __name__ == '__main__':

    FLAGS = flags.FLAGS

    flags.DEFINE_string('output_dir', os.path.join('..', 'outputs'), "")
    flags.DEFINE_string('experiment_name', 'test', "")

    flags.DEFINE_enum('dataset', 'mnist', ['mnist'], "")
    flags.DEFINE_enum('model', 'mlp', ['mlp', 'cnn'], "")
    flags.DEFINE_integer('width_multiplier', 64, "")

    flags.DEFINE_integer('initial_step', 0, "")
    flags.DEFINE_integer('final_step', 5000, "")
    flags.DEFINE_integer('save_freq', 1000, "")
    flags.DEFINE_integer('eval_freq', 1000, "")

    flags.DEFINE_bool('restore', False, "")
    flags.DEFINE_integer('batch_size', 128, "")
    flags.DEFINE_float('learning_rate', 0.001, "")

    app.run(main)