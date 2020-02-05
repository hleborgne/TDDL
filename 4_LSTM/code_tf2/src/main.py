import os   
import numpy as np 
import tensorflow as tf

from operator import itemgetter
from absl import flags, logging, app



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

    # ============================== Read data =================================
    # see https://www.tensorflow.org/datasets/splits
    if FLAGS.dataset == 'mnist': from datasets import mnist as dataset

    train_dataset = dataset.load(FLAGS.batch_size, split="train[:90%]")
    valid_dataset = dataset.load(FLAGS.batch_size, split='train[-10%:]')
    test_dataset = dataset.load(FLAGS.batch_size, split='test')

    # ======================== Create computation graph ========================
    # create model losses and optimizer
    if FLAGS.model == 'mlp': from models.mlp import MLP as Model
    if FLAGS.model == 'cnn': from models.cnn import CNN as Model
    if FLAGS.model == 'gru': from models.gru import GRU as Model
    if FLAGS.model == 'lstm': from models.lstm import LSTM as Model
    if FLAGS.model == 'bilstm': from models.bilstm import BiLSTM as Model

    model = Model()
    model.build(input_shape=(FLAGS.batch_size,28,28,1))
    model.summary()

    loss_func = tf.losses.CategoricalCrossentropy()
    optimizer = tf.optimizers.Adam(FLAGS.learning_rate)

    # Manage checkpoints
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(0), model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(
        ckpt, os.path.join(experiment_dir, 'checkpoints'), max_to_keep=1)
    

    # define forward pass and training step
    @tf.function
    def forward(example, training=False):
        print('tracing `forward` graph!') # print on the first call when tracing graph
        predictions = model.call(example['image'], training=training)
        loss = tf.losses.categorical_crossentropy(
                y_true=example['label'],
                y_pred=predictions)
        return loss, predictions

    @tf.function
    def train_step(example):
        print('tracing `train` graph!') # print on the first call when tracing graph
        with tf.GradientTape() as tape:
            loss, predictions = forward(example, training=True)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # ============================ train the model =============================
    # Restore checkpoint.
    if FLAGS.restore: 
        ckpt.restore(manager.latest_checkpoint)

    if 'train' in FLAGS.mode:

        # Define metrics.
        train_loss = tf.metrics.Mean(name='train_accuracy')
        train_accuracy = tf.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        valid_loss = tf.metrics.Mean(name='test_accuracy')
        valid_accuracy = tf.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        for step, example in train_dataset.enumerate():

            train_step(example)

            if step % FLAGS.eval_freq == 0:
                for example in train_dataset.take(10):
                    loss, predictions = forward(example)
                    train_accuracy(
                        tf.math.argmax(example['label'], axis=-1), predictions)
                    train_loss(loss)
                for example in test_dataset.take(10):
                    loss, predictions = forward(example)
                    valid_accuracy(
                        tf.math.argmax(example['label'], axis=-1), predictions)
                    valid_loss(loss)

                template = 'step: {:06d} - train loss/acc: {:3.2f}/{:2.2%} - valid loss/acc: {:3.2f}/{:2.2%}'
                message = template.format(step, 
                    train_loss.result(), train_accuracy.result(), 
                    valid_loss.result(), valid_accuracy.result())
                logging.info(message)
                print(message)
                
                # Reset the metrics for the next epoch.
                train_loss.reset_states()
                train_accuracy.reset_states()
                valid_loss.reset_states()
                valid_accuracy.reset_states()
            
            if step % FLAGS.save_freq == 0 and step != 0:
                manager.save()  
            
            if step == FLAGS.final_step: break
            else: ckpt.step.assign_add(1)

        # Save model at the end of training.
        model.save_weights(os.path.join(
            saved_model_dir, '{}_{:06d}.h5'.format(FLAGS.model, step)))
    
    if 'test' in FLAGS.mode:

        # Restore saved model
        model.load_weights(os.path.join(
            saved_model_dir, '{}_{:06d}.h5'.format(FLAGS.model, FLAGS.final_step)))

        # Define metrics.
        test_loss = tf.metrics.Mean(name='test_accuracy')
        test_accuracy = tf.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        for step, example in test_dataset.enumerate():
            loss, predictions = forward(example)
            test_accuracy(tf.math.argmax(example['label'], axis=-1), predictions)
            test_loss(loss)
        
        template = 'test loss/acc: {:3.2f}/{:2.2%}'
        message = template.format(test_loss.result(), test_accuracy.result())
        logging.info(message) 
        print(message)

if __name__ == '__main__':

    FLAGS = flags.FLAGS

    flags.DEFINE_string('output_dir', os.path.join('..', 'outputs'), "")
    flags.DEFINE_string('experiment_name', 'test', "")

    flags.DEFINE_enum('dataset', 'mnist', ['mnist'], "")
    flags.DEFINE_enum('model', 'mlp', ['mlp', 'cnn', 'bilstm', 'gru', 'lstm'], "")
    flags.DEFINE_enum('mode', 'train', ['train', 'test', 'train+test'], "")

    flags.DEFINE_integer('batch_size', 100, "")
    flags.DEFINE_float('learning_rate', 0.001, "")
    flags.DEFINE_integer('initial_step', 0, "")
    flags.DEFINE_integer('final_step', 5000, 'set to `-1` to train indefinitely')
    flags.DEFINE_integer('save_freq', 1000, "")
    flags.DEFINE_integer('eval_freq', 1000, "")

    flags.DEFINE_bool('restore', False, "")

    app.run(main)
