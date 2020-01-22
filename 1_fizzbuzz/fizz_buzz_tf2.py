# -*- coding: utf-8 -*-
# Fizz Buzz in Tensorflow
# see http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/
# Tensorflow 2.0 code (eager mode): herve.le-borgne@cea.fr
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

# codage binaire d'un chiffre (max NUM_DIGITS bits)
NUM_DIGITS = 10
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

# ground truth
def fizz_buzz_encode(i):
    if   i % 15 == 0: return 3
    elif i % 5  == 0: return 2
    elif i % 3  == 0: return 1
    else:             return 0

# données d'entraînement (X) et labels (Y)
trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = np.array([fizz_buzz_encode(i)          for i in range(101, 2 ** NUM_DIGITS)])

# [exercice 1.2] données de validation (tirage aléatoire du train)
NUM_VAL=100 # nombre de données de validation
p = np.random.permutation(range(len(trX)))
trX, trY = trX[p], trY[p]
valX, valY = trX[0:NUM_VAL].copy(), trY[0:NUM_VAL].copy()
trX, trY   = trX[NUM_VAL:], trY[NUM_VAL:]

trX = trX[..., tf.newaxis].astype('float32')
valX = valX[..., tf.newaxis].astype('float32')

train_ds = tf.data.Dataset.from_tensor_slices(( trX, trY ))
val_ds   = tf.data.Dataset.from_tensor_slices(( valX, valY ))

BATCH_SIZE = 32
batched_train_ds = train_ds.batch(BATCH_SIZE)
batched_val_ds   = val_ds.batch(BATCH_SIZE)

# définition du MLP à 1 couche cachée (non linearite ReLU)
NUM_HIDDEN = 100
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(10,1)))
model.add(tf.keras.layers.Dense(NUM_HIDDEN, activation='relu'))
model.add(tf.keras.layers.Dense(4, activation='softmax')) # activation='sigmoid'

# loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer   = tf.keras.optimizers.SGD(learning_rate=0.05)

# metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

@tf.function
def train_step(samples, labels):
  with tf.GradientTape() as tape:
    predictions = model(samples)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(samples, labels):
  predictions = model(samples)
  t_loss = loss_object(labels, predictions)
  val_loss(t_loss)
  val_accuracy(labels, predictions)

EPOCHS = 1000
for epoch in range(EPOCHS):

  for images, labels in batched_train_ds:
    train_step(images, labels)

  for val_images, val_labels in batched_val_ds:
    test_step(val_images, val_labels)

  template = 'Epoch {}, Loss: {:1.4}, Accuracy: {:2.2%}, Val Loss: {:1.4}, Val Accuracy: {:2.2%}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result(),
                        val_loss.result(),
                        val_accuracy.result()))

  # Reset the metrics for the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  val_loss.reset_states()
  val_accuracy.reset_states()

# affichage attendu
def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

# Affichage valeurs de test
raw_data_test = np.arange(1, 101) # valeurs de test
teX = np.transpose(binary_encode(raw_data_test, NUM_DIGITS))
teY = model(teX)
output = np.vectorize(fizz_buzz)(raw_data_test, tf.math.argmax(teY,1))
print('======================')
print(output)

# [exercice 1.1] performances en test
gtY = np.array([fizz_buzz_encode(i) for i in raw_data_test])
print("test perf: ", 100*np.mean(gtY == tf.math.argmax(teY,1)))

