# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

# pour affichage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf

# chargement des données MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# affichage des 10 premières images du "trains set"
for i in range(1,10):
    img = mnist.train.images[i].reshape(28,28)
    plt.imshow(img,cmap="gray")
    plt.title( 'Label {}'.format(str(mnist.train.labels[i] )) )
    plt.pause(0.5) # plt.show() pour attendre fermeture fenêtre
plt.close()

#------------------------------------------------------
# définition du modèle de regression linéaire
#------------------------------------------------------
D_in  = mnist.train.images[0].size
D_out = mnist.train.labels[0].size

x = tf.placeholder(tf.float32, [None, D_in])   # entrée
W = tf.Variable(tf.zeros([D_in, D_out]))       # poids à apprendre
b = tf.Variable(tf.zeros([D_out]))             # biais
y = tf.matmul(x, W) + b                        # prédiction
y_ = tf.placeholder(tf.float32, [None, D_out]) # vérité terrain

# définition de la fonction de coût
# et de la méthode d'apprentissage (descente de gradient)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# session interactive
sess = tf.InteractiveSession()
# tf.initialize_all_variables().run()
tf.global_variables_initializer().run()

# apprentissage en 1000 epochs
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# test du modèle
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

