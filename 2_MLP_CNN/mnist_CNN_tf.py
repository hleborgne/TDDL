#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

# >>>> Loading MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# >>>> Launching interactive TF session
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# >>>> Weight variable function
"""
Randomly initializes the weights of a variable of
shape = 'shape_var'. This function returns a tensor of
the specified shape filled with random values.
"""
def weight_variable(shape_var):
  initial = tf.truncated_normal(shape_var, stddev=0.1)
  return tf.Variable(initial)

# >>>> Bias variable function
"""
Creates a constant tensor of shape = 'shape_bias' with all
elements equal to the value 0.1.
"""
def bias_variable(shape_bias):
  initial = tf.constant(0.1, shape=shape_bias)
  return tf.Variable(initial)

# >>>> Conv2d function
"""
Computes the convolution between a filter W and an image x.
Parameters: stride=1, padding=0.
"""
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# >>>> Max-pooling function
"""
Computes the max-pooling for every patches of size 2x2 of an
input image x.
"""
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# >>>> Reshape input data vectors 
"""
Reshape a vector of size 784x1 into a matrix of size 28x28x1.
The parameter '-1' indicates that the size of the dimension at
that index of the parameter, remains the same.
"""
x_image = tf.reshape(x, [-1,28,28,1])

# >>>> Convolutional layer 1
"""
Random initialization of the weights W_conv1 (filters of conv1)
This layer will compute the convolution of 32 filters (of size 5x5)
with the input image (third dimension = 1 indicates that the input
tensor is one image, corresponding to the input grayscale images).
"""
W_conv1 = weight_variable([5, 5, 1, 32])

# > Bias of convolutional layer 1
"""
Initialize the bias of conv-layer 1 with a constant value of 0.1.
The value 32 indicates that we have 32 filters in conv1 and
thus, we will add a bias in each of these filters.
"""
b_conv1 = bias_variable([32])

# > Computing the output values of conv1 (feature maps)
"""
This will output a set of 32 feature maps of size 28x28x1.
Each feature map will be the output of the convolution of
one filter (among the 32 filters) with the input image.
"""
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# > Computing the output values of max-pool 1 (feature maps)
"""
Application of the max-pooling function on the 32 feature-maps (of size 28x28)
obtained from previous convolutional-layer.
This will output 32 feature-maps of size 14x14 (because we max-pool every 2x2 patches).
"""
h_pool1 = max_pool_2x2(h_conv1)

# >>>> Convolutional layer 2
"""
Random initialization of the weights W_conv2 (filters of conv2).
This layer will compute the convolution of 64 filters (of size 5x5)
with the input images (third dimension = 32 indicates that the input
tensor is a set of 32 images, corresponding to the feature maps of
size 14x14 obtained after max-pool1).
"""
W_conv2 = weight_variable([5, 5, 32, 64]) # declaration of the weights of conv2
b_conv2 = bias_variable([64]) # declaration of the weights of bias of conv2

# > Computing the output values of conv2 (feature maps)
"""
output: 64 feature maps of size 14x14x1
"""
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# > Computing the output values of max-pool2 (feature maps)
"""
output: 64 images of size 7x7x1
"""
h_pool2 = max_pool_2x2(h_conv2)

# >>>> Fully-connected layer 1
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# > Reshape the feature maps of max-pool2
"""
This will reshape the 64 feature maps of size 7x7x1
into a vector of size 7x7x1x64 (=3136).
"""
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# > Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# >>>> Fully-connected layer 2
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# >>>> Cost function and optimization algorithm
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
optimization_algorithm = tf.train.AdamOptimizer(1e-4).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
# >>>> Training the network
for i in range(10000):
   batch = mnist.train.next_batch(50)
   if i%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
      print("epoch: %d, training accuracy: %g"%(i, train_accuracy))
   optimization_algorithm.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# >>>> Testing the network on the Test data
print("\n\nTest accuracy: %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
# if there is not enough memory for the 10,000 images:
# ok=0
# for b in range(0,10000,500):
#     ok += tf.reduce_sum(tf.cast(correct_prediction.eval(feed_dict={x: mnist.test.images[b:b+499], y_: mnist.test.labels[b:b+499], keep_prob: 1.0}),tf.float32)).eval()
# print("test accuracy : ",ok/10000,"%")

# >>>> Displaying some feature maps

# > Normalize feature maps function 
def normalize_feat_map(feat_map, width_feat_map, height_feat_map):
   max_feat_map = np.amax(feat_map)
   min_feat_map = np.amin(feat_map)
   diff_max_min = max_feat_map - min_feat_map
   for i in range(0,width_feat_map):
      for j in range(0,height_feat_map):
         feature_map[i, j] = (feature_map[i, j] - min_feat_map) / diff_max_min
   return feature_map

layer_to_extract = h_conv1 # layer from which we want to extract the feature maps

# for the 5 first test images
for img_to_test in range(1, 5):
   extracted_feature = sess.run(layer_to_extract, feed_dict={x: mnist.test.images[img_to_test:img_to_test+1], keep_prob: 1.0}) # extract feature at layer_to_extract for test image 1

   # for the 5 first filters
   for nbr_feat_map in range(0,4):
      feature_map = extracted_feature[:,:,:,nbr_feat_map].reshape((28,28))
      normalize_feat_map(feature_map, 27, 27) # normalize the values of the feature map
      plt.imshow(feature_map, cmap='gray') # display the output feature map
      plt.show()
