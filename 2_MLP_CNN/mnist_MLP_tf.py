from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

# Loading the MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# >>>> Displaying some images and their labels form MNIST dataset
# for i in range(1,10):
#    print ("Label: " + str(mnist.train.labels[i])) # label of i-th element of training data
#    img = mnist.train.images[i].reshape((28, 28)) # saving in 'img', the reshaped i-th element of the training dataset
#    plt.imshow(img, cmap='gray') # displaying the image
#    plt.pause(0.5)


DATA_SIZE = 784
NUM_HIDDEN_1 = 256
NUM_HIDDEN_2 = 256
NUM_CLASSES = 10

# >>>> Define input and ground-truth variables
X = tf.placeholder(tf.float32, [None, DATA_SIZE]) # input data
Y = tf.placeholder(tf.float32, [None, NUM_CLASSES]) # ground-truth

# >>>> Randomly intialize the variables
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
w_h1 = init_weights([DATA_SIZE, NUM_HIDDEN_1])
w_h2 = init_weights([NUM_HIDDEN_1, NUM_HIDDEN_2])
w_o = init_weights([NUM_HIDDEN_2, NUM_CLASSES])

# >>>> Define the network model
def model(X, w_h1, w_h2, w_o):
    h1 = tf.nn.relu(tf.matmul(X, w_h1))
    h2 = tf.nn.relu(tf.matmul(h1, w_h2))
    return tf.matmul(h2, w_o)

# > Compute the predicted Y_p for an imput vector X
Y_p = model(X, w_h1, w_h2, w_o)

# > Define the cost function and the optimizer
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_p, labels=Y))
optimization_algorithm = tf.train.GradientDescentOptimizer(0.5).minimize(cost_function)

# >>>> Launch an interactive tensorflow session
sess = tf.InteractiveSession()
# tf.global_variables_initializer().run() # deprecated!
sess.run(tf.global_variables_initializer())

# >>>> For accuracy
correct_prediction = tf.equal(tf.argmax(Y_p,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# >>>> Train the network
for epoch in range(20000):
   batch = mnist.train.next_batch(50) # every batch of 50 images
   if epoch%250 == 0:
      train_accuracy = accuracy.eval(feed_dict={X:batch[0], Y: batch[1]})
      print("epoch: %d, training accuracy: %g"%(epoch, train_accuracy))
   optimization_algorithm.run(feed_dict={X: batch[0], Y: batch[1]})

# >>>> Save the learned model
# > Add ops to save and restore all the variables.
saver = tf.train.Saver()
# > Variables to save 
tf.add_to_collection('vars', w_h1)
tf.add_to_collection('vars', w_h2)
tf.add_to_collection('vars', w_o)
# > Save the variables to disk 
save_path = saver.save(sess, "./tensorflow_model.ckpt")
print("Model saved in file: %s" % save_path)


# >>>> Restore variables saved in learned model
new_saver = tf.train.import_meta_graph('./tensorflow_model.ckpt.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
all_vars = tf.get_collection('vars')
i = 0
for v in all_vars:
    v_ = sess.run(v)
    if i == 0:
       w_h1 = v_ # restore w_h1 
    if i == 1:
       w_h2 = v_ # restore w_h2
    if i == 2:
       w_o = v_ # restore w_o 
    i = i + 1
print("Model restored correctly!")

# >>>> Test the trained model
print("\n\nTest accuracy: %g"%accuracy.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

