import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import os

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784])
y = tf.placeholder("float", shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())
y_pred = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = -tf.reduce_mean(y * tf.log(y_pred))
train_step = tf.train.GradientDescentOptimizer(0.01 * 50).minimize(cross_entropy)

for i in range(10000):
    batch = mnist.train.next_batch(2000)
    train_step.run(feed_dict={x: batch[0], y: batch[1]})

    correct_prediction = tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1)), 'float')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    a = accuracy.eval(feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
    print(a)
