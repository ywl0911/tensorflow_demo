#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 18-9-29 下午3:25 
# @Author : ywl
# @File : dnn_demo.py 
# @Software: PyCharm


import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import os

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder("float", [None, 10])

W1 = tf.Variable(tf.truncated_normal([784, 500]))
b1 = tf.Variable(tf.truncated_normal([500]))
y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([500, 300]))
b2 = tf.Variable(tf.truncated_normal([300]))
y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)

W3 = tf.Variable(tf.truncated_normal([300, 400]))
b3 = tf.Variable(tf.truncated_normal([400]))
y3 = tf.nn.sigmoid(tf.matmul(y2, W3) + b3)

W4 = tf.Variable(tf.truncated_normal([400, 150]))
b4 = tf.Variable(tf.truncated_normal([150]))
y4 = tf.nn.sigmoid(tf.matmul(y3, W4) + b4)

W5 = tf.Variable(tf.truncated_normal([150, 10]))
b5 = tf.Variable(tf.truncated_normal([10]))
y_pred = tf.nn.softmax(tf.matmul(y4, W5) + b5)

# weight=tf.constant(value=[0.3,0.7])


cross_entropy = -tf.reduce_mean(y * tf.log(y_pred))
#
# weight = tf.constant([0.1,0.1,0.1,0.1,0.1,0.1,0.05,0.05,0.05,0.25])
# cross_entropy = -tf.reduce_mean(y * tf.log(y_pred) * weight)

train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(100000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    a = sess.run(cross_entropy, feed_dict={x: batch_xs, y: batch_ys})
    b = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print('loss:', a, 'accuracy:', b)
    print(cross_entropy.shape)
