# coding:utf-8
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import os
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])


def weight_variable(shape):
    # initial =
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])
# 5,5为卷积核patch的尺寸，1为输入的channel数量，32为输出的channel数量
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])
# [batch, in_height, in_width, in_channels]最后一维为通道数量，可以理解为RGB，黑白图只有一个通道

# return

h_conv1 = tf.nn.sigmoid(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
# [1, 1, 1, 1]第一位核最后一位为1，第二位为向右移动步长，第三位为向下移动步长
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, [1, 1, 1, 1], 'SAME') + b_conv2)

h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

# calculating test accuracy
test_data = mnist.test.images[:2000].reshape((-1, 28, 28))
test_data_new = []
for value in test_data:
    temp = np.rot90(np.rot90(value))
    test_data_new.append(temp.reshape([784]))
test_label = mnist.test.labels[:2000]

for i in range(20000):
    batch = mnist.train.next_batch(500)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 0.5})
        print "step %d:\n training accuracy %g" % (i, train_accuracy)

        # print "test accuracy %g" % accuracy.eval(feed_dict={
        #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 0.5})
        print "test accuracy_new %g" % accuracy.eval(feed_dict={
            x: test_data_new, y_: test_label, keep_prob: 0.5})
