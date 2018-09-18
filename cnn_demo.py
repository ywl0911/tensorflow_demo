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
    # initial = tf.constant(0.1, shape=shape)
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


with tf.name_scope("conv1"):
    W_conv1_fliter = weight_variable([5, 5, 1, 32])
    # 5,5为卷积核patch的尺寸，1为输入的channel数量，32为输出的channel数量
    # [filter_height, filter_width, in_channels, out_channels]这样的shape，
    # [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数] # 第三维为in_channels，就是参数x_image的第四维  第四维为卷积核的个数
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28])
    # [batch, in_height, in_width, in_channels]最后一维为通道数量，可以理解为RGB，黑白图只有一个通道
    x_image_input = tf.expand_dims(x_image, -1)  # -1参数表示第几维，-1表示扩展最后一维
    # 给x_image加一维，从[-1, 28, 28]变为[-1, 28, 28,1]，最后一维为通道数量，等价于x_image_input = tf.reshape(x, [-1, 28, 28,1])

    h_conv1_z = tf.nn.conv2d(x_image_input, W_conv1_fliter, strides=[1, 1, 1, 1], padding='SAME')
    # # 第三个参数strides：卷积时在图像X每一维的步长，这是一个一维的向量，长度4
    # string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。”SAME”是考虑边界，不足的时候用0去填充周围，”VALID”则不考虑

    h_conv1_a = tf.nn.sigmoid(h_conv1_z + b_conv1)

    h_pool1 = tf.nn.max_pool(h_conv1_z, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，维度为[batch, height, width, channels]
    # 第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    # 第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride, stride, 1]
    # 第四个参数padding：和卷积类似，可以取    # 'VALID'    # 或者    # 'SAME'
    # 返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]  这种形式

with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, [1, 1, 1, 1], 'SAME') + b_conv2)

    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('flat'):
    # cnn之后的全连接层
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    # h_pool2的维度为[batchsize，每张图片总共的像素点的个数]，经过两次卷积和两次pooling后，图片的变成了7×7，图片的通道变成了64
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    #  经过全连接层后h_fc1的维度为 [batchsize，1024]

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # 虽然经过dropout层h_fc1_drop的维度依旧为为 [batchsize，1024]

with tf.name_scope('softmax'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_predict))
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

label_index_predict = tf.argmax(y_predict, 1)
label_index_true = tf.argmax(y_, 1)

correct_prediction = tf.equal(label_index_predict, label_index_true)
# true和false 通过cast进行转换得到01序列

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# true和false 通过cast进行转换得到01序列

sess.run(tf.initialize_all_variables())

# calculating test accuracy
test_batch_size = 1000
test_data = mnist.test.data[:test_batch_size].reshape((-1, 28, 28))
test_data_new = []
for value in test_data:
    temp = value
    test_data_new.append(temp.reshape([784]))
test_label = mnist.test.labels[:test_batch_size]


def get_variable_value(v):
    return (sess.run(v, feed_dict={x: test_data_new, y_: test_label, keep_prob: 0.5}))


for i in range(20000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    if i % 1 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 0.5})
        print("step %d:\n training accuracy %g" % (i, train_accuracy))

        # print "test accuracy %g" % accurac[0][0][0][0]y.eval(feed_dict={
        #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 0.5})
        print("test accuracy_new %g" % get_variable_value(accuracy))
        # print get_variable_value(y_predict)
        # print get_variable_value(label_index_predict)
        #
        # print get_variable_value(y_)
        # print get_variable_value(label_index_true)

        print('>>>>>')
        # print(
        #     sess.run([h_conv1_z[0][0][0][:], b_conv1, (h_conv1_z + b_conv1)[0][0][0][:],
        #               h_conv1_a[0][10][0][:]],
        #              feed_dict={
        #                  x: test_data_new, y_: test_label, keep_prob: 0.5})
        # )
        # print(get_variable_value(tf.shape(y_predict)))

        # print(
        #     sess.run((h_conv1_temp[0][10]),
        #              feed_dict={
        #                  x: test_data_new, y_: test_label, keep_prob: 0.5})
        # )
