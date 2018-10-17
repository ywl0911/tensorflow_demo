import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
import os
import numpy as np

os.chdir('/home/ywl/Documents/python/tensorflow_demo')
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

train_rate = 0.001
train_step = 10000
batch_size = 1280
display_step = 10

frame_size = 28
sequence_length = 28
hidden_num = 100
n_classes = 10

# 定义输入,输出
x = tf.placeholder(dtype=tf.float32, shape=[None, sequence_length * frame_size], name="inputx")
y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name="expected_y")
# 定义权值
weights = tf.Variable(tf.truncated_normal(shape=[hidden_num, n_classes]))
bias = tf.Variable(tf.zeros(shape=[n_classes]))

x_input_time_major_false = tf.reshape(x, shape=[-1, sequence_length, frame_size])
# 此时x_input的shape为[batch_size * time_step_length * frame_size],此shape对应time_major=False

x_input_time_major_true = tf.stack(tf.unstack(x_input_time_major_false, axis=1), axis=0)
# x_input = tf.reshape(x, shape=[sequence_length, -1, frame_size])

# print(x_input.shape)
# 先把输入转换为dynamic_rnn接受的形状：batch_size,sequence_length,frame_size这样子的

# 以下两种写法可以互换
# rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_num, forget_bias=1.0, state_is_tuple=True)
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_num, forget_bias=1.0, state_is_tuple=True)# rnn_cell.output_size为rnn隐藏层节点个数，即hidden_num

output, states = tf.nn.dynamic_rnn(rnn_cell, x_input_time_major_true, dtype=tf.float32, time_major=True)
# 此时output就是一个[batch_size,sequence_length,rnn_cell.output_size]形状的tensor
# 此时state为tuple，是最后一个时刻的（c_state,h_state），其中c_state和h_state的size均为[batch_size,rnn_cell.output_size]
# c_state=state.c/state[0] # h_state=state.h/state[1]
# h_state=output[:,-1,:], 一般认为h_state为样本的representation

c_state, h_state = states

outputs = tf.stack(tf.unstack(output, num=batch_size, axis=1), axis=0)

y_pred = tf.nn.softmax(tf.matmul(outputs[:, -1, :], weights) + bias, 1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))

train = tf.train.AdamOptimizer(train_rate).minimize(cost)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1)), 'float'))

sess = tf.Session()
sess.run(tf.initialize_all_variables())
step = 1
testx, testy = mnist.test.next_batch(batch_size)
while step < train_step:
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    #    batch_x=tf.reshape(batch_x,shape=[batch_size,sequence_length,frame_size])
    _loss, __ = sess.run([cost, train], feed_dict={x: batch_x, y: batch_y})
    if step % display_step == 0:
        acc, loss, s, o = sess.run([accuracy, cost, h_state[1][:5], outputs[:, -1, :][1][:5]],
                                   feed_dict={x: testx, y: testy})
        print('step:', step, '\tacc:', round(acc, 4), '\tloss:', loss)
        print(s)
        print(o)
    step += 1
