import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
import os

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

# 定义RNN网络
# def RNN(x):
x = tf.reshape(x, shape=[-1, sequence_length, frame_size])
# 先把输入转换为dynamic_rnn接受的形状：batch_size,sequence_length,frame_size这样子的

# 以下两种写法可以互换
# rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_num, forget_bias=1.0, state_is_tuple=True)
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_num, forget_bias=1.0, state_is_tuple=True)
# rnn_cell.output_size为rnn隐藏层节点个数，即hidden_num

output, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
# 此时output就是一个[batch_size,sequence_length,rnn_cell.output_size]形状的tensor
# 此时state为tuple，（c_state,h_state）,其中c_state和h_state的size均为[batch_size,rnn_cell.output_size]
# c_state=state.c/state[0] # h_state=state.h/state[1]
# h_state=output[:,-1,:], 一般认为h_state为样本的representation

c_state, h_state = states

outputs = h_state
y_pred = tf.nn.softmax(tf.matmul(outputs, weights) + bias, 1)

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
    _loss, __ = sess.run([cost, train], feed_dict={x: batch_x.reshape([-1, sequence_length, frame_size]), y: batch_y})
    if step % display_step == 0:
        acc, loss, s, o = sess.run([accuracy, cost, h_state[1][:5], output[:, -1, :][1][:5]],
                                   feed_dict={x: testx.reshape([-1, sequence_length, frame_size]), y: testy})
        print('step:', step, '\tacc:', round(acc, 4), '\tloss:', loss)
        print(s)
        print(o)
    step += 1
