import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
import os
import numpy as np

os.chdir('/home/ywl/Documents/python/tensorflow_demo')
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

train_rate = 0.01
train_step = 12
batch_size = 1280
display_step = 1

frame_size = 28
time_step = 28
hidden_num = 100  # rnn层神经元节点数量
n_classes = 10

# 定义输入,输出
x_input = tf.placeholder(tf.float32, (None, None, 28), 'input_x')
# x_input的size为【batch_size*time_step*feature_dimension】
y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name="expected_y")

# 先把输入转换为dynamic_rnn接受的形状：batch_size,sequence_length,feature_dimension
# rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_num)
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_num, forget_bias=1.0, state_is_tuple=True)
# num_unit为lstm层节点数量

outputs, states = tf.nn.dynamic_rnn(rnn_cell, x_input, dtype=tf.float32)
# 有一个参数sequence_length=x_input.get_shape()[1].value,可以不传
# 此时output就是一个[batch_size,sequence_length,hidden_num]形状的tensor，即为每一个time_step的hidden state,time_step可以不固定，长度可以变化


# states：states表示最终的状态，也就是序列中最后一个cell输出的状态。一般情况下states的形状为 [batch_size, cell.output_size ]，
# 但当输入的cell为BasicLSTMCell时，state的形状为[2，batch_size, cell.output_size ]，
# 其中2也对应着LSTM中的cell state和hidden state。
h_state = states[1]
c_state = states[0]
# h_state = output[:, -1, :]

print('》》》', outputs.shape)
print('》》》', states.h.shape)
print('》》》', states.c.shape)

# return h_state
# 我们取出最后每一个序列的最后一个分量的输出output[:,-1,:],它的形状为[batch_size,rnn_cell.output_size]也就是:[batch_size,hidden_num]所以它可以和weights相乘。这就是2.5中weights的形状初始化为[hidden_num,n_classes]的原因。然后再经softmax归一化。

# 定义权值
weights = tf.Variable(tf.truncated_normal(shape=[hidden_num, n_classes]))
bias = tf.Variable(tf.zeros(shape=[n_classes]))

out = outputs[:, -1, :]
# out = h_state 两种写法均可，等价
pred = tf.nn.softmax(tf.matmul(out, weights) + bias, 1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train = tf.train.AdamOptimizer(train_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.to_float(correct_pred))

sess = tf.Session()
sess.run(tf.initialize_all_variables())
step = 1
testx, testy = mnist.test.next_batch(batch_size)

while step < train_step:
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    # batch_x = batch_x[:, :28 * stepp]
    #    batch_x=tf.reshape(batch_x,shape=[batch_size,sequence_length,frame_size])
    _loss, __ = sess.run([cost, train],
                         feed_dict={x_input: batch_x.reshape([-1, time_step, frame_size]), y: batch_y})
    if step % display_step == 0:
        acc, loss, s, o = sess.run([accuracy, cost, states[1][1][:5], outputs[:, -1, :][1][:5]],
                                   feed_dict={x_input: testx.reshape([-1, time_step, frame_size]),
                                              y: testy,
                                              })
        print('step:', step, '\tacc:', round(acc, 4), '\tloss:', loss)
        print(s)
        print(o)
    step += 1

# stepp = 21
# batch_x = batch_x[:, :4 * stepp].reshape([-1, stepp, frame_size])
# acc, loss, s, o = sess.run([accuracy, cost, states, output],
#                            feed_dict={x_input: batch_x, y: testy})
# print(1)
