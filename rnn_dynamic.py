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
# rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_num)
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_num, forget_bias=1.0, state_is_tuple=True)
# rnn_cell = rnn.B.static_rnn(hidden_num)

# 生成hidden_num个隐层的RNN网络,rnn_cell.output_size等于隐层个数，state_size也是等于隐层个数，但是对于LSTM单元来说这两个size又是不一样的。
# 这是一个深度RNN网络,对于每一个长度为sequence_length的序列[x1,x2,x3,...,]的每一个xi,都会在深度方向跑一遍RNN,每一个都会被这hidden_num个隐层单元处理。

output, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
# 此时output就是一个[batch_size,sequence_length,rnn_cell.output_size]形状的tensor

h_state = states[0]
# h_state = output[:, -1, :]

print('》》》', output.shape)
print('》》》', states.h.shape)
print('》》》', states.c.shape)

# return h_state
# 我们取出最后每一个序列的最后一个分量的输出output[:,-1,:],它的形状为[batch_size,rnn_cell.output_size]也就是:[batch_size,hidden_num]所以它可以和weights相乘。这就是2.5中weights的形状初始化为[hidden_num,n_classes]的原因。然后再经softmax归一化。


# outputs = RNN(x)
outputs = h_state
predy = tf.nn.softmax(tf.matmul(outputs, weights) + bias, 1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predy, labels=y))
train = tf.train.AdamOptimizer(train_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(predy, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.to_float(correct_pred))

sess = tf.Session()
sess.run(tf.initialize_all_variables())
step = 1
testx, testy = mnist.test.next_batch(batch_size)
while step < train_step:
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    #    batch_x=tf.reshape(batch_x,shape=[batch_size,sequence_length,frame_size])
    _loss, __ = sess.run([cost, train], feed_dict={x: batch_x.reshape([-1, sequence_length, frame_size]), y: batch_y})
    if step % display_step == 0:
        acc, loss, s, o = sess.run([accuracy, cost, states.h[1][:5], output[:, -1, :][1][:5]],
                                   feed_dict={x: testx.reshape([-1, sequence_length, frame_size]), y: testy})
        print('step:', step, '\tacc:', round(acc, 4), '\tloss:', loss)
        print(s)
        print(o)
    step += 1
