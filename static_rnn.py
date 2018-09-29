# coding:utf8
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import os
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# define constants
# unrolled through 28 time steps
time_steps = 28
# hidden LSTM units
num_units = 128
# rows of 28 pixels
n_input = 28
# learning rate for adam
learning_rate = 0.001
# mnist is meant to be classified in 10 classes(0-9).
n_classes = 10
# size of batch
batch_size = 128

# weights and biases of appropriate shape to accomplish above task
out_weights = tf.Variable(tf.random_normal([num_units, n_classes]))
out_bias = tf.Variable(tf.random_normal([n_classes]))

# defining placeholders
# input image placeholder
x = tf.placeholder("float", [None, time_steps, n_input])
# input label placeholder
y = tf.placeholder("float", [None, n_classes])

# processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
# input = tf.reshape(x, [-1,n_input])
input = tf.unstack(x, num=time_steps, axis=1)
# unstack，相对于reshape函数，但是reshape函数能够reshape成任意合法的形状
# unstack只能在固定的维度进行reshape，例如[a,b,c,d]只能reshape成a*[b,c,d]或b*[a,c,d]或c*[a,b,d]或d*[a,b,c]


# defining the network
# lstm_layer = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1)
lstm_layer = tf.nn.rnn_cell.GRUCell(num_units)

outputs, _ = tf.contrib.rnn.static_rnn(lstm_layer, input, dtype="float32")

# converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
prediction = tf.sigmoid(tf.matmul(outputs[-1], out_weights) + out_bias)

# loss_function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
# optimization
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# model evaluation
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# calculating test accuracy
test_data = mnist.test.images.reshape((-1, time_steps, n_input))
test_data_new = []
for value in test_data:
    test_data_new.append(np.rot90(np.rot90(value)))

test_label = mnist.test.labels

# initialize variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    iter = 1
    while iter < 3000:
        batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)

        batch_x = batch_x.reshape((batch_size, time_steps, n_input))

        sess.run(opt, feed_dict={x: batch_x, y: batch_y})

        if iter % 10 == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            los = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
            print("For iter ", iter)
            print("Accuracy ", acc)
            print("Loss ", los)
            print("__________________")
            print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
            print("Testing Accuracy_new:", sess.run(accuracy, feed_dict={x: test_data_new, y: test_label}))

        iter = iter + 1
