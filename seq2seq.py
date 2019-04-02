#!/usr/bin/env python
# coding: utf-8

# # Date Converter时间转化器
# 
# We will be translating from one date format to another. In order to do this we need to connect two set of LSTMs (RNNs). The diagram looks as follows: Each set respectively sharing weights (i.e. each of the 4 green cells have the same weights and similarly with the blue cells). The first is a many to one LSTM, which summarises the question at the last hidden layer (and cell memory).
# 
# The second set (blue) is a Many to Many LSTM which has different weights to the first set of LSTMs. The input is simply the answer sentence while the output is the same sentence shifted by one. Ofcourse during testing time there are no inputs for the `answer` and is only used during training.
# ![seq2seq_diagram](https://i.stack.imgur.com/YjlBt.png) 
# 
# **20th January 2017 => 20th January 2009**
# ![troll](../images/troll_face.png)
# 
# ## References:
# 1. Plotting Tensorflow graph: https://stackoverflow.com/questions/38189119/simple-way-to-visualize-a-tensorflow-graph-in-jupyter/38192374#38192374
# 2. The generation process was taken from: https://github.com/datalogue/keras-attention/blob/master/data/generate.py
# 3. 2014 paper with 2000+ citations: https://arxiv.org/pdf/1409.3215.pdf


import numpy as np
import random
import time
from faker import Faker
import babel
from babel.dates import format_date
import tensorflow as tf
from sklearn.model_selection import train_test_split

# In[ ]:
fake = Faker()
fake.seed(42)
random.seed(42)

FORMATS = ['short',
           'medium',
           'long',
           'full',
           'd MMM YYY',
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY',
           ]

# change this if you want it to work with only a single language
LOCALES = babel.localedata.locale_identifiers()
LOCALES = [lang for lang in LOCALES if 'en' in str(lang)]


# In[ ]:


def create_date():
    """
        Creates some fake dates 
        :returns: tuple containing 
                  1. human formatted string
                  2. machine formatted string
                  3. date object.
    """
    dt = fake.date_object()

    # wrapping this in a try catch because
    # the locale 'vo' and format 'full' will fail
    try:
        human = format_date(dt,
                            format=random.choice(FORMATS),
                            locale=random.choice(LOCALES))

        case_change = random.randint(0, 3)  # 1/2 chance of case change
        if case_change == 1:
            human = human.upper()
        elif case_change == 2:
            human = human.lower()

        machine = dt.isoformat()
    except AttributeError as e:
        return None, None, None

    return human, machine  # , dt


data = [create_date() for _ in range(10000 * 10)]

# 生成数据集
print(data)

# In[ ]:


x = [x for x, y in data]
y = [y for x, y in data]

u_characters = set(' '.join(x))
char2numX = dict(zip(u_characters, range(len(u_characters))))
# 将X的每个字符与num对应

u_characters = set(' '.join(y))
char2numY = dict(zip(u_characters, range(len(u_characters))))
# 将y的每个字符与num对应

# Pad all sequences that are shorter than the max length of the sequence

# In[ ]:
char2numX['<PAD>'] = len(char2numX)
# 在X的字符集中加入表示空字符的pad
num2charX = dict(zip(char2numX.values(), char2numX.keys()))
max_len = max([len(date) for date in x])

x = [[char2numX['<PAD>']] * (max_len - len(date)) + [char2numX[x_] for x_ in date] for date in x]
print(''.join([num2charX[x_] for x_ in x[4]]))
x = np.array(x)

# In[9]:
char2numY['<GO>'] = len(char2numY)
# 将起始字符<GO>加入到Y的字符集中
num2charY = dict(zip(char2numY.values(), char2numY.keys()))

y = [[char2numY['<GO>']] + [char2numY[y_] for y_ in date] for date in y]
# 将y转为index list，在头部加上<GO>
print(''.join([num2charY[y_] for y_ in y[4]]))
y = np.array(y)

# In[10]:


x_seq_length = len(x[0])
# incoder的time_step，为x中最长样本的长度
y_seq_length = len(y[0]) - 1


# decoder因为y加上了<GO>所以这里time_step的长度需要减1，


# In[11]:


def batch_data(x, y, batch_size):
    shuffle = np.random.permutation(len(x))
    start = 0
    #     from IPython.core.debugger import Tracer; Tracer()()
    x = x[shuffle]
    y = y[shuffle]
    while start + batch_size <= len(x):
        yield x[start:start + batch_size], y[start:start + batch_size]
        start += batch_size


# In[12]:


epochs = 2
batch_size = 128
nodes = 32  # lstm隐含层节点数量
embed_size = 10

tf.reset_default_graph()
sess = tf.InteractiveSession()

# Tensor where we will feed the data into graph
inputs = tf.placeholder(tf.int32, shape=(None, x_seq_length), name='inputs')
outputs = tf.placeholder(tf.int32, (None, None), 'output')  # decoder的input
targets = tf.placeholder(tf.int32, (None, None), 'targets')  # decoder的output

# 初始化input和output层的Embedding
input_embedding = tf.Variable(tf.random_uniform((len(char2numX), embed_size), -1.0, 1.0), name='enc_embedding')
output_embedding = tf.Variable(tf.random_uniform((len(char2numY), embed_size), -1.0, 1.0), name='dec_embedding')
date_input_embed = tf.nn.embedding_lookup(input_embedding, inputs)
date_output_embed = tf.nn.embedding_lookup(output_embedding, outputs)

with tf.variable_scope("encoding") as encoding_scope:
    lstm_enc = tf.contrib.rnn.BasicLSTMCell(nodes)
    enc_outputs, last_state = tf.nn.dynamic_rnn(lstm_enc, inputs=date_input_embed, dtype=tf.float32)

with tf.variable_scope("decoding") as decoding_scope:
    lstm_dec = tf.contrib.rnn.BasicLSTMCell(nodes)
    dec_outputs, _ = tf.nn.dynamic_rnn(lstm_dec, inputs=date_output_embed, initial_state=last_state)
# connect outputs to
logits = tf.contrib.layers.fully_connected(dec_outputs, num_outputs=len(char2numY), activation_fn=None)
with tf.name_scope("optimization"):
    # Loss function
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, y_seq_length]))
    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

# In[13]:
dec_outputs.get_shape().as_list()
last_state[0].get_shape().as_list()
inputs.get_shape().as_list()
date_input_embed.get_shape().as_list()
# Train the graph above:

# In[18]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# In[19]:

sess.run(tf.global_variables_initializer())
epochs = 20
for epoch_i in range(epochs):
    start_time = time.time()
    for batch_i, (x_batch, y_batch) in enumerate(batch_data(X_train, y_train, batch_size)):
        _, batch_loss, batch_logits = sess.run([optimizer, loss, logits],
                                               feed_dict={inputs: x_batch,
                                                          outputs: y_batch[:, :-1],
                                                          targets: y_batch[:, 1:]})
    accuracy = np.mean(batch_logits.argmax(axis=-1) == y_batch[:, 1:])
    print('Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss,
                                                                                          accuracy,
                                                                                          time.time() - start_time))

# Translate on test set

# In[20]:


x_batch, y_batch = next(batch_data(X_test, y_test, batch_size))

dec_input = np.zeros((len(x_batch), 1)) + char2numY['<GO>']
for i in range(y_seq_length):
    batch_logits = sess.run(logits,
                            feed_dict={inputs: x_batch,
                                       outputs: dec_input})
    prediction = batch_logits[:, -1].argmax(axis=-1)
    dec_input = np.hstack([dec_input, prediction[:, None]])
# test的时候因为没有正确结果，所以取每一步的预测概率最大的output作为下一个时刻decoder的输入
print('Accuracy on test set is: {:>6.3f}'.format(np.mean(dec_input == y_batch)))


# In[21]:
# Let's randomly take two from this test set and see what it spits out:
num_preds = 2
source_chars = [[num2charX[l] for l in sent if num2charX[l] != "<PAD>"] for sent in x_batch[:num_preds]]
dest_chars = [[num2charY[l] for l in sent] for sent in dec_input[:num_preds, 1:]]

for date_in, date_out in zip(source_chars, dest_chars):
    print(''.join(date_in) + ' => ' + ''.join(date_out))
