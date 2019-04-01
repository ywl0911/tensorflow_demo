#!/usr/bin/env python
# coding: utf-8

# # Date Converter
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
import json
import os
import time

from faker import Faker
import babel
from babel.dates import format_date

import tensorflow as tf

import tensorflow.contrib.legacy_seq2seq as seq2seq
# from utilities import show_graph

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


data = [create_date() for _ in range(50000)]

# See below what we are trying to do in this lesson. We are taking dates of various formats and converting them into a standard date format:

# In[ ]:


print(data)

# In[ ]:


x = [x for x, y in data]
y = [y for x, y in data]

u_characters = set(' '.join(x))
char2numX = dict(zip(u_characters, range(len(u_characters))))

u_characters = set(' '.join(y))
char2numY = dict(zip(u_characters, range(len(u_characters))))

# Pad all sequences that are shorter than the max length of the sequence

# In[ ]:


char2numX['<PAD>'] = len(char2numX)
num2charX = dict(zip(char2numX.values(), char2numX.keys()))
max_len = max([len(date) for date in x])

x = [[char2numX['<PAD>']] * (max_len - len(date)) + [char2numX[x_] for x_ in date] for date in x]
print(''.join([num2charX[x_] for x_ in x[4]]))
x = np.array(x)

# In[9]:


char2numY['<GO>'] = len(char2numY)
num2charY = dict(zip(char2numY.values(), char2numY.keys()))

y = [[char2numY['<GO>']] + [char2numY[y_] for y_ in date] for date in y]
print(''.join([num2charY[y_] for y_ in y[4]]))
y = np.array(y)

# In[10]:


x_seq_length = len(x[0])
y_seq_length = len(y[0]) - 1


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
nodes = 32
embed_size = 10

tf.reset_default_graph()
sess = tf.InteractiveSession()

# Tensor where we will feed the data into graph
inputs = tf.placeholder(tf.int32, shape=(None, x_seq_length), name='inputs')
outputs = tf.placeholder(tf.int32, (None, None), 'output')
targets = tf.placeholder(tf.int32, (None, None), 'targets')

# Embedding layers
input_embedding = tf.Variable(tf.random_uniform((len(char2numX), embed_size), -1.0, 1.0), name='enc_embedding')
output_embedding = tf.Variable(tf.random_uniform((len(char2numY), embed_size), -1.0, 1.0), name='dec_embedding')
date_input_embed = tf.nn.embedding_lookup(input_embedding, inputs)
date_output_embed = tf.nn.embedding_lookup(output_embedding, outputs)

with tf.variable_scope("encoding") as encoding_scope:
    lstm_enc = tf.contrib.rnn.BasicLSTMCell(nodes)
    _, last_state = tf.nn.dynamic_rnn(lstm_enc, inputs=date_input_embed, dtype=tf.float32)

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

# In[14]:


last_state[0].get_shape().as_list()

# In[15]:


inputs.get_shape().as_list()

# In[16]:


date_input_embed.get_shape().as_list()

# Train the graph above:

# In[18]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# In[19]:


sess.run(tf.global_variables_initializer())
epochs = 10
for epoch_i in range(epochs):
    start_time = time.time()
    for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_train, y_train, batch_size)):
        _, batch_loss, batch_logits = sess.run([optimizer, loss, logits],
                                               feed_dict={inputs: source_batch,
                                                          outputs: target_batch[:, :-1],
                                                          targets: target_batch[:, 1:]})
    accuracy = np.mean(batch_logits.argmax(axis=-1) == target_batch[:, 1:])
    print('Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss,
                                                                                          accuracy,
                                                                                          time.time() - start_time))

# Translate on test set

# In[20]:


source_batch, target_batch = next(batch_data(X_test, y_test, batch_size))

dec_input = np.zeros((len(source_batch), 1)) + char2numY['<GO>']
for i in range(y_seq_length):
    batch_logits = sess.run(logits,
                            feed_dict={inputs: source_batch,
                                       outputs: dec_input})
    prediction = batch_logits[:, -1].argmax(axis=-1)
    # prediction = batch_logits.reshape(128,-1).argmax(axis=-1)
    dec_input = np.hstack([dec_input, prediction[:, None]])

print('Accuracy on test set is: {:>6.3f}'.format(np.mean(dec_input == target_batch)))

# Let's randomly take two from this test set and see what it spits out:

# In[21]:


num_preds = 2
source_chars = [[num2charX[l] for l in sent if num2charX[l] != "<PAD>"] for sent in source_batch[:num_preds]]
dest_chars = [[num2charY[l] for l in sent] for sent in dec_input[:num_preds, 1:]]

for date_in, date_out in zip(source_chars, dest_chars):
    print(''.join(date_in) + ' => ' + ''.join(date_out))

# In[22]:


source_batch[0]

# In[ ]:
