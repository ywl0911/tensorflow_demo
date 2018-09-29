# coding:utf8
import tensorflow as tf

# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# 1.placeholder
v1 = tf.placeholder(tf.float32, shape=[2, 3, 4])
print(v1.name)
v1 = tf.placeholder(tf.float32, shape=[2, 3, 4], name='ph')
print(v1.name)
v1 = tf.placeholder(tf.float32, shape=[2, 3, 4], name='ph')
print(v1.name)
print(type(v1))
print(v1)

# 2. tf.Variable()
v2 = tf.Variable([1, 2], dtype=tf.float32)
print(v2.name)
v2 = tf.Variable([1, 2], dtype=tf.float32, name='V')
print(v2.name)
v2 = tf.Variable([1, 2], dtype=tf.float32, name='V')
print(v2.name)
print(type(v2))
print(v2)

# 3.tf.get_variable() 创建变量的时候必须要提供 name
v3 = tf.get_variable(name='gv', shape=[])
print(v3.name)
v4 = tf.get_variable(name='gv1', shape=[2, 1])
print(v4.name)

with tf.variable_scope("foo"):
    v = tf.get_variable("v", [2, 3], dtype=tf.float32)
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [2, 3], dtype=tf.float16)
assert v1 == v


def get_get_variable(x):
    return tf.get_variable('name1', initializer=tf.Variable(tf.random_normal([1], stddev=2))) + x


def get_variable(x):
    return tf.Variable(tf.random_normal([1], stddev=2), name='name') + x


with tf.variable_scope('bb') as scope:
    a = get_get_variable(1)
    scope.reuse_variables()
    b = get_get_variable(2)

with tf.variable_scope('aaa') as scope:
    a1 = get_variable(1)
    # scope.reuse_variables()
    b1 = get_variable(2)
    c1 = get_variable(2)
    d1 = get_variable(2)

aa = tf.random_uniform_initializer(1, 2)

aa = tf.Variable(tf.random_normal([1], stddev=2))

sess = tf.Session(config=config)
sess.run(tf.initialize_all_variables())
sess.run(aa)
