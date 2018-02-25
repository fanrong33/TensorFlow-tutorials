# encoding:utf-8
# 逻辑回归用于解决分类问题

import tensorflow as tf
import numpy as np

# 加载数据集
xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]


W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

x  = tf.placeholder(tf.float32, name='x-input')
y_ = tf.placeholder(tf.float32, name='y-input')


h = tf.matmul(W, x)
hypothesis = tf.div(1., 1. + tf.exp(-h))

# loss最优化目标函数
cost = -tf.reduce_mean(y_ * tf.log(hypothesis) + (1 - y_) * tf.log(1- hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(2000):
        sess.run(train, feed_dict={x: x_data, y_: y_data})

        if step % 20 == 0:
            print(step, sess.run(cost, feed_dict={x: x_data, y_: y_data}), sess.run(W))
