# encoding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weight    = tf.Variable(tf.random_normal([in_size, out_size]))
    biase     = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weight) + biase
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 模拟一些真实数据
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise  = np.random.normal(0, 0.05, x_data.shape).astype(np.float32) # 加一点noise,这样看起来会更像真实情况
y_data = np.square(x_data) - 0.5 + noise



# 
xs = tf.placeholder(tf.float32, [None, 1]) # 函数的输入参数
ys = tf.placeholder(tf.float32, [None, 1])
# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
hypothesis = add_layer(l1, 10, 1, activation_function=None)


cost      = tf.reduce_mean(tf.reduce_sum(tf.square(ys-hypothesis), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train     = optimizer.minimize(cost)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for step in range(1000):
    sess.run(train, feed_dict={xs: x_data, ys: y_data})
    if step % 50 == 0:
        print(sess.run(cost, feed_dict={xs: x_data, ys: y_data}))



