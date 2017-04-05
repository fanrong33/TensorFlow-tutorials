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
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise  = np.random.normal(0, 0.05, x_data.shape) # 加一点noise,这样看起来会更像真实情况
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


# plot the real data
fig = plt.figure()

ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data) # 散点图

# 这种方式结束会自动关闭，注释掉
# plt.ion() # 注意：plt.ion()用于连续显示。
# plt.show()


for step in range(1000):
    sess.run(train, feed_dict={xs: x_data, ys: y_data})
    if step % 20 == 0:
        # 可视化结果和进展
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        hypothesis_value = sess.run(hypothesis, feed_dict={xs: x_data})
        # plot the hypothesis
        lines = ax.plot(x_data, hypothesis_value, 'r-', lw=5) # 这些参数命名有点不好理解- -|||
        plt.pause(0.5)


plt.show()

