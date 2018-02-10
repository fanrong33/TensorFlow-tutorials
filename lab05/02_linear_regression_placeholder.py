# encoding: utf-8
import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777) # for reproducibility

# 模拟生成X和Y数据, 一阶张量: 向量(vector)
x_data = [1, 2, 3]
y_data = [1, 2, 3]


# 通过 hypothesis = W * x +b 去试图查找 W 权重和 b 偏值，也就是我们要求的值
# 我们知道 W 应该为 1 ，b 应该为 0
# 但是这里我们让 Tensofrlow 来指出
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))


x  = tf.placeholder(tf.float32, name='x-input')
y_ = tf.placeholder(tf.float32, name='y-input')

# 假设，线性模型
hypothesis = W * x + b

# 成本函数 最小化方差
cost = tf.reduce_mean(tf.square(hypothesis - y_))

# Minimize
learning_rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

# 启动Graph
sess = tf.Session()
# 初始化变量
init = tf.global_variables_initializer()
sess.run(init)


# 拟合平面
for step in range(2000):
    sess.run(train, feed_dict={x: x_data, y_: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={x: x_data, y_: y_data}), sess.run(W), sess.run(b))

# 得到最佳拟合结果 W: [1.00], b: [0.00]


# 使用得到的结果进行预测
print(sess.run(hypothesis, feed_dict={x: 5}))
print(sess.run(hypothesis, feed_dict={x: 2.5}))
