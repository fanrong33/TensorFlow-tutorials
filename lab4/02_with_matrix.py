# encoding: utf-8
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

# 模拟生成 x_data 和 y_data 数据, 二阶张量 (matrix)
x_data = [[1., 0., 3., 0., 5.],
          [0., 2., 0., 4., 0.]]

y_data = [1, 2, 3, 4, 5]

# 我们要就的变量
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='Weight')
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='biase')

# 假设，线性模型
hypothesis = tf.matmul(W, x_data) + b

# 成本函数 最小化方差
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train = optimizer.minimize(cost)


# 启动Graph
sess = tf.Session()
# 初始化变量
init = tf.global_variables_initializer()
sess.run(init)

# 拟合平面
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
