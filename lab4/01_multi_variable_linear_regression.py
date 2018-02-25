# encoding: utf-8
import tensorflow as tf
tf.set_random_seed(777) # for reproducibility

# 模拟生成X和Y数据，一阶张量: 向量 (vector)
x1_data = [1, 0, 3, 0, 5]
x2_data = [0, 2, 0, 4, 0]
y_data  = [1, 2, 3, 4, 5]


# 通过 hypothesis = W1 * x1_data + W2 * x2_data + b 去试图求解 W 权重和 b 偏值，也就是我们要求的值
# 我们知道 W1 应该为 1, W2 应该为 1, b 应该为 0
# 但我们还是让 TensorFlow 来帮我们找出来
W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # 我们要求的变量
W2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b  = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# 假设为多变量的线性模型
hypothesis = W1 * x1_data + W2 * x2_data + b


# 成本函数 最小化方差
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)


# 启动 Graph
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


# 拟合平面
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W1), sess.run(W2), sess.run(b))

# 得到最佳拟合结果 W: [1.00], b: [0.00]
