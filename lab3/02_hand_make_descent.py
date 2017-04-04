# encoding: utf-8
# 手工制做梯度下降函数
import tensorflow as tf
tf.set_random_seed(777) # for reproducibility

# 模拟生成X和Y数据，一阶张量: 向量 (vector)
x_data = [1, 2, 3]
y_data = [1, 2, 3]


# 通过 hypothesis = W * x_data 去试图查找 W 权重和 b 偏值，也就是我们要求的值
# 我们知道 W 应该为 1
# 但是这里我们让 Tensofrlow 来指出
W = tf.Variable(tf.random_uniform([1], -10.0, 10.0), name='Weight') # 我们要求的变量

X = tf.placeholder(tf.float32) # 参数X
Y = tf.placeholder(tf.float32) # 参数Y

# 假设，线性模型
hypothesis = W * X


# 成本函数 最小化方差
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
learning_rate = 0.1
descent = W - tf.multiply(learning_rate, tf.reduce_mean(tf.multiply((tf.multiply(W, X) - Y), X)))
train = W.assign(descent)


# 启动Graph
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


# 拟合平面
for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))


print(sess.run(hypothesis, feed_dict={X: 5}))
print(sess.run(hypothesis, feed_dict={X: 2.5}))
