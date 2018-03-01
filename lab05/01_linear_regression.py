# encoding: utf-8
# 线性回归模型
# @version: v1.0.1 build 20180301

import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

# 模拟生成X和Y数据，一阶张量: 向量 (vector)
x  = [1, 2, 3]
y_ = [1, 2, 3]


# 通过 y = W * x +b 去试图查找 W 权重和 b 偏值，也就是我们要求的值
# 我们知道 W 应该为 1 ，b 应该为 0
# 但我们还是让 TensorFlow 来帮我们找出来
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='Weight')
b = tf.Variable(tf.zeros([1]), name='bias')

# 假设，线性模型
y = W * x + b

# 回归问题最常用的损失函数为均方误差MSE（反向传播的算法）
loss = tf.reduce_mean(tf.square(y_ - y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)



# 启动Graph
with tf.Session() as sess:
    # 初始化变量
    init = tf.global_variables_initializer()
    sess.run(init)


    # 拟合平面
    for step in range(2001):
        sess.run(train)
        if step % 100 == 0:
            print(step, sess.run(loss), sess.run(W), sess.run(b))

    # 得到最佳拟合结果 W: [1.00], b: [0.00]


'''
(0, 0.25807598, array([ 0.87556314], dtype=float32), array([ 0.74662161], dtype=float32))
(100, 0.00049794879, array([ 0.97408277], dtype=float32), array([ 0.05891602], dtype=float32))
(200, 3.8342546e-06, array([ 0.99772584], dtype=float32), array([ 0.00516972], dtype=float32))
(300, 2.9523624e-08, array([ 0.99980038], dtype=float32), array([ 0.00045367], dtype=float32))
(400, 2.2644997e-10, array([ 0.99998248], dtype=float32), array([  3.98281773e-05], dtype=float32))
(500, 1.7479351e-12, array([ 0.99999851], dtype=float32), array([  3.51702943e-06], dtype=float32))
(600, 6.1580373e-14, array([ 0.99999982], dtype=float32), array([  5.28851160e-07], dtype=float32))
(700, 0.0, array([ 1.], dtype=float32), array([  5.20138101e-08], dtype=float32))
(800, 0.0, array([ 1.], dtype=float32), array([  5.20138101e-08], dtype=float32))
(900, 0.0, array([ 1.], dtype=float32), array([  5.20138101e-08], dtype=float32))
(1000, 0.0, array([ 1.], dtype=float32), array([  5.20138101e-08], dtype=float32))
(1100, 0.0, array([ 1.], dtype=float32), array([  5.20138101e-08], dtype=float32))
(1200, 0.0, array([ 1.], dtype=float32), array([  5.20138101e-08], dtype=float32))
(1300, 0.0, array([ 1.], dtype=float32), array([  5.20138101e-08], dtype=float32))
(1400, 0.0, array([ 1.], dtype=float32), array([  5.20138101e-08], dtype=float32))
(1500, 0.0, array([ 1.], dtype=float32), array([  5.20138101e-08], dtype=float32))
(1600, 0.0, array([ 1.], dtype=float32), array([  5.20138101e-08], dtype=float32))
(1700, 0.0, array([ 1.], dtype=float32), array([  5.20138101e-08], dtype=float32))
(1800, 0.0, array([ 1.], dtype=float32), array([  5.20138101e-08], dtype=float32))
(1900, 0.0, array([ 1.], dtype=float32), array([  5.20138101e-08], dtype=float32))
(2000, 0.0, array([ 1.], dtype=float32), array([  5.20138101e-08], dtype=float32))
'''