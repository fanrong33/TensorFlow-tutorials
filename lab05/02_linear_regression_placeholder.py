# encoding: utf-8
# 线性回归
# @version: v1.0.3 build 20180225
# 参考 https://github.com/aymericdamien/TensorFlow-Examples/

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
tf.set_random_seed(777) # for reproducibility


# 定义参数
learning_rate   = 0.01
training_epochs = 2000
display_step    = 100

# 训练数据 模拟生成X和Y数据, 一阶张量: 向量(vector)
train_x = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
# 输入变量定义约定 train_xs test_xs validation_xs batch_xs
# 或者不用加s，也可以，更简单
n_samples = train_x.shape[0] # 样本数
# print(n_samples)
''' 17 '''


# 通过 y = W * x +b 去试图查找 W 权重和 b 偏值，也就是我们要求的值
# 我们知道 W 应该为 1 ，b 应该为 0
# 但是这里我们让 Tensofrlow 来指出
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='Weight')
b = tf.Variable(tf.zeros([1]), name='bias')


# y_ 就像输入光标一样用下划线来代表，👍
x  = tf.placeholder(tf.float32, name='x-input')
y_ = tf.placeholder(tf.float32, name='y-input')


# 构建一个线性模型
# y = W * x + b
y = tf.add(tf.multiply(x, W), b)


# Mean squared error
# cost = tf.reduce_sum(tf.pow(y-y_, 2))/(2*n_samples)
# 成本函数 最小化方差
cost = tf.reduce_mean(tf.square(y - y_))


# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)


# 启动Graph
with tf.Session() as sess:

    # 初始化变量
    init = tf.global_variables_initializer()
    sess.run(init)

    # 拟合平面
    for epoch in range(training_epochs+1):
        sess.run(train, feed_dict={x: train_x, y_: train_y})

        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={x: train_x, y_: train_y})
            print("Epoch: %04d cost=%.9f W=%s b=%s " % (epoch, loss, sess.run(W), sess.run(b)))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={x: train_x, y_: train_y})
    print("Training cost=%s, W=%s b=%s" % (training_cost, sess.run(W), sess.run(b)))
    # 得到最佳拟合结果 W: [1.00], b: [0.00]
    

    # 拟合绘图，线性模型
    plt.plot(train_x, train_y, 'ro', label='Original data')
    plt.plot(train_x, sess.run(W) * train_x + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()


    # 测试数据
    test_x = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
    loss = sess.run(cost, feed_dict={x: test_x, y_: test_y})
    print("Testing cost=%s" % loss)

    pred_y = sess.run(y, feed_dict={x: test_x})

    plt.plot(test_x, test_y, 'bo', label='Testing data')
    # plt.plot(test_x, sess.run(W) * test_x + sess.run(b), label='Fitted line')
    plt.plot(test_x, pred_y, label='Fitted line')
    plt.legend()
    plt.show()


    # 使用得到的结果进行预测
    print(sess.run(y, feed_dict={x: np.asarray([5])}))
    print(sess.run(y, feed_dict={x: np.asarray([2.5])}))



'''
Epoch: 0000 cost=0.830925822 W=[ 0.22327769] b=[ 0.15528001]
Epoch: 0100 cost=0.183793530 W=[ 0.32099679] b=[ 0.30705699]
Epoch: 0200 cost=0.172268465 W=[ 0.30603015] b=[ 0.41316369]
Epoch: 0300 cost=0.165180445 W=[ 0.29429296] b=[ 0.49637499]
Epoch: 0400 cost=0.160821259 W=[ 0.28508842] b=[ 0.5616312]
Epoch: 0500 cost=0.158140317 W=[ 0.27786994] b=[ 0.61280686]
Epoch: 0600 cost=0.156491548 W=[ 0.27220902] b=[ 0.65294015]
Epoch: 0700 cost=0.155477524 W=[ 0.26776963] b=[ 0.68441343]
Epoch: 0800 cost=0.154853895 W=[ 0.26428819] b=[ 0.70909542]
Epoch: 0900 cost=0.154470339 W=[ 0.26155794] b=[ 0.72845173]
Epoch: 1000 cost=0.154234484 W=[ 0.25941679] b=[ 0.74363148]
Epoch: 1100 cost=0.154089406 W=[ 0.25773764] b=[ 0.75553578]
Epoch: 1200 cost=0.154000223 W=[ 0.25642085] b=[ 0.76487118]
Epoch: 1300 cost=0.153945327 W=[ 0.25538817] b=[ 0.77219254]
Epoch: 1400 cost=0.153911591 W=[ 0.25457832] b=[ 0.77793384]
Epoch: 1500 cost=0.153890848 W=[ 0.2539432] b=[ 0.78243673]
Epoch: 1600 cost=0.153878078 W=[ 0.25344512] b=[ 0.78596777]
Epoch: 1700 cost=0.153870210 W=[ 0.25305453] b=[ 0.78873688]
Epoch: 1800 cost=0.153865367 W=[ 0.25274822] b=[ 0.79090852]
Epoch: 1900 cost=0.153862417 W=[ 0.25250801] b=[ 0.7926116]
Epoch: 2000 cost=0.153860584 W=[ 0.25231957] b=[ 0.79394746]
Optimization Finished!
Training cost=0.153861, W=[ 0.25231957] b=[ 0.79394746]
Testing cost=0.1563
[ 2.05554533]
[ 1.42474639]
'''
