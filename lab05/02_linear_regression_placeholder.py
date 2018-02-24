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
display_step    = 50

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


# 通过 prediction = W * x +b 去试图查找 W 权重和 b 偏值，也就是我们要求的值
# 我们知道 W 应该为 1 ，b 应该为 0
# 但是这里我们让 Tensofrlow 来指出
# W = tf.Variable(np.random.randn(), name='Weight')
# b = tf.Variable(np.random.randn(), name='bias')
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='Weight')
b = tf.Variable(tf.zeros([1]), name='bias')


# y_ 就像输入光标一样用下划线来代表，👍
x  = tf.placeholder(tf.float32, name='x-input')
y_ = tf.placeholder(tf.float32, name='y-input')


# 构建一个线性模型
# prediction = W * x + b
prediction = tf.add(tf.multiply(x, W), b)


# Mean squared error
# cost = tf.reduce_sum(tf.pow(prediction-y_, 2))/(2*n_samples)
# 成本函数 最小化方差
cost = tf.reduce_mean(tf.square(prediction - y_))


# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)


# 启动Graph
with tf.Session() as sess:

    # 初始化变量
    init = tf.global_variables_initializer()
    sess.run(init)

    # 拟合平面
    for epoch in range(training_epochs):
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

    pred_y = sess.run(prediction, feed_dict={x: test_x})

    plt.plot(test_x, test_y, 'bo', label='Testing data')
    # plt.plot(test_x, sess.run(W) * test_x + sess.run(b), label='Fitted line')
    plt.plot(test_x, pred_y, label='Fitted line')
    plt.legend()
    plt.show()


    # 使用得到的结果进行预测
    print(sess.run(prediction, feed_dict={x: np.asarray([5])}))
    print(sess.run(prediction, feed_dict={x: np.asarray([2.5])}))



'''
Epoch: 0000 cost=0.175565556 W=0.248852 b=0.668943
Epoch: 0050 cost=0.155422091 W=0.267491 b=0.686388
Epoch: 0100 cost=0.155084535 W=0.265677 b=0.699252
Epoch: 0150 cost=0.154819801 W=0.26407 b=0.710644
Epoch: 0200 cost=0.154612198 W=0.262647 b=0.720732
Epoch: 0250 cost=0.154449388 W=0.261387 b=0.729666
Epoch: 0300 cost=0.154321745 W=0.260271 b=0.737578
Epoch: 0350 cost=0.154221594 W=0.259282 b=0.744584
Epoch: 0400 cost=0.154143050 W=0.258407 b=0.750788
Epoch: 0450 cost=0.154081479 W=0.257632 b=0.756283
Epoch: 0500 cost=0.154033184 W=0.256946 b=0.761149
Epoch: 0550 cost=0.153995350 W=0.256338 b=0.765458
Epoch: 0600 cost=0.153965607 W=0.2558 b=0.769273
Epoch: 0650 cost=0.153942347 W=0.255323 b=0.772652
Epoch: 0700 cost=0.153924078 W=0.254901 b=0.775644
Epoch: 0750 cost=0.153909743 W=0.254528 b=0.778294
Epoch: 0800 cost=0.153898522 W=0.254196 b=0.780641
Epoch: 0850 cost=0.153889701 W=0.253903 b=0.782719
Epoch: 0900 cost=0.153882772 W=0.253644 b=0.784559
Epoch: 0950 cost=0.153877378 W=0.253414 b=0.786189
Epoch: 1000 cost=0.153873131 W=0.25321 b=0.787632
Epoch: 1050 cost=0.153869808 W=0.25303 b=0.788911
Epoch: 1100 cost=0.153867170 W=0.25287 b=0.790042
Epoch: 1150 cost=0.153865129 W=0.252729 b=0.791045
Epoch: 1200 cost=0.153863519 W=0.252604 b=0.791932
Epoch: 1250 cost=0.153862283 W=0.252493 b=0.792718
Epoch: 1300 cost=0.153861254 W=0.252395 b=0.793414
Epoch: 1350 cost=0.153860509 W=0.252308 b=0.794031
Epoch: 1400 cost=0.153859898 W=0.252231 b=0.794577
Epoch: 1450 cost=0.153859422 W=0.252163 b=0.79506
Epoch: 1500 cost=0.153859034 W=0.252102 b=0.795488
Epoch: 1550 cost=0.153858781 W=0.252049 b=0.795868
Epoch: 1600 cost=0.153858542 W=0.252001 b=0.796203
Epoch: 1650 cost=0.153858304 W=0.251959 b=0.796501
Epoch: 1700 cost=0.153858200 W=0.251922 b=0.796764
Epoch: 1750 cost=0.153858081 W=0.251889 b=0.796997
Epoch: 1800 cost=0.153858021 W=0.25186 b=0.797204
Epoch: 1850 cost=0.153857931 W=0.251835 b=0.797386
Epoch: 1900 cost=0.153857872 W=0.251812 b=0.797548
Epoch: 1950 cost=0.153857827 W=0.251791 b=0.797692
Optimization Finished!
Training cost=0.153858, W=0.251774 b=0.797816
Testing cost=0.156519
[ 2.05668545]
[ 1.42725086]
'''
