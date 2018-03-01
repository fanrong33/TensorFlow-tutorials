# encoding: utf-8
# 逻辑回归 MNIST分类
# @version: v1.0.0 build 20180225
# 参考 https://github.com/aymericdamien/TensorFlow-Examples/

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
tf.set_random_seed(777) # for reproducibility


# 载入 MNIST 数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


# 定义参数
learning_rate   = 0.01
training_epochs = 5000
batch_size      = 100
display_step    = 250


# 定义计算精度函数
def compute_accuracy(xs, ys):
    global pred
    y_pred = sess.run(pred, feed_dict={x: xs})

    # 预测的y_与实际数据集的y进行对比判断是否相等
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(ys, 1))
    # 统计正确的比例
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    result = sess.run(accuracy, feed_dict={x: xs, y_: ys})
    return result


# tf Graph Input 输入
x  = tf.placeholder(tf.float32, shape=[None, 784], name='x-input') # mnist data image of shape 28*28=784
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y-input') # 0-9 digits recognition => 10 classes


# Set model weights 权重 和 b 偏值
W = tf.Variable(tf.zeros([784, 10]), name='Weight')
b = tf.Variable(tf.zeros([10]), name='bias')



# 构建一个线性模型
# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax


# Minimize error using cross entropy
# 成本函数 交叉熵算法，分类任务！
cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(pred), reduction_indices=1))

# Minimize 最优化算法）采用梯度下降法
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)


# 启动Graph
with tf.Session() as sess:

    # 初始化变量
    init = tf.global_variables_initializer()
    sess.run(init)



    # 拟合平面
    for epoch in range(training_epochs):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={x: batch_xs, y_: batch_ys})
            print("Epoch: %04d cost=%.9f" % (epoch, loss))
            # 这里不打印 W 和 b，因为它们是多维度的[784]



    print("Optimization Finished!")
    

    # 测试模型正确率
    accuracy = compute_accuracy(mnist.test.images, mnist.test.labels)
    print('Accuracy: %s' % (accuracy))


'''
Epoch: 0000 cost=2.283365726
Epoch: 0250 cost=1.197209716
Epoch: 0500 cost=0.699818432
Epoch: 0750 cost=0.670874953
Epoch: 1000 cost=0.610565841
Epoch: 1250 cost=0.564508259
Epoch: 1500 cost=0.584423840
Epoch: 1750 cost=0.438512951
Epoch: 2000 cost=0.506308198
Epoch: 2250 cost=0.425798118
Epoch: 2500 cost=0.322104096
Epoch: 2750 cost=0.370575935
Epoch: 3000 cost=0.464264363
Epoch: 3250 cost=0.345506579
Epoch: 3500 cost=0.431796581
Epoch: 3750 cost=0.502613783
Epoch: 4000 cost=0.354659319
Epoch: 4250 cost=0.591821969
Epoch: 4500 cost=0.316032857
Epoch: 4750 cost=0.328866512
Optimization Finished!
Accuracy: 0.9016
'''
