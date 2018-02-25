# encoding: utf-8

''' Neural Network

使用 TensorFlow 实现一个2个隐藏层的全连接神级网络

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
@version: v1.0.0 build 20180225
'''

from __future__ import print_function

import tensorflow as tf
import time

# 载入 MNIST 数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../datasets/MNIST/', one_hot=True)


# 定义参数
learning_rate   = 0.1
training_steps  = 1000
display_step    = 100
batch_size      = 128

# 定义神级网络参数
n_hidden_1  = 256 # 第一个隐藏层网络节点数
n_hidden_2  = 256 # 第二个隐藏层网络节点数
num_input   = 784 # MNIST 数据输入 (img shape: 28*28)
num_classes = 10 # MNIST 总分类 (0-9 digits)


# tf Graph input
x  = tf.placeholder(tf.float32, shape=[None, num_input], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name='y-input')

# Store layers weight 权重 & bias 偏值
weights = {
    'w1' : tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'w2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
    'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes])),
}


# Create model 定义全连接神级网络模型
def neural_net(x):
    # 隐藏层1
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    # 隐藏层2
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    # 输出层
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(x)
prediction = tf.nn.softmax(logits) # 感觉其实放在定义网络模型中更好，隐藏细节？错



# 定义成本函数, 使用tf内置定义的交叉熵函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)


# Evaluate model 评估模型
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# 启动图
with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    start = time.time()

    for step in range(training_steps):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

        if step % display_step == 0:
            loss           = sess.run(cost, feed_dict={x: batch_xs, y_: batch_ys})
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
            # accuracy = compute_accuracy(mnist.test.images, mnist.test.labels)
            print('Step %s, Training Accuracy: %.4f , Minibatch Loss: %.3f' % (step, train_accuracy, loss))
            

    cost_time = time.time() - start
    print("Optimization Finished! Cost Time: %.3fs" % cost_time)

    # 计算在测试集上的正确率
    test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print("Testing Accuracy: %s" % test_accuracy)


'''
Step 0, Training Accuracy: 0.1953 , Minibatch Loss: 12572.660
Step 100, Training Accuracy: 0.8125 , Minibatch Loss: 433.816
Step 200, Training Accuracy: 0.8906 , Minibatch Loss: 232.595
Step 300, Training Accuracy: 0.8906 , Minibatch Loss: 95.142
Step 400, Training Accuracy: 0.8672 , Minibatch Loss: 74.698
Step 500, Training Accuracy: 0.8906 , Minibatch Loss: 40.579
Step 600, Training Accuracy: 0.8906 , Minibatch Loss: 16.594
Step 700, Training Accuracy: 0.8828 , Minibatch Loss: 27.803
Step 800, Training Accuracy: 0.8281 , Minibatch Loss: 33.419
Step 900, Training Accuracy: 0.8672 , Minibatch Loss: 43.278
Optimization Finished! Cost Time: 10.409s
Testing Accuracy: 0.856
'''


