# encoding: utf-8

''' Convolutional Neural Network

使用 TensorFlow 实现一个卷积神级网络，手写数字的 MNIST 数据集

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

@version: v1.0.2 build 20180225
'''

from __future__ import print_function

import tensorflow as tf
import time

# 载入 MNIST 数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../datasets/MNIST/', one_hot=True)


# 定义参数
learning_rate   = 0.0001 # 1e-4
training_steps  = 1000
display_step    = 100
batch_size      = 128

# 定义神级网络参数
num_input   = 784  # MNIST 数据输入 (img shape: 28*28)
num_classes = 10   # MNIST 总分类 (0-9 digits)
dropout     = 0.75 # Dropout, probability to keep units

# tf Graph input
x  = tf.placeholder(tf.float32, shape=[None, num_input], name='x-input')/255.
y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name='y-input')
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# Store layers weight 权重 & bias 偏值
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32], stddev=0.1)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.1)),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024], stddev=0.1)), # dense
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes], stddev=0.1))
}
biases = {
    'bc1': tf.Variable(tf.constant(0.1, shape=[32])),
    'bc2': tf.Variable(tf.constant(0.1, shape=[64])),
    'bd1': tf.Variable(tf.constant(0.1, shape=[1024])),
    'out': tf.Variable(tf.constant(0.1, shape=[num_classes])),
}

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation 激活函数
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# Create model 定义卷积神经网络模型
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    h_conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    h_pool1 = maxpool2d(h_conv1, k=2)

    # Convolution Layer
    h_conv2 = conv2d(h_pool1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    h_pool2 = maxpool2d(h_conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    # h_pool2_flat = tf.reshape(h_pool2, [-1, weights['wd1'].get_shape().as_list()[0]])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights['wd1']) + biases['bd1']) # d -> dense

    # dropout 一般只在全连接层而不是卷积层或者池化层使用
    h_fc1_drop = tf.nn.dropout(h_fc1, dropout)

    # Output, class prediction
    out_layer = tf.add(tf.matmul(h_fc1_drop, weights['out']), biases['out'])
    return out_layer



# Construct model
logits = conv_net(x, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits) # 感觉其实放在定义网络模型中更好，隐藏细节? 错,有一定的意义



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
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5}) # dropout

        if step % display_step == 0:
            loss           = sess.run(cost, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
            print('Step %s, Training Accuracy: %.4f , Minibatch Loss: %.3f' % (step, train_accuracy, loss))
            
    cost_time = time.time() - start
    print("Optimization Finished! Cost Time: %.3fs" % cost_time)


    # 计算在测试集上的正确率
    test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    print("Testing Accuracy: %s" % test_accuracy)


'''
Step 0, Training Accuracy: 0.0547 , Minibatch Loss: 18.959
Step 100, Training Accuracy: 0.7891 , Minibatch Loss: 1.399
Step 200, Training Accuracy: 0.8516 , Minibatch Loss: 0.934
Step 300, Training Accuracy: 0.9531 , Minibatch Loss: 0.271
Step 400, Training Accuracy: 0.9297 , Minibatch Loss: 0.375
Step 500, Training Accuracy: 0.9453 , Minibatch Loss: 0.276
Step 600, Training Accuracy: 0.9609 , Minibatch Loss: 0.225
Step 700, Training Accuracy: 0.9766 , Minibatch Loss: 0.195
Step 800, Training Accuracy: 0.9531 , Minibatch Loss: 0.203
Step 900, Training Accuracy: 0.9766 , Minibatch Loss: 0.116
Optimization Finished! Cost Time: 556.263s
Testing Accuracy: 0.9701
'''


