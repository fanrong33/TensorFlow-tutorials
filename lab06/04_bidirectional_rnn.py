# encoding: utf-8

''' Bi-directional Recurrent Neural Network 双向循环神经网络

使用 TensorFlow 实现一个循环神经网络（LSTM长短时记忆模型），分类手写数字的 MNIST 数据集

Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

@version: v1.0.0 build 20180225
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import time

# 载入 MNIST 数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../datasets/MNIST/', one_hot=True)

'''
To classify images using a bidirectional recurrent neural network, we consider
every image row as a sequence of pixels. Because MNIST image shape is 28*28px,
we will then handle 28 sequences of 28 steps for every sample.
'''

# 定义训练参数
learning_rate   = 0.001 # 1e-3
training_steps  = 2000
batch_size      = 128
display_step    = 200

# 定义神经网络参数
num_input   = 28  # MNIST 数据输入 (img shape: 28*28)
timesteps   = 28  # timesteps
num_hidden  = 128 # hidden layer num of features
num_classes = 10  # MNIST 总分类 (0-9 digits)


# tf Graph input
x  = tf.placeholder(tf.float32, shape=[None, timesteps, num_input], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name='y-input')



# Store layers weight 权重 & bias 偏值
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*num_hidden, num_classes], stddev=0.1))
}
biases = {
    # 'out': tf.Variable(tf.random_normal([num_classes]))
    'out': tf.Variable(tf.constant(0.1, shape=[num_classes])),
}


# Create model 定义卷积神经网络模型
def BiRNN(x, weights, biases):
    
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    
    # Get lstm cell output
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']    


# Construct model
logits = BiRNN(x, weights, biases)
prediction = tf.nn.softmax(logits)



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
        # Reshape data to get 28 seq of 28 elements
        batch_xs = batch_xs.reshape((batch_size, timesteps, num_input))
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

        if step % display_step == 0:
            loss           = sess.run(cost, feed_dict={x: batch_xs, y_: batch_ys})
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
            # accuracy = compute_accuracy(mnist.test.images, mnist.test.labels)
            print('Step %s, Training Accuracy: %.4f , Minibatch Loss: %.3f' % (step, train_accuracy, loss))
            
    cost_time = time.time() - start
    print("Optimization Finished! Cost Time: %.3fs" % cost_time)


    # 计算在测试集上的正确率
    test_xs = mnist.test.images[:128].reshape((-1, timesteps, num_input))
    test_ys = mnist.test.labels[:128]
    test_accuracy = sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys})
    print("Testing Accuracy: %s" % test_accuracy)


'''
Step 0, Training Accuracy: 0.2031 , Minibatch Loss: 2.282
Step 200, Training Accuracy: 0.8438 , Minibatch Loss: 0.435
Step 400, Training Accuracy: 0.8906 , Minibatch Loss: 0.306
Step 600, Training Accuracy: 0.9297 , Minibatch Loss: 0.260
Step 800, Training Accuracy: 0.9766 , Minibatch Loss: 0.094
Step 1000, Training Accuracy: 0.9844 , Minibatch Loss: 0.052
Step 1200, Training Accuracy: 0.9844 , Minibatch Loss: 0.058
Step 1400, Training Accuracy: 0.9922 , Minibatch Loss: 0.043
Step 1600, Training Accuracy: 0.9766 , Minibatch Loss: 0.082
Step 1800, Training Accuracy: 1.0000 , Minibatch Loss: 0.030
Optimization Finished! Cost Time: 221.310s
Testing Accuracy: 1.0 😱❗️
'''

