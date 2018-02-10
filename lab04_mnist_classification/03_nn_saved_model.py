# encoding: utf-8
# version 1.0.0 build 20170426

"""
完整神经网络训练和测试代码
最佳实践
"""

from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入 MNIST 数据集
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


# 定义神经元层
def add_layer(input_tensor, in_size, out_size, activation_function=None,):
    Weights   = tf.Variable(tf.random_normal([in_size, out_size], stddev=0.1))
    biases    = tf.Variable(tf.zeros([out_size]) + 0.1)
    Wx_plus_b = tf.matmul(input_tensor, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 定义计算精度函数
def compute_accuracy(v_xs, v_ys):
    global y
    y_pred = sess.run(y, feed_dict={x: v_xs})

    # 预测的y_与实际数据集的y进行对比判断是否相等
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(v_ys, 1))
    # 统计正确的比例
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    result = sess.run(accuracy, feed_dict={x: v_xs, y_: v_ys})
    return result


# 定义要输入神经网络的 placeholder
x  = tf.placeholder(tf.float32, [None, 784], name='x-input')
y_ = tf.placeholder(tf.float32, [None, 10] , name='y-input')


# add output layer
# 隐藏层 500 个节点
layer1 = add_layer(x,      784, 500, activation_function=tf.nn.softmax) # 加入激励函数去线性化，relu效果很差，不知道为什么
y      = add_layer(layer1, 500, 10 , activation_function=tf.nn.softmax)


# 这里有一个问题，学习率！如果从中断的训练结果继续训练，如何继续上次的学习率？
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.8, global_step, mnist.train.num_examples/100, 0.96, staircase=True)


# 定义损失函数的反向传播的算法 交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)


# 模型持久化
saver = tf.train.Saver()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(30000):

        # 训练数据集
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

        # 每100轮输出一次在验证数据集上的测试结果
        if step % 1000 == 0:
            
            loss = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})

            v_xs = mnist.validation.images # 原来v_xs 是validation验证数据集，t_xs 是test测试数据集
            v_ys = mnist.validation.labels

            validation_accuracy = compute_accuracy(v_xs, v_ys)
            print("After %d training step(s), validation accuracy is %g, loss is %s" % (step, validation_accuracy, loss))

            # 持久化模型
            saver.save(sess, 'save/model.mod')


    # 在训练结束之后，在测试数据上检查神经网络模型的最终正确率
    t_xs = mnist.test.images
    t_ys = mnist.test.labels

    loss = sess.run(cross_entropy, feed_dict={x: t_xs, y_: t_ys})

    test_accuracy = compute_accuracy(t_xs, t_ys)
    print("After %d training step(s), test accuracy is %g, loss is %s" % (10000, test_accuracy, loss))

'''
After 0 training step(s), validation accuracy is 0.1356, loss is 2.29982
After 1000 training step(s), validation accuracy is 0.3328, loss is 1.3475
After 2000 training step(s), validation accuracy is 0.4084, loss is 1.39038
After 3000 training step(s), validation accuracy is 0.5138, loss is 0.993487
After 4000 training step(s), validation accuracy is 0.7154, loss is 0.587315
After 5000 training step(s), validation accuracy is 0.7098, loss is 0.530843
After 6000 training step(s), validation accuracy is 0.8428, loss is 0.729604
After 7000 training step(s), validation accuracy is 0.824, loss is 0.438647
After 8000 training step(s), validation accuracy is 0.8788, loss is 0.508588
After 9000 training step(s), validation accuracy is 0.9036, loss is 0.281562
After 10000 training step(s), validation accuracy is 0.8954, loss is 0.231724
After 11000 training step(s), validation accuracy is 0.9156, loss is 0.253607
After 12000 training step(s), validation accuracy is 0.9202, loss is 0.238587
After 13000 training step(s), validation accuracy is 0.9246, loss is 0.216087
After 14000 training step(s), validation accuracy is 0.9206, loss is 0.188054
After 15000 training step(s), validation accuracy is 0.923, loss is 0.319637
After 16000 training step(s), validation accuracy is 0.9204, loss is 0.154523
'''
