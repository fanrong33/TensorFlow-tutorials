# encoding: utf-8
# version 1.1.0 build 20180209

""" 一个完整很简单的训练神经网络分类问题的示例代码
使用train数据集训练
再用test数据集来测试精确度

注意：分类的推荐是这种
View more https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-01-classifier/
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# digits image data from 0 to 9
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


# 定义神经元层
def add_layer(inputs, in_size, out_size, activation_function=None,):
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    biase  = tf.Variable(tf.random_normal([1, out_size]) + 0.1)

    Wx_plus_b = tf.matmul(inputs, Weight) + biase
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

    # print(v_ys[0])
    # [ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]

    # print("y prediction %s" % y_pred[0])
    #y prediction [  1.21815374e-05   6.98716380e-04   1.38545025e-03   1.68449292e-03
    #   1.16552451e-06   1.75877038e-04   3.51117092e-06   9.89122689e-01
    #   1.00920763e-04   6.81501720e-03]

    # print('argmax: %s' % sess.run(tf.argmax(y_pred, 1)))
    # argmax: [7 2 1 ..., 4 5 6]
    # print('argmax: %s' % sess.run(tf.argmax(y_pred, 1))[0])
    # argmax: 7

    # y_correct_pre = sess.run(correct_prediction, feed_dict={x: t_xs, y_: t_ys})
    # print('y correct prediction %s' % y_correct_pre[0])
    # y correct prediction True


# 数据中包含55000张训练图片，每张图片的分辨率是28×28，所以我们的训练网络输入应该是28×28=784个像素数据。
# 定义要输入神经网络的 placeholder 
x  = tf.placeholder(tf.float32, [None, 784], name='x-input') # 28x28
y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')


# add output layer
y = add_layer(x, 784, 10, activation_function=tf.nn.softmax)


# loss最优化目标函数 选用 交叉熵函数。交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，它们的交叉熵等于零。
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# train方法（最优化算法）采用梯度下降法
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    

    for step in range(5000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
        if step % 50 == 0:
            loss = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
            accuracy = compute_accuracy(mnist.test.images, mnist.test.labels)
            print('accuracy: %s , loss: %s' % (accuracy, loss))
            # Testing data size: 10000


