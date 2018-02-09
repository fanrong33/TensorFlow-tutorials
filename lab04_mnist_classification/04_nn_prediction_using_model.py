# encoding: utf-8
# version 1.0.1 build 20180209

""" 使用上次导出的模型并继续训练 Softmax 回归模型
定义神经网络结构
恢复模型
接着上次训练的模型继续训练
输入图片数据进行预测
"""

# 完整神经网络训练和测试代码, 使用上次训练的模型继续训练

from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import os
from PIL import Image
import numpy as np


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
y_ = tf.placeholder(tf.float32, [None, 10],  name='y-input')


# add output layer
# prediction = add_layer(x, 784, 10, activation_function=tf.nn.softmax) # 加入激励函数去线性化，relu效果很差，不知道为什么
layer1 = add_layer(x, 784, 500, activation_function=tf.nn.softmax) # 加入激励函数去线性化，relu效果很差，不知道为什么
y = add_layer(layer1, 500, 10, activation_function=tf.nn.softmax)


# loss函数 选用交叉熵函数。交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，它们的交叉熵等于零。
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# 模型持久化
saver = tf.train.Saver()

with tf.Session() as sess:

    # saver.restore 一定要绝对路径，否则会报错找不到文件
    model_file = tf.train.latest_checkpoint('save/')
    if model_file:
        saver.restore(sess, model_file)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)



    # 在训练结束之后，在测试数据上检查神经网络模型的最终正确率
    t_xs = mnist.test.images
    t_ys = mnist.test.labels

    test_accuracy = compute_accuracy(t_xs, t_ys)
    print("test accuracy is %g" % (test_accuracy))




    dir_name = 'test_num'
    files = os.listdir(dir_name)
    '''
    ['1.png', '2.png', '5.png', '6.png', '7.png', '7_1.png']
    '''
    for i in range(len(files)):
        files[i] = dir_name+"/"+files[i]
        img = Image.open(files[i]).convert('L') # convert('L') ?
        test_image = np.array(Image.open(files[i]).convert("L"))
        # print(shape(test_image))
        # (28, 28)
        
        # 展平为一维
        test_image_flat = np.reshape(test_image, [1, 784]) # [-1, 784] 注意这里！！！
        # print(test_image_flat)
        """
        [[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0  34  26   3   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
      205 255  52   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0 224 255  33   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0 243 255  13   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   7 255 250   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0  24 255 232   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0  40 255 216   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0  57 255 200   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  73 255 184
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0  88 255 168   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 103
      255 153   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0 117 255 138   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0 132 255 124   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0 152 255 106   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0 204 255  63   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0  31 254 244  10   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0 125 255 165   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   3 226 255  67   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0  78 255 217   1   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 182 255 114
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0  33 253 247  18   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 129 255
      161   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   7  89  46   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]]
        """
        y_pred = sess.run(y, feed_dict={x: test_image_flat})
        print("prediction  %s" % y_pred)
        '''
        prediction  [[  6.12981748e-05   9.11273897e-01   2.99222331e-04   6.58253339e-05
        6.14197124e-05   9.67298474e-05   5.88818628e-04   5.81238477e-04
        8.68683234e-02   1.03363418e-04]]
        '''
        print('argmax: %s' % sess.run(tf.argmax(y_pred, 1)))
        '''
        argmax: [1]
        '''















