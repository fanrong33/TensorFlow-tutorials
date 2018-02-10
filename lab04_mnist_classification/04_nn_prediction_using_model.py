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
y_ = tf.placeholder(tf.float32, [None, 10] , name='y-input')


# add output layer
layer1 = add_layer(x     , 784, 500, activation_function=tf.nn.softmax) # 加入激励函数去线性化，relu效果很差，不知道为什么
y      = add_layer(layer1, 500, 10 , activation_function=tf.nn.softmax)


# 定义损失函数的反向传播的算法 交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


# 模型持久化
saver = tf.train.Saver()

with tf.Session() as sess:

    # saver.restore 一定要绝对路径，否则会报错找不到文件
    model_file = tf.train.latest_checkpoint('save/')
    if model_file:
        # 加载模型
        saver.restore(sess, model_file)
    else:
        print('No checkpoint file found')
        sys.exit()



    # 在训练结束之后，在测试数据上检查神经网络模型的最终正确率
    t_xs = mnist.test.images
    t_ys = mnist.test.labels

    test_accuracy = compute_accuracy(t_xs, t_ys)

    loss = sess.run(cross_entropy, feed_dict={x: t_xs, y_: t_ys})
    print("test accuracy is %g, loss is %s" % (test_accuracy, loss))
    '''
    test accuracy is 0.9224, loss is 0.295354
    '''



    dir_name = 'test_num'
    files = os.listdir(dir_name)
    print(files)
    '''
    ['1.png', '2.png', '5.png', '6.png', '7.png']
    '''
    for i in range(len(files)):
        files[i] = dir_name+"/"+files[i]
        img = Image.open(files[i]).convert('L') # convert('L') ?
        test_image = np.array(Image.open(files[i]).convert("L"))
        # print(shape(test_image))
        # (28, 28)
        
        # 展平为一维
        test_image_flat = np.reshape(test_image, [1, 784]) # [-1, 784] 注意这里！！！
        test_image_flat = test_image_flat / 255.
        """
        [[ 0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.13333333  0.10196078  0.01176471  0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.80392157  1.          0.20392157  0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.87843137  1.          0.12941176  0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.95294118  1.          0.05098039  0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.
           0.02745098  1.          0.98039216  0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.09411765  1.          0.90980392  0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.15686275  1.          0.84705882  0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.22352941  1.          0.78431373  0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.28627451  1.          0.72156863  0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.34509804  1.          0.65882353  0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.40392157  1.          0.6         0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.45882353  1.          0.54117647  0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.51764706  1.          0.48627451  0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.59607843  1.          0.41568627  0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.8         1.          0.24705882  0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.
           0.12156863  0.99607843  0.95686275  0.03921569  0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.49019608  1.          0.64705882  0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.
           0.01176471  0.88627451  1.          0.2627451   0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.30588235  1.          0.85098039  0.00392157  0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.71372549  1.          0.44705882  0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.
           0.12941176  0.99215686  0.96862745  0.07058824  0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.50588235  1.          0.63137255  0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.02745098  0.34901961  0.18039216  0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.          0.          0.          0.
           0.          0.          0.          0.        ]]
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















