# encoding: utf-8
# author 蔡繁荣
# version 1.0.0 build 02170426

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
# import input_data
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



# 定义要输入神经网络的 placeholder
x  = tf.placeholder(tf.float32, [None, 784], name='x-input')
y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')


# add output layer
# prediction = add_layer(x, 784, 10, activation_function=tf.nn.softmax) # 加入激励函数去线性化，relu效果很差，不知道为什么
layer1 = add_layer(x, 784, 500, activation_function=tf.nn.softmax) # 加入激励函数去线性化，relu效果很差，不知道为什么
prediction = add_layer(layer1, 500, 10, activation_function=tf.nn.softmax)


# loss函数 选用交叉熵函数。交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，它们的交叉熵等于零。
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(prediction), reduction_indices=[1]))

train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# 模型持久化
saver = tf.train.Saver()


sess = tf.Session()

# saver.restore 一定要绝对路径，否则会报错找不到文件
saver.restore(sess, "/Users/fanrong33/kuaipan/github/TensorFlow-tutorials/lab04_mnist_classification/model.ckpt")


# 接着上次训练的模型继续训练
validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
test_feed = {x: mnist.test.images, y_: mnist.test.labels}

with open('current_step.txt', 'r') as fp:
    current_step = int(fp.read())
'''

# 0 101    0   - 100
# 101 201  101 - 200

for step in range(current_step+1, current_step+1+10000):

    # 每次训练100个样本数据集
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train, feed_dict={x: batch_x, y_: batch_y})

    # 每1000轮 输出一次 在 验证数据集上 的 测试结果
    if step % 1000 == 0:
        # 计算精度
        v_xs  = mnist.validation.images
        v_ys_ = mnist.validation.labels

        ## 定义计算精确度公式 ########
        y_prediction = sess.run(prediction, feed_dict = {x: v_xs})

        correct_prediction = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(v_ys_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        ###########################

        validation_accuracy = sess.run(accuracy, feed_dict={x: v_xs, y_: v_ys_})
        print("After %d training step(s), validation accuracy is %g" % (step, validation_accuracy))

        # 记住当前训练的step(s)
        with open('current_step.txt', 'w') as fp:
            fp.write('%s' % step)

        saver.save(sess, 'model.ckpt')

'''


# 在训练结束之后，在测试数据上检查神经网络模型的最终正确率
t_xs = mnist.test.images
t_ys = mnist.test.labels

## 定义计算精确度公式 ########
y_prediction = sess.run(prediction, feed_dict={x: t_xs})

correct_prediction = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(t_ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
###########################
test_accuracy = sess.run(accuracy, feed_dict={x: t_xs, y_: t_ys})
print("After %d training step(s), test accuracy is %g" % (current_step, test_accuracy))


print(t_ys[0])
# [ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]

print("y prediction %s" % y_prediction[0])
#y prediction [  1.21815374e-05   6.98716380e-04   1.38545025e-03   1.68449292e-03
#   1.16552451e-06   1.75877038e-04   3.51117092e-06   9.89122689e-01
#   1.00920763e-04   6.81501720e-03]

print('argmax: %s' % sess.run(tf.argmax(y_prediction, 1)[0]))
# argmax: 7

y_correct_pre = sess.run(correct_prediction, feed_dict={x: t_xs, y_: t_ys})
print('y correct prediction %s' % y_correct_pre[0])
# y correct prediction True


# 暂时没有用到
def compute_accuracy(xs, ys):
    global prediction
    y_prediction = sess.run(prediction, feed_dict={x: xs})

    correct_prediction = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    result = sess.run(accuracy, feed_dict={x: xs, y: ys})
    return result



dir_name = 'test_num'
files = os.listdir(dir_name)
for i in range(len(files)):
    files[i] = dir_name+"/"+files[i]
    test_image = np.array(Image.open(files[i]).convert("L"))
    # print(shape(test_image))
    # (28, 28)

    # 下面这句报错 AttributeError: 'module' object has no attribute 'DataSet'
    # mnist.test2 = input_data.DataSet(test_images1, test_labels1)
    # print(compute_accuracy(mnist.test2.images, mnist.test2.lables))

    # test_images1 : shape (1, 1, 28, 28, 1)
    # x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    # print(test_images[0])
    
    test_image_flat = np.reshape(test_image, [-1, 784]) # [-1, 784] 注意这里！！！

    prediction_result = sess.run(prediction, feed_dict={x: test_image_flat})
    print("prediction  %s" % prediction_result)

    print('argmax: %s' % sess.run(tf.argmax(prediction_result, 1)))















