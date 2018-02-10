# encoding: utf-8
# 经典卷积神经网络模型 LeNet-5模型
# version 1.0.5 build 20180210

from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 定义计算精度函数
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])/255. # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image=tf.reshape(xs, [-1,28,28,1])
# print(x_image.shape)  # [n_samples, 28,28, 1]
# 调整输入数据placeholder的格式, 输入为一个四维矩阵

# 第一层卷积层的尺寸5x5，深度为32的过滤器，过滤器移动的步长为1，且使用全0填充
# conv1 layer
W_conv1 = weight_variable([5, 5, 1, 32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x32

# 第二层池化层前向传播过程，最大池化层，过滤器边长为2，使用全0填充且移动的步长为2.
# pool1 layer
h_pool1 = max_pool_2x2(h_conv1)                           # output size 14x14x32

# 第三层卷积层 输入为 14x14x32 的矩阵，输出为 14x14x64 的矩阵
# conv2 layer
W_conv2 = weight_variable([5, 5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64

# 第四层池化层的前向传播过程，这一层和第二层的结构是一样的。输入为 14x14x64 的矩阵，输出为 7x7x64 的矩阵 
# pool2 layer
h_pool2 = max_pool_2x2(h_conv2)                          # output size 7x7x64

# 将第四层池化层的输出转化为第五层全连接层的输入格式。第四层的输出为 7x7x64 的矩阵，
# 然而第五层全连接层需要的输入格式为向量，所以在这里需要将这个 7x7x64 的矩阵拉直成
# 一个向量。
# fc1 layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# dropout 一般只在全连接层而不是卷积层或者池化层使用
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第六层全连接层，输入为一组长度为 512 的向量，输出为一组长度为 10 的向量。这一层的输出
# 通过 softmax 之后就得到了最后的分类结果。
# fc2 layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# 定义损失函数的反向传播的算法 交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# 模型持久化
saver = tf.train.Saver()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(2000):

        # 训练数据集
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        
        # 每50 epoch 输出一次在验证数据集上的准确率
        if step % 100 == 0:

            loss = sess.run(cross_entropy, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})

            v_xs = mnist.test.images
            v_ys = mnist.test.labels

            validation_accuracy = compute_accuracy(v_xs, v_ys)
            print("After %d training step(s), validation accuracy is %g, loss is %s" % (step, validation_accuracy, loss))

            # 持久化模型
            saver.save(sess, 'save/model.mod')




'''
After 0 training step(s), validation accuracy is 0.18, loss is 8.68867
After 100 training step(s), validation accuracy is 0.8877, loss is 0.555782
After 200 training step(s), validation accuracy is 0.92, loss is 0.420055
After 300 training step(s), validation accuracy is 0.9351, loss is 0.232828
After 400 training step(s), validation accuracy is 0.9472, loss is 0.32939
After 500 training step(s), validation accuracy is 0.9502, loss is 0.219405
After 600 training step(s), validation accuracy is 0.954, loss is 0.346043
After 700 training step(s), validation accuracy is 0.9602, loss is 0.186519
After 800 training step(s), validation accuracy is 0.9643, loss is 0.169351
After 900 training step(s), validation accuracy is 0.9669, loss is 0.161793
After 1000 training step(s), validation accuracy is 0.9667, loss is 0.107727
After 1100 training step(s), validation accuracy is 0.9688, loss is 0.209269
After 1200 training step(s), validation accuracy is 0.9706, loss is 0.0954445
After 1300 training step(s), validation accuracy is 0.9707, loss is 0.145269
After 1400 training step(s), validation accuracy is 0.9744, loss is 0.12195
After 1500 training step(s), validation accuracy is 0.9764, loss is 0.146877
After 1600 training step(s), validation accuracy is 0.9756, loss is 0.0507634
After 1700 training step(s), validation accuracy is 0.9769, loss is 0.156197
After 1800 training step(s), validation accuracy is 0.978, loss is 0.10809
After 1900 training step(s), validation accuracy is 0.9791, loss is 0.0862329
'''
