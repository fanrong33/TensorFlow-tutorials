# encoding: utf-8
# View more https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-01-classifier/
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# digits image data from 0 to 9
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


# 定义神经元层
def add_layer(inputs, in_size, out_size, activation_function=None,):
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    biase = tf.Variable(tf.random_normal([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weight) + biase
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 定义计算精度函数
# 代码实现没看懂o(╯□╰)o
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})

    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

# 定义要输入神经网络的 placeholder 
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])


# add output layer
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)


# loss函数 选用 交叉熵函数。交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，它们的交叉熵等于零。
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))

train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


for step in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train, feed_dict={xs: batch_xs, ys: batch_ys})
    if step % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))


