# encoding: utf-8
# author 蔡繁荣
# version 1.0.0 build 02170426

""" 完整神经网络训练和测试代码
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
init = tf.global_variables_initializer()
sess.run(init)

validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
test_feed = {x: mnist.test.images, y_: mnist.test.labels}

for step in range(10001): # 0 - 10000

    # 训练数据集
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train, feed_dict={x: batch_x, y_: batch_y})

    # 每1000轮输出一次在验证数据集上的测试结果
    if step % 1000 == 0:
        # 计算精度
        v_xs = mnist.validation.images
        v_ys = mnist.validation.labels

        y_prediction = sess.run(prediction, feed_dict = {x: v_xs})

        correct_prediction = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(v_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        validation_accuracy = sess.run(accuracy, feed_dict={x: v_xs, y_: v_ys})
        print("After %d training step(s), validation accuracy is %g" % (step, validation_accuracy))

        # 记住当前训练的step(s)
        with open('current_step.txt', 'w') as fp:
            fp.write('%s' % step)

        # 删除之前的，保留最新的
        saver.save(sess, 'model.ckpt')


# 在训练结束之后，在测试数据上检查神经网络模型的最终正确率
t_xs = mnist.test.images
t_ys = mnist.test.labels

y_prediction = sess.run(prediction, feed_dict={x: t_xs})

correct_prediction = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(t_ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

test_accuracy = sess.run(accuracy, feed_dict={x: t_xs, y_: t_ys})
print("After %d training step(s), test accuracy is %g" % (10000, test_accuracy))

