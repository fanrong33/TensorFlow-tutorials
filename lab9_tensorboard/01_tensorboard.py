# encoding: utf-8
# tensorboard Graph 结构
import tensorflow as tf


def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('Weight'):
            Weight = tf.Variable(tf.random_normal([in_size, out_size]), name='Weight')
        with tf.name_scope('biase'):
            biase = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='biase')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weight) + biase
        if(activation_function is None):
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(prediction - ys))

with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


sess = tf.Session()

write = tf.summary.FileWriter('logs/', sess.graph)

init = tf.global_variables_initializer()
sess.run(init)


# 定位到本地目录并在终端运行以下命令
# $ tensorboard --logdir=logs
