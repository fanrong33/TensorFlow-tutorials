# encoding: utf-8
# tensorboard Graph 结构
import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('Weight'):
            Weight = tf.Variable(tf.random_normal([in_size, out_size]), name='Weight')
            tf.summary.histogram(layer_name+'/Weight', Weight)
        with tf.name_scope('biase'):
            biase = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='biase')
            tf.summary.histogram(layer_name+'/biase', biase)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weight) + biase
        if(activation_function is None):
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

        tf.summary.histogram(layer_name+'/outputs', outputs)
        return outputs

# 模拟真实数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # 看不懂啊，看不懂 o(╯□╰)o
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise


with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)


with tf.name_scope('loss'):
    loss= tf.reduce_mean(tf.reduce_sum(
        tf.square(ys- prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', loss) # 纯向量

with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


sess = tf.Session()
merged = tf.summary.merge_all()

writer = tf.summary.FileWriter('logs/', sess.graph)

init = tf.global_variables_initializer()
sess.run(init)


for step in range(1000):
    sess.run(train, feed_dict={xs: x_data, ys: y_data})
    if step % 20 == 0:
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, step)
        # print(step, sess.run(loss, feed_dict={xs: x_data, ys: y_data}))


# 定位到本地目录并在终端运行以下命令
# $ tensorboard --logdir=logs
