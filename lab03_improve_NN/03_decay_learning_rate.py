# encoding: utf-8
# 优化3：使用指数衰减学习率

import tensorflow as tf
from numpy.random import RandomState

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128

X = rdm.rand(dataset_size, 2)  #  X为shape=(128, 2)，128个数据集
# 定义规则来给出样本的标签。在这里所有x1+x2<1的样例都被认为是正样本（比如零件合格），
# 而其他为负样本（比如零件不合格）。和TensorFlow游乐场中的表示法不一样的地方是，
# 在这里使用0来表示负样本，1来表示正样本。大部分解决分类问题的神经网络都会采用0和1的表示方法。
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]


# 定义训练数据 batch 的大小
batch_size = 8

# 定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
biases1 = tf.Variable(tf.zeros([3]) + 0.1)
biases2 = tf.Variable(tf.zeros([1]) + 0.1)

x  = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义神经网络前向传播的过程, 加入偏置项和激励函数
a = tf.nn.relu(tf.matmul(x, w1) + biases1)
y = tf.nn.relu(tf.matmul(a, w2) + biases2)


global_step = tf.Variable(0)
# 通过exponential_decay函数生成学习率（初始学习率为0.1，每训练100轮后学习率乘以0.96 进行衰减）
# 100 epoch 的思考总结：如果总的训练数据有50000，每次批量训练100个，则50000/100=500，也就是500 epoch后所有数据训练一遍
# 则建议设置为 500，然后用更低的学习率进行训练学习
learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=True)
# 使用指数衰减的学习率。在minimize函数中传入global_step将自动更新global_step参数，从而使得学习率也得到相应更新。

# 定义损失函数的反向传播的算法 交叉熵
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
# train = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
# AdamOptimizer
# GradientDescentOptimizer

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    print(sess.run(w1))
    print(sess.run(w2))
    print(sess.run(biases1))
    print(sess.run(biases2))
    '''
    在训练之前神经网络参数的值：
    [[-0.81131822  1.48459876  0.06532937]
     [-2.44270396  0.0992484   0.59122431]]
    [[-0.81131822]
     [ 1.48459876]
     [ 0.06532937]]
    '''

    # 设定训练的轮数
    STEPS = 10000
    for i in range(STEPS):
        # 每次选取batch_size个样本进行训练。
        start = (i * batch_size) % dataset_size
        end   = min(start+batch_size, dataset_size)

        batch_xs, batch_ys = X[start:end], Y[start:end]

        # 通过选取的样本训练神经网络病更新参数。
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
        if i % 1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出。
            loss = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
            print("After %d training step(s), cross entropy on all data is %g" % (i, loss))
            '''
            After 0 training step(s), cross entropy on all data is 0.0674925
            After 1000 training step(s), cross entropy on all data is 0.0163385
            After 2000 training step(s), cross entropy on all data is 0.00907547
            After 3000 training step(s), cross entropy on all data is 0.00714436
            After 4000 training step(s), cross entropy on all data is 0.00578471
            '''

    print(sess.run(w1))
    print(sess.run(w2))
    print(sess.run(biases1))
    print(sess.run(biases2))
    '''
    在训练之后神经网络参数的值：
    [[-1.9618274   2.58235407  1.68203783]
     [-3.4681716   1.06982327  2.11788988]]
    [[-1.8247149 ]
     [ 2.68546653]
     [ 1.41819501]]
    [[ 17.50401878]]
    可以发现这两个参数的取值已经发生了变化，这个变化就是训练的结果。
    它使得这个神经网络能更好的你和提供的训练数据。
    '''


