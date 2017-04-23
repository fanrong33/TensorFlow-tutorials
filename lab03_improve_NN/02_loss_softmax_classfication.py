# encoding: utf-8
# 优化2：通过softmax层将神经网络输出变成一个概率分布，分类问题
# TODO待实现，应该是 one-hot encoding

import tensorflow as tf
from numpy.random import RandomState

# 定义训练数据 batch 的大小
batch_size = 8

# 定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
biases1 = tf.Variable(tf.zeros([1, 3]) + 0.1)
biases2 = tf.Variable(tf.zeros([1, 1]) + 0.1)


x  = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义神经网络前向传播的过程, 加入偏置项和激励函数
a = tf.nn.relu(tf.matmul(x, w1) + biases1)
y = tf.nn.relu(tf.matmul(a, w2) + biases2)

# 定义损失函数的反向传播的算法
# cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, y_)
train = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)


# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128

X = rdm.rand(dataset_size, 2)  #  X为shape=(128, 2)，128个数据集
# 定义规则来给出样本的标签。在这里所有x1+x2<1的样例都被认为是正样本（比如零件合格），
# 而其他为负样本（比如零件不合格）。和TensorFlow游乐场中的表示法不一样的地方是，
# 在这里使用0来表示负样本，1来表示正样本。大部分解决分类问题的神经网络都会采用0和1的表示方法。
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

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
    STEPS = 5000
    for i in range(STEPS):
        # 每次选取batch_size个样本进行训练。
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)

        # 通过选取的样本训练神经网络病更新参数。
        sess.run(train, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i %1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出。
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
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


