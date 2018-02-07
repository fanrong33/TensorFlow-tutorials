# encoding: utf-8
# 定义神经网络的结构和前向传播的过程
 
import tensorflow as tf 

"""
输入层   隐藏层    输出层
x * w1 = a
         a * w2 = y
"""

# 声明w1、w2两个变量。2 -> 3 -> 1
# 通过seed 参数设定了随机种子, 这样可以保证每次运行得到的结果是一样的。
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 暂时将输入的特征向量定义为一个常量。注意这里 x 是一个 1*2 的矩阵，。
x = tf.constant([ [0.7, 0.9] ])

# 定义神经网络前向传播的过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)


with tf.Session() as sess:
    # 因为w1和w2都还没有运行初始化过程。下面的两行分别初始化了w1和w2两个变量
    # sess.run(w1.initializer)
    # sess.run(w2.initializer)

    init = tf.global_variables_initializer()
    sess.run(init)

    print(sess.run(y))
    '''
    [[ 3.95757794]]
    '''

    print(sess.run(w1))
    '''
    [[-0.81131822  1.48459876  0.06532937]
     [-2.44270396  0.0992484   0.59122431]]
    '''
    print(sess.run(w2))
    '''
    [[-0.81131822]
     [ 1.48459876]
     [ 0.06532937]]
     '''
