# encoding: utf-8
# 定义神经网络的结构和前向传播的过程，并使用placeholder来作为input参数
 
import tensorflow as tf 

# 声明w1、w2两个变量。这里还通过 seed 参数设定了随机种子，
# 这样可以保证每次运行得到的结果是一样的。
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 暂时将输入的特征向量定义为一个常量。注意这里 x 是一个 1*2 的矩阵。
# x = tf.constant([[0.7, 0.9]])
x = tf.placeholder(tf.float32, shape=(1, 2), name='input')

# 定义神经网络前向传播的过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 下面一行将报错 InvalidArgumentError (see above for traceback): You must feed a value for placeholder 
# tensor 'input' with dtype float and shape [1,2]
print(sess.run(y, feed_dict={x: [[0.7, 0.9]]}))

# 输出 [[ 3.95757794]]

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
sess.close()
