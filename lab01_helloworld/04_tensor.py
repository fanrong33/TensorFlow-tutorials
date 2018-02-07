# encoding: utf-8
# tensor张量

import tensorflow as tf

# tf.constant 是一个计算，这个计算的结果为一个张量，保存在变量a中
a = tf.constant([1., 2.], name='a') # 常量a为一个长度为2的一维数组
b = tf.constant([2., 3.], name='b')

result = tf.add(a, b, name='add')
print(result)
'''
Tensor("add:0", shape=(2,), dtype=float32)
shape=(2,) 说明了张量result是一个一维数组，这个数组的长度为2
维度是张量一个很重要的属性，围绕张量的维度TensorFlow给出了很多有用的运算
'''

# 在卷积神经网络中，卷积层或池化层有可能改变张量的维度，
# 通过result.get_shape() 函数来获取结果张量的维度信息可以免去人工计算的麻烦
print(result.get_shape())
print(result.dtype)

with tf.Session() as sess:
    print(sess.run(result))
    '''
    [ 3.  5.]
    '''
