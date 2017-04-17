# encoding: utf-8

import tensorflow as tf

# tf.constant 是一个计算，这个计算的结果为一个张量，保存在变量a中
v1 = tf.constant([[1., 2.],
                  [3., 4.]], name='v1')
v2 = tf.constant([[5., 6.],
                  [7., 8.]], name='v2')

result = tf.matmul(v1, v2, name='matmul')
print result
'''
Out:
Tensor("matmul:0", shape=(2, 2), dtype=float32)
'''

with tf.Session() as sess:
    print(sess.run(result))
    '''
    [[ 19.  22.]
     [ 43.  50.]]
    '''
