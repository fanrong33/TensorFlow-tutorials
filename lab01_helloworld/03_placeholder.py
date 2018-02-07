# encoding: utf-8

import tensorflow as tf

# add(a+b)

a = tf.placeholder(tf.int16) # 其实就是函数的输入参数！占位符😄
b = tf.placeholder(tf.int16)

add = tf.add(a, b) 
mul = tf.multiply(a, b)

print(add)
''' Tensor("Add_1:0", dtype=int16) '''
print(a + b)
''' Tensor("add_1:0", dtype=int16) '''
print(mul)
print(a * b)

# Launch the default graph
sess = tf.Session()
result = sess.run(add, feed_dict={a: 2, b: 3})
print(result)
''' 5 '''

feed = {a: 3, b: 5}
result = sess.run(mul, feed_dict=feed)
print(result)
''' 15 '''




