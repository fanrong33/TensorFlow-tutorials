# encoding: utf-8

import tensorflow as tf

a = tf.placeholder(tf.int16) # 其实就是函数的输入参数！[尴尬]
b = tf.placeholder(tf.int16) 

add = tf.add(a, b)  # 定义add函数，需要a和b参数
mul = tf.multiply(a, b)

# Same op?
print(add)
print(a + b)
print(mul)
print(a * b)


# Launch the default graph
sess = tf.Session()
print(sess.run(add, feed_dict={a: 2, b: 3}))

# it's work!
feed = { a: 3, b: 5 }
print(sess.run(mul, feed_dict=feed))
