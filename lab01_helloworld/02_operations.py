# encoding: utf-8
import tensorflow as tf

# Start tf session
sess = tf.Session()

a = tf.constant(2)
b = tf.constant(3)

c = a+b

# Print out operation everything is operation
print(a)
''' Tensor("Const:0", shape=(), dtype=int32) '''
print(b)
print(c)

print(a+b)


# Print out the result of operation
print(sess.run(a))
''' 2 '''
print(sess.run(b))
''' 3 '''
print(sess.run(c))
''' 5 '''
print(sess.run(a+b))
''' 5 '''
