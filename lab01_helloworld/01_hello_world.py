# encoding: utf-8

from __future__ import print_function


import tensorflow as tf

hello = tf.constant('Hello world!')

# Start tf session
sess = tf.Session()

print(sess.run(hello))
