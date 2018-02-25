# encoding: utf-8

from __future__ import print_function

import tensorflow as tf

hello = tf.constant('Hello world!') # 注意：此处常量为小写的

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print(sess.run(hello))
