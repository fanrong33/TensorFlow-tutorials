# encoding: utf-8
import tensorflow as tf
import os
tf.set_random_seed(777)  # for reproducibility

# 先创建 W, b 的容器
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), dtype=tf.float32, name='Weight')
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0), dtype=tf.float32, name='biase')

X = tf.placeholder(tf.float32)

# 假设，线性模型
hypothesis = W * X + b


# 这里不需要初始化步骤 

saver = tf.train.Saver()
with tf.Session() as sess:
    # 提取变量
    saver.restore(sess, "my_net/save_net.ckpt");
    print("Weight: ", sess.run(W))
    print('biase: ', sess.run(b))

    # 使用得到的结果进行预测
    print(sess.run(hypothesis, feed_dict={X: 6}))
