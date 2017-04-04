# encoding: utf-8
import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777) # for reproducibility

# Graph Input
X = [1., 2., 3.]
Y = [1., 2., 3.]
m = n_samples = len(X)

# model weight
W = tf.placeholder(tf.float32) # 参数W

# Construct a linear model
hypothesis = tf.multiply(X, W)

# Cost function
cost = tf.reduce_sum(tf.pow(hypothesis - Y, 2)) / m


# for graphs
W_val = []
cost_val = []

# Launch the graphs
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(-30, 50):
    print(i * -0.1, sess.run(cost, feed_dict={W: i * 0.1}))
    W_val.append(i * 0.1)
    cost_val.append(sess.run(cost, feed_dict={W: i * 0.1}))

plt.plot(W_val, cost_val, 'ro')
plt.xlabel('Weight')
plt.ylabel('cost')
plt.show()

