import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3


### create tensorflow structure start ###
Weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biase = tf.Variable(tf.zeros([1]))

hypothesis = Weight*x_data + biase

coss = tf.reduce_mean(tf.square(hypothesis-y_data))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(coss)

init = tf.initialize_all_variables()
### create tensorflow structure end ###


sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(coss), sess.run(Weight), sess.run(biase))
