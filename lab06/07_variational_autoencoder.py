# encoding: utf-8
""" Variational Auto-Encoder 可变自动编码

Using a variational auto-encoder to generate digits images from noise.
MNIST handwritten digits are used as training examples.

References:
    - Auto-Encoding Variational Bayes The International Conference on Learning
    Representations (ICLR), Banff, 2014. D.P. Kingma, M. Welling
    - Understanding the difficulty of training deep feedforward neural networks.
    X Glorot, Y Bengio. Aistats 9, 249-256
    - Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    - [VAE Paper] https://arxiv.org/abs/1312.6114
    - [Xavier Glorot Init](www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.../AISTATS2010_Glorot.pdf).
    - [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

# 载入 MNIST 数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../datasets/MNIST/', one_hot=True)


# 定义训练参数
learning_rate  = 0.001
training_steps = 10000
batch_size     = 64
display_step   = 1000

# 定义神级网络参数
image_dim  = 784 # MNIST 数据输入 (img shape: 28*28)
hidden_dim = 512
latent_dim = 2   # 潜在的

# A custom initialization (see Xavier Glorot init)
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

# tf Graph input
x = tf.placeholder(tf.float32, shape=[None, image_dim], name='x-input')

# Store layers weight 权重 & bias 偏值
weights = {
    'encoder_w1' : tf.Variable(glorot_init([image_dim, hidden_dim])),
    'z_mean'     : tf.Variable(glorot_init([hidden_dim, latent_dim])),
    'z_std'      : tf.Variable(glorot_init([hidden_dim, latent_dim])),
    'decoder_w1' : tf.Variable(glorot_init([latent_dim, hidden_dim])),
    'decoder_out': tf.Variable(glorot_init([hidden_dim, image_dim]))
}
biases = {
    'encoder_b1' : tf.Variable(glorot_init([hidden_dim])),
    'z_mean' : tf.Variable(glorot_init([latent_dim])),
    'z_std' : tf.Variable(glorot_init([latent_dim])),
    'decoder_b1' : tf.Variable(glorot_init([hidden_dim])),
    'decoder_out': tf.Variable(glorot_init([image_dim]))
}


# Building the encoder
encoder = tf.matmul(x, weights['encoder_w1'] + biases['encoder_b1'])
encoder = tf.nn.tanh(encoder)
z_mean  = tf.matmul(encoder, weights['z_mean']) + biases['z_mean']
z_std   = tf.matmul(encoder, weights['z_std']) + biases['z_std']

# Sampler: Normal (gaussian) random distribution
eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
z = z_mean + tf.exp(z_std / 2) * eps


# Building the decoder (with scope to re-use these layers later)
decoder = tf.matmul(z, weights['decoder_w1']) + biases['decoder_b1']
decoder = tf.nn.tanh(decoder)
decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
decoder = tf.nn.sigmoid(decoder)


# Define VAE Loss
def vae_loss(x_reconstructed, x_true):
    # Reconstruction loss
    encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) \
                         + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
    encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
    # KL Divergence loss
    kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    return tf.reduce_mean(encode_decode_loss + kl_div_loss)


# Define loss and optimizer, minimize the squared error
cost = vae_loss(decoder, x)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)


# 启动图
with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    start = time.time()

    for step in range(training_steps):
        # Prepare Data
        # Get the next batch of MNIST data (only images are need, not labels)
        batch_xs, _ = mnist.train.next_batch(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        sess.run(train, feed_dict={x: batch_xs})

        if step % display_step == 0:
            loss = sess.run(cost, feed_dict={x: batch_xs})
            
            print('Step %s, Minibatch Loss: %.3f' % (step, loss))

    cost_time = time.time() - start
    print("Optimization Finished! Cost Time: %.3fs" % cost_time)


    # Testing
    # Generator takes noise as input
    noise_input = tf.placeholder(tf.float32, shape=[None, latent_dim])
    # Rebuild the decoder to create image from noise
    decoder = tf.matmul(noise_input, weights['decoder_w1']) + biases['decoder_b1']
    decoder = tf.nn.tanh(decoder)
    decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
    decoder = tf.nn.sigmoid(decoder)

    # Building a manifold of generated digits
    n = 20
    x_axis = np.linspace(-3, 3, n)
    y_axis = np.linspace(-3, 3, n)

    canvas = np.empty((28 * n, 28 * n))
    for i, yi in enumerate(x_axis):
        for j, xi in enumerate(y_axis):
            z_mu = np.array([[xi, yi]] * batch_size)
            x_mean = sess.run(decoder, feed_dict={noise_input: z_mu})
            canvas[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = \
            x_mean[0].reshape(28, 28)

    plt.figure(figsize=(8, 10))
    Xi, Yi = np.meshgrid(x_axis, y_axis)
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.show()



'''
Step 0, Minibatch Loss: 630.802
Step 1000, Minibatch Loss: 170.974
Step 2000, Minibatch Loss: 165.659
Step 3000, Minibatch Loss: 157.435
Step 4000, Minibatch Loss: 160.248
Step 5000, Minibatch Loss: 165.659
Step 6000, Minibatch Loss: 166.100
Step 7000, Minibatch Loss: 145.598
Step 8000, Minibatch Loss: 156.237
Step 9000, Minibatch Loss: 155.230
Optimization Finished! Cost Time: 211.555s
'''




