# encoding: utf-8

''' Dynamic Recurrent Neural Network 动态循环神经网络

TensorFlow implementation of a Recurrent Neural Network (LSTM) that performs
dynamic computation over sequences with variable length. This example is using
a toy dataset to classify linear sequences. The generated sequences have
variable length.

Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

@version: v1.0.0 build 20180225
'''

from __future__ import print_function

import tensorflow as tf
import random
import time


# ====================
#  TOY DATA GENERATOR
# ====================
class ToySequenceData(object):
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])

    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3, max_value=1000):
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(n_samples):
            # Random sequence length
            len = random.randint(min_seq_len, max_seq_len)
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(len)
            # Add a random or linear int sequence (50% prob)
            if random.random() < .5:
                # Generate a linear sequence
                rand_start = random.randint(0, max_value - len)
                s = [[float(i)/max_value] for i in
                     range(rand_start, rand_start + len)]
                # Pad sequence for dimension consistency
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([1., 0.])
            else:
                # Generate a random sequence
                s = [[float(random.randint(0, max_value))/max_value]
                     for i in range(len)]
                # Pad sequence for dimension consistency
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([0., 1.])
        self.batch_id = 0

    def next_batch(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen



# 定义训练参数
learning_rate   = 0.01 # 1e-2
training_steps  = 2000
batch_size      = 128
display_step    = 200

# 定义神经网络参数
seq_max_len = 20 # Sequence max length
num_hidden  = 64 # hidden layer num of features
num_classes = 2  # linear sequence or not


trainset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)
testset = ToySequenceData(n_samples=500, max_seq_len=seq_max_len)


# tf Graph input
x  = tf.placeholder(tf.float32, shape=[None, seq_max_len, 1], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name='y-input')
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])


# Store layers weight 权重 & bias 偏值
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes], stddev=0.1))
}
biases = {
    # 'out': tf.Variable(tf.random_normal([num_classes]))
    'out': tf.Variable(tf.constant(0.1, shape=[num_classes])),
}


# Create model 定义动态循环神经网络模型
def dynamicRNN(x, seqlen, weights, biases):
    
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden)
    
    # Get lstm cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, num_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']


# Construct model
prediction = dynamicRNN(x, seqlen, weights, biases)



# 定义成本函数, 使用tf内置定义的交叉熵函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)


# Evaluate model 评估模型
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# 启动图
with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    start = time.time()

    for step in range(training_steps):
        batch_xs, batch_ys, batch_seqlen = trainset.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys, seqlen: batch_seqlen})

        if step % display_step == 0:
            loss           = sess.run(cost, feed_dict={x: batch_xs, y_: batch_ys, seqlen: batch_seqlen})
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, seqlen: batch_seqlen})
            # accuracy = compute_accuracy(mnist.test.images, mnist.test.labels)
            print('Step %s, Training Accuracy: %.4f , Minibatch Loss: %.3f' % (step, train_accuracy, loss))
            
    cost_time = time.time() - start
    print("Optimization Finished! Cost Time: %.3fs" % cost_time)


    # 计算在测试集上的正确率
    test_xs     = testset.data
    test_ys     = testset.labels
    test_seqlen = testset.seqlen
    test_accuracy = sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys, seqlen: test_seqlen})
    print("Testing Accuracy: %s" % test_accuracy)


'''
Step 0, Training Accuracy: 0.5547 , Minibatch Loss: 0.690
Step 200, Training Accuracy: 0.8828 , Minibatch Loss: 0.253
Step 400, Training Accuracy: 0.9766 , Minibatch Loss: 0.059
Step 600, Training Accuracy: 0.9844 , Minibatch Loss: 0.024
Step 800, Training Accuracy: 0.9922 , Minibatch Loss: 0.018
Step 1000, Training Accuracy: 0.9922 , Minibatch Loss: 0.018
Step 1200, Training Accuracy: 0.9688 , Minibatch Loss: 0.110
Step 1400, Training Accuracy: 0.9844 , Minibatch Loss: 0.020
Step 1600, Training Accuracy: 0.9922 , Minibatch Loss: 0.015
Step 1800, Training Accuracy: 0.9922 , Minibatch Loss: 0.013
Optimization Finished! Cost Time: 113.161s
Testing Accuracy: 0.996
'''

