# encoding: utf-8
# author 蔡繁荣
# version 1.0.2 build 02170426

""" 通过 input_data.read_data_sets 函数生成的类会自动将 MNIST 数据集划分为 train、validation 和 test 三个数据集
思路事物的本质（之前没考虑到mnist的结构）
View more http://yann.lecun.com/exdb/mnist
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入MNIST数据集 digits image data from 0 to 9
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


print("Traning data sise: %s" % mnist.train.num_examples)
# Traning data sise: 55000

print("Validating data size: %s" % mnist.validation.num_examples)
# Validating data size: 5000

print("Testing data size: %s" % mnist.test.num_examples)
# Testing data size: 10000

print("Example traning data: %s" % mnist.train.images[0])
# [0. 0. ... 784]  28x28

print("Example training data label: %s" % mnist.train.labels[0])
# Example training data label: [ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]


batch_size = 100

xs, ys = mnist.train.next_batch(batch_size)
print "X shape: " , xs.shape
# X shape:  (100, 784)

print "Y shape: " , ys.shape
# Y shape:  (100, 10)