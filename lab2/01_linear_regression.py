# encoding: utf-8
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

# 模拟生成X和Y数据，一阶张量: 向量 (vector)
x_data = [1, 2, 3]
y_data = [1, 2, 3]


# 通过 hypothesis = W * x_data +b 去试图查找 W 权重和 b 偏值，也就是我们要求的值
# 我们知道 W 应该为 1 ，b 应该为 0
# 但我们还是让 TensorFlow 来帮我们找出来
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='Weight')
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='bias')

# 假设，线性模型
hypothesis = W * x_data + b

# 成本函数 最小化方差
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)


# 启动Graph
sess = tf.Session()
# 初始化变量
# init = tf.initialize_all_variables() # tf 马上就要废弃这种写法
init = tf.global_variables_initializer()  # 替换成这样就好
sess.run(init)



# 拟合平面
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

# 得到最佳拟合结果 W: [1.00], b: [0.00]