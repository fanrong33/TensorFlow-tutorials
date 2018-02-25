# encoding: utf-8
import numpy as np
import matplotlib.pyplot as plt
# 用 Keras 构建回归神经网络的步骤:
# 1、导入模块并创建数据
from keras.models import Sequential # 用来一层一层去建立神级层
from keras.layers import Dense # 意思是这个神级层是全连接层
np.random.seed(1337)


# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X) # randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))

# plot data
plt.scatter(X, Y)
#plt.show()

X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]

# 2、建立模型
model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))

# 3、激活模型
model.compile(loss='mse', optimizer='sgd')

# 4、训练模型
print('Training ---')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost: ', cost)

# 5、检验模型
print('\nTesting ---')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost: ', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# 6、可视化结果
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()



