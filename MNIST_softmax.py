import tensorflow as tf
import numpy as np
import os

# 下载图片
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# x代表一个占位符，代表等待识别的图片
x = tf.placeholder(tf.float32, [None, 784])

# W是softmax模型的参数，将一个784维的输入转化为一个10维的输出，其实就是矩阵相乘
# 在Tenforflow中，变量的参数用tf.Variable表示
W = tf.Variable(tf.zeros([784, 10]))

# b即为偏置bias
b = tf.Variable(tf.zeros([10]))

# y表示模型输出
y = tf.nn.softmax(tf.matmul(x, W) + b)

# y_表示实际标签,同样用占位符表示
y_ = tf.placeholder(tf.float32, [None, 10])

# 根据y和y_设计交叉熵损失函数
# tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

# 使用梯度下降法进行优化
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 创建一个session，只有在session中才能运行优化步骤train_step
sess = tf.InteractiveSession()

# 运行之前初始化所有变量，分配内存
tf.global_variables_initializer().run()

# 进行1000步梯度下降
for _ in range(1000):
    # 在mnist中读取100个训练数据
    # batch_xs是形状为（100， 784）的图像数据，batch_ys是（100， 10）的实际标签
    # batch_xs和batch_ys分别对应着两个占位符x和y_
    batch_xs, batch_ys = mnist.train.next_batch(100)

    # 在session中运行train_step，运行时要传入占位符的值
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 正确的预测结果
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 计算正确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 在Session中运行Tensor可以得到Tensor的值
# 计算最终的准确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print(sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels}))
