import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# 函数weight_variable可以返回一个给定形状的变量并自动以截断正态分布初始化，
# 可以用于创建卷积核
def weight_variabel(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# 卷积需要转化成28*28的图片，-1表示第一维的大小是根据x自动确定的
# 改变x的格式转为4D的向量[batch,in_height,in_width,in_channels]
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一层卷积层 卷积标配：卷积 激活 池化
W_conv1 = weight_variabel([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
W_conv2 = weight_variabel([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 28*28的图片第一次卷积后还是28*28，第一次池化后变为14*14
# 第二次卷积后为14*14，第二次池化后变为7*7
# 经过上面操作后得到64张7*7的平面

# 全连接层，输出为1024维向量
# 上一场有7*7*64个神经元，全连接层有1024个神经元
W_fc1 = weight_variabel([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# dropout keep_prob是一个占位符，训练时为0.5 测试时为1
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 在加一个全连接层，把上一层全连接层的输出转化为10个类别的打分
# 把1024维向量转化为10维向量，对应10个类别
W_fc2 = weight_variabel([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 使用tf.nn.softmax_cross_entropy_with_logits接口直接优化
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

# 定义train_step
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 定义准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 创建Session 对变量初始化
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

batch_size = 50
n_test_num = mnist.test.num_examples // batch_size
cnt = 0
i = 0

# 训练20000步
for i in range(20000):
    batch = mnist.train.next_batch(batch_size)
    # 每一百步报告一次在验证集和测试集上的准确率
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, train_accuracy %g' % (i, train_accuracy))

        accuracy_sum = 0
        for cnt in range(n_test_num):
            batch_test_xs, batch_test_ys = mnist.test.next_batch(batch_size)
            accuracy_sum += accuracy.eval(feed_dict={x: batch_test_xs, y_: batch_test_ys, keep_prob: 1.0})
        print('step %d, test_accuracy %g' % (i,  accuracy_sum / (cnt + 1)))

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 训练结束 打印测试集准确率
# print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
