#  coding=utf-8

import tensorflow as tf
import input_data
import time

flags = tf.app.flags
flags.DEFINE_string(
    'summaries_dir', '/home/xxd/Downloads/cnn_logs', 'Summaries directory')
	
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 加载数据
mnist = input_data.read_data_sets("./data", one_hot=True)

# 输入占位符
with tf.name_scope('input'):
	#  定义占位符+ 初始化变量
	x = tf.placeholder("float", [None, 784])
	y_ = tf.placeholder("float", [None, 10])

# 添加一个名称确保图层的逻辑分组
with tf.name_scope('h_conv1'):
	# 这个变量将保存层的权重
	with tf.name_scope('W_conv1'):
		W_conv1 = weight_variable([5, 5, 1, 32])

	# 这个变量将保存状态层的偏置量
	with tf.name_scope('b_conv1'):
		b_conv1 = bias_variable([32])
		
	# 重塑x为4维张量28*28的图片，和颜色通道数量为1
	with tf.name_scope('image_reshape'):
		x_image = tf.reshape(x, [-1, 28, 28, 1])

	# 对于输入的input_tensor加权求卷积，再分别加上一个偏置量
	with tf.name_scope('Wx_plus_b'):
		# 应用ReLU激活函数
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		
		h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

with tf.name_scope('pool2_reshape'):
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))

# 训练操作
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
# train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.InteractiveSession()
print time.ctime()
sess.run(tf.initialize_all_variables())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(
            feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print "step %d,  training accuracy %g" % (i, train_accuracy)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print "test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

# 完成任务，关闭会话
sess.close()
print time.ctime()
