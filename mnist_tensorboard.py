# coding=utf-8

"""这是一个在TensorBoard上展现总结的一个简单的MNIST分类器
 这是一个普通的MNIST模型，但是是一个很好的例子，特别是在使用
 tf.name_scope来在TensorBoard图形浏览器上构造一个清晰图型，和
 在TensorBoard上命名总结标签执行非常有意义的分组上来说都是一个很好的例子。
 它演示了TensorBoard面板上的每个功能.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', './data', 'Directory for storing data')
flags.DEFINE_string(
    'summaries_dir', '/home/zte/Downloads/mnist_logs', 'Summaries directory')


def weight_variable(shape):
    """创建一个权重变量，并做适当的初始化"""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """创建一个偏执变量，并做适当的初始化"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var, name):
    """在一个张量上附加大量summaries."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """该函数使用了可重用的代码做一个简单的神经网络层。
    该代码使用矩阵相乘加上偏置量后应用ReLU激活，也设置上了名称，便于在图形化结
    果展示中易于阅读和添加总结操作数量"""

    # 添加一个名称确保图层的逻辑分组
    with tf.name_scope(layer_name):
        # 这个变量将保存层的权重
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights, layer_name + '/weights')

        # 这个变量将保存状态层的偏置量
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases, layer_name + '/biases')

        # 对于输入的input_tensor加权求和，再分别加上一个偏置量
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            # tf.histogram_summary(layer_name + '/pre_activations', preactivate)
        # 应用ReLU激活函数
        activations = act(preactivate, 'activation')
        # tf.histogram_summary(layer_name + '/activations', activations)
        return activations


def train():
    # 输入数据
    mnist = input_data.read_data_sets(FLAGS.data_dir,
                                      one_hot=True,
                                      fake_data=FLAGS.fake_data)
    # 在一个会话中首先创建图并登陆
    sess = tf.InteractiveSession()

    # 创建一个多层模型

    # 输入占位符
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    # 重塑x为4维张量28*28的图片，和颜色通道数量为1
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.image_summary('input', image_shaped_input, 20)

    # 为了权重在初始化时加入少量的噪声来打破对称性以及避免0梯度,初始化这变量为非0
    # 由于我们使用的是ReLU神经元，因此比较好的做法是用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0的问题
    # 为了不在建立模型的时候反复做初始化操作，定义两个函数用于初始化

    hidden1 = nn_layer(x, 784, 500, 'layer1')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.scalar_summary('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)

    y = nn_layer(dropped, 500, 10, 'layer2', act=tf.nn.softmax)

    # 计算交叉熵
    with tf.name_scope('cross_entropy'):
        diff = y_ * tf.log(y)
        with tf.name_scope('total'):
            cross_entropy = -tf.reduce_mean(diff)
        tf.scalar_summary('cross entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
            cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)

    # 合并所有的summaries汇总并写入日志目录/mnist_logs
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train',
                                          sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')

    tf.initialize_all_variables().run()

    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data, & add training
    # summaries
    # 训练这模型并写总结
    # 每10步，度量下测试集的精确性并写下测试总结
    # 所有其他的步骤为在训练数据集上运行并添加训练总结
    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        """构造一个流张量feed_dict:映射数据到张量占位符上."""
        if train or FLAGS.fake_data:
            xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
            k = FLAGS.dropout
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    for i in range(FLAGS.max_steps):
        # Record summaries and test-set accuracy
        # 记录总结和测试集的精确度
        if i % 10 == 0:
            summary, acc = sess.run(
                [merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        # Record train set summaries, and train
        # 记录训练集总结，继续训练
        else:
            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = sess.run(
                    [merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)


def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
