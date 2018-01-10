import tensorflow as tf
import numpy as np

data = np.loadtxt('./data.csv', delimiter=',', unpack=True, dtype='float32')
# 행, 열 값 변경
x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
b1 = tf.Variable(tf.zeros([10]))
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))

W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))
b2 = tf.Variable(tf.zeros([20]))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))

W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))
b3 = tf.Variable(tf.zeros([3]))

model = tf.add(tf.matmul(L2, W3), b3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost, global_step=global_step)

with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables())

    check_path = tf.train.get_checkpoint_state('./model')
    if check_path and tf.train.checkpoint_exists(check_path.model_checkpoint_path):
        saver.restore(sess, check_path.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    for step in range(2):
        sess.run(train_op, feed_dict={X: x_data, Y: y_data})

        print('step: %d' % sess.run(global_step), 'cost: %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    saver.save(sess, './model/dnn.ckpt', global_step=global_step)
    prediction = tf.argmax(model, 1)
    target = tf.argmax(Y, 1)
    print('예측값:', sess.run(prediction, feed_dict={X: x_data}), '실제값:', sess.run(target, feed_dict={Y: y_data}))

    is_correct = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('정확도:', sess.run(accuracy*100, feed_dict={X: x_data, Y: y_data}))
