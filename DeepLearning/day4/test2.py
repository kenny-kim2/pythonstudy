import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
import time

start = time.time()
mnist = input_data.read_data_sets('./mnist/data/', one_hot=True)

# [x1, x2, x3, x4] -> x1: 입력 갯수, x2 x3: 입력 데이터 차원, x4: 특징(회색조 이미지라 데이터가 밝기만 필요)
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool)

# LAYER 1
# 3 * 3 커널을 32개 가진 컨볼루션 계층
# 2 * 2 커널을 가진 풀링 계층
L1 = tf.layers.conv2d(X, 32, [3, 3])
# max_pooling2d(inputdata, poolsize, strides)
L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])
L1 = tf.layers.dropout(L1, 0.8, is_training)

# LAYER 2
L2 = tf.layers.conv2d(L1, 64, [3, 3])
L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
L2 = tf.layers.dropout(L2, 0.8, is_training)

# LAYER3
# 완전 연결 계층
L3 = tf.contrib.layers.flatten(L2)
L3 = tf.layers.dense(L3, 256, activation=tf.nn.relu)
L3 = tf.layers.dropout(L3, 0.8, is_training)

# LAYER4
W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L3, W4)

# Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_size = 100
    total_batch = int(mnist.train.num_examples / batch_size)

    for epoch in range(15):
        total_cost = 0

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # 입력 데이터 차원을 늘리는 작업, 1차원 백터에서 2차원 행렬로 전환
            batch_x = batch_x.reshape(-1, 28, 28, 1)

            _, cost_val = sess.run([train_op, cost], feed_dict={X: batch_x, Y: batch_y, is_training: True})

            total_cost += cost_val
        print('epoch:', (epoch+1), 'avg Cost:', '{:.3f}'.format(total_cost/total_batch))

    print('최적화 완료')

    # 결과 확인
    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('정확도:', sess.run(accuracy, feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1), Y: mnist.test.labels, is_training: False}))
    print('소요시간:', time.time() - start, 's')
