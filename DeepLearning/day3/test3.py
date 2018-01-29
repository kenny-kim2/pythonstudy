import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

start = time.time()
mnist = input_data.read_data_sets('./mnist/data/', one_hot=True)

# 신경망 모델 구축
X = tf.placeholder(tf.float32, [None, 28*28])
Y = tf.placeholder(tf.float32, [None, 10])

# 표준편차가 0.01인 정규분포를 가지는 임의의 값으로 초기화
W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    batch_size = 100
    total_batch = int(mnist.train.num_examples / batch_size)

    for epoch in range(50):
        total_cost = 0

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            _, cost_val = sess.run([train_op, cost], feed_dict={X: batch_x, Y: batch_y})
            total_cost += cost_val
        print('epoch:', '%04d' % (epoch + 1), 'avg cost:', '{:3f}'.format(total_cost / total_batch))

    print('최적화 완료!')

    # 결과 확인
    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('정확도:', sess.run(accuracy*100, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
    print('소요시간:', time.time() - start, 's')
