import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist/data', one_hot=True)

# 하이퍼 파라미터 구조화
learning_rate = 0.01
training_epoch = 20
batch_size = 100
# 은닉층 뉴런 갯수
n_hidden = 256
n_input = 28*28

# 비지도 이기 떄문에 Y값이 없음
X = tf.placeholder(tf.float32, [None, n_input])

# 인코더
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
b_encode = tf.Variable(tf.random_normal([n_hidden], stddev=0.01))

encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, W_encode), b_encode))

# 디코더
W_decode = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
b_decode = tf.Variable(tf.random_normal([n_input], stddev=0.01))
decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, W_decode), b_decode))

cost = tf.reduce_mean(tf.pow(X - decoder, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    total_batch = int(mnist.train.num_examples / batch_size)

    for epoch in range(training_epoch):
        total_cost = 0

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, cost_val = sess.run([train_op, cost], feed_dict={X: batch_x})
            total_cost += cost_val
        print('Epoch:', (epoch+1), '%04d'%(epoch + 1), 'Avg. cost=', '{:.4f}'.format(total_cost/total_batch))
    
    print('최적화 완료')

    # 결과 확인

    sample_size = 10

    samples = sess.run(decoder, feed_dict={X: mnist.test.images[10:sample_size+10]})
    fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

    for i in range(sample_size):
        ax[0][i].set_axis_off()
        ax[1][i].set_axis_off()
        ax[0][i].imshow(np.reshape(mnist.test.images[i+sample_size], (28, 28)))
        ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

    plt.show()
