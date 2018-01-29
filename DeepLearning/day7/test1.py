import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

start = time.time()
mnist = input_data.read_data_sets('./mnist/data', one_hot=True)

# 하이퍼 파라미터
learning_rate = 0.001
total_epoch = 30
batch_size = 128

n_input = 28
# RNN 설정 시 몇단계의 데이터를 받을것인지 설정이 필요
n_step = 28
n_hidden = 128
n_class = 10

X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

# 'RNN Cell'을 'n_hidden'개 생성
# 긴 문장일 경우 마지막 단어가 첫 단어를 기억 못할 경우도 있다.
# 이때는 BasicRNNCell 대신 LSTM 신경망을 사용
cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)

# 원래는 지속적으로 동작하는 루프를 개발해야함
# states = tf.zeros(batch_size)
# for i in range(n_step):
#     outputs, states = cell(X[[:, i]], states)
#     ...
# 하지만 기본 로직이 탑재되어있음 tf.nn.dynamic_rnn
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32, time_major=True)

print(outputs)

# output 데이터는 [batch_size, n_step, n_hidden]
# Y 값과 동일한 [batch_size, n_class] 형태로 변경이 필요함
# dynamic_rnn time_major 옵션을 True로 주게되면 [n_step, batch_size, n_hidden] 형태로 나옴

# outputs: [batch_size, n_step, n_hidden] -> [n_step, batch_size, n_hidden]
outputs = tf.transpose(outputs, [1, 0, 2])
print(outputs)

# -> [batch_size, n_hidden]
outputs = outputs[-1]

# 기본 식 사용
model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    total_batch = int(mnist.train.num_examples / batch_size)

    for epoch in range(total_epoch):
        total_cost = 0

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # 입력 데이터 포멧이 [batch_size, n_step, n_input] 이므로 수정
            batch_x = batch_x.reshape((batch_size, n_step, n_input))

            _, cost_val = sess.run([train_op, cost], feed_dict={X: batch_x, Y: batch_y})
            total_cost += cost_val
        print('epoch:', epoch, 'Avg.cost:', '{:.3f}'.format(total_cost / total_batch))

    print('최적화 완료!')

    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    test_batch_size = len(mnist.test.images)
    test_x = mnist.test.images.reshape(test_batch_size, n_step, n_input)
    test_y = mnist.test.labels

    print('정확도:', '{:.4f}'.format(sess.run(accuracy, feed_dict={X:test_x, Y:test_y})))
    print('소요시간:', time.time() - start, 's')