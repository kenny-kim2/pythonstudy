import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('./mnist/data/', one_hot=True)

# 신경망 모델 구축
X = tf.placeholder(tf.float32, [None, 28*28])
Y = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32)

global_step = tf.Variable(0, trainable=False, name='global_step')

with tf.name_scope('layer1'):
    # 표준편차가 0.01인 정규분포를 가지는 임의의 값으로 초기화
    W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
    L1 = tf.nn.relu(tf.matmul(X, W1))
    L1 = tf.nn.dropout(L1, keep_prob)

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
    L2 = tf.nn.relu(tf.matmul(L1, W2))
    L2 = tf.nn.dropout(L2, keep_prob)

with tf.name_scope('output'):
    W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
    model = tf.matmul(L2, W3)

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(cost, global_step=global_step)
    tf.summary.scalar('cost', cost)

with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables())

    check_path = tf.train.get_checkpoint_state('./model6')
    if check_path and tf.train.checkpoint_exists(check_path.model_checkpoint_path):
        saver.restore(sess, check_path.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./log', sess.graph)

    batch_size = 100
    total_batch = int(mnist.train.num_examples / batch_size)

    for epoch in range(10):
        total_cost = 0

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            _, cost_val = sess.run([train_op, cost], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
            total_cost += cost_val
        print('epoch:', '%04d' % (epoch + 1), 'avg cost:', '{:3f}'.format(total_cost / total_batch))

    print('최적화 완료!')
    print(mnist.test.labels)
    # summary = sess.run(merged, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
    # writer.add_summary(summary, global_step=sess.run(global_step))

    # 결과 확인
    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    # 마지막 정확도 측정시에는 전체 뉴런을 모두 사용해야 한다
    print('정확도:', sess.run(accuracy*100, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

    saver.save(sess, './model6/dnn.ckpt', global_step=global_step)

    # 결과 확인 matplotlib
    labels = sess.run(model, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1})
    # 그래프 준비
    fig = plt.figure()

    for i in range(10):
        # 2행 5열 그래프를 만들고, i+1 번째에 숫자 이미지를 출력
        subplot = fig.add_subplot(2, 5, i+1)
        # 이미지를 깨끗하게 출력하기 위하여 눈금 제거
        subplot.set_xticks([])
        subplot.set_yticks([])
        # 출력한 이미지 위에 예측한 숫자를 출력
        # argmax는 numpy나 tensorflow나 동일한 함수가 들어있음
        # 결과값인 labels 에는 one-hot encoding으로 데이터가 들어있으므로 가장 높은 값을 가진 데이터를 예측 문자로 출력해야함
        subplot.set_title('%d' % np.argmax(labels[i]))
        # 1차원 배열로 되어있는 i번째 이미지 데이터를 28*28 형식의 2차원 배열로 변형하여 이미지 형태로 출력
        # cmap 파라미터를 통하여 이미지를 그레이스케일로 출력
        subplot.imshow(mnist.test.images[i].reshape((28,28)), cmap=plt.cm.gray_r)

    plt.show()
