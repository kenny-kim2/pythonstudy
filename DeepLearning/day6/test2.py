import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./mnist/data', one_hot=True)

# 하이퍼 파라미터
total_epoch = 100
batch_size = 100
n_hidden = 256
n_input = 28*28
n_noise = 128
# 결과 학습을 위하여 추가
n_class = 10

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])
Z = tf.placeholder(tf.float32, [None, n_noise])

# 데이터 생성자 변수
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden]))

G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

# 데이터 구분자 변수
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))
# 데이터와 얼마나 가까운 값인지 출력하기 위하여 하나의 값만 출력
D_W2 = tf.Variable(tf.random_normal([n_hidden, 1]))
D_b2 = tf.Variable(tf.zeros([1]))

# 데이터 생성자 신경망 구성
def generator(noise_z, labels):
    with tf.variable_scope('generator'):
        inputs = tf.concat([noise_z, labels], 1)
        hidden = tf.layers.dense(inputs, n_hidden, activation=tf.nn.relu)
        output = tf.layers.dense(hidden, n_input, activation=tf.nn.relu)
    return output

# 데이터 구분자 신경망 구성
def discriminator(inputs, labels, reuse=None):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()
        inputs = tf.concat([inputs, labels], 1)
        hidden = tf.layers.dense(inputs, n_hidden, activation=tf.nn.relu)
        # 손실 계산에 sigmoid entropy 관련 함수를 사용하기 위하여 활성 함수는 선언하지 않음
        output = tf.layers.dense(hidden, 1, activation=None)
    return output
# 무작위 노이즈 생성
def get_noise(batch_size, n_noise):
    return np.random.uniform(-1., 1., size=[batch_size, n_noise])

# 노이즈 Z를 이용하여 가짜 이미지를 만들 G, G가 만든 이미지와 진짜 이미지 X를 비교하여 이미지가 진짜 인지 확인
G = generator(Z, Y)
D_real = discriminator(X, Y)
D_gene = discriminator(G, Y, True)


# 데이터 생성자의 손실값은 실제 값으로 학습한 결과는 1에 가까워야 하고 생성한 값으로 학습한 결과는 0에 가까워야함
# D_real은 1에 가까워야 하고 D_gene은 0에 가까워야 함
# 경찰의 학습 손실값
loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
loss_D_gene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene, labels=tf.ones_like(D_gene)))
loss_D = loss_D_real + loss_D_gene

# 위조 지폐범의 학습 손실값
loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene, labels=tf.zeros_like(D_gene)))

# 각 신경망에 사용하는 변수 묶음 처리
vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

# GAN이론에 따르면 loss를 최대로 해야하지만 사용할 수 있는 함수가 minimize뿐이여서 함수에 -값을 붙임
train_D = tf.train.AdamOptimizer().minimize(loss_D, var_list=vars_D)
train_G = tf.train.AdamOptimizer().minimize(loss_G, var_list=vars_G)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    total_batch = int(mnist.train.num_examples / batch_size)

    loss_val_D, loss_val_G = 0, 0

    for epoch in range(total_epoch):
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            noise = get_noise(batch_size, n_noise)

            _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X:batch_x, Y:batch_y, Z:noise})
            _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z:noise, Y:batch_y})

        print('epoch:', epoch, 'D loss:', '{:.4f}'.format(loss_val_D))
        print('epoch:', epoch, 'G loss:', '{:.4f}'.format(loss_val_G))

        if epoch == 0 or (epoch + 1) % 10 == 0:
            sample_size = 10
            noise = get_noise(sample_size, n_noise)
            samples = sess.run(G, feed_dict={Z:noise, Y:mnist.test.labels[:sample_size]})

            fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

            for i in range(sample_size):
                ax[0][i].set_axis_off()
                ax[1][i].set_axis_off()

                ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
                ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

            plt.savefig('samples2/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)
    print('최적화 완료')
