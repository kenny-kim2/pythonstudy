import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./mnist/data', one_hot=True)

# 하이퍼 파라미터
total_epoch = 100
batch_size = 100
learning_rate = 0.0002
n_hidden = 256
n_input = 28*28
n_noise = 128

X = tf.placeholder(tf.float32, [None, n_input])
# 노이즈
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
def generator(noise_z):
    hidden = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, G_W2)+ G_b2)
    return output

# 데이터 구분자 신경망 구성
def discriminator(inputs):
    hidden = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, D_W2) + D_b2)

    return output
# 무작위 노이즈 생성
def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))

# 노이즈 Z를 이용하여 가짜 이미지를 만들 G, G가 만든 이미지와 진짜 이미지 X를 비교하여 이미지가 진짜 인지 확인
G = generator(Z)
D_gene = discriminator(G)
D_real = discriminator(X)

# 데이터 생성자의 손실값은 실제 값으로 학습한 결과는 1에 가까워야 하고 생성한 값으로 학습한 결과는 0에 가까워야함
# D_real은 1에 가까워야 하고 D_gene은 0에 가까워야 함
# 경찰의 학습 손실값
loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1-D_gene))

# 위조 지폐범의 학습 손실값
loss_G = tf.reduce_mean(tf.log(D_gene))

# 각 신경망에 사용하는 변수 묶음 처리
D_var_list = [D_W1, D_b1, D_W2, D_b1]
G_var_list = [G_W1, G_b1, G_W2, G_b2]


# GAN이론에 따르면 loss를 최대로 해야하지만 사용할 수 있는 함수가 minimize뿐이여서 함수에 -값을 붙임
train_D = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-loss_G, var_list=G_var_list)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    total_batch = int(mnist.train.num_examples / batch_size)

    loss_val_D, loss_val_G = 0, 0

    for epoch in range(total_epoch):
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            noise = get_noise(batch_size, n_noise)

            _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X:batch_x, Z:noise})
            _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z:noise})

        print('epoch:', epoch, 'D loss:', '{:.4f}'.format(loss_val_D))
        print('epoch:', epoch, 'G loss:', '{:.4f}'.format(loss_val_G))

        if epoch == 0 or (epoch + 1) % 10 == 0:
            sample_size = 10
            noise = get_noise(sample_size, n_noise)
            samples = sess.run(G, feed_dict={Z:noise})

            fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

            for i in range(sample_size):
                ax[i].set_axis_off()
                ax[i].imshow(np.reshape(samples[i], (28, 28)))

            plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)
    print('최적화 완료')
