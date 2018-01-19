import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import numpy as np
import time

start = time.time()

img = mpimage.imread('../paint/8-1.jpg')

# 고흐 데이터
test1data = tf.convert_to_tensor(img, np.float32)

img2 = mpimage.imread('../paint/test-1.jpg')

# 실제 사진 데이터
resultdata = tf.convert_to_tensor(img2, np.float32)

# 하이퍼 파라미터
total_epoch = 10000
learning_rate = 0.00001
n_hidden1 = 1400
n_hidden2 = 300
n_input = 200 * 200 * 3
n_input2 = 13 * 13 * 64
n_noise = 200 * 200 * 3

version = 1
selected_version = 1

X1 = tf.placeholder(tf.float32, [None, n_input])
X2 = tf.placeholder(tf.float32, [None, n_input])
Z = tf.placeholder(tf.float32, [None, n_noise])
keep_prob = tf.placeholder(tf.float32)
global_step1 = tf.Variable(0, trainable=False, name='global_step1')
global_step2 = tf.Variable(0, trainable=False, name='global_step2')
global_step3 = tf.Variable(0, trainable=False, name='global_step3')

# 데이터 생성자 변수
with tf.name_scope('Generator_Variable'):
    G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden1], stddev=0.01))
    G_b1 = tf.Variable(tf.zeros([n_hidden1]))

    G_W2 = tf.Variable(tf.random_normal([n_hidden1, n_input], stddev=0.01))
    G_b2 = tf.Variable(tf.zeros([n_input]))

# 데이터 구분자 변수
with tf.name_scope('Discriminator_Variable'):
    D_W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
    D_W2 = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.01))
    D_W3 = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.01))
    D_W4 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    D_W5 = tf.Variable(tf.random_normal([n_input2, n_hidden2], stddev=0.01))

    # 데이터와 얼마나 가까운 값인지 출력하기 위하여 하나의 값만 출력
    D_W6 = tf.Variable(tf.random_normal([n_hidden2, 1]))
    D_b6 = tf.Variable(tf.zeros([1]))

# 데이터 생성자 신경망 구성
def generator(noise_z):
    with tf.name_scope('generator'):
        hidden = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1)
        output = tf.nn.sigmoid(tf.matmul(hidden, G_W2) + G_b2)
        return output

# 데이터 구분자 신경망 구성
def discriminator(inputs):
    with tf.name_scope('Discriminator'):
        # 이미지 하나를 학습 하기때문에 1
        inputs_reshape = tf.reshape(inputs, [1, 200, 200, 3])

        # 200 * 200 * 3 -> 100 * 100 * 3
        L1 = tf.nn.conv2d(inputs_reshape, D_W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # 100 * 100 -> 50 * 50 * 3
        L2 = tf.nn.conv2d(L1, D_W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # 50 * 50 * 3 -> 25 * 25 * 3
        L3 = tf.nn.conv2d(L2, D_W3, strides=[1, 1, 1, 1], padding='SAME')
        L3 = tf.nn.relu(L3)
        L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # 25 * 25 * 3 -> 13 * 13 * 3
        L4 = tf.nn.conv2d(L3, D_W4, strides=[1, 1, 1, 1], padding='SAME')
        L4 = tf.nn.relu(L4)
        L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        L5 = tf.reshape(L4, [1, n_input2])
        L5 = tf.nn.dropout(tf.nn.relu(tf.matmul(L5, D_W5)), keep_prob=0.8)
        output = tf.nn.sigmoid(tf.matmul(L5, D_W6) + D_b6)
        return output

# 무작위 노이즈 생성
def get_noise(n_noise):
    with tf.name_scope('Make_noise'):
        return np.random.normal(size=(n_noise))

with tf.name_scope('Make_Data'):
    G = generator(Z)
    D_gene = discriminator(G)
    D_real1 = discriminator(X1)
    D_real2 = discriminator(X2)

with tf.name_scope('Cost'):
    loss_D1 = tf.reduce_mean(tf.log(D_real1) + tf.log(1 - D_gene))
    loss_D2 = tf.reduce_mean(tf.log(D_real2) + tf.log(1 - D_gene))
    loss_G = tf.reduce_mean(tf.log(D_gene))

    tf.summary.scalar('loss_D1', loss_D1)
    tf.summary.scalar('loss_D2', loss_D2)
    tf.summary.scalar('loss_G', loss_G)

    D_var_list = [D_W1, D_W2, D_W3, D_W4, D_W5, D_W6, D_b6]
    G_var_list = [G_W1, G_b1, G_W2, G_b2]

with tf.name_scope('Train'):
    train_D1 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-loss_D1, var_list=D_var_list, global_step=global_step1)
    train_D2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-loss_D2, var_list=D_var_list, global_step=global_step2)
    train_G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-loss_G, var_list=G_var_list, global_step=global_step3)

with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables())

    check_path = tf.train.get_checkpoint_state('./model'+str(selected_version))
    if check_path and tf.train.checkpoint_exists(check_path.model_checkpoint_path):
        saver.restore(sess, check_path.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    # sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./log'+str(version), sess.graph)

    loss_val_D, loss_val_G = 0, 0

    # for epoch in range(total_epoch):
    #     noise = get_noise(n_noise)
    #     print(epoch)
    #     print(np.reshape(noise, (1, n_noise)) / 255)
    #     print(sess.run(G, feed_dict={Z: np.reshape(noise, (1, n_noise)) / 255}).shape)
    #     print(sess.run(X, feed_dict={X: np.reshape(imgdata.eval(), (1, n_input)) / 255}).shape)
    #     print(sess.run(D_gene, feed_dict={Z: np.reshape(noise, (1, n_noise)) / 255}))
    #     print(sess.run(D_real, feed_dict={X: np.reshape(imgdata.eval(), (1, n_input)) / 255}))
    #     print(sess.run(loss_D, feed_dict={X: np.reshape(imgdata.eval(), (1, n_input)) / 255, Z: np.reshape(noise, (1, n_noise)) / 255}))
    #     print(sess.run(loss_G, feed_dict={X: np.reshape(imgdata.eval(), (1, n_input)) / 255, Z: np.reshape(noise, (1, n_noise)) / 255}))
    # exit(0)

    for epoch in range(total_epoch):
        local_time = time.time()
        noise = get_noise(n_noise)
        # print(epoch)
        # print(np.reshape(noise, (1, n_noise)) / 255)
        # print('D_gene', sess.run(D_gene, feed_dict={Z: np.reshape(noise, (1, n_noise)) / 255}).shape)
        # print('D_real', sess.run(D_real, feed_dict={X: np.reshape(imgdata.eval(), (1, n_input)) / 255}).shape)
        # print(sess.run(loss_D, feed_dict={X: np.reshape(imgdata.eval(), (1, n_input)) / 255, Z: np.reshape(noise, (1, n_noise)) / 255}))
        # print(sess.run(loss_G, feed_dict={X: np.reshape(imgdata.eval(), (1, n_input)) / 255, Z: np.reshape(noise, (1, n_noise)) / 255}))

        _, loss_val_D1 = sess.run([train_D1, loss_D1], feed_dict={X1: np.reshape(test1data.eval(), (1, n_input)) / 255, X2: np.reshape(resultdata.eval(), (1, n_input)) / 255, Z: np.reshape(noise, (1, n_noise))/255, keep_prob: 0.8})
        _, loss_val_D2 = sess.run([train_D2, loss_D2], feed_dict={X1: np.reshape(test1data.eval(), (1, n_input)) / 255, X2: np.reshape(resultdata.eval(), (1, n_input)) / 255, Z: np.reshape(noise, (1, n_noise)) / 255, keep_prob: 0.8})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: np.reshape(noise, (1, n_noise))/255, keep_prob: 0.8})
        print('epoch:', '%d' % sess.run(global_step1), 'loss_val_D1:', loss_val_D1, 'loss_val_D2:', loss_val_D2, 'loss_val_G:', loss_val_G, '소요시간:', time.time() - local_time, 's')

        summary = sess.run(merged, feed_dict={X1: np.reshape(test1data.eval(), (1, n_input)) / 255, X2: np.reshape(resultdata.eval(), (1, n_input)) / 255, Z: np.reshape(noise, (1, n_noise))/255, keep_prob: 0.6})
        writer.add_summary(summary, global_step=sess.run(global_step1))

        if epoch % 10 == 0:
            noise = get_noise(n_noise)
            returndata = sess.run(G, feed_dict={Z: np.reshape(noise, (1, n_noise)) / 255})

            plt.imshow(np.reshape(returndata, (200, 200, 3)) * 255)
            plt.savefig('sample'+str(version)+'/{}.png'.format(str(sess.run(global_step1)).zfill(3)), bbox_inches='tight')

            plt.close()

    noise = get_noise(n_noise)
    returndata = sess.run(G, feed_dict={Z: np.reshape(noise, (1, n_noise)) / 255})

    plt.imshow(np.reshape(returndata, (200, 200, 3)) * 255)
    plt.savefig('sample' + str(version) + '/last.png', bbox_inches='tight')

    plt.close()

    print('소요시간:', time.time() - start, 's')

    # plt.show()

    saver.save(sess, './model'+str(version)+'/ganCheck.ckpt', global_step=global_step1)
