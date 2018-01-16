import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import numpy as np

img = mpimage.imread('../paint/1-1.jpg')

# 실제 데이터
imgdata = tf.convert_to_tensor(img, np.float32)

# 실험 데이터
result = mpimage.imread('../paint/test.jpg')
resultdata = tf.convert_to_tensor(result, np.float32)


# 하이퍼 파라미터
total_epoch = 10
learning_rate = 0.0002
n_hidden1 = 1400
n_hidden2 = 300
n_input = 230 * 334 * 3
n_input2 = 15 * 21 * 3
n_noise = 152 * 116 * 3

X = tf.placeholder(tf.float32, [None, n_input])
Z = tf.placeholder(tf.float32, [None, n_noise])

# 데이터 생성자 변수
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden1], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden1]))

G_W2 = tf.Variable(tf.random_normal([n_hidden1, n_input], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

# 데이터 구분자 변수

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
    hidden = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, G_W2) + G_b2)
    return output

# 데이터 구분자 신경망 구성
def discriminator(inputs):
    inputs_reshape = tf.reshape(inputs, [-1, 230, 334, 3])

    # 230 * 334 * 3 -> 115 * 167 * 3
    L1 = tf.nn.conv2d(inputs_reshape, D_W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # 115 * 167 -> 57 * 84 * 3
    L2 = tf.nn.conv2d(L1, D_W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 57 * 84 * 3 -> 29 * 42 * 3
    L3 = tf.nn.conv2d(L2, D_W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 29 * 42 * 3 -> 15 * 21 * 3
    L4 = tf.nn.conv2d(L3, D_W4, strides=[1, 1, 1, 1], padding='SAME')
    L4 = tf.nn.relu(L4)
    L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    L5 = tf.reshape(L4, [64, n_input2])
    L5 = tf.nn.dropout(tf.nn.relu(tf.matmul(L5, D_W5)), keep_prob=0.8)

    output = tf.nn.sigmoid(tf.matmul(L5, D_W6) + D_b6)
    return output

# 무작위 노이즈 생성
def get_noise(n_noise):
    return np.random.normal(size=(n_noise))

G = generator(Z)
D_gene = discriminator(G)
D_real = discriminator(X)

loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1-D_gene))
loss_G = tf.reduce_mean(tf.log(D_gene))

D_var_list = [D_W1, D_W2, D_W3, D_W4, D_W5, D_W6, D_b6]
G_var_list = [G_W1, G_b1, G_W2, G_b2]

train_D = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-loss_G, var_list=G_var_list)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    loss_val_D, loss_val_G = 0, 0

    # for epoch in range(total_epoch):
    #     noise = get_noise(n_noise)
    #     print(epoch)
    #     print(np.reshape(noise, (1, n_noise)) / 255)
    #     print(sess.run(D_gene, feed_dict={Z: np.reshape(noise, (1, n_noise)) / 255}))
    #     print(sess.run(D_real, feed_dict={X: np.reshape(imgdata.eval(), (1, n_input)) / 255}))
    #     print(sess.run(loss_D, feed_dict={X: np.reshape(imgdata.eval(), (1, n_input)) / 255, Z: np.reshape(noise, (1, n_noise)) / 255}))
    #     print(sess.run(loss_G, feed_dict={X: np.reshape(imgdata.eval(), (1, n_input)) / 255, Z: np.reshape(noise, (1, n_noise)) / 255}))
    #
    # exit(0)
    for epoch in range(total_epoch):
        noise = get_noise(n_noise)
        print(epoch)
        print(np.reshape(noise, (1, n_noise)) / 255)
        print(sess.run(D_gene, feed_dict={Z: np.reshape(noise, (1, n_noise)) / 255}))
        print(sess.run(D_real, feed_dict={X: np.reshape(imgdata.eval(), (1, n_input)) / 255}))
        print(sess.run(loss_D, feed_dict={X: np.reshape(imgdata.eval(), (1, n_input)) / 255, Z: np.reshape(noise, (1, n_noise)) / 255}))
        print(sess.run(loss_G, feed_dict={X: np.reshape(imgdata.eval(), (1, n_input)) / 255, Z: np.reshape(noise, (1, n_noise)) / 255}))

        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: np.reshape(imgdata.eval(), (1, n_input))/255, Z: np.reshape(noise, (1, n_noise))/255})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: np.reshape(noise, (1, n_noise))/255})
        print('epoch:', epoch, 'lossD:', loss_val_D, 'loss_val_G:', loss_val_G)

        print()
        print()

    returndata = sess.run(G, feed_dict={Z: np.reshape(resultdata.eval(), (1, n_noise))/255})

    fig, ax = plt.subplots(1)
    returndata = np.reshape(returndata, (247, 260, 3))
    print(returndata)
    ax[1].imshow(returndata/255)
    plt.show()
