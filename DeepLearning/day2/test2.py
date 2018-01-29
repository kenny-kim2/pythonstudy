# -*- coding: utf-8 -*-
import tensorflow as tf

# None 은 크기가 정해지지 않았음을 의미
X = tf.placeholder(tf.float32, [None, 3])
print(X)

x_data = [[1,2,3],[4,5,6]]

# 무작위 값으로 초기화
W = tf.Variable(tf.random_normal([3,2]))
b = tf.Variable(tf.random_normal([2,1]))

# 정해진 값으로 초기화
# W = tf.Variable([[0.1,0.1],[0.2,0.2],[0.3,0.3]])

expr = tf.matmul(X, W) + b

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('===x_data===')
    print(x_data)
    print('===W===')
    print(sess.run(W))
    print('===b===')
    print(sess.run(b))
    print(sess.run(expr, feed_dict={X:x_data}))
