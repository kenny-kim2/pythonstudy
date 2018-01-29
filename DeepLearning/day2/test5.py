import tensorflow as tf

a = tf.constant([[1, 2], [1, 2]])
b = tf.constant([[3, 4], [3, 4]])

w = tf.matmul(a, b)
z = a * b

with tf.Session() as sess:
    print(sess.run(w))
    print(sess.run(z))