import tensorflow as tf

data = [[[1, 2, 3, 4], [5, 6, 7, 8]],
        [[9, 10, 11, 12], [13, 14, 15, 16]],
        [[24, 23, 22, 21], [20, 19, 18, 17]]]

X = tf.placeholder(tf.float32, shape=[3, 2, 4])
arg0 = tf.argmax(X, axis=0)
arg1 = tf.argmax(X, axis=1)
arg2 = tf.argmax(X, axis=2)

with tf.Session() as sess:
    print(sess.run(arg0, feed_dict={X: data}))
    print()
    print(sess.run(arg1, feed_dict={X: data}))
    print()
    print(sess.run(arg2, feed_dict={X: data}))
