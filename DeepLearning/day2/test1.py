# -*- coding: utf-8 -*-
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow')
print(hello)
print()
print()
# 변수
a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a,b)
print(c)

with tf.Session() as sess:
    print(sess.run(c))
