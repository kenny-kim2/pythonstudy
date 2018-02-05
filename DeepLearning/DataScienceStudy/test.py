# import tensorflow as tf
# import matplotlib.image as mpimage
# import matplotlib.pyplot as plt
# import numpy as np
#
# img = mpimage.imread('../paint/1-1.jpg')
#
# localplt = plt.imshow(img)
# plt.show(localplt)

# import urllib.request
#
# image = urllib.request.urlopen(urllib.request.Request('http://img.danawa.com/prod_img/500000/426/889/img/4889426_1.jpg?shrink=160:160&_v=20170406180457'))
# # print(urllib.request.urlopen("http://gilugi407.blog.me"))
#
# print(image.read())
#
# img = mpimage.imread(image.read())
#
# print(img)

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

X = tf.placeholder(tf.float32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    with tf.device('/gpu:1'):
        print('/gpu:1',sess.run(X, feed_dict={X:3}))

    with tf.device('/cpu:0'):
        print('/cpu:0', sess.run(X, feed_dict={X: 3}))