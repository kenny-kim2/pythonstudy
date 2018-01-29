import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import tensorflow as tf

img = mpimage.imread('../paint/1.jpg')
print(img)
print('----------------------------------')

resizedata = tf.image.resize_images(img, (200, 200))
# print(resizedata)

with tf.Session() as sess:
    fig, ax = plt.subplots(2, 1, figsize=(1, 2))

    ax[0].set_axis_off()
    ax[1].set_axis_off()

    ax[0].imshow(img)
    ax[1].imshow(resizedata.eval())
    print(resizedata.eval())
    plt.show()
    plt.close()
