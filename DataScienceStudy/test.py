import tensorflow as tf
import matplotlib.image as mpimage
import matplotlib.pyplot as plt
import numpy as np

img = mpimage.imread('../paint/1-1.jpg')

localplt = plt.imshow(img)
plt.show(localplt)