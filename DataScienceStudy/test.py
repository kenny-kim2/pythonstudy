import tensorflow as tf
import matplotlib.image as mpimage
import numpy as np


data = np.random.normal(scale = 10, size=(10,10))
dataresult = tf.reshape(data, (2, 5, 10))

print(dataresult)