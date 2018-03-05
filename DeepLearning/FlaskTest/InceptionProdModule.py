# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Simple transfer learning with Inception v3 or Mobilenet models.

With support for TensorBoard.

This example shows how to take a Inception v3 or Mobilenet model trained on
ImageNet images, and train a new top layer that can recognize other classes of
images.

The top layer receives as input a 2048-dimensional vector (1001-dimensional for
Mobilenet) for each image. We train a softmax layer on top of this
representation. Assuming the softmax layer contains N labels, this corresponds
to learning N + 2048*N (or 1001*N)  model parameters corresponding to the
learned biases and weights.

Here's an example, which assumes you have a folder containing class-named
subfolders, each full of images for each label. The example folder flower_photos
should have a structure like this:

~/flower_photos/daisy/photo1.jpg
~/flower_photos/daisy/photo2.jpg
...
~/flower_photos/rose/anotherphoto77.jpg
...
~/flower_photos/sunflower/somepicture.jpg

The subfolder names are important, since they define what label is applied to
each image, but the filenames themselves don't matter. Once your images are
prepared, you can run the training with a command like this:


```bash
bazel build tensorflow/examples/image_retraining:retrain && \
bazel-bin/tensorflow/examples/image_retraining/retrain \
    --image_dir ~/flower_photos
```

Or, if you have a pip installation of tensorflow, `retrain.py` can be run
without bazel:

```bash
python tensorflow/examples/image_retraining/retrain.py \
    --image_dir ~/flower_photos
```

You can replace the image_dir argument with any folder containing subfolders of
images. The label for each image is taken from the name of the subfolder it's
in.

This produces a new model file that can be loaded and run by any TensorFlow
program, for example the label_image sample code.

By default this script will use the high accuracy, but comparatively large and
slow Inception v3 model architecture. It's recommended that you start with this
to validate that you have gathered good training data, but if you want to deploy
on resource-limited platforms, you can try the `--architecture` flag with a
Mobilenet model. For example:

Run floating-point version of mobilenet:
```bash
python tensorflow/examples/image_retraining/retrain.py \
    --image_dir ~/flower_photos --architecture mobilenet_1.0_224
```

Run quantized version of mobilenet:
```bash
python tensorflow/examples/image_retraining/retrain.py \
    --image_dir ~/flower_photos/   --architecture mobilenet_1.0_224_quantized
```

There are 32 different Mobilenet models to choose from, with a variety of file
size and latency options. The first number can be '1.0', '0.75', '0.50', or
'0.25' to control the size, and the second controls the input image size, either
'224', '192', '160', or '128', with smaller sizes running faster. See
https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html
for more information on Mobilenet.

To use with TensorBoard:

By default, this script will log summaries to /tmp/retrain_logs directory

Visualize the summaries with this command:

tensorboard --logdir /tmp/retrain_logs

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import tarfile

import urllib.request

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.contrib.quantize.python import quant_ops
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

class InceptionProdModule:

    def __init__(self, architecture='inception_v3',
                 eval_step_interval=10, final_tensor_name='final_result', flip_left_right=False,
                 how_many_training_steps=2000, image_dir='',
                 intermediate_store_frequency=0, learning_rate=0.8, model_dir='./inception_prod/imagenet', output_graph='./inception_prod/output_graph.pb',
                 output_labels='./inception_prod/output_labels.txt', print_misclassified_test_images=False, random_brightness=0,
                 random_crop=0, random_scale=0, test_batch_size=-1,
                 train_batch_size=200, validation_batch_size=100, check_point_path='./inception_prod/checkpoint', main_path='./inception_prod'):
        self.architecture = architecture
        self.eval_step_interval = eval_step_interval
        self.final_tensor_name = final_tensor_name
        self.flip_left_right = flip_left_right
        self.how_many_training_steps = how_many_training_steps
        self.image_dir = image_dir
        self.intermediate_store_frequency = intermediate_store_frequency
        self.learning_rate = learning_rate
        self.model_dir = model_dir
        self.output_graph = output_graph
        self.output_labels = output_labels
        self.print_misclassified_test_images = print_misclassified_test_images
        self.random_brightness = random_brightness
        self.random_crop = random_crop
        self.random_scale = random_scale
        self.test_batch_size = test_batch_size
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size

        self.MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

        # 학습 모델 저장
        self.check_point_path = check_point_path
        self.main_path = main_path

        if not os.path.exists(main_path):
            print('not exist main_path')
        else:
            try:
                with gfile.FastGFile(main_path+'/label_size.txt', 'r') as f:
                    self.label_size = int(f.read())
            except Exception as e:
                print('label data error')

        print('inception load OK')

    def create_model_graph(self, model_info):
        """" Creates a graph from saved GraphDef file and returns a Graph object.

        Args:
        model_info: Dictionary containing information about the model architecture.

        Returns:
            Graph holding the trained Inception network, and various tensors we'll be
            manipulating.
        """
        with tf.Graph().as_default() as graph:
            model_path = os.path.join(self.model_dir, model_info['model_file_name'])
            with gfile.FastGFile(model_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
                    graph_def,
                    name='',
                    return_elements=[
                        model_info['bottleneck_tensor_name'],
                        model_info['resized_input_tensor_name'],
                    ]))
        return graph, bottleneck_tensor, resized_input_tensor


    def run_bottleneck_on_image(self, sess, image_data, image_data_tensor,
                                decoded_image_tensor, resized_input_tensor,
                                bottleneck_tensor):
        """ Runs inference on an image to extract the 'bottleneck' summary layer.
        Args:
            sess: Current active TensorFlow Session.
            image_data: String of raw JPEG data.
            image_data_tensor: Input data layer in the graph.
            decoded_image_tensor: Output of initial image resizing and preprocessing.
            resized_input_tensor: The input node of the recognition graph.
            bottleneck_tensor: Layer before the final softmax.

        Returns:
            Numpy array of bottleneck values.
        """
        # First decode the JPEG image, resize it, and rescale the pixel values.
        resized_input_values = sess.run(decoded_image_tensor,
                                        {image_data_tensor: image_data})
        # Then run it through the recognition network.
        bottleneck_values = sess.run(bottleneck_tensor,
                                     {resized_input_tensor: resized_input_values})
        bottleneck_values = np.squeeze(bottleneck_values)
        return bottleneck_values

    def variable_summaries(self, var):
        """ Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


    def add_final_training_ops(self, class_count, final_tensor_name, bottleneck_tensor,
                               bottleneck_tensor_size, quantize_layer):
        """ Adds a new softmax and fully-connected layer for training.

        We need to retrain the top layer to identify our new classes, so this function
        adds the right operations to the graph, along with some variables to hold the
        weights, and then sets up all the gradients for the backward pass.

        The set up for the softmax and fully-connected layers is based on:
        https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

        Args:
            class_count: Integer of how many categories of things we're trying to recognize.
            final_tensor_name: Name string for the new final node that produces results.
            bottleneck_tensor: The output of the main CNN graph.
            bottleneck_tensor_size: How many entries in the bottleneck vector.
            quantize_layer: Boolean, specifying whether the newly added layer should be quantized.

        Returns:
            The tensors for the training and cross entropy results, and tensors for the
            bottleneck input and ground truth input.
        """
        with tf.name_scope('input'):
            bottleneck_input = tf.placeholder_with_default(
                bottleneck_tensor,
                shape=[None, bottleneck_tensor_size],
                name='BottleneckInputPlaceholder')

            ground_truth_input = tf.placeholder(
                tf.int64, [None], name='GroundTruthInput')

            global_step = tf.Variable(0, trainable=False, name='global_step')

        # Organizing the following ops as `final_training_ops` so they're easier
        #  to see in TensorBoard
        layer_name = 'final_training_ops'
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                initial_value = tf.truncated_normal(
                    [bottleneck_tensor_size, class_count], stddev=0.001)
                layer_weights = tf.Variable(initial_value, name='final_weights')
                if quantize_layer:
                    quantized_layer_weights = quant_ops.MovingAvgQuantize(
                        layer_weights, is_training=True)
                    self.variable_summaries(quantized_layer_weights)

                self.variable_summaries(layer_weights)
            with tf.name_scope('biases'):
                layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
                if quantize_layer:
                    quantized_layer_biases = quant_ops.MovingAvgQuantize(
                        layer_biases, is_training=True)
                    self.variable_summaries(quantized_layer_biases)
                self.variable_summaries(layer_biases)

            with tf.name_scope('Wx_plus_b'):
                if quantize_layer:
                    logits = tf.matmul(bottleneck_input,
                                       quantized_layer_weights) + quantized_layer_biases
                    logits = quant_ops.MovingAvgQuantize(
                        logits,
                        init_min=-32.0,
                        init_max=32.0,
                        is_training=True,
                        num_bits=8,
                        narrow_range=False,
                        ema_decay=0.5)
                    tf.summary.histogram('pre_activations', logits)
                else:
                    logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
                    tf.summary.histogram('pre_activations', logits)
        final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

        tf.summary.histogram('activations', final_tensor)

        with tf.name_scope('cross_entropy'):
            cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
                labels=ground_truth_input, logits=logits)
        tf.summary.scalar('cross_entropy', cross_entropy_mean)

        with tf.name_scope('train'):
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            train_step = optimizer.minimize(cross_entropy_mean, global_step=global_step)

        return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
                final_tensor, global_step)


    def add_evaluation_step(self, result_tensor, ground_truth_tensor):
        """ Inserts the operations we need to evaluate the accuracy of our results.

        Args:
            result_tensor: The new final node that produces results.
            ground_truth_tensor: The node we feed ground truth data into.

        Returns:
            Tuple of (evaluation step, prediction).
        """
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                prediction = tf.argmax(result_tensor, 1)
                correct_prediction = tf.equal(prediction, ground_truth_tensor)
            with tf.name_scope('accuracy'):
                evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', evaluation_step)
        return evaluation_step, prediction

    def create_model_info(self, architecture):
        """Given the name of a model architecture, returns information about it.

        There are different base image recognition pretrained models that can be
        retrained using transfer learning, and this function translates from the name
        of a model to the attributes that are needed to download and train with it.

        Args:
            architecture: Name of a model architecture.

        Returns:
            Dictionary of information about the model, or None if the name isn't recognized

        Raises:
            ValueError: If architecture name is unknown.
        """
        architecture = architecture.lower()
        is_quantized = False
        if architecture == 'inception_v3':
            # pylint: disable=line-too-long
            data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
            # pylint: enable=line-too-long
            bottleneck_tensor_name = 'pool_3/_reshape:0'
            bottleneck_tensor_size = 2048
            input_width = 299
            input_height = 299
            input_depth = 3
            resized_input_tensor_name = 'Mul:0'
            model_file_name = 'classify_image_graph_def.pb'
            input_mean = 128
            input_std = 128
        elif architecture.startswith('mobilenet_'):
            parts = architecture.split('_')
            if len(parts) != 3 and len(parts) != 4:
                tf.logging.error("Couldn't understand architecture name '%s'",
                                 architecture)
                return None

            version_string = parts[1]
            if (version_string != '1.0' and version_string != '0.75' and
                    version_string != '0.50' and version_string != '0.25'):
                tf.logging.error(
                    """"The Mobilenet version should be '1.0', '0.75', '0.50', or '0.25',
                    but found '%s' for architecture '%s'""",
                    version_string, architecture)
                return None

            size_string = parts[2]
            if (size_string != '224' and size_string != '192' and
                    size_string != '160' and size_string != '128'):
                tf.logging.error(
                    """The Mobilenet input size should be '224', '192', '160', or '128',
                     but found '%s' for architecture '%s'""",
                    size_string, architecture)
                return None

            if len(parts) == 3:
                is_quantized = False
            else:
                if parts[3] != 'quantized':
                    tf.logging.error(
                        "Couldn't understand architecture suffix '%s' for '%s'", parts[3],
                        architecture)
                    return None
                is_quantized = True

            if is_quantized:
                data_url = 'http://download.tensorflow.org/models/mobilenet_v1_'
                data_url += version_string + '_' + size_string + '_quantized_frozen.tgz'
                bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
                resized_input_tensor_name = 'Placeholder:0'
                model_dir_name = ('mobilenet_v1_' + version_string + '_' + size_string +
                                  '_quantized_frozen')
                model_base_name = 'quantized_frozen_graph.pb'
            else:
                data_url = 'http://download.tensorflow.org/models/mobilenet_v1_'
                data_url += version_string + '_' + size_string + '_frozen.tgz'
                bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
                resized_input_tensor_name = 'input:0'
                model_dir_name = 'mobilenet_v1_' + version_string + '_' + size_string
                model_base_name = 'frozen_graph.pb'

            bottleneck_tensor_size = 1001
            input_width = int(size_string)
            input_height = int(size_string)
            input_depth = 3
            model_file_name = os.path.join(model_dir_name, model_base_name)
            input_mean = 127.5
            input_std = 127.5
        else:
            tf.logging.error("Couldn't understand architecture name '%s'", architecture)
            raise ValueError('Unknown architecture', architecture)

        return {
            'data_url': data_url,
            'bottleneck_tensor_name': bottleneck_tensor_name,
            'bottleneck_tensor_size': bottleneck_tensor_size,
            'input_width': input_width,
            'input_height': input_height,
            'input_depth': input_depth,
            'resized_input_tensor_name': resized_input_tensor_name,
            'model_file_name': model_file_name,
            'input_mean': input_mean,
            'input_std': input_std,
            'quantize_layer': is_quantized,
        }


    def add_jpeg_decoding(self, input_width, input_height, input_depth, input_mean,
                          input_std):
        """ Adds operations that perform JPEG decoding and resizing to the graph..

        Args:
            input_width: Desired width of the image fed into the recognizer graph.
            input_height: Desired width of the image fed into the recognizer graph.
            input_depth: Desired channels of the image fed into the recognizer graph.
            input_mean: Pixel value that should be zero in the image for the graph.
            input_std: How much to divide the pixel values by before recognition.

        Returns:
            Tensors for the node to feed JPEG data into, and the output of the preprocessing steps.
        """
        jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
        decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
        decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
        decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
        resize_shape = tf.stack([input_height, input_width])
        resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
        resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                                 resize_shape_as_int)
        offset_image = tf.subtract(resized_image, input_mean)
        mul_image = tf.multiply(offset_image, 1.0 / input_std)
        return jpeg_data, mul_image

    def predict(self, predict_images):
        label_data = []
        result = []
        with gfile.FastGFile(self.output_labels, 'r') as f:
            label_data = str(f.read()).split('\n')

        # Gather information about the model architecture we'll be using.
        model_info = self.create_model_info(self.architecture)
        if not model_info:
            tf.logging.error('Did not recognize architecture flag')
            return -1

        graph, bottleneck_tensor, resized_image_tensor = (self.create_model_graph(model_info))

        with tf.Session(graph=graph) as sess:
            # Set up the image decoding sub-graph.
            jpeg_data_tensor, decoded_image_tensor = self.add_jpeg_decoding(
                model_info['input_width'], model_info['input_height'],
                model_info['input_depth'], model_info['input_mean'],
                model_info['input_std'])

            # Add the new layer that we'll be training.
            (train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor, global_step) = self.add_final_training_ops(
                self.label_size, self.final_tensor_name, bottleneck_tensor,
                model_info['bottleneck_tensor_size'], model_info['quantize_layer'])

            # Create the operations we need to evaluate the accuracy of our new layer.
            evaluation_step, prediction = self.add_evaluation_step(
                final_tensor, ground_truth_input)

            # Set up all our weights to their initial default values.
            check_point = tf.train.get_checkpoint_state(self.check_point_path)
            saver = tf.train.Saver(tf.global_variables())
            if check_point and tf.train.checkpoint_exists(check_point.model_checkpoint_path):
                saver.restore(sess, check_point.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            predict_input_list = []

            for image in predict_images:
                req = urllib.request.Request(image)
                response = urllib.request.urlopen(req)
                image_data = response.read()
                predict_input_list.append(self.run_bottleneck_on_image(
                    sess, image_data, jpeg_data_tensor, decoded_image_tensor,
                    resized_image_tensor, bottleneck_tensor))

            predictions, final_tensor_data = sess.run([prediction, final_tensor], feed_dict={bottleneck_input: predict_input_list})

            print(predict_images)
            print(predictions)

            result = []
            for i in range(len(predictions)):
                print(str(predict_images[i]) , '->', str(label_data[predictions[i]]))
                print(final_tensor_data[i][predictions[i]])
                if final_tensor_data[i][predictions[i]] > 0.5:
                    result.append(label_data[predictions[i]])
                else:
                    result.append(0)
        return result
