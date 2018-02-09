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

class InceptionModule:

    def __init__(self, architecture='inception_v3', bottleneck_dir='./inception/bottleneck',
                 eval_step_interval=10, final_tensor_name='final_result', flip_left_right=False,
                 how_many_training_steps=1000, image_dir='', intermediate_output_graphs_dir='./inception/intermediate_graph/',
                 intermediate_store_frequency=0, learning_rate=0.01, model_dir='./inception/imagenet', output_graph='./inception/output_graph.pb',
                 output_labels='./inception/output_labels.txt', print_misclassified_test_images=False, random_brightness=0,
                 random_crop=0, random_scale=0, summaries_dir='./inception/retrain_logs', test_batch_size=-1, testing_percentage=10,
                 train_batch_size=500, validation_batch_size=100, validation_percentage=10, check_point_path='./inception/checkpoint', main_path='./inception'):
        self.architecture = architecture
        self.bottleneck_dir = bottleneck_dir
        self.eval_step_interval = eval_step_interval
        self.final_tensor_name = final_tensor_name
        self.flip_left_right = flip_left_right
        self.how_many_training_steps = how_many_training_steps
        self.image_dir = image_dir
        self.intermediate_output_graphs_dir = intermediate_output_graphs_dir
        self.intermediate_store_frequency = intermediate_store_frequency
        self.learning_rate = learning_rate
        self.model_dir = model_dir
        self.output_graph = output_graph
        self.output_labels = output_labels
        self.print_misclassified_test_images = print_misclassified_test_images
        self.random_brightness = random_brightness
        self.random_crop = random_crop
        self.random_scale = random_scale
        self.summaries_dir = summaries_dir
        self.test_batch_size = test_batch_size
        self.testing_percentage = testing_percentage
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.validation_percentage = validation_percentage

        self.MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

        # 학습 모델 저장
        self.check_point_path = check_point_path
        self.main_path = main_path

        if not os.path.exists(main_path):
            print('not exist main_path')
            self.init_data()
        else:
            try:
                with gfile.FastGFile(main_path+'/label_size.txt', 'r') as f:
                    self.label_size = int(f.read())
            except Exception as e:
                self.init_data()

        print('inception load OK')


    def create_image_lists(self, train_list, test_list, validation_list):
        # 수정 사항
        # 원본은 이미지 파일을 체크하여 레이블과 이미지 디렉토리 path 넘김
        # 이미지 url을 체크하여 존재하는 이미지 url만 처리하도록 수정
        # 우선적으로 파일에서 읽도록 처리, 추후에 DB에서 조회하도록 수정 필요
        # 테스트 데이터, 중간 체크 데이터 파일로 구분하도록 수정

        result = {}

        for image in train_list.keys():
            label = train_list[image]
            if label not in result.keys():
                training_images = []
                testing_images = []
                validation_images = []
                result[label] = {
                    'training': training_images,
                    'testing': testing_images,
                    'validation': validation_images,
                }
            checkdata = result[label]
            checkdata['training'].append(image)
            result[label] = checkdata

        for image in test_list.keys():
            label = test_list[image]
            if label not in result.keys():
                training_images = []
                testing_images = []
                validation_images = []
                result[label] = {
                    'training': training_images,
                    'testing': testing_images,
                    'validation': validation_images,
                }
            checkdata = result[label]
            checkdata['testing'].append(image)
            result[label] = checkdata

        for image in validation_list.keys():
            label = validation_list[image]
            if label not in result.keys():
                training_images = []
                testing_images = []
                validation_images = []
                result[label] = {
                    'training': training_images,
                    'testing': testing_images,
                    'validation': validation_images,
                }
            checkdata = result[label]
            checkdata['validation'].append(image)
            result[label] = checkdata

        return result

    def get_image_path(self, image_lists, label_name, index, image_dir, category):
        """" Returns a path to an image for a label at the given index.
        Args:
            image_lists: Dictionary of training images for each label.
            label_name: Label string we want to get an image for.
            index: Int offset of the image we want. This will be moduloed by the
            available number of images for the label, so it can be arbitrarily large.
            image_dir: Root folder string of the subfolders containing the training
            images.
            category: Name string of set to pull images from - training, testing, or
            validation.

        Returns:
            File system path string to an image that meets the requested parameters.
        """
        if label_name not in image_lists:
            tf.logging.fatal('Label does not exist %s.', label_name)
        label_lists = image_lists[label_name]
        if category not in label_lists:
            tf.logging.fatal('Category does not exist %s.', category)
        category_list = label_lists[category]
        if not category_list:
            tf.logging.fatal('Label %s has no images in the category %s.',
                             label_name, category)
        mod_index = index % len(category_list)
        base_name = category_list[mod_index]
        sub_dir = label_lists['dir']
        full_path = os.path.join(image_dir, sub_dir, base_name)
        return full_path

    def get_bottleneck_path(self, bottleneck_dir, label_name, index, category, architecture):
        """" Returns a path to a bottleneck file for a label at the given index.
        Args:
            image_lists: Dictionary of training images for each label.
            label_name: Label string we want to get an image for.
            index: Integer offset of the image we want. This will be moduloed by the
            available number of images for the label, so it can be arbitrarily large.
            bottleneck_dir: Folder string holding cached files of bottleneck values.
            category: Name string of set to pull images from - training, testing, or
            validation.
            architecture: The name of the model architecture.

        Returns:
            File system path string to an image that meets the requested parameters.
        """
        return bottleneck_dir + '/' + label_name + '_' + str(index) + '_' + category + '_' + architecture + '.txt'

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


    def maybe_download_and_extract(self, data_url):
        """ Download and extract model tar file.
        If the pretrained model we're using doesn't already exist, this function
        downloads it from the TensorFlow.org website and unpacks it into a directory.

        Args:
            data_url: Web location of the tar file containing the pretrained model.
        """
        dest_directory = self.model_dir
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = data_url.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' %
                                 (filename,
                                  float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
            statinfo = os.stat(filepath)
            tf.logging.info('Successfully downloaded', str(filename), str(statinfo.st_size), 'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)
        else:
            print('Not extracting or downloading files, model already present in disk')

    def ensure_dir_exists(self, dir_name):
        """ Makes sure the folder exists on disk.

        Args:
            dir_name: Path string to the folder we want to create.
        """
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    bottleneck_path_2_bottleneck_values = {}

    def create_bottleneck_file(self, bottleneck_path, image_url, sess, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor,
                               bottleneck_tensor):
        """Create a single bottleneck file."""
        tf.logging.info('Creating bottleneck at ' + bottleneck_path)
        # 이미지 데이터 파일 조회 -> url 조회로 변경

        try:
            req = urllib.request.Request(image_url)
            response = urllib.request.urlopen(req)
            image_data = response.read()
        except Exception as e:
            print(image_url)
            exit()

        try:
            bottleneck_values = self.run_bottleneck_on_image(
                sess, image_data, jpeg_data_tensor, decoded_image_tensor,
                resized_input_tensor, bottleneck_tensor)
        except Exception as e:
            raise RuntimeError('Error during processing file %s (%s)' % (image_url, str(e)))
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)

    def get_or_create_bottleneck(self, sess, label_name, index, image_url, category, bottleneck_dir, jpeg_data_tensor,
                                 decoded_image_tensor, resized_input_tensor,
                                 bottleneck_tensor, architecture):
        """ Retrieves or calculates bottleneck values for an image.

        If a cached version of the bottleneck data exists on-disk, return that,
        otherwise calculate the data and save it to disk for future use.

        Args:
            sess: The current active TensorFlow Session.
            image_lists: Dictionary of training images for each label.
            label_name: Label string we want to get an image for.
            index: Integer offset of the image we want. This will be modulo-ed by the
            available number of images for the label, so it can be arbitrarily large.
            image_dir: Root folder string of the subfolders containing the training images.
            category: Name string of which set to pull images from - training, testing, or validation.
            bottleneck_dir: Folder string holding cached files of bottleneck values.
            jpeg_data_tensor: The tensor to feed loaded jpeg data into.
            decoded_image_tensor: The output of decoding and resizing the image.
            resized_input_tensor: The input node of the recognition graph.
            bottleneck_tensor: The output tensor for the bottleneck values.
            architecture: The name of the model architecture.

        Returns:
            Numpy array of values produced by the bottleneck layer for the image.
        """

        # 이미지 사용을 위하여 이미지 데이터 로드 하는부분 수정
        bottleneck_path = self.get_bottleneck_path(bottleneck_dir, label_name, index, category, architecture)

        if not os.path.exists(bottleneck_path):
            self.create_bottleneck_file(bottleneck_path, image_url, sess, jpeg_data_tensor,
                                        decoded_image_tensor, resized_input_tensor,
                                        bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        did_hit_error = False
        try:
            bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
        except ValueError:
            tf.logging.warning('Invalid float found, recreating bottleneck')
            did_hit_error = True
        if did_hit_error:
            self.create_bottleneck_file(bottleneck_path, image_url, sess, jpeg_data_tensor,
                                        decoded_image_tensor, resized_input_tensor,
                                        bottleneck_tensor)
            with open(bottleneck_path, 'r') as bottleneck_file:
                bottleneck_string = bottleneck_file.read()
            # Allow exceptions to propagate here, since they shouldn't happen after a
            #  fresh creation
            bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
        return bottleneck_values

    def cache_bottlenecks(self, sess, image_lists, bottleneck_dir,
                          jpeg_data_tensor, decoded_image_tensor,
                          resized_input_tensor, bottleneck_tensor, architecture):
        """ Ensures all the training, testing, and validation bottlenecks are cached.

        Because we're likely to read the same image multiple times (if there are no
        distortions applied during training) it can speed things up a lot if we
        calculate the bottleneck layer values once for each image during
        preprocessing, and then just read those cached values repeatedly during
        training. Here we go through all the images we've found, calculate those
        values, and save them off.

        Args:
            sess: The current active TensorFlow Session.
            image_lists: Dictionary of training images for each label.
            image_dir: Root folder string of the subfolders containing the training images.
            bottleneck_dir: Folder string holding cached files of bottleneck values.
            jpeg_data_tensor: Input tensor for jpeg data from file.
            decoded_image_tensor: The output of decoding and resizing the image.
            resized_input_tensor: The input node of the recognition graph.
            bottleneck_tensor: The penultimate output layer of the graph.
            architecture: The name of the model architecture.

        Returns:
            Nothing.
        """
        how_many_bottlenecks = 0
        self.ensure_dir_exists(bottleneck_dir)
        for label_name, label_lists in image_lists.items():
            for category in ['training', 'testing', 'validation']:
                category_list = label_lists[category]
                for index, unused_base_name in enumerate(category_list):
                    self.get_or_create_bottleneck(
                        sess, label_name, index, unused_base_name, category,
                        bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                        resized_input_tensor, bottleneck_tensor, architecture)
                    how_many_bottlenecks += 1
                    if how_many_bottlenecks % 100 == 0:
                        tf.logging.info(str(how_many_bottlenecks) + ' bottleneck files created.')


    def get_random_cached_bottlenecks(self, sess, image_lists, how_many, category,
                                      bottleneck_dir, jpeg_data_tensor,
                                      decoded_image_tensor, resized_input_tensor,
                                      bottleneck_tensor, architecture):
        """ Retrieves bottleneck values for cached images.

        If no distortions are being applied, this function can retrieve the cached
        bottleneck values directly from disk for images. It picks a random set of
        images from the specified category.

        Args:
            sess: Current TensorFlow Session.
            image_lists: Dictionary of training images for each label.
            how_many: If positive, a random sample of this size will be chosen. If negative, all bottlenecks will be retrieved.
            category: Name string of which set to pull from - training, testing, or validation.
            bottleneck_dir: Folder string holding cached files of bottleneck values.
            image_dir: Root folder string of the subfolders containing the training images.
            jpeg_data_tensor: The layer to feed jpeg image data into.
            decoded_image_tensor: The output of decoding and resizing the image.
            resized_input_tensor: The input node of the recognition graph.
            bottleneck_tensor: The bottleneck output layer of the CNN graph.
            architecture: The name of the model architecture.

        Returns:
            List of bottleneck arrays, their corresponding ground truths, and the
            relevant filenames.
        """
        class_count = len(image_lists.keys())
        bottlenecks = []
        ground_truths = []
        if how_many >= 0:
            # Retrieve a random sample of bottlenecks.
            for unused_i in range(how_many):
                label_index = random.randrange(class_count)
                label_name = list(image_lists.keys())[label_index]

                image_index = random.randrange(len(image_lists[label_name][category]))
                bottleneck = self.get_or_create_bottleneck(
                    sess, label_name, image_index, image_lists[label_name][category][image_index], category,
                    bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                    resized_input_tensor, bottleneck_tensor, architecture)

                bottlenecks.append(bottleneck)
                ground_truths.append(label_index)

        else:
            # Retrieve all bottlenecks.
            for label_index, label_name in enumerate(image_lists.keys()):
                for image_index, image_name in enumerate(image_lists[label_name][category]):
                    bottleneck = self.get_or_create_bottleneck(
                        sess, label_name, image_index, image_name, category,
                        bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                        resized_input_tensor, bottleneck_tensor, architecture)
                    bottlenecks.append(bottleneck)
                    ground_truths.append(label_index)

        return bottlenecks, ground_truths


    def get_random_distorted_bottlenecks(
            self, sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
            distorted_image, resized_input_tensor, bottleneck_tensor):
        """ Retrieves bottleneck values for training images, after distortions.

        If we're training with distortions like crops, scales, or flips, we have to
        recalculate the full model for every image, and so we can't use cached
        bottleneck values. Instead we find random images for the requested category,
        run them through the distortion graph, and then the full graph to get the
        bottleneck results for each.

        Args:
            sess: Current TensorFlow Session.
            image_lists: Dictionary of training images for each label.
            how_many: The integer number of bottleneck values to return.
            category: Name string of which set of images to fetch - training, testing, or validation.
            image_dir: Root folder string of the subfolders containing the training images.
            input_jpeg_tensor: The input layer we feed the image data to.
            distorted_image: The output node of the distortion graph.
            resized_input_tensor: The input node of the recognition graph.
            bottleneck_tensor: The bottleneck output layer of the CNN graph.

        Returns:
            List of bottleneck arrays and their corresponding ground truths.
        """
        class_count = len(image_lists.keys())
        bottlenecks = []
        ground_truths = []
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(self.MAX_NUM_IMAGES_PER_CLASS + 1)
            image_path = self.get_image_path(image_lists, label_name, image_index, image_dir,
                                        category)
            if not gfile.Exists(image_path):
                tf.logging.fatal('File does not exist %s', image_path)
            jpeg_data = gfile.FastGFile(image_path, 'rb').read()
            # Note that we materialize the distorted_image_data as a numpy array before
            # sending running inference on the image. This involves 2 memory copies and
            # might be optimized in other implementations.
            distorted_image_data = sess.run(distorted_image,
                                            {input_jpeg_tensor: jpeg_data})
            bottleneck_values = sess.run(bottleneck_tensor,
                                         {resized_input_tensor: distorted_image_data})
            bottleneck_values = np.squeeze(bottleneck_values)
            bottlenecks.append(bottleneck_values)
            ground_truths.append(label_index)
        return bottlenecks, ground_truths


    def should_distort_images(self, flip_left_right, random_crop, random_scale,
                              random_brightness):
      """Whether any distortions are enabled, from the input flags.

      Args:
        flip_left_right: Boolean whether to randomly mirror images horizontally.
        random_crop: Integer percentage setting the total margin used around the
        crop box.
        random_scale: Integer percentage of how much to vary the scale by.
        random_brightness: Integer range to randomly multiply the pixel values by.

      Returns:
        Boolean value indicating whether any distortions should be applied.
      """
      return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
              (random_brightness != 0))


    def add_input_distortions(self, flip_left_right, random_crop, random_scale,
                              random_brightness, input_width, input_height,
                              input_depth, input_mean, input_std):
        """ Creates the operations to apply the specified distortions.

        During training it can help to improve the results if we run the images
        through simple distortions like crops, scales, and flips. These reflect the
        kind of variations we expect in the real world, and so can help train the
        model to cope with natural data more effectively. Here we take the supplied
        parameters and construct a network of operations to apply them to an image.

        Cropping
        ~~~~~~~~

        Cropping is done by placing a bounding box at a random position in the full
        image. The cropping parameter controls the size of that box relative to the
        input image. If it's zero, then the box is the same size as the input and no
        cropping is performed. If the value is 50%, then the crop box will be half the
        width and height of the input. In a diagram it looks like this:

        <       width         >
        +---------------------+
        |                     |
        |   width - crop%     |
        |    <      >         |
        |    +------+         |
        |    |      |         |
        |    |      |         |
        |    |      |         |
        |    +------+         |
        |                     |
        |                     |
        +---------------------+

        Scaling
        ~~~~~~~

        Scaling is a lot like cropping, except that the bounding box is always
        centered and its size varies randomly within the given range. For example if
        the scale percentage is zero, then the bounding box is the same size as the
        input and no scaling is applied. If it's 50%, then the bounding box will be in
        a random range between half the width and height and full size.

        Args:
            flip_left_right: Boolean whether to randomly mirror images horizontally.
            random_crop: Integer percentage setting the total margin used around the crop box.
            random_scale: Integer percentage of how much to vary the scale by.
            random_brightness: Integer range to randomly multiply the pixel values by. graph.
            input_width: Horizontal size of expected input image to model.
            input_height: Vertical size of expected input image to model.
            input_depth: How many channels the expected input image should have.
            input_mean: Pixel value that should be zero in the image for the graph.
            input_std: How much to divide the pixel values by before recognition.

        Returns:
            The jpeg input layer and the distorted result tensor.
        """

        jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
        decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
        decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
        decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
        margin_scale = 1.0 + (random_crop / 100.0)
        resize_scale = 1.0 + (random_scale / 100.0)
        margin_scale_value = tf.constant(margin_scale)
        resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
                                               minval=1.0,
                                               maxval=resize_scale)
        scale_value = tf.multiply(margin_scale_value, resize_scale_value)
        precrop_width = tf.multiply(scale_value, input_width)
        precrop_height = tf.multiply(scale_value, input_height)
        precrop_shape = tf.stack([precrop_height, precrop_width])
        precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
        precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                                    precrop_shape_as_int)
        precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
        cropped_image = tf.random_crop(precropped_image_3d,
                                       [input_height, input_width, input_depth])
        if flip_left_right:
            flipped_image = tf.image.random_flip_left_right(cropped_image)
        else:
            flipped_image = cropped_image
        brightness_min = 1.0 - (random_brightness / 100.0)
        brightness_max = 1.0 + (random_brightness / 100.0)
        brightness_value = tf.random_uniform(tensor_shape.scalar(),
                                             minval=brightness_min,
                                             maxval=brightness_max)
        brightened_image = tf.multiply(flipped_image, brightness_value)
        offset_image = tf.subtract(brightened_image, input_mean)
        mul_image = tf.multiply(offset_image, 1.0 / input_std)
        distort_result = tf.expand_dims(mul_image, 0, name='DistortResult')
        return jpeg_data, distort_result


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


    def save_graph_to_file(self, sess, graph, graph_file_name):
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), [self.final_tensor_name])

        with gfile.FastGFile(graph_file_name, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        return


    def prepare_file_system(self):
        # Setup the directory we'll write summaries to for TensorBoard
        if tf.gfile.Exists(self.summaries_dir):
            tf.gfile.DeleteRecursively(self.summaries_dir)
        tf.gfile.MakeDirs(self.summaries_dir)
        if self.intermediate_store_frequency > 0:
            self.ensure_dir_exists(self.intermediate_output_graphs_dir)
        return


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

    def init_model(self):
        # Gather information about the model architecture we'll be using.
        model_info = self.create_model_info(self.architecture)
        if not model_info:
            tf.logging.error('Did not recognize architecture flag')
            return -1

        # Set up the pre-trained graph.
        self.maybe_download_and_extract(model_info['data_url'])

    def train(self, image_lists, is_init_data=False):
        print('train start =', str(len(image_lists)))
        '''
        학습 로직
        :param image_lists: 학습 이미지 데이터(create_image_lists을 이용해서 생성)
        :param is_init_data: 초기화일 경우 label명을 파일에 써야하기 때문에 초기화일 경우 True
        :return:
        '''
        # Gather information about the model architecture we'll be using.
        model_info = self.create_model_info(self.architecture)
        if not model_info:
            tf.logging.error('Did not recognize architecture flag')
            return -1

        # Set up the pre-trained graph.
        self.maybe_download_and_extract(model_info['data_url'])

        graph, bottleneck_tensor, resized_image_tensor = (self.create_model_graph(model_info))
        class_count = len(image_lists.keys())
        if class_count == 0:
            tf.logging.error('No valid folders of images found at ' + self.image_dir)
            return -1
        if class_count == 1:
            tf.logging.error('Only one valid folder of images found at ' +
                             self.image_dir +
                             ' - multiple classes are needed for classification.')
            return -1

        with tf.Session(graph=graph) as sess:
            # Set up the image decoding sub-graph.
            jpeg_data_tensor, decoded_image_tensor = self.add_jpeg_decoding(
                model_info['input_width'], model_info['input_height'],
                model_info['input_depth'], model_info['input_mean'],
                model_info['input_std'])

            # We'll make sure we've calculated the 'bottleneck' image summaries and
            # cached them on disk.
            self.cache_bottlenecks(sess, image_lists,
                                   self.bottleneck_dir, jpeg_data_tensor,
                                   decoded_image_tensor, resized_image_tensor,
                                   bottleneck_tensor, self.architecture)

            # Add the new layer that we'll be training.
            (train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor, global_step) = self.add_final_training_ops(
                len(image_lists.keys()), self.final_tensor_name, bottleneck_tensor,
                model_info['bottleneck_tensor_size'], model_info['quantize_layer'])

            # Create the operations we need to evaluate the accuracy of our new layer.
            evaluation_step, prediction = self.add_evaluation_step(
                final_tensor, ground_truth_input)

            # Merge all the summaries and write them out to the summaries_dir
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.summaries_dir + '/train',
                                                 sess.graph)

            validation_writer = tf.summary.FileWriter(self.summaries_dir + '/validation')

            # Set up all our weights to their initial default values.
            check_point = tf.train.get_checkpoint_state(self.check_point_path)
            saver = tf.train.Saver(tf.global_variables())
            if check_point and tf.train.checkpoint_exists(check_point.model_checkpoint_path) and not is_init_data:
                saver.restore(sess, check_point.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            # Run the training for as many cycles as requested on the command line.
            for i in range(self.how_many_training_steps):
                # Get a batch of input bottleneck values, either calculated fresh every
                #  time with distortions applied, or from the cache stored on disk.
                (train_bottlenecks, train_ground_truth) = self.get_random_cached_bottlenecks(
                        sess, image_lists, self.train_batch_size, 'training',
                        self.bottleneck_dir, jpeg_data_tensor,
                        decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                        self.architecture)

                # Feed the bottlenecks and ground truth into the graph, and run a training
                # step. Capture training summaries for TensorBoard with the `merged` op.
                train_summary, _ = sess.run(
                    [merged, train_step],
                    feed_dict={bottleneck_input: train_bottlenecks,
                               ground_truth_input: train_ground_truth})

                train_writer.add_summary(train_summary, i)

                # Every so often, print out how well the graph is training.
                is_last_step = (i + 1 == self.how_many_training_steps)
                if (i % self.eval_step_interval) == 0 or is_last_step:
                    train_accuracy, cross_entropy_value = sess.run(
                        [evaluation_step, cross_entropy],
                        feed_dict={bottleneck_input: train_bottlenecks,
                                   ground_truth_input: train_ground_truth})
                    tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
                                    (datetime.now(), i, train_accuracy * 100))
                    tf.logging.info('%s: Step %d: Cross entropy = %f' %
                                    (datetime.now(), sess.run(global_step), cross_entropy_value))
                    validation_bottlenecks, validation_ground_truth = (
                        self.get_random_cached_bottlenecks(
                            sess, image_lists, self.validation_batch_size, 'validation',
                            self.bottleneck_dir, jpeg_data_tensor,
                            decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                            self.architecture))
                    # Run a validation step and capture training summaries for TensorBoard
                    # with the `merged` op.
                    validation_summary, validation_accuracy, prediction_result = sess.run(
                        [merged, evaluation_step, prediction],
                        feed_dict={bottleneck_input: validation_bottlenecks,
                                   ground_truth_input: validation_ground_truth})
                    validation_writer.add_summary(validation_summary, i)
                    tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                                    (datetime.now(), i, validation_accuracy * 100,
                                     len(validation_bottlenecks)))
                # Store intermediate results
                intermediate_frequency = self.intermediate_store_frequency

                if (intermediate_frequency > 0 and (i % intermediate_frequency == 0)
                        and i > 0):
                    intermediate_file_name = (self.intermediate_output_graphs_dir +
                                              'intermediate_' + str(i) + '.pb')
                    tf.logging.info('Save intermediate result to : ' +
                                    intermediate_file_name)
                    self.save_graph_to_file(sess, graph, intermediate_file_name)

            # We've completed all our training, so run a final test evaluation on
            # some new images we haven't used before.
            test_bottlenecks, test_ground_truth = (
                self.get_random_cached_bottlenecks(
                    sess, image_lists, self.test_batch_size, 'testing',
                    self.bottleneck_dir, jpeg_data_tensor,
                    decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                    self.architecture))
            test_accuracy = sess.run(
                evaluation_step,
                feed_dict={bottleneck_input: test_bottlenecks,
                           ground_truth_input: test_ground_truth})
            tf.logging.info('Final test accuracy = %.1f%% (N=%d)' %
                            (test_accuracy * 100, len(test_bottlenecks)))

            self.save_graph_to_file(sess, graph, self.output_graph)

            if is_init_data:
                with gfile.FastGFile(self.output_labels, 'w') as f:
                    f.write('\n'.join(image_lists.keys()) + '\n')
                self.label_size = len(image_lists.keys())
                with gfile.FastGFile(self.main_path+'/label_size.txt', 'w') as f:
                    f.write(str(self.label_size))

            saver.save(sess, self.check_point_path+'/inception.ckpt', global_step=global_step)

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

        # Set up the pre-trained graph.
        self.maybe_download_and_extract(model_info['data_url'])

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

            predictions = sess.run(prediction, feed_dict={bottleneck_input: predict_input_list})

            print(predict_images)
            print(predictions)

            result = []
            for i in range(len(predictions)):
                print(str(predict_images[i]) , '->', str(label_data[predictions[i]]))
                result.append(label_data[predictions[i]])
        return result


    def init_data(self):
        # Needed to make sure the logging output is visible.
        #  See https://github.com/tensorflow/tensorflow/issues/3047
        tf.logging.set_verbosity(tf.logging.INFO)

        # Prepare necessary directories that can be used during training
        # 기존 데이터 삭제
        self.prepare_file_system()

        # 모델이 없는 경우 모델 다운로드
        self.init_model()

        # 테스트시 사용할 이미지 데이터 조회
        # 대량 테스트를 위해서 추후 DB연결로 변경
        print('data load')
        train_image_data = {}
        try:
            with open("./trainlist_15253_check.txt", 'r') as f:
                while True:
                    line = f.readline()
                    if not line: break
                    line = line.replace('\n','')
                    line = line.replace(' ', '')
                    key, value = line.split('\t')

                    # try:
                    #     urllib.request.urlopen(value)
                    # except Exception:
                    #     print('trainlist invalid url', value)
                    #     continue

                    train_image_data[value] = key
                    if len(train_image_data) % 100 == 0:
                        print('train process in', str(len(train_image_data)))
        except FileNotFoundError:
            return -1

        test_image_data = {}
        try:
            with open("./testlist_15253_check.txt", 'r') as f:
                while True:
                    line = f.readline()
                    if not line: break
                    line = line.replace('\n', '')
                    line = line.replace(' ', '')
                    key, value = line.split('\t')

                    # try:
                    #     urllib.request.urlopen(value)
                    # except Exception:
                    #     print('testlist invalid url', value)
                    #     continue

                    test_image_data[value] = key
                    if len(test_image_data) % 100 == 0:
                        print('test process in', str(len(test_image_data)))
        except FileNotFoundError:
            return -1

        validation_image_data = {}
        try:
            with open("./validlist_15253_check.txt", 'r') as f:
                while True:
                    line = f.readline()
                    if not line: break
                    line = line.replace('\n', '')
                    line = line.replace(' ', '')
                    key, value = line.split('\t')

                    # try:
                    #     urllib.request.urlopen(value)
                    # except Exception:
                    #     print('validlist invalid url', value)
                    #     continue

                    validation_image_data[value] = key
                    if len(validation_image_data) % 100 == 0:
                        print('valid process in', str(len(validation_image_data)))
        except FileNotFoundError:
            return -1

        # Look at the folder structure, and create lists of all the images.
        image_lists = self.create_image_lists(train_image_data, test_image_data, validation_image_data)

        self.train(image_lists, True)
