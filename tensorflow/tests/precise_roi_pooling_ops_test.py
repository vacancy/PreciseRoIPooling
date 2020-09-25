#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : precise_roi_pooling_ops_test.py
# Author : Kanghee Lee
# Email  : lerohiso@gmail.com
# Date   : 09/25/2020
#
# This file is part of PreciseRoIPooling.
# Distributed under terms of the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util
from prroi_pool import PreciseRoIPooling


class PreciseRoIPoolingTest(test.TestCase):
    @test_util.run_gpu_only
    def test_forward(self):
        with self.test_session():
            with ops.device("/gpu:0"):
                pooled_width = 7
                pooled_height = 7
                spatial_scale = 0.5
                data_format = 'channels_first'
                pool = PreciseRoIPooling(pooled_height,
                                         pooled_width,
                                         spatial_scale=spatial_scale,
                                         data_format=data_format)
                features = tf.random.uniform([4, 16, 24, 32], dtype=tf.float32)
                rois = tf.constant([[0, 0, 0, 14, 14], [1, 14, 14, 28, 28]], dtype=tf.float32)
                operation_outputs = pool([features, rois])
                real_outputs = tf.keras.layers.AveragePooling2D(data_format=data_format, strides=1)(features)
                real_outputs = tf.stack([real_outputs[0, :, :7, :7], real_outputs[1, :, 7:14, 7:14]], axis=0)
                self.assertAllClose(operation_outputs, real_outputs)

    @test_util.run_gpu_only
    def test_backward(self):
        with self.test_session():
            with ops.device("/gpu:0"):
                pooled_width = 2
                pooled_height = 2
                spatial_scale = 0.5
                data_format = 'channels_first'
                base_directory = os.path.dirname(os.path.realpath(__file__))

                # binaries from pytorch prroi_pool module
                features = np.load(os.path.join(base_directory, 'test_binaries/2_2_0.5/features.npy'))
                rois = np.load(os.path.join(base_directory, 'test_binaries/2_2_0.5/rois.npy'))

                real_outputs = np.load(os.path.join(base_directory, 'test_binaries/2_2_0.5/real_outputs.npy'))
                real_gradients0 = np.load(os.path.join(base_directory, 'test_binaries/2_2_0.5/gradients0.npy'))
                real_gradients1 = np.load(os.path.join(base_directory, 'test_binaries/2_2_0.5/gradients1.npy'))
                features = tf.convert_to_tensor(features)
                rois = tf.convert_to_tensor(rois)
                with tf.GradientTape() as tape:
                    tape.watch([features, rois])
                    outputs = PreciseRoIPooling(pooled_height=pooled_height,
                                                pooled_width=pooled_width,
                                                spatial_scale=spatial_scale,
                                                data_format=data_format)([features, rois])
                    loss = tf.reduce_sum(outputs)

                gradients = tape.gradient(loss, [features, rois])

                self.assertAllClose(outputs, real_outputs)
                self.assertAllClose(gradients[0], real_gradients0)
                self.assertAllClose(gradients[1], real_gradients1)


if __name__ == '__main__':
    test.main()
