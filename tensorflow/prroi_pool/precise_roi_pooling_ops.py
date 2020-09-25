#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : precise_roi_pooling_ops.py
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
import platform
import tensorflow as tf

__all__ = ['PreciseRoIPooling']

os_type = platform.system()
if os_type == 'Windows':
    MODULE_NAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'module/precise_roi_pooling.dll')
elif os_type == 'Linux':
    MODULE_NAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'module/libprecise_roi_pooling.so')

_precise_roi_pooling_ops = tf.load_op_library(MODULE_NAME)

def _precise_roi_pooling(features,
                         rois,
                         pooled_height,
                         pooled_width,
                         spatial_scale,
                         data_format,
                         name=None):
    with tf.name_scope(name or "precise_roi_pooling"):
        op_call = _precise_roi_pooling_ops.precise_ro_i_pooling

        if data_format == 'channels_last':
            inputs = tf.transpose(features, [0, 3, 1, 2])
        elif data_format == "channels_first":
            inputs = features
        else:
            raise ValueError('`data_format` must be either `channels_last` or `channels_first`')

        outputs = op_call(inputs,
                          rois,
                          pooled_height=pooled_height,
                          pooled_width=pooled_width,
                          spatial_scale=spatial_scale,
                          data_format='NCHW')

        if data_format == 'channels_last':
            return tf.transpose(outputs, [0, 2, 3, 1])

        return outputs

class PreciseRoIPooling(tf.keras.layers.Layer):
    def __init__(self,
                 pooled_height: int,
                 pooled_width: int,
                 spatial_scale: float,
                 data_format: str = 'channels_first',
                 **kwargs):
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        self.spatial_scale = spatial_scale

        if data_format != 'channels_last' and data_format != 'channels_first':
            raise ValueError('`data_format` must be either `channels_last` or'
                             '`channels_first`, instead got %s' % data_format)

        self.data_format = data_format

        super().__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('Input must be a list of two Tensors to process')
        super().build(input_shape)

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('Input must be a list of two Tensors to process')

        features = tf.convert_to_tensor(inputs[0])
        rois = tf.convert_to_tensor(inputs[1])

        return _precise_roi_pooling(features,
                                    rois,
                                    pooled_height=self.pooled_height,
                                    pooled_width=self.pooled_width,
                                    spatial_scale=self.spatial_scale,
                                    data_format=self.data_format)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)

        #  Input validation
        if len(input_shape) != 2:
            raise ValueError('Input must be a list of two shapes')

        number_of_rois = input_shape[1][0]

        if self.data_format == 'channels_first':
            number_of_channels = input_shape[0][1]
            return [(number_of_rois, number_of_channels, self.pooled_height, self.pooled_width)]

        elif self.data_format == 'channels_last':
            number_of_channels = input_shape[0][3]
            return [(number_of_rois, self.pooled_height, self.pooled_width, number_of_channels)]
        else:
            raise ValueError(
                '`data_format` must be either `channels_last` or `channels_first`'
            )

    def get_config(self):
        config = {
            'pooled_height': self.pooled_height,
            'pooled_width': self.pooled_width,
            'spatial_scale': self.spatial_scale,
            'data_format': self.data_format,
        }

        base_config = super().get_config()
        return {**base_config, **config}

@tf.RegisterGradient('PreciseRoIPooling')
def _precise_roi_pooling_grad(op, grad_output):
    pooled_height = op.get_attr('pooled_height')
    pooled_width = op.get_attr('pooled_width')
    spatial_scale = op.get_attr('spatial_scale')
    data_format = op.get_attr('data_format')

    features = tf.convert_to_tensor(op.inputs[0], name='features')
    rois = tf.convert_to_tensor(op.inputs[1], name='rois')
    pooled_features = tf.convert_to_tensor(op.outputs[0], name='pooled_features')
    grad_output = tf.convert_to_tensor(grad_output, name='grad_output')

    op_call = _precise_roi_pooling_ops.precise_ro_i_pooling_grad
    grads = op_call(features,
                    rois,
                    pooled_features,
                    grad_output,
                    pooled_height=pooled_height,
                    pooled_width=pooled_width,
                    spatial_scale=spatial_scale,
                    data_format=data_format)

    features_gradient = tf.convert_to_tensor(grads[0], name='features_gradient')
    rois_gradient = tf.convert_to_tensor(grads[1], name='rois_gradient')
    return [features_gradient, rois_gradient]
