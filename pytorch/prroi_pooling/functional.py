#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : functional.py
# Author : Jiayuan Mao, Tete Xiao
# Email  : maojiayuan@gmail.com, jasonhsiao97@gmail.com
# Date   : 07/13/2018
# 
# This file is part of PreciseRoIPooling.
# Distributed under terms of the MIT license.
# Copyright (c) 2017 Megvii Technology Limited.

import torch
import torch.autograd as ag

try:
    from . import _prroi_pooling
except ImportError:
    raise ImportError('Can not found the compiled Precise RoI Pooling library. Run ./travis.sh in the directory first')

__all__ = ['prroi_pooling2d']


class PrRoIPooling2DFunction(ag.Function):
    @staticmethod
    def forward(ctx, features, rois, pooled_height, pooled_width, spatial_scale):
        pooled_height = int(pooled_height)
        pooled_width = int(pooled_width)
        spatial_scale = float(spatial_scale)

        params = (pooled_height, pooled_width, spatial_scale)
        batch_size, nr_channels, data_height, data_width = features.size()
        nr_rois = rois.size(0)
        output = torch.zeros(
            (nr_rois, nr_channels, pooled_height, pooled_width),
            dtype=features.dtype, device=features.device
        )

        if features.is_cuda:
            output = _prroi_pooling.roi_pooling_forward_cuda(features, rois, output, *params)
            ctx.params = params
            ctx.save_for_backward(features, rois, output)
        else:
            raise NotImplementedError('Precise RoI Pooling only supports GPU (cuda) implememtations.')

        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, rois, output = ctx.saved_tensors

        grad_input = torch.zeros_like(features)
        grad_coor = torch.zeros_like(rois)
        _prroi_pooling.roi_pooling_backward_cuda(features, rois, output, grad_output, grad_input, *ctx.params)
        _prroi_pooling.roi_pooling_coor_backward_cuda(features, rois, output, grad_output, grad_coor, *ctx.params)

        return grad_input, grad_coor, None, None, None


prroi_pooling2d = PrRoIPooling2DFunction.apply

