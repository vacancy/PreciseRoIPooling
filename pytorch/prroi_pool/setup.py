#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : setup.py
# Author : Jiayuan Mao, Tete Xiao
# Email  : maojiayuan@gmail.com, jasonhsiao97@gmail.com
# Date   : 07/13/2018
#
# This file is part of PreciseRoIPooling.
# Distributed under terms of the MIT license.
# Copyright (c) 2017 Megvii Technology Limited.

import os
import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

include_dirs = ['src/']
sources = []
defines = []

if torch.cuda.is_available():
    sources += ['src/prroi_pooling_gpu.c', 'src/prroi_pooling_gpu_impl.cu']
    defines += [('WITH_CUDA', None)]
else:
    # TODO(Jiayuan Mao @ 07/13): remove this restriction after we support the cpu implementation.
    raise NotImplementedError('Precise RoI Pooling only supports GPU (cuda) implememtations.')


if __name__ == '__main__':
    setup(
        name='_prroi_pooling',
        ext_modules=[
            CUDAExtension(
                name='_prroi_pooling',
                sources=sources,
                define_macros=defines,
                include_dirs=include_dirs
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )

