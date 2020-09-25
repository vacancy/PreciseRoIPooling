#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : build_cuda.py
# Author : Kanghee Lee
# Email  : lerohiso@gmail.com
# Date   : 09/25/2020
#
# This file is part of PreciseRoIPooling.
# Distributed under terms of the MIT license.

import os
import platform
import shutil
import subprocess

import tensorflow as tf

CUDA_SRCS = []
CUDA_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'build')

if not os.path.isdir(CUDA_OUTPUT_DIR):
  os.makedirs(CUDA_OUTPUT_DIR)

for file in os.listdir(os.path.dirname(os.path.realpath(__file__))):
  if file.endswith('.cu.cc'):
    CUDA_SRCS.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), file))

CUDA_COMPILER = shutil.which('nvcc')
if CUDA_COMPILER == None:
  raise ValueError('CUDA Compiler Not Found')

TF_CFLAGS = ' '.join(tf.sysconfig.get_compile_flags())
TF_LFLAGS = ' '.join(tf.sysconfig.get_link_flags())

CUDA_NVCC_FLAGS = TF_CFLAGS + ' ' + TF_LFLAGS + ' -D GOOGLE_CUDA=1 -x cu --expt-relaxed-constexpr'

os_type = platform.system()
if os_type == 'Windows':
  CUDA_NVCC_FLAGS += ' -Xcompiler -MD -cudart=shared -D_WINSOCKAPI_'
  CUDA_OUTPUT_FILENAME = 'precise_roi_pooling_cuda.lib'
elif os_type == 'Linux':
  CUDA_NVCC_FLAGS += ' -Xcompiler -fPIC -DNDEBUG'
  CUDA_OUTPUT_FILENAME = 'precise_roi_pooling_cuda.so'

COMMAND = CUDA_COMPILER
COMMAND += ' -c -o ' + os.path.join(CUDA_OUTPUT_DIR, CUDA_OUTPUT_FILENAME)
COMMAND += ' ' + ' '.join(CUDA_SRCS)
COMMAND += ' ' + CUDA_NVCC_FLAGS

process = subprocess.Popen(COMMAND, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
process_output = process.communicate()[0]
print(process_output.decode())

if process.returncode is not 0:
  raise ValueError('Fail to CUDA Compile')
