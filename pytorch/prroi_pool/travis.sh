#! /bin/bash -e
# File   : travis.sh
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
#
# Distributed under terms of the MIT license.
# Copyright (c) 2017 Megvii Technology Limited.

echo "Working directory: " `pwd`
echo "Building python libraries..."
python3 setup.py build_ext

