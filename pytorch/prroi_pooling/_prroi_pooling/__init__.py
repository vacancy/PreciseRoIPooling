#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao, Tete Xiao
# Email  : maojiayuan@gmail.com, jasonhsiao97@gmail.com
# Date   : 07/13/2018
# 
# This file is part of PreciseRoIPooling.
# Distributed under terms of the MIT license.
# Copyright (c) 2017 Megvii Technology Limited.

from torch.utils.ffi import _wrap_function
from ._prroi_pooling import lib as _lib, ffi as _ffi

def _import_symbols(locals):
    for symbol in dir(_lib):
        fn = getattr(_lib, symbol)
        if callable(fn):
            locals[symbol] = _wrap_function(fn, _ffi)
        else:
            locals[symbol] = fn
        __all__.append(symbol)

_import_symbols(locals())

