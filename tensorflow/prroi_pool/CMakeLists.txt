# File   : CMakeLists.txt
# Author : Kanghee Lee
# Email  : lerohiso@gmail.com
# Date   : 09/25/2020
#
# This file is part of PreciseRoIPooling.
# Distributed under terms of the MIT license.

CMAKE_MINIMUM_REQUIRED(VERSION 3.17 FATAL_ERROR)

PROJECT(precise_roi_pooling)
FIND_PACKAGE(CUDA)
FIND_PACKAGE(PythonInterp 3)

if (MSVC)
    SET(GPU_LIB ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/build/precise_roi_pooling_cuda.lib)
elseif (UNIX)
    SET(GPU_LIB ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/build/precise_roi_pooling_cuda.so)
endif()

if (NOT EXISTS ${GPU_LIB})
    EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/build_cuda.py" RESULTS_VARIABLE RET_CODE)
    if (NOT "${RET_CODE}" STREQUAL "0")
        MESSAGE(FATAL_ERROR "Fail to Complie CUDA codes")
    endif ()
endif ()

if (NOT DEFINED TF_PATH)
    EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; print(tf.sysconfig.get_include(), end='', flush=True)"  OUTPUT_VARIABLE TF_INC)
    EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; print(tf.sysconfig.get_lib(), end='', flush=True)"  OUTPUT_VARIABLE TF_LIB)
    MESSAGE(STATUS "TF_INC: " ${TF_INC})
    MESSAGE(STATUS "TF_LIB: " ${TF_LIB})
    SET(TF_PATH 1)
endif ()

if (NOT DEFINED TF_FLAGS)
    EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()), end='', flush=True)"  OUTPUT_VARIABLE TF_CFLAGS)
    EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()), end='', flush=True)"  OUTPUT_VARIABLE TF_LFLAGS)
    MESSAGE(STATUS "TF_CFLAGS: " ${TF_CFLAGS})
    MESSAGE(STATUS "TF_LFLAGS: " ${TF_LFLAGS})
    SET(TF_FLAGS 1)
endif ()

INCLUDE_DIRECTORIES(${TF_INC})
LINK_DIRECTORIES(${TF_LIB})
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

LIST(APPEND CMAKE_CXX_FLAGS "${TF_CFLAGS} ${TF_LFLAGS} -O2 -D GOOGLE_CUDA=1 -std=c++11 -shared")
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    LIST(APPEND CMAKE_CXX_FLAGS " -lcudart -DNOMINMAX")
endif ()

MESSAGE(STATUS "CMAKE_CXX_COMPILER_ID: " ${CMAKE_CXX_COMPILER_ID})
MESSAGE(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})

ADD_LIBRARY(precise_roi_pooling SHARED src/kernels/precise_roi_pooling.h
                                       src/kernels/precise_roi_pooling_kernels.cc
                                       src/ops/precise_roi_pooling_ops.cc)
TARGET_COMPILE_FEATURES(precise_roi_pooling PUBLIC cxx_std_11)
SET_TARGET_PROPERTIES(precise_roi_pooling PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/module/"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/module/"
)

ADD_LIBRARY(precise_roi_pooling_gpu SHARED IMPORTED)
if (MSVC)
    SET_TARGET_PROPERTIES(precise_roi_pooling_gpu PROPERTIES IMPORTED_IMPLIB ${GPU_LIB})
elseif (UNIX)
    SET_TARGET_PROPERTIES(precise_roi_pooling_gpu PROPERTIES IMPORTED_LOCATION ${GPU_LIB})
endif()

ADD_LIBRARY(tensorflow_internal SHARED IMPORTED)
if (MSVC)
    SET_TARGET_PROPERTIES(tensorflow_internal PROPERTIES
        IMPORTED_IMPLIB ${TF_LIB}/python/_pywrap_tensorflow_internal.lib)
elseif (UNIX)
    SET_TARGET_PROPERTIES(tensorflow_internal PROPERTIES
        IMPORTED_LOCATION ${TF_LIB}/python/_pywrap_tensorflow_internal.so)
endif()

TARGET_LINK_LIBRARIES(precise_roi_pooling tensorflow_internal
                                          precise_roi_pooling_gpu
                                          ${CUDA_LIBRARIES})

ADD_CUSTOM_TARGET(precise_roi_pooling_test ALL
                  COMMAND ${CMAKE_COMMAND} -E env
                  "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../"
                  ${PYTHON_EXECUTABLE} tests/precise_roi_pooling_ops_test.py
                  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../")

ADD_DEPENDENCIES(precise_roi_pooling_test precise_roi_pooling)
