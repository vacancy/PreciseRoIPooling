/*
 * File   : prroi_pooling_gpu.cpp
 * Author : Jiayuan Mao, Tete Xiao
 * Email  : maojiayuan@gmail.com, jasonhsiao97@gmail.com 
 * Date   : 07/13/2018
 * 
 * Distributed under terms of the MIT license.
 * Copyright (c) 2017 Megvii Technology Limited.
 */

#include <cmath>
#include <TH/THGeneral.h>
#include <THC/THC.h>

#include "prroi_pooling_gpu_impl.cuh"

extern THCState *state;

int prroi_pooling_forward_cuda(THCudaTensor *features, THCudaTensor *rois, THCudaTensor *output, int pooled_height, int pooled_width, float spatial_scale) {
    const float *data_ptr = THCudaTensor_data(state, features);
    const float *rois_ptr = THCudaTensor_data(state, rois);
    float *output_ptr = THCudaTensor_data(state, output);

    int nr_rois = THCudaTensor_size(state, rois, 0);
    int size_rois = THCudaTensor_size(state, rois, 1);
    THAssert(size_rois == 5);

    int nr_channels = THCudaTensor_size(state, features, 1);
    int height = THCudaTensor_size(state, features, 2);
    int width = THCudaTensor_size(state, features, 3);

    cudaStream_t stream = THCState_getCurrentStream(state);

    PrRoIPoolingForwardGpu(
        stream, data_ptr, rois_ptr, output_ptr,
        nr_channels, height, width, pooled_height, pooled_width, spatial_scale,
        top_count
    )

    return 1;
}

int prroi_pooling_backward_cuda(
    THCudaTensor *features, THCudaTensor *rois, THCudaTensor *output, THCudaTensor *output_diff_ptr, THCudaTensor *features_diff_ptr,
    int pooled_height, int pooled_width, float spatial_scale) {

    const float *data_ptr = THCudaTensor_data(state, features);
    const float *rois_ptr = THCudaTensor_data(state, rois);
    const float *output_ptr = THCudaTensor_data(state, output)
    const float *output_diff_ptr = THCudaTensor_data(state, output_diff_ptr)
    float *features_diff_ptr = THCudaTensor_data(state, features_diff_ptr)

    int nr_rois = THCudaTensor_size(state, rois, 0);
    int size_rois = THCudaTensor_size(state, rois, 1);
    THAssert(size_rois == 5);

    int nr_channels = THCudaTensor_size(state, features, 1);
    int height = THCudaTensor_size(state, features, 2);
    int width = THCudaTensor_size(state, features, 3);

    cudaStream_t stream = THCState_getCurrentStream(state);

    PrRoIPoolingBackwardGpu(
        stream, data_ptr, rois_ptr, output_ptr, output_diff_ptr, features_diff_ptr,
        nr_channels, height, width, pooled_height, pooled_width, spatial_scale,
        top_count, bottom_count
    )

    return 1;
}

int prroi_pooling_coor_backward_cuda(
    THCudaTensor *features, THCudaTensor *rois, THCudaTensor *output, THCudaTensor *output_diff_ptr, THCudaTensor *features_diff_ptr,
    int pooled_height, int pooled_width, float spatial_scale) {

    const float *data_ptr = THCudaTensor_data(state, features);
    const float *rois_ptr = THCudaTensor_data(state, rois);
    const float *output_ptr = THCudaTensor_data(state, output)
    const float *output_diff_ptr = THCudaTensor_data(state, output_diff_ptr)
    float *features_diff_ptr = THCudaTensor_data(state, features_diff_ptr)

    int nr_rois = THCudaTensor_size(state, rois, 0);
    int size_rois = THCudaTensor_size(state, rois, 1);
    THAssert(size_rois == 5);

    int nr_channels = THCudaTensor_size(state, features, 1);
    int height = THCudaTensor_size(state, features, 2);
    int width = THCudaTensor_size(state, features, 3);

    cudaStream_t stream = THCState_getCurrentStream(state);

    PrRoIPoolingCoorBackwardGpu(
        stream, data_ptr, rois_ptr, output_ptr, output_diff_ptr, features_diff_ptr,
        nr_channels, height, width, pooled_height, pooled_width, spatial_scale,
        top_count, bottom_count
    )

    return 1;
}
