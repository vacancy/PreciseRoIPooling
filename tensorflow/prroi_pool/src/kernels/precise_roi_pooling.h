/*
 * File   : precise_roi_pooling.h
 * Author : Kanghee Lee
 * Email  : lerohiso@gmail.com
 *
 * Distributed under terms of the MIT license.
 */

#ifndef KERNEL_PRECISE_ROI_POOLING_H_
#define KERNEL_PRECISE_ROI_POOLING_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct PreciseRoIPoolingFunctor {
    Status operator()(OpKernelContext* context,
                      const Tensor& features,
                      const Tensor& rois,
                      Tensor* pooled_features,
                      int pooled_height,
                      int pooled_width,
                      float spatial_scale,
                      TensorFormat data_format);
};

template <typename Device, typename T>
struct PreciseRoIPoolingGradFunctor {
    Status operator()(OpKernelContext* context,
                      const Tensor& features,
                      const Tensor& rois,
                      const Tensor& pooled_features,
                      const Tensor& pooled_features_diff,
                      Tensor* features_gradient,
                      Tensor* rois_gradient,
                      int pooled_height,
                      int pooled_width,
                      float spatial_scale,
                      TensorFormat data_format);
};

}  // namespace functor

}  // namespace tensorflow

#endif // KERNEL_PRECISE_ROI_POOLING_H_