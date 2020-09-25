/*
 * File   : precise_roi_pooling_kernels.cu.cc
 * Author : Kanghee Lee
 * Email  : lerohiso@gmail.com
 *
 * Distributed under terms of the MIT license.
 */

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "precise_roi_pooling.h"
#include "external/prroi_pooling_gpu_impl.cu"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// Define the GPU implementation that launches the CUDA kernel.
template <typename Dtype>
struct PreciseRoIPoolingFunctor<GPUDevice, Dtype> {
    Status operator()(OpKernelContext *context,
                      const Tensor& features,
                      const Tensor& rois,
                      Tensor* pooled_features,
                      int pooled_height,
                      int pooled_width,
                      float spatial_scale,
                      TensorFormat data_format) {
        const int32 batch_size = GetTensorDim(features, data_format, 'N');
        const int32 number_of_channels = GetTensorDim(features, data_format, 'C');
        const int32 features_height = GetTensorDim(features, data_format, 'H');
        const int32 features_width = GetTensorDim(features, data_format, 'W');

        const int32 number_of_rois = rois.dim_size(0);

        const int top_count = number_of_rois * number_of_channels * pooled_height * pooled_width;
        const GPUDevice &d = context->eigen_gpu_device();

        PrRoIPoolingForwardGpu(d.stream(),
                               features.flat<Dtype>().data(),
                               rois.flat<Dtype>().data(),
                               pooled_features->flat<Dtype>().data(),
                               number_of_channels,
                               features_height,
                               features_width,
                               pooled_height,
                               pooled_width,
                               spatial_scale,
                               top_count);

        return Status::OK();
    }
};

template <typename Dtype>
struct PreciseRoIPoolingGradFunctor<GPUDevice, Dtype> {
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
                      TensorFormat data_format) {
        const int32 batch_size = GetTensorDim(features, data_format, 'N');
        const int32 number_of_channels = GetTensorDim(features, data_format, 'C');
        const int32 features_height = GetTensorDim(features, data_format, 'H');
        const int32 features_width = GetTensorDim(features, data_format, 'W');

        const int32 number_of_rois = rois.dim_size(0);

        const int top_count = number_of_rois * number_of_channels * pooled_height * pooled_width;
        const GPUDevice &d = context->eigen_gpu_device();

        const int features_gradient_size = batch_size * number_of_channels * features_height * features_width;
        const int rois_gradient_size = number_of_rois * 5;

        PrRoIPoolingBackwardGpu(d.stream(),
                                features.flat<Dtype>().data(),
                                rois.flat<Dtype>().data(),
                                pooled_features.flat<Dtype>().data(),
                                pooled_features_diff.flat<Dtype>().data(),
                                features_gradient->flat<Dtype>().data(),
                                number_of_channels,
                                features_height,
                                features_width,
                                pooled_height,
                                pooled_width,
                                spatial_scale,
                                top_count,
                                features_gradient_size);

        PrRoIPoolingCoorBackwardGpu(d.stream(),
                                    features.flat<Dtype>().data(),
                                    rois.flat<Dtype>().data(),
                                    pooled_features.flat<Dtype>().data(),
                                    pooled_features_diff.flat<Dtype>().data(),
                                    rois_gradient->flat<Dtype>().data(),
                                    number_of_channels,
                                    features_height,
                                    features_width,
                                    pooled_height,
                                    pooled_width,
                                    spatial_scale,
                                    top_count,
                                    rois_gradient_size);

        return Status::OK();
    }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct PreciseRoIPoolingFunctor<GPUDevice, float>;
template struct PreciseRoIPoolingGradFunctor<GPUDevice, float>;

}  // end namespace functor

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA