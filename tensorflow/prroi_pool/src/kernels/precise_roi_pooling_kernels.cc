/*
 * File   : precise_roi_pooling_kernels.cc
 * Author : Kanghee Lee
 * Email  : lerohiso@gmail.com
 *
 * Distributed under terms of the MIT license.
 */

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "precise_roi_pooling.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"


namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// CPU specialization of actual computation.
template <typename Dtype>
struct PreciseRoIPoolingFunctor<CPUDevice, Dtype> {
    Status operator()(OpKernelContext* context,
                      const Tensor& features,
                      const Tensor& rois,
                      Tensor* pooled_features,
                      int pooled_height,
                      int pooled_width,
                      float spatial_scale,
                      TensorFormat data_format) {
        return Status(errors::Internal("CPU mode is not implemented"));
    }
};


template <typename Dtype>
struct PreciseRoIPoolingGradFunctor<CPUDevice, Dtype> {
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
        return Status(errors::Internal("CPU mode is not implemented"));
    }
};

} // end namespace functor


// OpKernel definition.
template <typename Device, typename T>
class PreciseRoIPoolingOp : public OpKernel {
    public:
        explicit PreciseRoIPoolingOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("pooled_height", &pooled_height));
            OP_REQUIRES_OK(context, context->GetAttr("pooled_width", &pooled_width));
            OP_REQUIRES_OK(context, context->GetAttr("spatial_scale", &spatial_scale));

            OP_REQUIRES(context,
                        pooled_height > 0,
                        errors::InvalidArgument("pooled_height should be > 0, but was: ", pooled_height));
            OP_REQUIRES(context,
                        pooled_width > 0,
                        errors::InvalidArgument("pooled_width should be > 0, but was: ", pooled_width));
            OP_REQUIRES(context,
                        spatial_scale > 0.f,
                        errors::InvalidArgument("spatial_scale should be > 0., but was: ", spatial_scale));
            string data_format_string;
            OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_string));
            OP_REQUIRES(context, FormatFromString(data_format_string, &data_format),
                        errors::InvalidArgument("Invalid data format"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& features = context->input(0);
            const Tensor& rois = context->input(1);

            // we didn't check the batch-dimension during "SetShapeFn"
            OP_REQUIRES(context,
                        features.shape().dims() == 4,
                        errors::InvalidArgument("features must be 4-dimensional", features.shape().DebugString()));
            OP_REQUIRES(context,
                        rois.shape().dims() == 2 && rois.shape().dim_size(1) == 5,
                        errors::InvalidArgument("rois must be in '[batch_index, x0, y0, x1, y1]' format", rois.shape().DebugString()));

            const int32 number_of_channels = GetTensorDim(features, data_format, 'C');
            const int32 height = GetTensorDim(features, data_format, 'H');
            const int32 width = GetTensorDim(features, data_format, 'W');

            const int32 number_of_rois = rois.dim_size(0);

            Tensor* pooled_features;
            OP_REQUIRES_OK(context,
                           context->allocate_output(0, TensorShape({number_of_rois, number_of_channels, pooled_height, pooled_width}),
                           &pooled_features));

            functor::PreciseRoIPoolingFunctor<Device, T> PreciseRoIPoolingFunc;
            Status s = PreciseRoIPoolingFunc(context,
                                             features,
                                             rois,
                                             pooled_features,
                                             pooled_height,
                                             pooled_width,
                                             spatial_scale,
                                             data_format);

            OP_REQUIRES_OK(context, s);
        }

    private:
        int pooled_height;
        int pooled_width;
        float spatial_scale;
        TensorFormat data_format;
};

template <typename Device, typename T>
class PreciseRoIPoolingGradOp : public OpKernel {
    public:
        explicit PreciseRoIPoolingGradOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("pooled_height", &pooled_height));
            OP_REQUIRES_OK(context, context->GetAttr("pooled_width", &pooled_width));
            OP_REQUIRES_OK(context, context->GetAttr("spatial_scale", &spatial_scale));

            OP_REQUIRES(context,
                        pooled_height > 0,
                        errors::InvalidArgument("pooled_height should be > 0, but was: ", pooled_height));
            OP_REQUIRES(context,
                        pooled_width > 0,
                        errors::InvalidArgument("pooled_width should be > 0, but was: ", pooled_width));
            OP_REQUIRES(context,
                        spatial_scale > 0.f,
                        errors::InvalidArgument("spatial_scale should be > 0., but was: ", spatial_scale));
            string data_format_string;
            OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_string));
            OP_REQUIRES(context, FormatFromString(data_format_string, &data_format),
                        errors::InvalidArgument("Invalid data format"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& features = context->input(0);
            const Tensor& rois = context->input(1);
            const Tensor& pooled_features = context->input(2);
            const Tensor& pooled_features_diff = context->input(3);

            // we didn't check the batch-dimension during "SetShapeFn"
            OP_REQUIRES(context,
                        features.shape().dims() == 4,
                        errors::InvalidArgument("features must be 4-dimensional", features.shape().DebugString()));
            OP_REQUIRES(context,
                        rois.shape().dims() == 2 && rois.shape().dim_size(1) == 5,
                        errors::InvalidArgument("rois must be in '[batch_index, x0, y0, x1, y1]' format", rois.shape().DebugString()));

            Tensor* features_gradient;
            OP_REQUIRES_OK(context,
                           context->allocate_output(0, features.shape(),
                           &features_gradient));

            Tensor* rois_gradient;
            OP_REQUIRES_OK(context,
                           context->allocate_output(1, rois.shape(),
                           &rois_gradient));

            functor::PreciseRoIPoolingGradFunctor<Device, T> PreciseRoIPoolingGradFunc;
            Status s = PreciseRoIPoolingGradFunc(context,
                                                 features,
                                                 rois,
                                                 pooled_features,
                                                 pooled_features_diff,
                                                 features_gradient,
                                                 rois_gradient,
                                                 pooled_height,
                                                 pooled_width,
                                                 spatial_scale,
                                                 data_format);

            OP_REQUIRES_OK(context, s);
        }

    private:
        int pooled_height;
        int pooled_width;
        float spatial_scale;
        TensorFormat data_format;
};

// Register the CPU kernels.
#define REGISTER_PRECISE_ROI_POOLING_OP_CPU(T)                      \
    REGISTER_KERNEL_BUILDER(Name("PreciseRoIPooling")               \
                                .Device(DEVICE_CPU)                 \
                                .TypeConstraint<T>("T"),            \
                            PreciseRoIPoolingOp<CPUDevice, T>)      \
    REGISTER_KERNEL_BUILDER(Name("PreciseRoIPoolingGrad")           \
                                .Device(DEVICE_CPU)                 \
                                .TypeConstraint<T>("T"),            \
                            PreciseRoIPoolingGradOp<CPUDevice, T>)

TF_CALL_float(REGISTER_PRECISE_ROI_POOLING_OP_CPU);
#undef REGISTER_PRECISE_ROI_POOLING_OP_CPU


// Register the GPU kernels.
#ifdef GOOGLE_CUDA

#define REGISTER_PRECISE_ROI_POOLING_OP_GPU(T)                      \
    REGISTER_KERNEL_BUILDER(Name("PreciseRoIPooling")               \
                                .Device(DEVICE_GPU)                 \
                                .TypeConstraint<T>("T"),            \
                            PreciseRoIPoolingOp<GPUDevice, T>)      \
    REGISTER_KERNEL_BUILDER(Name("PreciseRoIPoolingGrad")           \
                                .Device(DEVICE_GPU)                 \
                                .TypeConstraint<T>("T"),            \
                            PreciseRoIPoolingGradOp<GPUDevice, T>)

TF_CALL_float(REGISTER_PRECISE_ROI_POOLING_OP_GPU);
#undef REGISTER_PRECISE_ROI_POOLING_OP_GPU

#endif  // GOOGLE_CUDA

}  // namespace tensorflow