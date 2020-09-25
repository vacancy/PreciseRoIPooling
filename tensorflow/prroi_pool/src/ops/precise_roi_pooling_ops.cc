/*
 * File   : precise_roi_pooling_ops.cc
 * Author : Kanghee Lee
 * Email  : lerohiso@gmail.com
 *
 * Distributed under terms of the MIT license.
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("PreciseRoIPooling")
    .Input("features: T")
    .Input("rois: T")
    .Output("pooled_features: T")
    .Attr("pooled_height: int")
    .Attr("pooled_width: int")
    .Attr("spatial_scale: float")
    .Attr("data_format: {'NCHW'} = 'NCHW'")
    .Attr("T: realnumbertype")
    .SetShapeFn([](InferenceContext* c) {
         ShapeHandle features, rois;

         TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &features));
         TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &rois));

         // get input shapes
         int32 number_of_rois, number_of_channels;
         number_of_rois = c->Value(c->Dim(rois, 0));
         string data_format;
         Status s = c->GetAttr("data_format", &data_format);
         if (s.ok() && data_format == "NCHW") {
              number_of_channels = c->Value(c->Dim(features, 1));
         }
         else {
              number_of_channels = c->Value(c->Dim(features, 3));
         }

         int32 pooled_height;
         int32 pooled_width;

         TF_RETURN_IF_ERROR(c->GetAttr("pooled_height", &pooled_height));
         TF_RETURN_IF_ERROR(c->GetAttr("pooled_width", &pooled_width));

         // Note, the output is always NCHW (even when input is NHWC)
         c->set_output(0, c->MakeShape({number_of_rois, number_of_channels, pooled_height, pooled_width}));
         return Status::OK();
    })
    .Doc(R"doc(PreciseRoIPooling op.)doc");

REGISTER_OP("PreciseRoIPoolingGrad")
    .Input("features: T")
    .Input("rois: T")
    .Input("pooled_features: T")
    .Input("pooled_features_diff: T")
    .Output("features_gradient: T")
    .Output("rois_gradient: T")
    .Attr("pooled_height: int")
    .Attr("pooled_width: int")
    .Attr("spatial_scale: float")
    .Attr("data_format: {'NCHW'} = 'NCHW'")
    .Attr("T: realnumbertype")
    .SetShapeFn([](InferenceContext* c) {
         ShapeHandle features, rois;
         TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &features));
         TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &rois));
         c->set_output(0, features);
         c->set_output(1, rois);
         return Status::OK();
    })
    .Doc(R"doc(PreciseRoIPoolingGrad op.)doc");

}  // namespace tensorflow