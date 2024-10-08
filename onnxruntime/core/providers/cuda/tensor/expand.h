// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

class Expand final : public CudaKernel {
 public:
  Expand(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

Status FuncExpand(
    const CudaKernel* cuda_kernel,
    OpKernelContext* ctx,
    const Tensor* input_data_tensor,
    const Tensor* /*input_shape_tensor*/,
    Tensor* output_tensor);

std::unique_ptr<Tensor> FuncExpand(
    const CudaKernel* cuda_kernel,
    OpKernelContext* ctx,
    const Tensor* input_data_tensor,
    const Tensor* input_shape_tensor);

}  // namespace cuda
}  // namespace onnxruntime
