// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/bert/packed_attention.h"
#include "contrib_ops/cuda/bert/attention_kernel_options.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class PackedMultiHeadAttention final : public TrtFusedAttention<T> {
 public:
  PackedMultiHeadAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  Status CheckInputs(const TensorShape& query_shape,
                     const Tensor* key,
                     const Tensor* value,
                     const Tensor* bias,
                     const TensorShape& token_offset_shape,
                     const TensorShape& cu_seq_len_shape,
                     const Tensor* attention_bias,
                     PackedAttentionParameters& parameters) const;
  int GetNumHeads() const { return num_heads_; }
  float GetScale() const { return scale_; }

  int num_heads_;  // number of attention heads
  float scale_;    // the scale for softmax in memory efficient attention or unfused attention.

  bool disable_memory_efficient_attention_;
  bool disable_flash_attention_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
