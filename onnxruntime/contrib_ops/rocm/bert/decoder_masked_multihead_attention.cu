// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/bert/decoder_masked_multihead_attention.h"

#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
#include "contrib_ops/rocm/bert/batched_gemm_softmax_gemm_permute_pipelines.cuh"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/fpgeneric.h"
#include "core/platform/env_var_utils.h"

using namespace onnxruntime::rocm;
using namespace ::onnxruntime::common;
// using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace rocm {

// TODO: refactor
static constexpr int kPastSequenceLengthInputIndex = 7;
static constexpr int kBeamWidthInputIndex = 8;
static constexpr int kCacheIndirectionInputIndex = 9;
static constexpr int kPastInputIndex = 5;
static constexpr int kPresentOutputIndex = 1;

#define REGISTER_KERNEL_TYPED(T)                                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                              \
      DecoderMaskedMultiHeadAttention,                                        \
      kMSDomain,                                                              \
      1,                                                                      \
      T,                                                                      \
      kRocmExecutionProvider,                                                 \
      (*KernelDefBuilder::Create())                                           \
          .MayInplace(kPastInputIndex, kPresentOutputIndex)                   \
          .MayInplace(kPastInputIndex + 1, kPresentOutputIndex + 1)           \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())              \
          .InputMemoryType(OrtMemTypeCPUInput, kPastSequenceLengthInputIndex) \
          .InputMemoryType(OrtMemTypeCPUInput, kBeamWidthInputIndex),         \
      DecoderMaskedMultiHeadAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
DecoderMaskedMultiHeadAttention<T>::DecoderMaskedMultiHeadAttention(const OpKernelInfo& info) : RocmKernel(info) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = static_cast<int>(num_heads);
  mask_filter_value_ = info.GetAttrOrDefault<float>("mask_filter_value", -10000.0f);
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  past_present_share_buffer_ = info.GetAttrOrDefault<int64_t>("past_present_share_buffer", 0LL);
}

template <typename T>
Status DecoderMaskedMultiHeadAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* mask_index = context->Input<Tensor>(3);
  const Tensor* relative_position_bias = context->Input<Tensor>(4);
  const Tensor* past_key = context->Input<Tensor>(kPastInputIndex);
  const Tensor* past_value = context->Input<Tensor>(kPastInputIndex + 1);
  const Tensor* past_seq_len = context->Input<Tensor>(kPastSequenceLengthInputIndex);
  const Tensor* beam_width = context->Input<Tensor>(kBeamWidthInputIndex);
  const Tensor* cache_indir = context->Input<Tensor>(kCacheIndirectionInputIndex);

  // TODO:
  ORT_ENFORCE(cache_indir == nullptr);

  auto& device_prop = GetDeviceProp();
  AttentionParameters attn;
  ORT_RETURN_IF_ERROR(
      multihead_attention_helper::CheckInputs<Tensor>(
          query, key, value, /*bias=*/nullptr,
          mask_index, relative_position_bias,
          past_key, past_value, past_seq_len,
          &attn,
          num_heads_, mask_filter_value_, scale_,
          past_present_share_buffer_,
          device_prop.maxThreadsPerBlock));

  // This kernel is for decoding only (i.e.) sequence length has to be 1
  if (attn.sequence_length != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input sequence length should be 1 to use DecoderMaskedMultiHeadAttention");
  }

  if (attn.head_size != attn.v_head_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "QK head size should be same as V head size to use DecoderMaskedMultiHeadAttention");
  }

  if (attn.mask_type != AttentionMaskType::MASK_NONE) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "DecoderMaskedMultiHeadAttention currently does not implement mask support.");
  }

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(attn.batch_size);
  output_shape[1] = static_cast<int64_t>(attn.sequence_length);
  output_shape[2] = static_cast<int64_t>(attn.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);  // Bx1xNH

  std::vector<int64_t> present_dims{
      attn.batch_size, attn.num_heads,
      past_present_share_buffer_ ? attn.max_sequence_length : attn.total_sequence_length,
      attn.head_size};
  TensorShape present_shape(present_dims);
  Tensor* present_key = context->Output(kPresentOutputIndex, present_shape);
  Tensor* present_value = context->Output(kPresentOutputIndex + 1, present_shape);

  bool is_cross_attention = past_key == nullptr && present_key == nullptr;
  attn.qkv_format = is_cross_attention ? Q_B1NH_K_V_BNSH_CROSS : Q_K_V_B1NH_SELF;

  auto rocm_stream = Stream(context);

  using HipT = typename ToHipType<T>::MappedType;
  using AttentionTunableOp = GemmSoftmaxGemmPermuteTunableOp<HipT>;
  auto workspace_bytes = AttentionTunableOp::GetWorkspaceNumBytes(&attn);
  auto workspace = GetScratchBuffer<void>(workspace_bytes, context->GetComputeStream());

  GemmSoftmaxGemmPermuteParams<HipT> params;
  params.tuning_ctx = GetTuningContext();
  params.stream = Stream(context);
  params.handle = GetRocblasHandle(context);
  params.attention = &attn;
  params.device_prop = &device_prop;
  params.scale = scale_ == 0 ? 1.0f / sqrt(attn.head_size) : scale_;
  std::tie(params.q_buffer, params.k_buffer, params.v_buffer) = GetQkvBuffers<HipT>(
      &attn,
      query->DataRaw(),
      key == nullptr ? nullptr : key->DataRaw(),
      value == nullptr ? nullptr : value->DataRaw());
  params.out_buffer = reinterpret_cast<HipT*>(output->MutableDataRaw());

  if (relative_position_bias != nullptr) {
    params.bias_buffer = reinterpret_cast<const HipT*>(relative_position_bias->DataRaw());
  }

  params.workspace_buffer = reinterpret_cast<HipT*>(workspace.get());
  return AttentionTunableOp{}(&params);

  // Update the q buffers
  // FIXME: attn.q = const_cast<T1*>(query->Data<T1>());

  // Update the relative position bias for self attention
  if (relative_position_bias != nullptr) {
    // FIXME: attn.relative_attention_bias = const_cast<T1*>(relative_position_bias->Data<T1>());
  }

  if (is_cross_attention) {
    if (relative_position_bias != nullptr) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, NOT_IMPLEMENTED,
          "DecoderMaskedMultiHeadAttention does not support relative position bias for cross-attention");
    }

    attn.total_sequence_length = attn.kv_sequence_length;
    attn.max_sequence_length = attn.kv_sequence_length;
    // parameters.k and paraneters.v are nullptr
    // FIXME: attn.k_cache = const_cast<T1*>(key->Data<T1>());
    // FIXME: attn.v_cache = const_cast<T1*>(value->Data<T1>());
  } else {
    // Sanity check
    ORT_ENFORCE(past_present_share_buffer_);
    ORT_ENFORCE(past_key != nullptr && past_value != nullptr);

    auto* past_key_data = past_key->Data<T>();
    auto* past_value_data = past_value->Data<T>();
    auto* present_key_data = present_key->MutableData<T>();
    auto* present_value_data = present_value->MutableData<T>();

    // No production use-case will incur this copy cost as the implementation of
    // GreedySearch/BeamSearch is written in such a way that the past and present buffers
    // will be shared.
    // This is just to circumvent the OpTester's limitation of not being able to bind a specific
    // buffer to inputs/outputs.
    if (present_key_data != past_key_data) {
      HIP_RETURN_IF_ERROR(hipMemcpyAsync(present_key_data, past_key_data, past_key->SizeInBytes(),
                                         hipMemcpyDeviceToDevice, rocm_stream));
    }
    if (present_value_data != past_value_data) {
      HIP_RETURN_IF_ERROR(hipMemcpyAsync(present_value_data, past_value_data, past_value->SizeInBytes(),
                                         hipMemcpyDeviceToDevice, rocm_stream));
    }

    // FIXME: attn.is_cross_attention = false;

    // FIXME: attn.k = const_cast<T1*>(key->Data<T1>());
    // FIXME: attn.v = const_cast<T1*>(value->Data<T1>());
    // FIXME: attn.k_cache = present_key_data;
    // FIXME: attn.v_cache = present_value_data;
  }

  // FIXME: attn.out = output->MutableDataRaw();

  // Mask
  if (attn.mask_type == AttentionMaskType::MASK_2D_KEY_PADDING) {
    // FIXME: attn.mask = mask_index->Data<int32_t>();
  }

  // Beam width (in case we are using this op inside BeamSearch)
  if (beam_width != nullptr) {
    // FIXME: attn.beam_width = static_cast<int>(*beam_width->Data<int32_t>());
  }

  // Cache indirection (in case we are using this op inside BeamSearch)
  // if (attn.beam_width > 1) {
  //   // If beam width > 1, then cache indirection buffer MUST be present
  //   if (cache_indir == nullptr) {
  //     return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
  //                            "If beam width is greater than 1, then cache indirection buffer MUST be present");
  //   }

  //   // FIXME: attn.cache_indir = cache_indir->Data<int32_t>();
  // }

  return Status::OK();
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
