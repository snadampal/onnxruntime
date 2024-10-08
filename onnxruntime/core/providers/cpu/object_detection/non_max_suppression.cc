/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/* Modifications Copyright (c) Microsoft. */

#include "non_max_suppression.h"

#include <queue>
#include <utility>

#include "core/common/narrow.h"
#include "non_max_suppression_helper.h"

// TODO:fix the warnings
#ifdef _MSC_VER
#pragma warning(disable : 4244)
#endif
namespace onnxruntime {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    NonMaxSuppression,
    kOnnxDomain,
    10, 10,
    kCpuExecutionProvider,
    KernelDefBuilder(),
    NonMaxSuppression);

ONNX_OPERATOR_KERNEL_EX(
    NonMaxSuppression,
    kOnnxDomain,
    11,
    kCpuExecutionProvider,
    KernelDefBuilder(),
    NonMaxSuppression);

using namespace nms_helpers;

// This works for both CPU and GPU.
// CUDA kernel declare OrtMemTypeCPUInput for max_output_boxes_per_class(2), iou_threshold(3) and score_threshold(4)
Status NonMaxSuppressionBase::PrepareCompute(OpKernelContext* ctx, PrepareContext& pc) {
  const auto* boxes_tensor = ctx->Input<Tensor>(0);
  ORT_ENFORCE(boxes_tensor);
  pc.boxes_data_ = boxes_tensor->Data<float>();

  const auto* scores_tensor = ctx->Input<Tensor>(1);
  ORT_ENFORCE(scores_tensor);
  pc.scores_data_ = scores_tensor->Data<float>();

  const auto num_inputs = ctx->InputCount();

  if (num_inputs > 2) {
    const auto* max_output_boxes_per_class_tensor = ctx->Input<Tensor>(2);
    if (max_output_boxes_per_class_tensor != nullptr) {
      pc.max_output_boxes_per_class_ = max_output_boxes_per_class_tensor->Data<int64_t>();
    }
  }

  if (num_inputs > 3) {
    const auto* iou_threshold_tensor = ctx->Input<Tensor>(3);
    if (iou_threshold_tensor != nullptr) {
      pc.iou_threshold_ = iou_threshold_tensor->Data<float>();
    }
  }

  if (num_inputs > 4) {
    const auto* score_threshold_tensor = ctx->Input<Tensor>(4);
    if (score_threshold_tensor != nullptr) {
      pc.score_threshold_ = score_threshold_tensor->Data<float>();
    }
  }

  const auto& boxes_shape = boxes_tensor->Shape();
  pc.boxes_size_ = boxes_shape.Size();
  const auto& scores_shape = scores_tensor->Shape();
  pc.scores_size_ = scores_shape.Size();

  ORT_RETURN_IF_NOT(boxes_shape.NumDimensions() == 3, "boxes must be a 3D tensor.");
  ORT_RETURN_IF_NOT(scores_shape.NumDimensions() == 3, "scores must be a 3D tensor.");

  const auto& boxes_dims = boxes_shape.GetDims();
  const auto& scores_dims = scores_shape.GetDims();
  ORT_RETURN_IF_NOT(boxes_dims[0] == scores_dims[0], "boxes and scores should have same num_batches.");
  ORT_RETURN_IF_NOT(boxes_dims[1] == scores_dims[2], "boxes and scores should have same spatial_dimension.");
  ORT_RETURN_IF_NOT(boxes_dims[2] == 4, "The most inner dimension in boxes must have 4 data.");

  pc.num_batches_ = boxes_dims[0];
  pc.num_classes_ = scores_dims[1];
  pc.num_boxes_ = narrow<int>(boxes_dims[1]);

  return Status::OK();
}

Status NonMaxSuppressionBase::GetThresholdsFromInputs(const PrepareContext& pc,
                                                      int64_t& max_output_boxes_per_class,
                                                      float& iou_threshold,
                                                      float& score_threshold) {
  if (pc.max_output_boxes_per_class_ != nullptr) {
    max_output_boxes_per_class = std::max<int64_t>(*pc.max_output_boxes_per_class_, 0);
  }

  if (pc.iou_threshold_ != nullptr) {
    iou_threshold = *pc.iou_threshold_;
    ORT_RETURN_IF_NOT((iou_threshold >= 0 && iou_threshold <= 1.f), "iou_threshold must be in range [0, 1].");
  }

  if (pc.score_threshold_ != nullptr) {
    score_threshold = *pc.score_threshold_;
  }

  return Status::OK();
}

Status NonMaxSuppression::Compute(OpKernelContext* ctx) const {
  PrepareContext pc;
  ORT_RETURN_IF_ERROR(PrepareCompute(ctx, pc));

  int64_t max_output_boxes_per_class = 0;
  float iou_threshold = .0f;
  float score_threshold = .0f;

  ORT_RETURN_IF_ERROR(GetThresholdsFromInputs(pc, max_output_boxes_per_class, iou_threshold, score_threshold));

  if (0 == max_output_boxes_per_class) {
    ctx->Output(0, {0, 3});
    return Status::OK();
  }

  const auto* const boxes_data = pc.boxes_data_;
  const auto* const scores_data = pc.scores_data_;

  struct BoxInfoPtr {
    float score_{};
    int64_t index_{};

    BoxInfoPtr() = default;
    explicit BoxInfoPtr(float score, int64_t idx) : score_(score), index_(idx) {}
    inline bool operator<(const BoxInfoPtr& rhs) const {
      return score_ < rhs.score_ || (score_ == rhs.score_ && index_ > rhs.index_);
    }
  };

  const auto center_point_box = GetCenterPointBox();

  std::vector<SelectedIndex> selected_indices;
  std::vector<BoxInfoPtr> selected_boxes_inside_class;
  selected_boxes_inside_class.reserve(std::min<size_t>(static_cast<size_t>(max_output_boxes_per_class), pc.num_boxes_));

  for (int64_t batch_index = 0; batch_index < pc.num_batches_; ++batch_index) {
    for (int64_t class_index = 0; class_index < pc.num_classes_; ++class_index) {
      int64_t box_score_offset = (batch_index * pc.num_classes_ + class_index) * pc.num_boxes_;
      const float* batch_boxes = boxes_data + (batch_index * pc.num_boxes_ * 4);
      std::vector<BoxInfoPtr> candidate_boxes;
      candidate_boxes.reserve(pc.num_boxes_);

      // Filter by score_threshold_
      const auto* class_scores = scores_data + box_score_offset;
      if (pc.score_threshold_ != nullptr) {
        for (int64_t box_index = 0; box_index < pc.num_boxes_; ++box_index, ++class_scores) {
          if (*class_scores > score_threshold) {
            candidate_boxes.emplace_back(*class_scores, box_index);
          }
        }
      } else {
        for (int64_t box_index = 0; box_index < pc.num_boxes_; ++box_index, ++class_scores) {
          candidate_boxes.emplace_back(*class_scores, box_index);
        }
      }
      std::priority_queue<BoxInfoPtr, std::vector<BoxInfoPtr>> sorted_boxes(std::less<BoxInfoPtr>(), std::move(candidate_boxes));

      selected_boxes_inside_class.clear();
      // Get the next box with top score, filter by iou_threshold
      while (!sorted_boxes.empty() && static_cast<int64_t>(selected_boxes_inside_class.size()) < max_output_boxes_per_class) {
        const BoxInfoPtr& next_top_score = sorted_boxes.top();

        bool selected = true;
        // Check with existing selected boxes for this class, suppress if exceed the IOU (Intersection Over Union) threshold
        for (const auto& selected_index : selected_boxes_inside_class) {
          if (SuppressByIOU(batch_boxes, next_top_score.index_, selected_index.index_, center_point_box, iou_threshold)) {
            selected = false;
            break;
          }
        }

        if (selected) {
          selected_boxes_inside_class.push_back(next_top_score);
          selected_indices.emplace_back(batch_index, class_index, next_top_score.index_);
        }
        sorted_boxes.pop();
      }  // while
    }  // for class_index
  }  // for batch_index

  constexpr auto last_dim = 3;
  const auto num_selected = selected_indices.size();
  Tensor* output = ctx->Output(0, {static_cast<int64_t>(num_selected), last_dim});
  ORT_ENFORCE(output != nullptr);
  static_assert(last_dim * sizeof(int64_t) == sizeof(SelectedIndex), "Possible modification of SelectedIndex");
  memcpy(output->MutableData<int64_t>(), selected_indices.data(), num_selected * sizeof(SelectedIndex));

  return Status::OK();
}

}  // namespace onnxruntime
