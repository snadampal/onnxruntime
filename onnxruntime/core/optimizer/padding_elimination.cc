// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnx/defs/attr_proto_util.h>

#include "core/optimizer/initializer.h"
#include "core/optimizer/padding_elimination.h"
#include "core/graph/graph_utils.h"
#include "core/framework/random_seed.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {


void PushAllOutputNode(Graph& graph, std::queue<Node*>& q, Node* node, std::unordered_set<Node*>& visited) {
  for (auto iter = node->OutputNodesBegin(); iter != node->OutputNodesEnd(); ++iter) {
    Node* node = graph.GetNode(iter->Index());
    if (visited.find(node) == visited.end()) {
      q.push(node);
    }
  }
}

bool IsATenEmbedding(const Node* node) {
  for(auto kv : node->GetAttributes()) {
    if (kv.first == "operator" && kv.second.s() == "embedding") {
      return true;
    }
  }
  return false;
}

NodeArg* GetDimsValue(Graph& graph, NodeArg* input, NodeArg* indices_arg, Node& node) {
  InlinedVector<NodeArg*> shape_output_args{&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("shape_result"),
                                                                      nullptr)};
  Node& shape_node = graph.AddNode(graph.GenerateNodeName("shape"), "Shape", "", {input},
                                   shape_output_args, nullptr, kOnnxDomain);
  ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(shape_node), "Failed to get shape for " + shape_node.Name());
  shape_node.SetExecutionProviderType(node.GetExecutionProviderType());

  InlinedVector<NodeArg*> gather_input_args;
  gather_input_args.push_back(shape_output_args[0]);
  gather_input_args.push_back(indices_arg);

  InlinedVector<NodeArg*> gather_out_args{&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("gather_result"),
                                                                    nullptr)};

  Node& gather_node = graph.AddNode(graph.GenerateNodeName("gather_first_dim"), "GatherElements", "", gather_input_args,
                                   gather_out_args, nullptr, kOnnxDomain);
  ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(gather_node), "Failed to get shape for " + gather_node.Name());
  gather_node.SetExecutionProviderType(node.GetExecutionProviderType());

  return gather_out_args[0];
}

NodeArg* UpdateShape(Graph& graph, NodeArg* input, NodeArg* update_value, NodeArg* indices_arg, Node& node) {
  InlinedVector<NodeArg*> shape_output_args{&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("shape_result"),
                                                                      nullptr)};
  Node& shape_node = graph.AddNode(graph.GenerateNodeName("shape"), "Shape", "", {input},
                                   shape_output_args, nullptr, kOnnxDomain);
  ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(shape_node), "Failed to get shape for " + shape_node.Name());
  shape_node.SetExecutionProviderType(node.GetExecutionProviderType());

  InlinedVector<NodeArg*> scatter_input_args;
  scatter_input_args.push_back(shape_output_args[0]);
  scatter_input_args.push_back(indices_arg);
  scatter_input_args.push_back(update_value);

  InlinedVector<NodeArg*> scatter_out_args{&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("scatter_result"),
                                                                     nullptr)};

  Node& scatter_node = graph.AddNode(graph.GenerateNodeName("update_dim"), "ScatterElements", "", scatter_input_args,
                                     scatter_out_args, nullptr, kOnnxDomain);
  ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(scatter_node), "Failed to update shape for " + scatter_node.Name());
  scatter_node.SetExecutionProviderType(node.GetExecutionProviderType());

  return scatter_out_args[0];
}
namespace {
// TODO: Merge with InsertNodesForValidLabelIndices
NodeArg* InsertNodesForValidInputIndices(Graph& graph, Node& node, NodeArg* input_to_filter, NodeArg* reduce_index_input, std::string& token_dim_name) {
  InlinedVector<NodeArg*> sub_input_args{input_to_filter, reduce_index_input};

  InlinedVector<NodeArg*> sub_output_args{&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("sub_result"),
                                                                    input_to_filter->TypeAsProto())};

  Node& sub_node = graph.AddNode(graph.GenerateNodeName("sub"), "Sub", "sub padding idx", sub_input_args,
                                 sub_output_args, nullptr, kOnnxDomain);
  ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(sub_node), "Failed to set op schema for " + sub_node.Name());
  sub_node.SetExecutionProviderType(node.GetExecutionProviderType());

  auto non_zero_out_arg = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("filter_pad_result"),
                                                    input_to_filter->TypeAsProto());

  Node& non_zero_node = graph.AddNode(graph.GenerateNodeName("filter_pad"), "NonZero",
                                      "filtering padding idx",
                                      {sub_node.MutableOutputDefs()[0]},
                                      {non_zero_out_arg}, nullptr, kOnnxDomain);

  ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(non_zero_node),
              "Failed to set op schema for " + non_zero_node.Name());

  // 1D input NonZero generates output of shape (1,valid_token_count).
  ONNX_NAMESPACE::TensorShapeProto non_zero_output_shape;
  non_zero_output_shape.add_dim()->set_dim_value(1);
  non_zero_output_shape.add_dim()->set_dim_param(token_dim_name);
  non_zero_out_arg->SetShape(non_zero_output_shape);
  non_zero_node.SetExecutionProviderType(node.GetExecutionProviderType());

  InlinedVector<NodeArg*> squeeze_input_args;
  squeeze_input_args.push_back(non_zero_out_arg);

  bool opset_lower_than_13 = onnxruntime::optimizer::compute_optimizer::GetONNXOpSetVersion(graph) < 13;
  onnxruntime::NodeAttributes attributes;
  if (opset_lower_than_13) {
    attributes["axes"] = ONNX_NAMESPACE::MakeAttribute("axes", std::vector<int64_t>{0});
  } else {
    squeeze_input_args.push_back(onnxruntime::optimizer::compute_optimizer::CreateInitializerFromVector(
        graph, {1}, {0}, graph.GenerateNodeArgName("axes")));
  }

  auto squeeze_out_arg = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("squeeze_adaptor"),
                                                   non_zero_out_arg->TypeAsProto());
  Node& squeeze_node = graph.AddNode(graph.GenerateNodeName("squeeze_adaptor"), "Squeeze", "nonzero_squeezer",
                                     squeeze_input_args, {squeeze_out_arg}, &attributes, kOnnxDomain);
  ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(squeeze_node),
              "Failed to set op schema for " + squeeze_node.Name());

  // After Squeeze, the shape becomes (valid_token_count).
  ONNX_NAMESPACE::TensorShapeProto squeeze_output_shape;
  squeeze_output_shape.add_dim()->set_dim_param(token_dim_name);
  squeeze_out_arg->SetShape(squeeze_output_shape);
  squeeze_node.SetExecutionProviderType(node.GetExecutionProviderType());

  return squeeze_out_arg;
}
}  // namespace

NodeArg* InsertNodesForInput(Graph& graph, Node& node, int in_index, NodeArg* gather_index_arg, const logging::Logger& logger) {
  InlinedVector<NodeArg*> reshape_input_args;
  reshape_input_args.reserve(2);
  reshape_input_args.push_back(node.MutableInputDefs()[in_index]);
  std::vector<int64_t> new_shape;
  new_shape.push_back(-1); // only support flatten 0 and 1 dims
  auto input_shape = node.InputDefs()[in_index]->Shape();
  for (int k = 2; k < input_shape->dim_size(); k++) {
    ORT_ENFORCE(input_shape->dim(k).has_dim_value());
    new_shape.push_back(input_shape->dim(k).dim_value());
  }
  ONNX_NAMESPACE::TensorProto new_shape_const_tensor;
  new_shape_const_tensor.set_name(graph.GenerateNodeArgName("new_shape"));
  new_shape_const_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  new_shape_const_tensor.add_dims(new_shape.size());
  new_shape_const_tensor.set_raw_data(new_shape.data(), new_shape.size() * sizeof(int64_t));
  NodeArg* new_shape_arg = &graph_utils::AddInitializer(graph, new_shape_const_tensor);
  reshape_input_args.push_back(new_shape_arg);

  InlinedVector<NodeArg*> reshape_output_args;
  reshape_output_args.push_back(
    &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("inputs_reshape_result"), nullptr));

  Node* new_reshape_node = onnxruntime::optimizer::compute_optimizer::InsertIntermediateNodeOnDestInput(
      graph, node,
      in_index,
      0,
      0,
      graph.GenerateNodeName("Reshape"),
      "Reshape",
      "Reshape node to filter invalid tokens.",
      reshape_input_args,
      reshape_output_args,
      {},
      "",
      logger);

  new_reshape_node->SetExecutionProviderType(node.GetExecutionProviderType());
  auto reshape_out_arg = new_reshape_node->MutableOutputDefs()[0];

  ONNX_NAMESPACE::TensorShapeProto flattened_shape;
  for (auto dim_value : new_shape) {
      flattened_shape.add_dim()->set_dim_value(dim_value);
  }
  reshape_out_arg->SetShape(flattened_shape);

  InlinedVector<NodeArg*> gather_input_args;
  gather_input_args.reserve(2);
  gather_input_args.push_back(reshape_output_args[0]);
  gather_input_args.push_back(gather_index_arg);

  InlinedVector<NodeArg*> gather_output_args;
  gather_output_args.push_back(
      &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("padding_filter_result"),
                                reshape_out_arg->TypeAsProto()));

  Node* new_gather_node = onnxruntime::optimizer::compute_optimizer::InsertIntermediateNodeOnDestInput(
      graph, node,
      in_index,
      0,
      0,
      graph.GenerateNodeName("PaddingFilter"),
      "ShrunkenGather",
      "ShrunkenGather node to filter invalid tokens.",
      gather_input_args,
      gather_output_args,
      {},
      kMSDomain,
      logger);

  new_gather_node->SetExecutionProviderType(node.GetExecutionProviderType());
  auto gather_out_arg = new_gather_node->MutableOutputDefs()[0];
  std::cout<<gather_out_arg->Name()<<" 0.7"<<std::endl;
  return gather_out_arg;
}

NodeArg* InsertNodesForOutput(Graph& graph,
                              Node& node,
                              int in_index,
                              NodeArg* gathergrad_index_arg,
                              NodeArg* new_shape_arg,
                              NodeArg* first_two_dims_arg,
                              const logging::Logger& logger) {

  std::vector<int64_t> other_indices;
  auto input_shape = node.InputDefs()[in_index]->Shape();
  for (int k = 2; k < input_shape->dim_size(); k++) {
    //ORT_ENFORCE(input_shape->dim(k).has_dim_value());
    other_indices.push_back(k-1);
    // When executing, Shape of node here is flattened, so the indices should start from 1.
  }

  NodeArg* shape_arg = nullptr;
  if (other_indices.empty()) {
    shape_arg = first_two_dims_arg;
  } else {
    ONNX_NAMESPACE::TensorProto other_indices_const_tensor;
    other_indices_const_tensor.set_name(graph.GenerateNodeArgName("other_shape"));
    other_indices_const_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    other_indices_const_tensor.add_dims(other_indices.size());
    other_indices_const_tensor.set_raw_data(other_indices.data(), other_indices.size() * sizeof(int64_t));
    NodeArg* other_indices_arg = &graph_utils::AddInitializer(graph, other_indices_const_tensor);
    NodeArg* other_dims_arg = GetDimsValue(graph, node.MutableInputDefs()[in_index], other_indices_arg, node);

    InlinedVector<NodeArg*> concat_input_args;
    concat_input_args.push_back(first_two_dims_arg);
    concat_input_args.push_back(other_dims_arg);

    InlinedVector<NodeArg*> concat_output_args{&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("concat_shape_result"),
                                             nullptr)};

    onnxruntime::NodeAttributes attributes;
    attributes["axis"] = ONNX_NAMESPACE::MakeAttribute("axis", int64_t(0));

    Node& concat_node = graph.AddNode(graph.GenerateNodeName("concat_shape"), "Concat", "", concat_input_args,
                                    concat_output_args, &attributes, kOnnxDomain);
    ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(concat_node), "Failed to concat shape for " + concat_node.Name());
    concat_node.SetExecutionProviderType(node.GetExecutionProviderType());
    shape_arg = concat_output_args[0];
  }

  InlinedVector<NodeArg*> gathergrad_input_args;
  gathergrad_input_args.reserve(3);
  gathergrad_input_args.push_back(new_shape_arg);
  gathergrad_input_args.push_back(gathergrad_index_arg);
  gathergrad_input_args.push_back(node.MutableInputDefs()[in_index]);

  InlinedVector<NodeArg*> gathergrad_output_args;
  gathergrad_output_args.push_back(
      &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("padding_recover_result"),
                                nullptr));

  Node* new_gathergrad_node = onnxruntime::optimizer::compute_optimizer::InsertIntermediateNodeOnDestInput(
      graph, node,
      in_index,
      2,
      0,
      graph.GenerateNodeName("PaddingRecover"),
      "GatherGrad",
      "GatherGrad node to recover invalid tokens.",
      gathergrad_input_args,
      gathergrad_output_args,
      {},
      kMSDomain,
      logger);

  new_gathergrad_node->SetExecutionProviderType(node.GetExecutionProviderType());
  auto gathergrad_out_arg = new_gathergrad_node->MutableOutputDefs()[0];

  InlinedVector<NodeArg*> reshape_input_args;
  reshape_input_args.push_back(gathergrad_out_arg);
  reshape_input_args.push_back(shape_arg);
  InlinedVector<NodeArg*> reshape_output_args{&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("reshape_result"),
                                              nullptr)};
  Node* new_reshape_node = onnxruntime::optimizer::compute_optimizer::InsertIntermediateNodeOnDestInput(
      graph, node,
      in_index,
      0,
      0,
      graph.GenerateNodeName("RecoverShape"),
      "Reshape",
      "Reshape node to recover invalid tokens.",
      reshape_input_args,
      reshape_output_args,
      {},
      kOnnxDomain,
      logger);
  new_reshape_node->SetExecutionProviderType(node.GetExecutionProviderType());
  return new_reshape_node->MutableOutputDefs()[0];
}


Status PaddingElimination::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  Node* input_node = nullptr;
  NodeArg* input_ids_arg = nullptr;
  // make sure each egde in candidate_edges has two consecutive dims to be flattened
  std::unordered_map<NodeArg*, std::vector<int>> candidate_edges;
  // input edges of nodes in candidate_input should be in candidate_edges or to be added Reshape + Gather
  std::unordered_set<Node*> candidate_input;
  // input edges of nodes in candidate_output, if in candidate_edges, should be added GatherGrad + Reshape
  std::unordered_set<Node*> candidate_output;
  std::queue<Node*> to_visit;
  std::unordered_set<Node*> visited;

  // Find the valid embedding input node
  for (auto node_index : node_topology_list) {
    auto& node = *graph.GetNode(node_index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "ATen", {1}, kPytorchAtenDomain) &&
        graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) &&
        IsATenEmbedding(&node) &&
        node.InputDefs().size() >= 3 &&
        node.InputDefs()[2]->Exists() &&
        graph_utils::IsConstantInitializer(graph, node.InputDefs()[2]->Name()) &&
        node.InputDefs()[1]->Exists() &&
        graph_utils::IsGraphInput(graph, node.InputDefs()[1]) &&
        node.InputDefs()[1]->Shape()->dim(0).has_dim_param() &&
        node.InputDefs()[1]->Shape()->dim(1).has_dim_param()) {
          const TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[2]->Name());
          if (tensor_proto != nullptr &&
              tensor_proto->dims_size() == 0 &&
             ((tensor_proto->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT32) ||
              (tensor_proto->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64))) {
                int64_t padding_idx = *reinterpret_cast<const int64_t*>(tensor_proto->raw_data().data());
                if (padding_idx < 0) {
                  continue;
                }
                input_node = &node;
                input_ids_arg = input_node->MutableInputDefs()[1];
                PushAllOutputNode(graph, to_visit, input_node, visited);
                for (auto output_edges : input_node->MutableOutputDefs()) { // ???make sure there is only one output edge ???
                  candidate_edges[output_edges] = std::vector<int>({0,1});
                }
                break;
          }
    }
  }

  if (!input_node) {
    LOG_DEBUG_INFO(logger, "Exit PaddingElimination optimization for not finding any valid embedding input node.");
    return Status::OK();
  }

  while(!to_visit.empty()) {
    Node* cur = to_visit.front();
    to_visit.pop();
    visited.insert(cur);
    if (graph_utils::IsSupportedOptypeVersionAndDomain(*cur, "Add", {7, 13, 14})) {
      if(candidate_edges.find(cur->MutableInputDefs()[0]) != candidate_edges.end()){
        int dim0 = candidate_edges[cur->MutableInputDefs()[0]][0];
        int dim1 = candidate_edges[cur->MutableInputDefs()[0]][1];
        // Now only support the target two dims are absolutely same or the other shape dim size is less.
        // TODO: support other case such as one of the dims is 1.
        if (cur->InputDefs()[1]->Shape()->dim_size() < cur->InputDefs()[0]->Shape()->dim_size() - dim1 ||
             (cur->InputDefs()[0]->Shape()->dim_size() == cur->InputDefs()[1]->Shape()->dim_size() &&
               cur->InputDefs()[0]->Shape()->dim(dim0) == cur->InputDefs()[1]->Shape()->dim(dim0) &&
               cur->InputDefs()[0]->Shape()->dim(dim1) == cur->InputDefs()[1]->Shape()->dim(dim1))) {
          candidate_edges[cur->MutableOutputDefs()[0]] = candidate_edges[cur->MutableInputDefs()[0]];
          PushAllOutputNode(graph, to_visit, cur, visited);
          if (cur->InputDefs()[0]->Shape()->dim_size() == cur->InputDefs()[1]->Shape()->dim_size()) {
            candidate_input.insert(cur);
          }
        } else {
          LOG_DEBUG_INFO(logger, "PaddingElimination::Input shapes of node:" + cur->Name() +"are not compatible.");
          candidate_output.insert(cur);
          continue;
        }
      } else if(candidate_edges.find(cur->MutableInputDefs()[1]) != candidate_edges.end()){
        int dim0 = candidate_edges[cur->MutableInputDefs()[1]][0];
        int dim1 = candidate_edges[cur->MutableInputDefs()[1]][1];
        if (cur->InputDefs()[0]->Shape()->dim_size() < cur->InputDefs()[1]->Shape()->dim_size() - dim1 ||
             (cur->InputDefs()[0]->Shape()->dim_size() == cur->InputDefs()[1]->Shape()->dim_size() &&
              cur->InputDefs()[0]->Shape()->dim(dim0) == cur->InputDefs()[1]->Shape()->dim(dim0) &&
              cur->InputDefs()[0]->Shape()->dim(dim1) == cur->InputDefs()[1]->Shape()->dim(dim1))) {
          candidate_edges[cur->MutableOutputDefs()[0]] = candidate_edges[cur->MutableInputDefs()[1]];
          PushAllOutputNode(graph, to_visit, cur, visited);
          if (cur->InputDefs()[0]->Shape()->dim_size() == cur->InputDefs()[1]->Shape()->dim_size()) {
            candidate_input.insert(cur);
          }
        } else {
          LOG_DEBUG_INFO(logger, "PaddingElimination::Input shapes of node:" + cur->Name() +"are not compatible.");
          candidate_output.insert(cur);
          continue;
        }
      } else {
        LOGS(logger, WARNING) << "PaddingElimination::Can not found input edges of node:" << cur->Name() << " in candidate_edges.";
      }
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(*cur, "LayerNormalization", {1, 17}, kOnnxDomain)) {
      assert(candidate_edges.find(cur->MutableInputDefs()[0]) != candidate_edges.end());
      auto axis = static_cast<int64_t>(cur->GetAttributes().at("axis").i());
      axis = axis < 0 ? axis + cur->InputDefs()[0]->Shape()->dim_size() : axis;
      if (axis < 2) {
        LOG_DEBUG_INFO(logger, "PaddingElimination::axis of Normalization: " + cur->Name() + " is " +
                                   std::to_string(axis) + ", which blocks merging leading two dims.");
        candidate_output.insert(cur);
      } else {
        candidate_edges[cur->MutableOutputDefs()[0]] = candidate_edges[cur->MutableInputDefs()[0]];
        PushAllOutputNode(graph, to_visit, cur, visited);
      }
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(*cur, "Dropout", {12, 13})) {
      assert(candidate_edges.find(cur->MutableInputDefs()[0]) != candidate_edges.end());
      candidate_edges[cur->MutableOutputDefs()[0]] = candidate_edges[cur->MutableInputDefs()[0]];
      candidate_edges[cur->MutableOutputDefs()[1]] = candidate_edges[cur->MutableInputDefs()[0]];
      PushAllOutputNode(graph, to_visit, cur, visited);
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(*cur, "Cast", {9, 13}) ||
               graph_utils::IsSupportedOptypeVersionAndDomain(*cur, "BiasGelu", {1}, kMSDomain)){
      assert(candidate_edges.find(cur->MutableInputDefs()[0]) != candidate_edges.end());
      candidate_edges[cur->MutableOutputDefs()[0]] = candidate_edges[cur->MutableInputDefs()[0]];
      PushAllOutputNode(graph, to_visit, cur, visited);
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(*cur, "MatMul", {1, 9, 13})) {
      if(candidate_edges.find(cur->MutableInputDefs()[0]) != candidate_edges.end()){
        if (cur->InputDefs()[0]->Shape()->dim_size() > candidate_edges[cur->MutableInputDefs()[0]][1] + 1) {
          candidate_edges[cur->MutableOutputDefs()[0]] = candidate_edges[cur->MutableInputDefs()[0]];
          PushAllOutputNode(graph, to_visit, cur, visited);
        } else {
          LOG_DEBUG_INFO(logger, "PaddingElimination::dim size of left input of matmul must larger than the dim to be merged");
          candidate_output.insert(cur);
          continue;
        }
      }else {
        assert(candidate_edges.find(cur->MutableInputDefs()[1]) != candidate_edges.end());
        LOG_DEBUG_INFO(logger, "PaddingElimination::right edge of matmul would not included.");
        candidate_output.insert(cur);
        continue;
      }
    } else {
      candidate_output.insert(cur);
    }
  }

  // Add Reshape + Sub + NonZero + Squeeze to get the padding index to be gathered
  InlinedVector<NodeArg*> reshape_input_args;
  reshape_input_args.push_back(input_ids_arg);
  std::vector<int64_t> new_shape;
  new_shape.push_back(-1); // Assume input_ids only have two leading dims to be flattened.
  auto input_shape = input_node->InputDefs()[1]->Shape();
  for (int k = 2; k < input_shape->dim_size(); k++) {
    ORT_ENFORCE(input_shape->dim(k).has_dim_value());
    new_shape.push_back(input_shape->dim(k).dim_value());
  }
  ONNX_NAMESPACE::TensorProto new_shape_const_tensor;
  new_shape_const_tensor.set_name(graph.GenerateNodeArgName("new_shape"));
  new_shape_const_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  new_shape_const_tensor.add_dims(new_shape.size());
  new_shape_const_tensor.set_raw_data(new_shape.data(), new_shape.size() * sizeof(int64_t));
  NodeArg* new_shape_arg = &graph_utils::AddInitializer(graph, new_shape_const_tensor);
  reshape_input_args.push_back(new_shape_arg);

  InlinedVector<NodeArg*> reshape_output_args;
  reshape_output_args.push_back(
      &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("inputs_reshape_result"), nullptr));

  Node& reshape_node = graph.AddNode(graph.GenerateNodeName("inputs_reshape"), "Reshape", "input flatten first two dims", reshape_input_args,
                                     reshape_output_args, nullptr, kOnnxDomain);
  ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(reshape_node), "Failed to set op schema for " + reshape_node.Name());
  reshape_node.SetExecutionProviderType(input_node->GetExecutionProviderType());

  std::string token_dim_name = MakeString("valid_token_count_", utils::GetRandomSeed());
  NodeArg* squeeze_out_arg = InsertNodesForValidInputIndices(graph, *input_node, reshape_output_args[0], input_node->MutableInputDefs()[2], token_dim_name);


  InsertNodesForInput(graph, *input_node, 1, squeeze_out_arg, logger);
  for (auto& node: candidate_input) {
    for (size_t i = 0; i < node->InputDefs().size(); ++i) {
      if(candidate_edges.find(node->MutableInputDefs()[i]) == candidate_edges.end()) {
        InsertNodesForInput(graph, *node, i, squeeze_out_arg, logger);
        // onnxruntime::optimizer::compute_optimizer::UpdateSliceOutputShape(
        //     *gather_out_arg, 0, squeeze_out_arg->Shape()->dim(0));
      }
    }
  }


  std::vector<int64_t> indices;
  indices.push_back(0);
  ONNX_NAMESPACE::TensorProto indices_const_tensor;
  indices_const_tensor.set_name(graph.GenerateNodeArgName("indices"));
  indices_const_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  indices_const_tensor.add_dims(indices.size());
  indices_const_tensor.set_raw_data(indices.data(), indices.size() * sizeof(int64_t));
  NodeArg* first_index_arg = &graph_utils::AddInitializer(graph, indices_const_tensor);

  NodeArg* first_dim = GetDimsValue(graph, reshape_output_args[0], first_index_arg, *input_node);

  std::vector<int64_t> first_two_indices{0,1};
  ONNX_NAMESPACE::TensorProto first_two_indices_const_tensor;
  first_two_indices_const_tensor.set_name(graph.GenerateNodeArgName("first_two_indices"));
  first_two_indices_const_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  first_two_indices_const_tensor.add_dims(first_two_indices.size());
  first_two_indices_const_tensor.set_raw_data(first_two_indices.data(), first_two_indices.size() * sizeof(int64_t));
  NodeArg* first_two_indices_arg = &graph_utils::AddInitializer(graph, first_two_indices_const_tensor);
  NodeArg* first_two_dims_arg = GetDimsValue(graph, input_ids_arg, first_two_indices_arg, *input_node);

  for (const auto& node: candidate_output) {
    for (size_t i = 0; i < node->InputDefs().size(); ++i) {
      if(candidate_edges.find(node->MutableInputDefs()[i]) != candidate_edges.end()) {
        std::cout<<node->InputDefs()[i]->Name()<<" 0.9"<<std::endl;
        NodeArg* shape_arg = UpdateShape(graph, node->MutableInputDefs()[i], first_dim, first_index_arg, *node);
        InsertNodesForOutput(graph, *node, i, squeeze_out_arg, shape_arg, first_two_dims_arg, logger);
      }
    }
  }

  for (auto edge: candidate_edges) {
    ONNX_NAMESPACE::TensorShapeProto flattened_shape;
    flattened_shape.add_dim()->set_dim_param(token_dim_name);
    auto input_shape = edge.first->Shape();
    for (int k = 2; k < input_shape->dim_size(); k++) {
      ORT_ENFORCE(input_shape->dim(k).has_dim_value());
      flattened_shape.add_dim()->set_dim_value(input_shape->dim(k).dim_value());
      edge.first->SetShape(flattened_shape);
    }
  }
  return Status::OK();
}
}  // namespace onnxruntime
