// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include <memory>
#include <string>
#include <vector>

#include "dragnn/core/compute_session.h"
#include "dragnn/core/compute_session_pool.h"
#include "dragnn/core/ops/compute_session_op.h"
#include "dragnn/core/resource_container.h"
#include "dragnn/protos/data.pb.h"
#include "dragnn/protos/spec.pb.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"

using tensorflow::DEVICE_CPU;
using tensorflow::DT_BOOL;
using tensorflow::DT_FLOAT;
using tensorflow::DT_INT32;
using tensorflow::DT_INT64;
using tensorflow::DT_STRING;
using tensorflow::DataType;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::ResourceMgr;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::TensorShape;

namespace syntaxnet {
namespace dragnn {

typedef ResourceContainer<ComputeSession> ComputeSessionResource;
typedef ResourceContainer<ComputeSessionPool> ComputeSessionPoolResource;

// Given a MasterSpec proto, outputs a handle to a ComputeSession.
class GetSession : public OpKernel {
 public:
  explicit GetSession(OpKernelConstruction *context) : OpKernel(context) {
    string master_spec_str;
    string grid_point_spec_str;
    OP_REQUIRES_OK(context, context->GetAttr("master_spec", &master_spec_str));
    OP_REQUIRES_OK(context,
                   context->GetAttr("grid_point", &grid_point_spec_str));
    CHECK(master_spec_.ParseFromString(master_spec_str));
    CHECK(grid_point_.ParseFromString(grid_point_spec_str));
    OP_REQUIRES_OK(context, context->MatchSignature({DT_STRING}, {DT_STRING}));
  }

  void Compute(OpKernelContext *context) override {
    const string container = context->input(0).scalar<string>()();
    ResourceMgr *rmgr = context->resource_manager();

    // Create the pool for this container, or re-use one that was allocated in a
    // previous call.
    auto create_pool = [this,
                        &container](ComputeSessionPoolResource **resource) {
      LOG(INFO) << "Creating new ComputeSessionPool in container handle: "
      << container;
      std::unique_ptr<ComputeSessionPool> pool(
          new ComputeSessionPool(master_spec_, grid_point_));
      *resource = new ComputeSessionPoolResource(std::move(pool));
      return Status::OK();
    };

    ComputeSessionPoolResource *pool_resource;

    // Synchronize access to the resource manager when getting or creating the
    // ComputeSessionPool.
    // Scoping for minimal mutex locking.
    {
      tensorflow::mutex_lock lock(lock_);
      OP_REQUIRES_OK(context,
                     rmgr->LookupOrCreate<ComputeSessionPoolResource>(
                         container, "pool", &pool_resource, create_pool));
    }
    ComputeSessionPool *pool = pool_resource->get();
    CHECK(pool != nullptr);

    // Get a new Session for this computation from the pool.
    std::unique_ptr<ComputeSession> session = pool->GetSession();
    const string id = std::to_string(session->Id());

    // Store it in the ResourceManager.
    OP_REQUIRES_OK(
        context,
        rmgr->Create<ComputeSessionResource>(
            container, id, new ComputeSessionResource(std::move(session))));

    Tensor *output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({2}), &output));
    output->vec<string>()(0) = container;
    output->vec<string>()(1) = id;

    // Unref the pool so it gets destroyed properly.
    pool_resource->Unref();
    VLOG(1) << "Returning session: " << id;
  }

 private:
  MasterSpec master_spec_;
  GridPoint grid_point_;

  // Mutex that serializes accesses to the resource manager. (These would block
  // in the compute session pool anyways, so there's no regression there, and
  // we need to protect from racy multiple initialization.)
  tensorflow::mutex lock_;

  TF_DISALLOW_COPY_AND_ASSIGN(GetSession);
};

REGISTER_KERNEL_BUILDER(Name("GetSession").Device(DEVICE_CPU), GetSession);

// Given a handle to a ComputeSession, returns it to the pool. As long as we
// start with "GetSession", DRAGNN graphs are thread-safe and there is no need
// for explicit multi-thread logic. As long as we end with "ReleaseSession",
// then memory usage will be constrained to the maximum number of concurrent
// requests.
class ReleaseSession : public OpKernel {
 public:
  explicit ReleaseSession(OpKernelConstruction *context) : OpKernel(context) {
    string master_spec_str;
    OP_REQUIRES_OK(context, context->MatchSignature({DT_STRING}, {}));
  }

  void Compute(OpKernelContext *context) override {
    auto handle = context->input(0).vec<string>();
    const string &container = handle(0);
    const string &id = handle(1);
    VLOG(1) << "Releasing session: " << id;
    ResourceMgr *rmgr = context->resource_manager();

    // Get the pool for this container.
    ComputeSessionPoolResource *pool_resource;
    TF_CHECK_OK(rmgr->Lookup<ComputeSessionPoolResource>(container, "pool",
                                                         &pool_resource));
    auto *pool = pool_resource->get();
    CHECK(pool != nullptr);

    // Get the compute session.
    ComputeSessionResource *session_resource = nullptr;
    TF_CHECK_OK(
        rmgr->Lookup<ComputeSessionResource>(container, id, &session_resource));

    // We need to release the ComputeSession from both the ResourceMgr and
    // the ComputeSessionPool. The order of release is critical. If the
    // resource is not first Delete()-ed from the ResourceMgr, then another
    // thread may try to Create() the same resource, resulting in an
    // "Already exists" error.
    //
    // First, delete the ResourceMgr reference so it can be used in the future.
    TF_CHECK_OK(rmgr->Delete<ComputeSessionResource>(container, id));

    // Second, return the ComputeSession to the pool.
    pool->ReturnSession(session_resource->release());

    // Unref the resources so they get destroyed properly.
    session_resource->Unref();
    pool_resource->Unref();
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ReleaseSession);
};

REGISTER_KERNEL_BUILDER(Name("ReleaseSession").Device(DEVICE_CPU),
                        ReleaseSession);

/*******************************************************************************
 *                   ComputeSessionOps below here.
 ******************************************************************************/

// Advances a session based on the next oracle (gold) action.
class AdvanceFromOracle : public ComputeSessionOp {
 public:
  explicit AdvanceFromOracle(OpKernelConstruction *context)
      : ComputeSessionOp(context) {
    OP_REQUIRES_OK(context, context->MatchSignature({DT_STRING}, {DT_STRING}));
  }

  bool OutputsHandle() const override { return true; }
  bool RequiresComponentName() const override { return true; }

  void ComputeWithState(OpKernelContext *context,
                        ComputeSession *session) override {
    session->AdvanceFromOracle(component_name());
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(AdvanceFromOracle);
};

REGISTER_KERNEL_BUILDER(Name("AdvanceFromOracle").Device(DEVICE_CPU),
                        AdvanceFromOracle);

// Advances the session using predicted action scores.
// The tensor of scores has shape batch_size x num_actions.
class AdvanceFromPrediction : public ComputeSessionOp {
 public:
  explicit AdvanceFromPrediction(OpKernelConstruction *context)
      : ComputeSessionOp(context) {
    OP_REQUIRES_OK(context,
                   context->MatchSignature({DT_STRING, DT_FLOAT}, {DT_STRING}));
  }

  bool OutputsHandle() const override { return true; }
  bool RequiresComponentName() const override { return true; }

  void ComputeWithState(OpKernelContext *context,
                        ComputeSession *session) override {
    const Tensor &scores = context->input(1);
    session->AdvanceFromPrediction(component_name(),
                                   scores.tensor<float, 2>().data(),
                                   scores.NumElements());
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(AdvanceFromPrediction);
};

REGISTER_KERNEL_BUILDER(Name("AdvanceFromPrediction").Device(DEVICE_CPU),
                        AdvanceFromPrediction);

// Given a handle to a ComputeSession and a channel index, outputs fixed
// features for that channel. Recall that a channel can only have a single
// feature implementation, but that feature can output up to a prespecified
// maximum number of ids.
//
// Fixed features ids are output in a preallocated tensor of size
// batch_size * max_num_ids. Therefore the ids for the first batch element
// are in output in the first max_num_ids elements of the tensor and so on.
// Unused elements in the tensor are set to -1.
class ExtractFixedFeatures : public ComputeSessionOp {
 public:
  explicit ExtractFixedFeatures(OpKernelConstruction *context)
      : ComputeSessionOp(context) {
    OP_REQUIRES_OK(context, context->GetAttr("channel_id", &channel_id_));
    OP_REQUIRES_OK(context, context->GetAttr("max_num_ids", &max_num_ids_));
    OP_REQUIRES_OK(context, context->MatchSignature(
        {DT_STRING, DT_INT32}, {DT_INT64}));
  }

  bool OutputsHandle() const override { return false; }
  bool RequiresComponentName() const override { return true; }

  void ComputeWithState(OpKernelContext *context,
                        ComputeSession *session) override {
    int batch_size = context->input(1).scalar<int>()();
    int output_size = batch_size * max_num_ids_;
    Tensor *ids;
    CHECK(context->allocate_output(0, TensorShape({output_size}), &ids).ok());

    int64 *output = ids->vec<int64>().data();
    session->GetInputFeatures(component_name(), channel_id_, output);
  }

 private:
  int channel_id_;
  int max_num_ids_;
  TF_DISALLOW_COPY_AND_ASSIGN(ExtractFixedFeatures);
};

REGISTER_KERNEL_BUILDER(Name("ExtractFixedFeatures").Device(DEVICE_CPU),
                        ExtractFixedFeatures);

// Given a ComputeSession and a channel index, outputs link features.
// Link features are returned as two vectors of size: batch_size * channel_size:
//   - step_idx: specifies the element to read in a tensor array of activations,
//   - idx: specifies the row within the tensor array element.
class ExtractLinkFeatures : public ComputeSessionOp {
 public:
  explicit ExtractLinkFeatures(OpKernelConstruction *context)
      : ComputeSessionOp(context) {
    OP_REQUIRES_OK(context, context->GetAttr("channel_id", &channel_id_));
    OP_REQUIRES_OK(context,
                   context->MatchSignature({DT_STRING}, {DT_INT32, DT_INT32}));
  }

  bool OutputsHandle() const override { return false; }
  bool RequiresComponentName() const override { return true; }

  void ComputeWithState(OpKernelContext *context,
                        ComputeSession *session) override {
    auto features =
        session->GetTranslatedLinkFeatures(component_name(), channel_id_);

    // Computes output size.
    const int64 num_indices = features.size();

    // Allocates output tensors.
    Tensor *step_idx_output;
    Tensor *idx_output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({num_indices}),
                                            &step_idx_output));
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, TensorShape({num_indices}), &idx_output));

    // Clip step_idx for all features. If a feature is empty, set the step
    // index to -1.
    for (int i = 0; i < features.size(); ++i) {
      if (!features[i].has_step_idx() || features[i].step_idx() < -1) {
        features[i].set_step_idx(-1);
      }
    }

    // Fills output tensors.
    for (int i = 0; i < features.size(); ++i) {
      // Sets the element to read from a tensor array of activations.
      step_idx_output->vec<int32>()(i) = features[i].step_idx();

      // Within the tensor array element the id has to account for batch index.
      idx_output->vec<int32>()(i) = (features[i].step_idx() >= 0)
              ? features[i].batch_idx() : 0;

      VLOG(2) << "features[" << i << "]: " << features[i].ShortDebugString();
    }
  }

 private:
  int channel_id_;
  TF_DISALLOW_COPY_AND_ASSIGN(ExtractLinkFeatures);
};

REGISTER_KERNEL_BUILDER(Name("ExtractLinkFeatures").Device(DEVICE_CPU),
                        ExtractLinkFeatures);

// Emits a vector of gold labels of size batch_size.
class EmitOracleLabels : public ComputeSessionOp {
 public:
  explicit EmitOracleLabels(OpKernelConstruction *context)
      : ComputeSessionOp(context) {
    OP_REQUIRES_OK(context, context->MatchSignature({DT_STRING}, {DT_INT32}));
  }
  bool OutputsHandle() const override { return false; }
  bool RequiresComponentName() const override { return true; }

  void ComputeWithState(OpKernelContext *context,
                        ComputeSession *session) override {
    VLOG(2) << "state->BatchSize: " << session->BatchSize(component_name());
    Tensor *output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0,
                       TensorShape({session->BatchSize(component_name())}),
                       &output));
    std::vector<int> labels = session->EmitOracleLabels(component_name());
    int index = 0;
    for (const auto &label : labels) {
      output->vec<int32>()(index) = label;
      ++index;
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(EmitOracleLabels);
};

REGISTER_KERNEL_BUILDER(Name("EmitOracleLabels").Device(DEVICE_CPU),
                        EmitOracleLabels);

// Given a handle to a ComponentState, emits a single bool indicating
// whether all elements in the batch are in their final states.
class EmitAllFinal : public ComputeSessionOp {
 public:
  explicit EmitAllFinal(OpKernelConstruction *context)
      : ComputeSessionOp(context) {
    OP_REQUIRES_OK(context, context->MatchSignature({DT_STRING}, {DT_BOOL}));
  }

  bool OutputsHandle() const override { return false; }
  bool RequiresComponentName() const override { return true; }

  void ComputeWithState(OpKernelContext *context,
                        ComputeSession *session) override {
    Tensor *output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({1}), &output));
    const bool is_terminal = session->IsTerminal(component_name());
    VLOG(2) << "EmitAllFinal: is_terminal = " << is_terminal;
    output->vec<bool>()(0) = is_terminal;
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(EmitAllFinal);
};

REGISTER_KERNEL_BUILDER(Name("EmitAllFinal").Device(DEVICE_CPU), EmitAllFinal);

// Prepares the given component for computation.
class InitComponentData : public ComputeSessionOp {
 public:
  explicit InitComponentData(OpKernelConstruction *context)
      : ComputeSessionOp(context) {
    OP_REQUIRES_OK(context,
                   context->MatchSignature({DT_STRING}, {DT_STRING}));
  }

  bool OutputsHandle() const override { return true; }

  bool RequiresComponentName() const override { return true; }

  void ComputeWithState(OpKernelContext *context,
                        ComputeSession *session) override {
    session->InitializeComponentData(component_name());
  }
};

REGISTER_KERNEL_BUILDER(Name("InitComponentData").Device(DEVICE_CPU),
                        InitComponentData);

// Returns the given component's batch size.
class BatchSize : public ComputeSessionOp {
 public:
  explicit BatchSize(OpKernelConstruction *context)
      : ComputeSessionOp(context) {
    OP_REQUIRES_OK(context, context->MatchSignature({DT_STRING}, {DT_INT32}));
  }

  bool OutputsHandle() const override { return false; }
  bool RequiresComponentName() const override { return true; }

  void ComputeWithState(OpKernelContext *context,
                        ComputeSession *session) override {
    Tensor *output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
    output->scalar<int>()() = session->BatchSize(component_name());
  }
};

REGISTER_KERNEL_BUILDER(Name("BatchSize").Device(DEVICE_CPU), BatchSize);

// Attaches a data source to the master.
class AttachDataReader : public ComputeSessionOp {
 public:
  explicit AttachDataReader(OpKernelConstruction *context)
      : ComputeSessionOp(context) {
    OP_REQUIRES_OK(
        context, context->MatchSignature({DT_STRING, DT_STRING}, {DT_STRING}));
  }

  bool OutputsHandle() const override { return true; }
  bool RequiresComponentName() const override { return false; }

  // Calls SetInputData() on the ComputeSession.
  void ComputeWithState(OpKernelContext *context,
                        ComputeSession *session) override {
    auto input_data(context->input(1).vec<string>());

    std::vector<string> data;
    for (int i = 0; i < input_data.size(); ++i) {
      data.push_back(input_data(i));
    }
    session->SetInputData(data);
  }
};

REGISTER_KERNEL_BUILDER(Name("AttachDataReader").Device(DEVICE_CPU),
                        AttachDataReader);

class WriteAnnotations : public ComputeSessionOp {
 public:
  explicit WriteAnnotations(OpKernelConstruction *context)
      : ComputeSessionOp(context) {
    OP_REQUIRES_OK(context, context->MatchSignature({DT_STRING}, {DT_STRING}));
  }

  bool OutputsHandle() const override { return true; }
  bool RequiresComponentName() const override { return true; }

  void ComputeWithState(OpKernelContext *context,
                        ComputeSession *session) override {
    session->FinalizeData(component_name());
  }
};

REGISTER_KERNEL_BUILDER(Name("WriteAnnotations").Device(DEVICE_CPU),
                        WriteAnnotations);

// Given a handle to a ComponentState, emits a vector of strings
// corresponding to the serialized predictions of the model.
class EmitAnnotations : public ComputeSessionOp {
 public:
  explicit EmitAnnotations(OpKernelConstruction *context)
      : ComputeSessionOp(context) {
    OP_REQUIRES_OK(context, context->MatchSignature({DT_STRING}, {DT_STRING}));
  }

  bool OutputsHandle() const override { return false; }
  bool RequiresComponentName() const override { return true; }

  void ComputeWithState(OpKernelContext *context,
                        ComputeSession *session) override {
    // Get the annotations from the state.
    auto annotations = session->GetSerializedPredictions();

    // Copy annotations to the output.
    Tensor *output;
    const int64 output_size = annotations.size();
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({output_size}), &output));
    auto annotations_output = output->vec<string>();
    for (int i = 0; i < annotations.size(); ++i) {
      annotations_output(i) = annotations[i];
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(EmitAnnotations);
};

REGISTER_KERNEL_BUILDER(Name("EmitAnnotations").Device(DEVICE_CPU),
                        EmitAnnotations);

}  // namespace dragnn
}  // namespace syntaxnet
