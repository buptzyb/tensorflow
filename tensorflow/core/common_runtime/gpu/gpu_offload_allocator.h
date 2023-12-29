/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_OFFLOAD_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_OFFLOAD_ALLOCATOR_H_

#include <string>

#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/tsl/framework/allocator.h"

namespace tensorflow {

// An allocator for GPU-offload memory. Memory allocated with this allocator
// can be accessed from both host and device, and can be larger than the device
// memory capacity. The GPU driver transparently migrates accessed pages; this
// can be slow on x86 systems, but is much faster on NVIDIA Grace-Hopper
// systems where it provides an efficient means of offloading large allocations.
class GpuOffloadAllocator : public tsl::Allocator {
 public:
  // Note: stream_exec cannot be null.
  explicit GpuOffloadAllocator(se::StreamExecutor* stream_exec, int numa_node,
                               bool is_cpu_allocator = false);

  std::string Name() override { return "GpuOffloadAllocator"; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;

  void SetStreamAndPreallocateMemory(void* stream) override;

  tsl::AllocatorMemoryType GetMemoryType() const override {
    return tsl::AllocatorMemoryType::kMigratable;
  }

 private:
  se::StreamExecutor* stream_exec_;  // not owned, non-null
  CUstream cuda_stream_;
  int numa_node_;
  bool can_access_pageable_;
  bool can_migrate_on_demand_;
  int offload_level_;
};

class SubGpuOffloadAllocator : public tsl::SubAllocator,
                               public GpuOffloadAllocator {
 public:
  // Note: stream_exec cannot be null.
  explicit SubGpuOffloadAllocator(se::StreamExecutor* stream_exec,
                                  int numa_node, bool is_cpu_allocator = false)
      : SubAllocator(std::vector<Visitor>(), std::vector<Visitor>()),
        GpuOffloadAllocator(stream_exec, numa_node, is_cpu_allocator) {}
  ~SubGpuOffloadAllocator() override {}

  void* Alloc(size_t alignment, size_t num_bytes,
              size_t* bytes_received) override {
    *bytes_received = num_bytes;
    return AllocateRaw(alignment, num_bytes);
  }

  void Free(void* ptr, size_t num_bytes) override { DeallocateRaw(ptr); }

  void SetStreamAndPreallocateMemory(void* stream) override {
    GpuOffloadAllocator::SetStreamAndPreallocateMemory(stream);
  }

  bool SupportsCoalescing() const override { return false; }

  tsl::AllocatorMemoryType GetMemoryType() const override {
    return tsl::AllocatorMemoryType::kMigratable;
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_OFFLOAD_ALLOCATOR_H_
