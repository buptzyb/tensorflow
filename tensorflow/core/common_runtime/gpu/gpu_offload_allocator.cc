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

#if GOOGLE_CUDA
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/compiler/xla/stream_executor/platform.h"
#include "third_party/gpus/cuda/include/cuda.h"
#endif

#include "absl/strings/str_cat.h"
#include "tensorflow/core/common_runtime/gpu/gpu_offload_allocator.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/mem.h"
#include "tensorflow/tsl/platform/platform.h"

#ifdef __linux__
#include <sys/mman.h>  // For madvise
#endif                 // __linux__

namespace tensorflow {

namespace {

#if GOOGLE_CUDA
using GPUResult = CUresult;
#endif

#if GOOGLE_CUDA
std::string GetGpuErrorMessage(GPUResult result) {
  const char* error;
  cuGetErrorString(result, &error);
  const char* name;
  cuGetErrorName(result, &name);
  return absl::StrCat("CUDA error: ", error ? error : "<unknown>", " (",
                      name ? name : "Unknown", ")");
}
#endif

bool GpuPageableMemoryAccessSupported() {
#if GOOGLE_CUDA
  // We assume that all GPU devices in the system have the same result.
  const int device = 0;
  int result;
  const GPUResult gpu_ret = cuDeviceGetAttribute(
      &result, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, device);
  if (gpu_ret) {
    LOG(ERROR) << "Failed to query GPU direct pageable memory access support: "
               << GetGpuErrorMessage(gpu_ret);
    return false;
  }
  return static_cast<bool>(result);
#else
  return false;
#endif
}

bool GpuConcurrentManagedAccessSupported() {
#if GOOGLE_CUDA
  // We assume that all GPU devices in the system have the same result.
  const int device = 0;
  int result;
  const GPUResult gpu_ret = cuDeviceGetAttribute(
      &result, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, device);
  if (gpu_ret) {
    LOG(ERROR) << "Failed to query GPU direct pageable memory access support: "
               << GetGpuErrorMessage(gpu_ret);
    return false;
  }
  return static_cast<bool>(result);
#else
  return false;
#endif
}

int GetGpuOffloadLevel(bool is_cpu_allocator) {
  int64 offload_level;
  TF_CHECK_OK(ReadInt64FromEnvVar(
      is_cpu_allocator ? "TF_OFFLOAD_CPU_LEVEL" : "TF_OFFLOAD_GPU_LEVEL",
      /*default_val=*/0, &offload_level));
  if (offload_level <= 0) {
    LOG(FATAL) << "OffloadAllocator should not be used for offload level "
               << offload_level;
  }
  if (offload_level > 5) offload_level = 5;
  if (is_cpu_allocator) offload_level = 6 - offload_level;
  return offload_level;
}

}  // namespace

GpuOffloadAllocator::GpuOffloadAllocator(se::StreamExecutor* stream_exec,
                                         int numa_node, bool is_cpu_allocator)
    : stream_exec_(stream_exec), numa_node_(numa_node) {
  DCHECK(stream_exec_ != nullptr);
  cuda_stream_ = nullptr;
  se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
  can_access_pageable_ = GpuPageableMemoryAccessSupported();
  can_migrate_on_demand_ = GpuConcurrentManagedAccessSupported();
  offload_level_ = GetGpuOffloadLevel(is_cpu_allocator);
  VLOG(1) << "GpuOffloadAllocator::GpuOffloadAllocator can_access_pageable_="
          << can_access_pageable_
          << " can_migrate_on_demand_=" << can_migrate_on_demand_
          << " offload_level_=" << offload_level_;
}

void* GpuOffloadAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  VLOG(2) << "GpuOffloadAllocator::AllocateRaw alignment=" << alignment
          << ", num_bytes=" << num_bytes;
  void* ptr = nullptr;
#if GOOGLE_CUDA
  if (offload_level_ >= 5) {
    // The highest level offloading. Memory is pinned at the CPU side.
    if (num_bytes > 0) {
      ptr = stream_exec_->HostMemoryAllocate(num_bytes);
      if (ptr == nullptr) {
        LOG(WARNING) << "could not allocate pinned host memory of size: "
                     << num_bytes;
        return ptr;
      }
      VLOG(3)
          << "GpuOffloadAllocator::AllocateRaw HostMemory without migration: "
          << ptr;
    }
  } else if (can_access_pageable_) {
    constexpr size_t kHugePageAlignment = 2 * 1024 * 1024;  // 2 MiB
    // Can use regular CPU allocator.
    ptr = tsl::port::AlignedMalloc(num_bytes,
                                   std::max(alignment, kHugePageAlignment));
#ifdef __linux__
    // Enable Transparent Huge Pages (THP) for the allocation to reduce TLB
    // pressure.
    if (::madvise(ptr, num_bytes, MADV_HUGEPAGE) != 0) {
      LOG(WARNING) << "GpuOffloadAllocator: Could not enable Transparent Huge "
                      "Pages (THP) on allocation: "
                   << ::strerror(errno);
    }
#endif  // __linux__
  } else if (can_migrate_on_demand_) {
    // Must use GPU managed memory allocator.
    se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
    CUdeviceptr gpu_ptr;
    GPUResult gpu_ret =
        cuMemAllocManaged(&gpu_ptr, num_bytes, CU_MEM_ATTACH_GLOBAL);
    if (gpu_ret) {
      LOG(ERROR) << "cuMemAllocManaged failed: " << GetGpuErrorMessage(gpu_ret);
      return nullptr;
    }
    if (gpu_ptr & (alignment - 1)) {
      LOG(ERROR) << "Pointer returned by cuMemAllocManaged does not meet "
                    "required alignment of "
                 << alignment << " bytes.";
      return nullptr;
    }
    if (offload_level_ == 4) {
      // Use UM driver to manage tensors, but they are still pinned on the CPU
      // side.
      gpu_ret =
          cuMemAdvise(gpu_ptr, num_bytes, CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                      CU_DEVICE_CPU);
      if (gpu_ret) {
        LOG(ERROR) << "Failed to cuMemAdvise prefer GPU: "
                   << GetGpuErrorMessage(gpu_ret);
        return nullptr;
      }
    } else if (offload_level_ <= 2) {
      // Use UM driver to manage tensors, and migration is allowed.
      CUdevice device;
      gpu_ret = cuDeviceGet(
          &device,
          stream_executor::DeviceOrdinalHelper::DecodeDeviceFromOrdinal(
              stream_exec_->device_ordinal()));
      if (gpu_ret) {
        LOG(ERROR) << "Failed to get GPU device: "
                   << GetGpuErrorMessage(gpu_ret)
                   << ", device ordinal: " << stream_exec_->device_ordinal();
        return nullptr;
      }
      gpu_ret = cuMemAdvise(gpu_ptr, num_bytes,
                            CU_MEM_ADVISE_SET_PREFERRED_LOCATION, device);
      if (gpu_ret) {
        LOG(ERROR) << "Failed to cuMemAdvise prefer GPU: "
                   << GetGpuErrorMessage(gpu_ret);
        return nullptr;
      }
      if (offload_level_ == 1) {
        // Prefetch the memory.
        CHECK(cuda_stream_ != nullptr);
        gpu_ret = cuMemPrefetchAsync(gpu_ptr, num_bytes, device, cuda_stream_);
        if (gpu_ret) {
          LOG(ERROR) << "Failed to cuMemPrefetchAsync: "
                     << GetGpuErrorMessage(gpu_ret);
          return nullptr;
        }
      }
    }
    ptr = reinterpret_cast<void*>(gpu_ptr);
  } else {
    LOG(ERROR) << "System does not support GPU offload allocations.";
    return nullptr;
  }
#endif  // GOOGLE_CUDA
  VLOG(2) << "GpuOffloadAllocator::AllocateRaw returning ptr=" << ptr;
  return ptr;
}

void GpuOffloadAllocator::DeallocateRaw(void* ptr) {
  VLOG(2) << "GpuOffloadAllocator::DeallocateRaw ptr=" << ptr;
#if GOOGLE_CUDA
  if (offload_level_ >= 5) {
    if (ptr != nullptr) {
      stream_exec_->HostMemoryDeallocate(ptr);
    }
    VLOG(3)
        << "GpuOffloadAllocator::DeallocateRaw HostMemory without migration: "
        << ptr;
  } else if (can_access_pageable_) {
    tsl::port::AlignedFree(ptr);
  } else if (can_migrate_on_demand_) {
    se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
    const CUdeviceptr gpu_ptr = reinterpret_cast<CUdeviceptr>(ptr);
    const GPUResult gpu_ret = cuMemFree(gpu_ptr);
    if (gpu_ret) {
      LOG(ERROR) << "Failed to free memory allocated by cuMemAllocManaged: "
                 << GetGpuErrorMessage(gpu_ret);
    }
  }
#endif  // GOOGLE_CUDA
}

void GpuOffloadAllocator::SetStreamAndPreallocateMemory(void* stream) {
  CUstream new_cuda_stream = *(static_cast<CUstream*>(stream));
  // We don't need to re-set the CUDA stream if this is the same stream
  if (cuda_stream_ != nullptr && new_cuda_stream != cuda_stream_) {
    LOG(FATAL) <<  // Crash OK.
        "Trying to set the stream twice. This isn't supported. ";
  }
  cuda_stream_ = new_cuda_stream;
}

}  // namespace tensorflow
