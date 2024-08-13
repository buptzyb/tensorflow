#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::system_clock;

// The default batch size.
#define BATCH_SIZE 16
// The default infer iterations for each thread.
#define INFER_NUM 1000
// The default number of threads in parallel.
#define NUM_THREADS 3

namespace tensorflow {

// The GPU host allocator.
static Allocator *host_allocator = nullptr;

namespace example {

typedef std::vector<std::pair<std::string, Tensor>> InputsMap;

// The thread function for multiple-thread TF session run.
void TFRun(Session *sess, int infer_num,
           std::vector<std::pair<std::string, Tensor>> *inputs,
           std::vector<std::string> *output_names,
           std::vector<Tensor> *output_tensors) {
  for (int i = 0; i < infer_num; i++) {
    TF_CHECK_OK(sess->Run(*inputs, *output_names, {}, output_tensors));
  }
}

// The thread function for multiple-thread TF session run.
void TFRunWithCounter(Session *sess, int infer_num,
                      std::vector<std::pair<std::string, Tensor>> *inputs,
                      std::vector<std::string> *output_names,
                      std::vector<Tensor> *output_tensors,
                      std::vector<system_clock::time_point> *timestamp) {
  for (int i = 0; i < infer_num; i++) {
    timestamp->emplace_back(system_clock::now());
    TF_CHECK_OK(sess->Run(*inputs, *output_names, {}, output_tensors));
  }
  timestamp->emplace_back(system_clock::now());
}

void GetQPSLatency(
    std::vector<std::vector<system_clock::time_point>> *timestamps,
    std::string qtype = "Multiple Threads", int seperate = 10) {
  LOG(INFO) << "[TF] " << qtype << " finished.";
  std::vector<double> qps(seperate);
  std::vector<double> tp99(seperate);
  std::vector<double> tp95(seperate);
  std::vector<double> mean(seperate);
  double total_qps, total_tp99, total_tp95, total_mean;

  // Set the start and end time.
  auto start = timestamps->front().front();
  auto end = timestamps->front().back();
  for (int i = 1; i < timestamps->size(); ++i) {
    if ((*timestamps)[i].front() > start) {
      start = (*timestamps)[i].front();
    }
    if ((*timestamps)[i].back() < end) {
      end = (*timestamps)[i].back();
    }
  }

  // Create the buckets.
  auto bucket_len = (end - start) / seperate;
  std::vector<system_clock::time_point> buckets;
  for (int i = 0; i < seperate; ++i) {
    buckets.push_back(start + (i + 1) * bucket_len);
  }

  // Put latencies to the buckets.
  std::vector<std::vector<double>> latencies(seperate);
  for (int i = 0; i < timestamps->size(); ++i) {
    int bucket_idx = 0;
    for (int j = 1; j < (*timestamps)[i].size(); ++j) {
      if ((*timestamps)[i][j] > buckets[bucket_idx]) {
        ++bucket_idx;
        if (bucket_idx >= seperate) break;
      }
      latencies[bucket_idx].push_back(
          duration_cast<microseconds>((*timestamps)[i][j] -
                                      (*timestamps)[i][j - 1])
              .count() /
          1000.0);
    }
  }

  // Calculate qps, tp99, tp95, and mean of each bucket.
  for (int i = 0; i < seperate; ++i) {
    if (latencies[i].size() > 0) {
      qps[i] = 1000000.0 * latencies[i].size() /
               duration_cast<microseconds>(bucket_len).count();
      std::sort(latencies[i].begin(), latencies[i].end());
      tp99[i] = latencies[i][static_cast<size_t>(latencies[i].size() * 0.99)];
      tp95[i] = latencies[i][static_cast<size_t>(latencies[i].size() * 0.95)];
      double accumulate = 0;
      for (int j = 0; j < latencies[i].size(); ++j) {
        accumulate += latencies[i][j];
      }
      mean[i] = accumulate / latencies[i].size();
    } else {
      qps[i] = 0;
      tp99[i] = 0;
      tp95[i] = 0;
      mean[i] = 0;
    }
  }

  // Calculate total qps, tp99, tp95, and mean.
  std::vector<double> total_latencies;
  for (int i = 2; i < seperate - 3; ++i) {
    if (latencies[i].size() > 0) {
      total_latencies.insert(total_latencies.end(), latencies[i].begin(),
                             latencies[i].end());
    }
  }
  total_qps =
      1000000.0 * total_latencies.size() /
      (duration_cast<microseconds>(bucket_len).count() * (seperate - 5));
  std::sort(total_latencies.begin(), total_latencies.end());
  total_tp99 =
      total_latencies[static_cast<size_t>(total_latencies.size() * 0.99)];
  total_tp95 =
      total_latencies[static_cast<size_t>(total_latencies.size() * 0.95)];
  double accumulate = 0;
  for (int j = 0; j < total_latencies.size(); ++j) {
    accumulate += total_latencies[j];
  }
  total_mean = accumulate / total_latencies.size();

  // Print.
  auto duration_seconds =
      duration_cast<microseconds>(end - start).count() / 1000000.0;
  LOG(INFO) << "[TF] " << qtype << " Duration = " << duration_seconds
            << " seconds.";
  LOG(INFO) << "[TF] " << qtype << " QPS = " << total_qps;
  LOG(INFO) << "[TF] " << qtype << " TP99 Latency = " << total_tp99;
  LOG(INFO) << "[TF] " << qtype << " TP95 Latency = " << total_tp95;
  LOG(INFO) << "[TF] " << qtype << " Mean Latency = " << total_mean;
  std::string qps_str = "[";
  std::string tp99_str = "[";
  std::string tp95_str = "[";
  std::string mean_str = "[";
  for (int i = 0; i < seperate; ++i) {
    qps_str += std::to_string(qps[i]) + ",";
    tp99_str += std::to_string(tp99[i]) + ",";
    tp95_str += std::to_string(tp95[i]) + ",";
    mean_str += std::to_string(mean[i]) + ",";
  }
  qps_str += "]";
  tp99_str += "]";
  tp95_str += "]";
  mean_str += "]";
  VLOG(1) << "[TF] " << qtype << " QPS Vector = " << qps_str;
  VLOG(1) << "[TF] " << qtype << " TP99 Latency Vector = " << tp99_str;
  VLOG(1) << "[TF] " << qtype << " TP95 Latency Vector = " << tp95_str;
  VLOG(1) << "[TF] " << qtype << " Mean Latency Vector = " << mean_str;
}

// We assume that the first unkonwn dim is batch size.
TensorShape getNodeShape(const GraphDef &graph_def, const std::string name,
                         int batch_size) {
  for (int i = 0; i < graph_def.node_size(); i++) {
    auto n = graph_def.node(i);
    if (n.name() == name) {
      auto shape = n.attr().at("shape").shape();
      int dims = shape.dim_size();
      TensorShape tensorShape;
      for (int d = 0; d < dims; d++) {
        int dim_size = shape.dim(d).size();
        if (d == 0 && dim_size == -1) {
          int new_size = batch_size;
          dim_size = new_size;
        }
        tensorShape.AddDim(dim_size);
      }
      return tensorShape;
    }
  }
  LOG(ERROR) << "Cannot find the node" << name << std::endl;
  exit(1);
}

DataType getNodeType(const GraphDef &graph_def, const std::string name) {
  for (int i = 0; i < graph_def.node_size(); i++) {
    auto n = graph_def.node(i);
    if (n.name() == name) {
      auto dtype = n.attr().at("dtype").type();
      return dtype;
    }
  }
  LOG(ERROR) << "Cannot find the node" << name << std::endl;
  exit(1);
}

void RandomInitialize(Tensor &t) {
  int num_elements = t.NumElements();
  if (t.dtype() == DT_HALF) {
    auto *data = t.flat<Eigen::half>().data();
    for (int i = 0; i < num_elements; i++) {
      float value = static_cast<float>(rand() % 101 - 50) / 100.0f;
      data[i] = static_cast<Eigen::half>(value);
    }
  } else if (t.dtype() == DT_FLOAT) {
    float *data = t.flat<float>().data();
    for (int i = 0; i < num_elements; i++) {
      float value = static_cast<float>(rand() % 101 - 50) / 100.0f;
      data[i] = value;
    }
  } else if (t.dtype() == DT_INT32) {
    int *data = t.flat<int>().data();
    for (int i = 0; i < num_elements; i++) {
      int value = static_cast<int>(rand() % 10000);
      data[i] = value;
    }
  } else if (t.dtype() == DT_BOOL) {
    bool *data = t.flat<bool>().data();
    for (int i = 0; i < num_elements; i++) {
      bool value = static_cast<bool>(rand() % 2);
      data[i] = value;
    }
  } else if (t.dtype() == DT_INT64) {
    int64 *data = t.flat<int64>().data();
    for (int i = 0; i < num_elements; i++) {
      int64 value = static_cast<int64>(rand() % 10000);
      data[i] = value;
    }
  } else {
    LOG(WARNING) << "Unsupported data type.";
  }
  return;
}

void PrintTensorData(Tensor &t) {
  void *data;
  if (t.dtype() == DT_HALF) {
    data = static_cast<void *>(t.flat<Eigen::half>().data());
  } else if (t.dtype() == DT_FLOAT) {
    data = static_cast<void *>(t.flat<float>().data());
  } else if (t.dtype() == DT_BOOL) {
    data = static_cast<void *>(t.flat<bool>().data());
  } else if (t.dtype() == DT_INT32) {
    data = static_cast<void *>(t.flat<int>().data());
  } else if (t.dtype() == DT_INT64) {
    data = static_cast<void *>(t.flat<int64>().data());
  } else {
    LOG(WARNING) << "Print Tensor: Unsupported data type!";
    return;
  }

  int dims = t.dims();
  std::cout << "shape: " << std::endl;
  for (int i = 0; i < dims; i++) {
    std::cout << t.dim_size(i) << ", ";
  }
  std::cout << std::endl;

  // Print the first 32 elements.
  int size = t.NumElements();
  size = size > 32 ? 32 : size;

  for (int i = 0; i < size; i++) {
    float value;
    if (t.dtype() == DT_HALF) {
      value = static_cast<float>(static_cast<Eigen::half *>(data)[i]);
    } else if (t.dtype() == DT_INT32) {
      value = static_cast<int *>(data)[i];
    } else if (t.dtype() == DT_INT64) {
      value = static_cast<int64 *>(data)[i];
    } else if (t.dtype() == DT_BOOL) {
      value = static_cast<bool *>(data)[i];
    } else {
      value = static_cast<float *>(data)[i];
    }
    std::cout << value << ", ";
  }
  std::cout << std::endl;
}

void GenerateInputs(GraphDef &graph_def, const std::vector<string> &input_names,
                    std::vector<Tensor> &input_tensors, int batch_size) {
  input_tensors.clear();
  for (int i = 0; i < input_names.size(); i++) {
    auto tensorshape = getNodeShape(graph_def, input_names[i], batch_size);
    auto tensortype = getNodeType(graph_def, input_names[i]);
    Tensor t = host_allocator ? Tensor(host_allocator, tensortype, tensorshape)
                              : Tensor(tensortype, tensorshape);
    RandomInitialize(t);
    input_tensors.push_back(t);
  }
}

void FillInputsMap(InputsMap &inputs_map, std::vector<std::string> &input_names,
                   std::vector<Tensor> &input_tensors) {
  assert(input_names.size() == input_tensors.size());
  for (size_t i = 0; i < input_tensors.size(); i++) {
    inputs_map.push_back(
        std::pair<std::string, Tensor>(input_names[i], input_tensors[i]));
  }
}

void CopyTensorContents(Tensor &dst_tensor, Tensor &src_tensor) {
  int dst_eles = dst_tensor.NumElements();
  int src_eles = src_tensor.NumElements();
  if (dst_eles != src_eles) {
    LOG(ERROR) << "number of elements not match";
  }

  char *dst, *src;
  int ele_size;
  if (dst_tensor.dtype() == DT_HALF) {
    dst = reinterpret_cast<char *>(dst_tensor.flat<Eigen::half>().data());
    src = reinterpret_cast<char *>(src_tensor.flat<Eigen::half>().data());
    ele_size = 2;
  } else if (dst_tensor.dtype() == DT_FLOAT) {
    dst = reinterpret_cast<char *>(dst_tensor.flat<float>().data());
    src = reinterpret_cast<char *>(src_tensor.flat<float>().data());
    ele_size = 4;
  } else if (dst_tensor.dtype() == DT_INT32) {
    dst = reinterpret_cast<char *>(dst_tensor.flat<int>().data());
    src = reinterpret_cast<char *>(src_tensor.flat<int>().data());
    ele_size = 4;
  } else if (dst_tensor.dtype() == DT_BOOL) {
    dst = reinterpret_cast<char *>(dst_tensor.flat<bool>().data());
    src = reinterpret_cast<char *>(src_tensor.flat<bool>().data());
    ele_size = 1;
  } else if (dst_tensor.dtype() == DT_INT64) {
    dst = reinterpret_cast<char *>(dst_tensor.flat<int64>().data());
    src = reinterpret_cast<char *>(src_tensor.flat<int64>().data());
    ele_size = 8;
  } else {
    LOG(ERROR) << "Copy Tensor: Unsupported data type!" << std::endl;
    return;
  }

  for (int i = 0; i < src_eles * ele_size; i++) {
    dst[i] = src[i];
  }
}

inline void SetDevice(const string &device, GraphDef *graph_def) {
  for (int i = 0; i < graph_def->node_size(); ++i) {
    auto node = graph_def->mutable_node(i);
    if (node->device().empty()) {
      node->set_device(device);
    }
  }
}

void SetSessionOptions(SessionOptions &options, string &config_proto) {
  ConfigProto *config = &options.config;
  if (!config_proto.empty()) {
    Status s = ReadTextProto(Env::Default(), config_proto.c_str(), config);
    if (!s.ok()) {
      s = ReadBinaryProto(Env::Default(), config_proto.c_str(), config);
      if (!s.ok()) {
        LOG(ERROR) << "Read config proto from file " << config_proto
                   << " failed: " << s.ToString()
                   << ". Use default ConfigProto.";
        return;
      }
    }
    VLOG(1) << "Read config proto: " << config->DebugString();

    if (config->graph_options().optimizer_options().global_jit_level() ==
        tensorflow::OptimizerOptions::ON_1) {
      // Use XLA.
      auto *flags = tensorflow::GetMarkForCompilationPassFlags();
      flags->tf_xla_cpu_global_jit = true;
      flags->tf_xla_min_cluster_size = 1;
      tensorflow::SetXlaAutoJitFlagFromFlagString("single-gpu(2)");
    }
  }
}

Status Test(GraphDef &graph_def, std::vector<std::string> &input_names,
            std::vector<std::string> &output_names, int batch_size,
            int num_infers_per_thread, int num_threads, string config_proto) {
  SessionOptions options;
  SetSessionOptions(options, config_proto);

  std::unique_ptr<Session> session(NewSession(options));
  TF_CHECK_OK(session->Create(graph_def));

  const DeviceMgr *device_manager;
  TF_CHECK_OK(session->LocalDeviceManager(&device_manager));
  std::vector<Device *> devices = device_manager->ListDevices();
  for (auto *d : devices) {
    if (d->parsed_name().type == "CPU") {
      LOG(INFO) << "CPU device: " << d->name();
      AllocatorAttributes attr;
      attr.set_gpu_compatible(true);
      host_allocator = d->GetAllocator(attr);
      break;
    }
  }

  std::vector<Tensor> input_tensors_tf;
  GenerateInputs(graph_def, input_names, input_tensors_tf, batch_size);

  InputsMap inputs_tf;
  FillInputsMap(inputs_tf, input_names, input_tensors_tf);

  std::vector<std::vector<Tensor>> output_tensors_tf(num_threads);
  std::vector<std::thread> threads;

  // The first session run is slow due to resource initialization.
  TFRun(session.get(), num_infers_per_thread, &inputs_tf, &output_names,
        &output_tensors_tf[0]);
  sleep(1);

  // The second session run can be used to compare single thread performance.
  std::vector<std::vector<system_clock::time_point>> timestamp(1);
  TFRunWithCounter(session.get(), num_infers_per_thread, &inputs_tf,
                   &output_names, &output_tensors_tf[0], &timestamp[0]);
  GetQPSLatency(&timestamp, "Single Thread");
  sleep(1);

  // The third session run can be used to compare multiple threads performance.
  std::vector<std::vector<system_clock::time_point>> timestamps(num_threads);
  for (int i = 0; i < num_threads; i++) {
    timestamps[i].reserve(num_infers_per_thread + 1);
  }
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(
        TFRunWithCounter, session.get(), num_infers_per_thread, &inputs_tf,
        &output_names, &output_tensors_tf[i], &timestamps[i]));
  }
  for (auto &thread : threads) {
    thread.join();
  }

  for (int i = 0; i < num_threads; i++) {
    LOG(INFO) << "TF results: ";
    // Print the first output tensor.
    PrintTensorData(output_tensors_tf[i][0]);
    output_tensors_tf[i].clear();
  }

  GetQPSLatency(&timestamps, "Multiple Threads");

  TF_CHECK_OK(session->Close());
  return OkStatus();
}
}  // end namespace example
}  // end namespace tensorflow

using namespace tensorflow;

int main(int argc, char *argv[]) {
  // Example: ./application model_path in_num input_name [input_names]
  //                        out_num output_name [output_names]
  //                        batch_size infer_num_per_thread num_threads
  //                        config_path

  int arg_idx = 1;
  std::string model_path = argv[arg_idx++];
  int input_num = std::stoi(argv[arg_idx++]);

  std::vector<std::string> input_names;
  std::cout << input_num << " inputs: ";
  for (int i = 0; i < input_num; i++) {
    input_names.push_back(argv[arg_idx++]);
    std::cout << argv[arg_idx - 1] << ",";
  }
  std::cout << std::endl;

  int output_num = std::stoi(argv[arg_idx++]);
  std::vector<std::string> output_names;
  std::cout << output_num << " outputs: ";
  for (int i = 0; i < output_num; i++) {
    output_names.push_back(argv[arg_idx++]);
    std::cout << argv[arg_idx - 1] << ",";
  }
  std::cout << std::endl;

  int batch_size = BATCH_SIZE;
  if (argc > arg_idx) {
    batch_size = std::stoi(argv[arg_idx++]);
  }
  std::cout << "batch size = " << batch_size << std::endl;

  int num_infers_per_thread = INFER_NUM;
  if (argc > arg_idx) {
    num_infers_per_thread = std::stoi(argv[arg_idx++]);
  }
  std::cout << "num_infers_per_thread = " << num_infers_per_thread << std::endl;

  int num_threads = NUM_THREADS;
  if (argc > arg_idx) {
    num_threads = std::stoi(argv[arg_idx++]);
  }
  std::cout << "num_threads = " << num_threads << std::endl;

  std::string config_proto;
  if (argc > arg_idx) {
    config_proto = argv[arg_idx++];
  }
  std::cout << "config_proto = " << config_proto << std::endl;

  GraphDef graph_def;
  Status status;

  // Accept pb or pbtxt files here.
  if (model_path.find(".pbtxt") == std::string::npos) {
    status = ReadBinaryProto(Env::Default(), model_path, &graph_def);
  } else {
    status = ReadTextProto(Env::Default(), model_path, &graph_def);
  }

  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  TF_CHECK_OK(example::Test(graph_def, input_names, output_names, batch_size,
                            num_infers_per_thread, num_threads, config_proto));

  return 0;
}
