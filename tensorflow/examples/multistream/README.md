# **Multiple Stream TensorFlow**

## **Introduction**
Multiple Stream TensorFlow is developed based on the official [TensorFlow](https://github.com/tensorflow/tensorflow). It leverages the features of modern GPUs to accelerate deep learning training and inference. This Multi-Stream implementation has successfully helped several customers migrate their TF models to the GPU and go online.

See our talk at GTC2024: **"Accelerate Recommender Systems and Increase GPU Utilization With Multiple CUDA Streams"** ([Link](https://www.nvidia.com/en-us/on-demand/session/gtc24-s61278/)), and our talk at NVIDIA AI Open Day 2024 Summer (in Chinese, [Link](https://www.bilibili.com/video/BV1Py411q7Ym/)).

**Please note!!!** The multi-stream training feature is documented, but still in a private repo. This branch only contains multi-stream inference implementation. If you are interested in the training part, please reach out to <robinz@nvidia.com> and <ruotongw@nvidia.com> for the full access!

See the last section for code migration guidelines. Please reach out to <robinz@nvidia.com> and <ruotongw@nvidia.com> for technical support, thanks!

## **Key Features**

1. Multiple Streams Inference

The user can specify how many stream groups to create for each GPU. Each time the session.run() function is called, TF will schedule the most suitable stream group to execute. To reduce inter-stream dependencies at high concurrency, we also support different CUDA contexts for the stream groups. Eager mode (no explicit session) is not supported yet.

2. Multiple Streams Training

The user can dispatch different branches of the model to different stream groups. We provide a Python interface for this, like what is implemented in Pytorch, but more powerful. Users don't need to manage the cross-stream dependency manually. In eager mode, the Python interface is only supported inside a `tf.function`.

3. Stream Merging

In the native TF, one computation stream and multiple copy streams are introduced to a stream group to achieve parallelism of computation and copy. However, synchronization between the streams can cause significant overhead, especially in scenarios where copying is frequent. Now, with the Multi-Stream implementation, the parallelism of copy and computation can be done between stream groups, so we allow users to use only one stream for computation and copy within a stream group to alleviate the stream synchronization overhead.

4. Resources management among streams

In the design, we make all stream groups reuse the same set of model parameters to avoid taking up too much GPU memory, which is achieved by setting `TF_SEGMENT_OWN_CONST=true`. For other resources, we separate them between stream groups as much as possible to reduce the dependencies and achieve better speedups, e.g.

* Per-stream GPU allocator
* Per-stream host allocator if `TF_PER_STREAM_HOST_ALLOCATOR=true`
* Per-stream StreamExecutor
* Per-stream thread pool if multiple thread pools are set via ConfigProto in TF1-style code
* Per-stream GPU threads if `TF_GPU_THREAD_MODE=gpu_private`

The multi-stream GPU allocators can also share the same memory limit to avoid an imbalanced memory footprint by setting `TF_GPU_STREAM_GROUP_SHARE_MEM_POOL=true`.

Furthermore, we realize that sometimes memory consumption is critical in some cases. In those cases, `TF_GPU_STREAM_AWARE_BFC=true` can be set to let multiple streams use a shared single BFC allocator, to keep the memory consumption nearly the same as the vanilla BFC. However, there will be extra synchronization overhead between streams if memory is intensive.

## **Requirements**

Minimum CUDA Driver version: r465. Recommended CUDA Driver version: r525 or above.

Recommended CUDA Toolkit version: CUDA12.4 or above.

If you have customized operations in your system, please check whether your code follows the important TensorFlow and CUDA best practice: **DO NOT INVOLVE ANY CPU-GPU SYNCHRONIZATION IF NOT HAVE TO**. The synchronizations not only can result in large unnecessary CPU side overhead and be a blocker for multiple streams to take their best effort, but also can lead to deadlock in some cases especially when NCCL operations are involved. Here are the guidelines you should follow:

1. Avoid using `cudaDeviceSynchronize()` API in your customized op.

2. Avoid using APIs that cause implicit device synchronization, such as `cudaFree()`, `cudaMemcpy()`, `cudaMemset()`. Use the asynchronized version `cudaFreeAsync()`, `cudaMemcpyAsync()`, `cudaMemsetAsync()` instead.

3. Avoid using pageable memory HtoD/DtoH copy to transfer a large chunk of data. It's slow and will make the CPU thread wait until the data transfer is finished. Use pinned memory transfer instead.

4. Using `cudaStreamSynchronize()` is not that bad, but still not recommended. Because this API blocks the calling CPU thread. You can learn TensorFlow's practice that uses `cudaEventRecord()` + `cudaEventQuery()` to judge the stream status, and uses `cudaEventRecord()` + `cudaStreamWaitEvent()` to build the cross-stream dependency.

## **Build Instruction**
1. Recommended docker image:
 - tensorflow/build:2.13-python3.11

2. Configure TF

* Run ./configure
 - The generated .tf\_configure.bazelrc example:

```
build --action_env PYTHON_BIN_PATH="/usr/bin/python3"
build --action_env PYTHON_LIB_PATH="/usr/lib/python3/dist-packages"
build --python_path="/usr/bin/python3"
build --action_env CUDA_TOOLKIT_PATH="/usr/local/cuda-11.8"
build --action_env TF_CUDA_COMPUTE_CAPABILITIES="8.0"
build --action_env LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
build --action_env GCC_HOST_COMPILER_PATH="/usr/bin/x86_64-linux-gnu-gcc-9"
build --config=cuda
build:opt --copt=-Wno-sign-compare
build:opt --host_copt=-Wno-sign-compare
test --flaky_test_attempts=3
test --test_size_filters=small,medium
test --test_env=LD_LIBRARY_PATH
test:v1 --test_tag_filters=-benchmark-test,-no_oss,-oss_excluded,-no_gpu,-oss_serial
test:v1 --build_tag_filters=-benchmark-test,-no_oss,-oss_excluded,-no_gpu
test:v2 --test_tag_filters=-benchmark-test,-no_oss,-oss_excluded,-no_gpu,-oss_serial,-v1only
test:v2 --build_tag_filters=-benchmark-test,-no_oss,-oss_excluded,-no_gpu,-v1only
```

3. Build the Multi-Stream examples

* Run the following command to build:

```
# To build the training example
bazel build  --config=opt //tensorflow/examples/multistream:multistream_training

# To build the inference example
bazel build --config=opt //tensorflow/examples/multistream:multistream_inference
```

4. (optional) Build and install the TF whl package

```
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./bazel-bin/tensorflow/tools/pip_package/
pip install --force-reinstall --no-deps ./bazel-bin/tensorflow/tools/pip_package/tensorflow-2.13.1-cp311-cp311-linux_x86_64.whl
```

## **Run Testing**

### **Multi-Stream Training**

* Basic settings.

```
export TF_GPU_STREAM_GROUP_COUNT=4
export TF_NODE_LEVEL_MULTISTREAM=true

./bazel-bin/tensorflow/examples/multistream/multistream_training
```

Set `TF_GPU_STREAM_GROUP_COUNT=N` to create N stream groups. `N=1` will fall back to the original TF performance.

Set `TF_NODE_LEVEL_MULTISTREAM=true` to indicate that multi-stream training mode (node-level multiple streams) is enabled. Otherwise, the multi-stream inference mode (graph-level multiple streams) is enabled.

By setting the two environment variables above, operations in `tf.function` will be assigned to at most 4 stream groups for paralleled execution, according to the assignment rule in the file `multistream.py`. The coding specification is to specify stream groups in the form of TF scope:

```
with tf.cuda.stream_scope(tf.cuda.get_stream(2), include_grad=True):
 # Operations in this scope will be assigned to stream group 2.
 out = ...
```

Operations outside of any scope will be assigned to the default TF stream group, the same as using `with tf.cuda.stream_scope(tf.cuda.get_stream(0))`.

**Alert!!!** If you're using eager mode, keep in mind that you need to do manual management of the tensors in the function that needs to be used across stream groups to ensure that the lifecycle of this tensor lasts until the end of the step. See the `hold_tensors` comments in `multistream.py` for details. In non-eager mode (session-run style code), we already have support for automated tensor lifecycle management, so you don't need to worry about this. The eager mode automation is still under development.

The XLA support for multiple streams training is still experimental and some usages are not supported. The nodes in a TF function graph cannot follow the stream assignment scope for the function itself, because the stream assignment cannot be passed into the function graph, so nodes will be executed on the default stream. Also, using "@tf.function(jit_compile=True)" will lead to all the nodes inside the function being executed on the default stream.

* Advanced settings.

```
export TF_GPU_STREAM_GROUP_COUNT=4
export TF_NODE_LEVEL_MULTISTREAM=true
export TF_REDUCE_STREAM_WAIT=false
export TF_MULTI_STREAM_ALLOW_FORWARD_PROPAGATION=true
export TF_STREAM_FROM_FILE=/path/to/the/file.txt
export TF_GPU_STREAM_GROUP_SHARE_MEM_POOL=true

./bazel-bin/tensorflow/examples/multistream/multistream_training
```

The default `TF_REDUCE_STREAM_WAIT=true` will reduce unnecessary cross-stream waiting as much as possible while trying to keep the program running correctly. As it's an experimental feature, we allow disabling it by setting the environment variable to false.

Set `TF_MULTI_STREAM_ALLOW_FORWARD_PROPAGATION=true` to allow for forward propagation of the stream group assignment. For example, if one node is assigned to stream group 1, but one of its successor nodes is not assigned explicitly, we can move it to stream group 1 as well to reduce the cross-stream waiting overhead. However, turning on this feature does not guarantee performance gains, as the dependencies between graph nodes are often very complex.

Sometimes the stream groups set via the Python API are not necessarily accurate, for example, TF may internally optimize the graph and subsequently insert some new nodes. Therefore, we allow the user to set up stream groups for nodes on the final graph with regex via a file specified in the environment variable `TF_STREAM_FROM_FILE`. The file format should be

```
1:node_name_re1,node_name_re2
2:node_name_re3,node_name_re4
```

All nodes with regex matching `node_name_re1` or `node_name_re2` will be assigned to stream group 1. All nodes with regex matching `node_name_re3` or `node_name_re4` will be assigned to stream group 2.

Set `TF_GPU_STREAM_GROUP_SHARE_MEM_POOL=true` to let the multi-stream GPU allocators share the same memory limit to address imbalanced memory footprint. This usually happens when there are a lot of operations on one stream, but only a few operations on the other stream. By default, this option is set to false, and the allocators will evenly divide the memory to use.

Lastly, if you are in a memory-intensive situation or wish multiple streams behave like a single stream in the memory consumption aspect, you can set `TF_GPU_STREAM_AWARE_BFC=true` to let multiple streams share a single BFC allocator and thus memory space. Be aware that this may cause overhead if memory is intensive, as it needs to do stream synchronization with the former stream when the memory chunk is being used by a new different stream. And in the design of stream-aware BFC, we would like it to reduce these stream syncs as much as possible, thus it has an aggressive memory-allocating behavior. For example, suppose your model runs on a 40GB GPU, vanilla BFC consumes 32GB memory on a single stream, while stream-aware BFC may aggressively consume 40GB memory on multiple streams, because it favors allocating new chunks rather than borrowing existing chunks to reduce stream synchronization, if existing chunks are all on different streams. However, you can set:

```
 sess_config = tf.ConfigProto()
 sess_config.gpu_options.per_process_gpu_memory_fraction = 32.0 / 40.0
```

to set a hard limit for BFC. Stream-aware BFC should work under 32GB just fine, though some overhead may be added. In most cases, overhead should be negligible.

Also, there is another environment variable `TF_BFC_EXTEND_LIMIT_MB` can be set to limit the max memory space size every time stream aware extends. By default, BFC is greedy and always doubles the space when extending, which may cause a waste of space. The recommended env value is `2048` or `4096`. This can help to control total space under stream-aware BFC.

### **Multi-Stream Inference**

* Basic settings.

```
export TF_GPU_STREAM_GROUP_COUNT=4
export TF_NODE_LEVEL_MULTISTREAM=false

# command model_path number_of_input input_name number_of_output output_name batch_size iters number_of_threads config_path
./bazel-bin/tensorflow/examples/multistream/multistream_inference ./tensorflow/examples/multistream/data/model_test.pbtxt 1 input0 1 output 16 1000 4 ./tensorflow/examples/multistream/data/config_proto
```

* Advanced settings.

```
export TF_GPU_STREAM_GROUP_COUNT=4
export TF_NODE_LEVEL_MULTISTREAM=false
export TF_GPU_STREAM_MERGE=true
export TF_GPU_CONTEXT_COUNT=4
export TF_PER_STREAM_HOST_ALLOCATOR=true
export TF_SEGMENT_OWN_CONST=true
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=1

# enable MPS
nvidia-cuda-mps-control -d

# command model_path number_of_input input_name number_of_output output_name batch_size iters number_of_threads config_path
./bazel-bin/tensorflow/examples/multistream/multistream_inference ./tensorflow/examples/multistream/data/model_test.pbtxt 1 input0 1 output 16 1000 4 ./tensorflow/examples/multistream/data/config_proto

# disable MPS
echo quit | nvidia-cuda-mps-control
```

Set `TF_GPU_STREAM_MERGE=true` to merge the compute stream, H2D copy stream, and D2H copy stream in one stream group into one stream.

Set `TF_GPU_CONTEXT_COUNT=N` to create the N stream groups in N CUDA contexts to reduce CPU-side contention for the context lock. Enabling MPS by `nvidia-cuda-mps-control -d` is needed if multi-context is used. The performance gain is not guaranteed as MPS brings extra overheads.

Set `TF_PER_STREAM_HOST_ALLOCATOR=true` to create an exclusive GPU host allocator for every stream group to reduce the allocator contention overhead.

Set `TF_SEGMENT_OWN_CONST=true` to let the op_segment in the device hold the Constant op. This is helpful to save memory footprint when multi-stream is enabled. Otherwise, the constant tensor buffers would have to be copied multiple times. However, with multiple CUDA contexts on, this may cause cross-context memory copying, introducing more stream synchronization overhead.

In the config_proto file, if you create multiple `session_inter_op_thread_pool`, TF will create several thread pools in one session, rather than using a global thread pool. If you also use multi-stream, the `i`-th stream group will use the `i % number_of_threadpool`-th thread pool, so we recommend setting the number of the thread pool the same as the number of the stream groups. If multi-stream is not enabled, the user should specify which thread pool to use manually.

Set `TF_GPU_THREAD_MODE=gpu_private` and `TF_GPU_THREAD_COUNT=N` to use `N` additional threads for every stream group to execute GPU ops, instead of using the threads from the above session thread pool. Set `TF_GPU_THREAD_MODE=gpu_shared` to let all the stream groups share `N` additional threads.

See the **Best Practice** section in the [document](https://docs.google.com/document/d/1yL3lWk_iFKqLTyekkuaiKXZ78I0lPmD5kM1fghHRs4Y/edit?usp=sharing) for more information.

## **Migration Instruction**

You may need to migrate the multiple streams functionality to your own version of TensorFlow. Here are some considerations for migration.

Besides the examples and README commit, our multi-stream implementation consists of four commits:

1. stream merging

2. multiple streams for inference

3. multiple streams for training

4. stream-aware BFC allocator

If you need multiple streams for the inference system, migrating the second commit should be sufficient to enable the feature. But the first commit is also worth trying as it can help reduce the stream-switching overhead in one stream group, which can be the main overhead if you have a lot of small HtoD/DtoH data transfers or if you are using an old version of CUDA driver (older than r515).

If you need multiple streams for the training system, you have to migrate all the first three commits. There is a new class called `TensorHolder` in the first commit, and it will be used in the third commit. You can just migrate this class and its object in the OpKernel params if you don't want to migrate the whole first commit.

The fourth commit implements a new allocator that helps reduce the GPU memory consumption. If the GPU goes OOM after enabling multiple streams and using the environment variable `TF_SEGMENT_OWN_CONST=true` cannot help, this commit can help you reduce the memory usage to the same level as with single streams, but may also bring some synchronization overhead.
