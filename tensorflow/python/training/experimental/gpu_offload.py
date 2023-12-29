# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains functions to use mixed precision with the graph rewrite."""

import contextlib
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import ops
from tensorflow.python.eager import context as eager_context, def_function
from tensorflow.python.util.tf_export import tf_export


@tf_export('train.experimental.gpu_offload_scope')
@contextlib.contextmanager
def gpu_offload_scope(enabled=True):
  """Creates a context in which all op outputs are allocated using a
  GPU-offload allocator. This allows the allocations to exceed the size of GPU
  memory and be automatically migrated to (or from) the GPU on a per-page
  basis.
  Note: This is only supported when inside a tf.function.
  """
  if eager_context.executing_eagerly():
    raise RuntimeError(
        "gpu_offload_scope can only be used inside a tf.function")
  value = attr_value_pb2.AttrValue(b=enabled)
  attrs = {"_gpu_offload_outputs_enabled": value}
  # pylint: disable=protected-access
  with ops.get_default_graph()._attr_scope(attrs):
    yield


@tf_export('train.experimental.make_gpu_offload')
def make_gpu_offload(fn):
  """Calls fn() inside a context in which all op outputs are allocated using a
  GPU-offload allocator. This allows the allocations to exceed the size of GPU
  memory and be automatically migrated to (or from) the GPU on a per-page
  basis.
  Returns the result of fn().
  """
  @def_function.function  # Required for attribute scope to work
  def make():
    with gpu_offload_scope():
      return fn()
  return make()
