# Description:
#   TensorFlow C++ inference example with multiple CUDA streams execution.

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_cc_binary(
    name = "multistream_inference",
    srcs = ["main.cc"],
    linkopts = select({
        "//tensorflow:windows": [],
        "//tensorflow:macos": [
            "-lm",
            "-lpthread",
        ],
        "//tensorflow:ios": [
            "-lm",
            "-lpthread",
        ],
        "//conditions:default": [
            "-lm",
            "-lpthread",
            "-lrt",
        ],
    }),
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/compiler/jit:jit",
    ],
)

py_binary(
    name = "multistream_training",
    srcs = ["multistream.py"],
    main = "multistream.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)
