# SPDX-License-Identifier: Apache-2.0
# Qwen3-0.6B on the Tensor API. Link this directory into a LiteRT checkout as
# tensor/examples/qwen3-0.6b and build from the checkout root.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//litert/build_common:litert_build_defs.bzl", "commandline_flag_copts")
load(
    "//litert/build_common:special_rule.bzl",
    "litert_android_linkopts",
    "litert_gpu_accelerator_prebuilts",
)

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "qwen3_config",
    hdrs = ["qwen3_config.h"],
)

cc_library(
    name = "qwen3_graph",
    srcs = ["qwen3_graph.cc"],
    hdrs = ["qwen3_graph.h"],
    deps = [
        ":qwen3_config",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/tflite:arithmetic_tflite",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "qwen3_weights",
    srcs = ["qwen3_weights.cc"],
    hdrs = ["qwen3_weights.h"],
    deps = [
        ":qwen3_config",
        ":qwen3_graph",
        "//tensor",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/examples/gemma3:safetensor_loader",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "qwen3_main",
    srcs = ["qwen3_main.cc"],
    copts = select({
        "//litert:android": [],
        "//conditions:default": ["-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1"],
    }) + commandline_flag_copts(),
    data = litert_gpu_accelerator_prebuilts(),
    linkopts = litert_android_linkopts(),
    deps = [
        ":qwen3_config",
        ":qwen3_graph",
        ":qwen3_weights",
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_options",
        "//litert/cc/options:litert_gpu_options",
        "//tensor",
        "//tensor:datatypes",
        "//tensor/backends/tflite:tflite_flatbuffer_conversion",
        "//tensor/runners/litert:litert_dynamic_runner",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)
