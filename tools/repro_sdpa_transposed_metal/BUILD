# SPDX-License-Identifier: Apache-2.0
# Minimal reproduction of odml.sdpa_transposed on the GPU accelerator. Link
# this directory into a LiteRT checkout as tensor/examples/sdpa_transposed_repro
# and build from the checkout root (run.sh does both).

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("//litert/build_common:litert_build_defs.bzl", "commandline_flag_copts")
load(
    "//litert/build_common:special_rule.bzl",
    "litert_android_linkopts",
    "litert_gpu_accelerator_prebuilts",
)

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "sdpa_repro_main",
    srcs = ["sdpa_repro_main.cc"],
    copts = select({
        "//litert:android": [],
        "//conditions:default": ["-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1"],
    }) + commandline_flag_copts(),
    data = litert_gpu_accelerator_prebuilts(),
    linkopts = litert_android_linkopts(),
    deps = [
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_options",
        "//litert/cc:litert_tensor_buffer",
        "//litert/cc/options:litert_gpu_options",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/tflite:arithmetic_tflite",
        "//tensor/backends/tflite:tflite_flatbuffer_conversion",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
    ],
)
