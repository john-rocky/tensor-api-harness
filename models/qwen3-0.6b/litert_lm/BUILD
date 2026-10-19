# SPDX-License-Identifier: Apache-2.0
# Qwen3-0.6B on the Tensor API as a LiteRT-LM executor backend. overlay.sh
# links this directory's files and the graph sources of ../graph into a
# LiteRT-LM checkout as tensor_api/qwen3-0.6b; build from the checkout root.
# The graph headers keep their include paths through include_prefix.

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "qwen3_graph",
    srcs = [
        "qwen3_graph.cc",
        "qwen3_weights.cc",
    ],
    hdrs = [
        "qwen3_config.h",
        "qwen3_graph.h",
        "qwen3_weights.h",
    ],
    include_prefix = "tensor/examples/qwen3-0.6b",
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "@litert//tensor",
        "@litert//tensor:arithmetic",
        "@litert//tensor:buffer",
        "@litert//tensor:datatypes",
        "@litert//tensor/backends/tflite:arithmetic_tflite",
        "@litert//tensor/examples/gemma3:safetensor_loader",
    ],
)

cc_library(
    name = "tensor_api_executor",
    srcs = ["tensor_api_executor.cc"],
    hdrs = ["tensor_api_executor.h"],
    deps = [
        ":qwen3_graph",
        "//runtime/executor:llm_executor",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/executor:llm_executor_processed_tokens",
        "//runtime/executor:llm_executor_settings",
        "//runtime/util:convert_tensor_buffer",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/c:litert_common",
        "@litert//litert/cc:litert_buffer_ref",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_environment_options",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc/options:litert_cpu_options",
        "@litert//litert/cc/options:litert_gpu_options",
        "@litert//tensor",
        "@litert//tensor:datatypes",
        "@litert//tensor/backends/tflite:tflite_flatbuffer_conversion",
        "@litert//tensor/runners/litert:litert_dynamic_runner",
        "@litert//tflite/delegates/xnnpack:xnnpack_delegate_hdrs_only",
    ],
)

# Registers --backend=tensor_api_cpu and --backend=tensor_api_gpu.
cc_library(
    name = "tensor_api_backend",
    srcs = ["tensor_api_backend.cc"],
    deps = [
        ":qwen3_graph",
        ":tensor_api_executor",
        "//runtime/components:model_resources",
        "//runtime/executor:executor_backend_registry",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor",
        "//runtime/executor:llm_executor_settings",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

# Loads model paths ending in .greedy.litertlm with greedy sampler parameters
# and a tokenizer that logs ids (for the output match).
cc_library(
    name = "greedy_litertlm_resources",
    srcs = ["greedy_litertlm_resources.cc"],
    deps = [
        "//runtime/components:model_resources",
        "//runtime/components:model_resources_litert_lm",
        "//runtime/executor:executor_backend_registry",
        "//runtime/executor:executor_settings_base",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:litert_lm_loader",
        "//runtime/util:scoped_file",
        "//support/tokenizer",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = 1,
)

# //runtime/engine:litert_lm_advanced_main with the same source, link options
# and dependencies, plus the two libraries above.
cc_binary(
    name = "litert_lm_advanced_main_tensor_api",
    srcs = ["//runtime/engine:litert_lm_advanced_main.cc"],
    additional_linker_inputs = select({
        "@platforms//os:windows": [
            "@litert//litert/c:windows_exported_symbols.def",
        ],
        "@platforms//os:linux": ["//runtime/engine:litert_lm_main.exported_symbols"],
        "//conditions:default": [],
    }),
    linkopts = select({
        "@litert//litert:litert_runtime_link_mode_dynamic": [],
        # Export LiteRt* symbols for LiteRt accelerator shlibs.
        "@platforms//os:ios": ["-Wl,-exported_symbol,_LiteRt*"],
        "@platforms//os:macos": ["-Wl,-exported_symbol,_LiteRt*"],
        "@platforms//os:windows": [
            "/DEF:$(location @litert//litert/c:windows_exported_symbols.def)",
        ],
        "@platforms//os:linux": ["-Wl,--dynamic-list=$(location //runtime/engine:litert_lm_main.exported_symbols)"],
        "//conditions:default": ["-Wl,--export-dynamic-symbol=LiteRt*"],
    }) + select({
        "@platforms//os:android": [
            "-lEGL",
            "-lGLESv3",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":greedy_litertlm_resources",
        ":tensor_api_backend",
        "//runtime/components/constrained_decoding:no_repeat_ngram_config",
        "//runtime/components/constrained_decoding:repetition_penalty_config",
        "//runtime/components/constrained_decoding:suppress_tokens_config",
        "//runtime/engine:io_types",
        "//runtime/engine:litert_lm_lib",
        "//runtime/engine:shared_flags",
        "//runtime/executor:executor_settings_base",
        "//runtime/proto:litert_lm_metrics_cc_proto",
        "//runtime/util:litert_status_util",
        "//runtime/util:logging",
        "//runtime/util:metrics_util",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:marshalling",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)
