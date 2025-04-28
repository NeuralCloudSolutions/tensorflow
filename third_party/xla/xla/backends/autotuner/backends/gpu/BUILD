load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "gpu_codegen_backend",
    hdrs = ["gpu_codegen_backend.h"],
    deps = [
        "//xla:xla_proto_cc",
        "//xla/backends/autotuner:codegen_backend",
        "//xla/hlo/ir:hlo",
        "//xla/service:compiler",
        "//xla/service:executable",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "cublas",
    srcs = ["cublas.cc"],
    hdrs = ["cublas.h"],
    deps = [
        ":gpu_codegen_backend",
        "//xla:shape_util",
        "//xla/backends/autotuner:codegen_backend",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:compiler",
        "//xla/service:hlo_cost_analysis",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:cublas_cudnn",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/gpu:redzone_allocator",
        "//xla/tools:hlo_decomposer_lib",
        "//xla/tsl/lib/gtl:iterator_range",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ] + select({
        "@local_config_cuda//cuda:FALSE": [],
        "//conditions:default": [
            "//xla/service/gpu:matmul_utils",
            "//xla/service/gpu/transforms:dot_algorithm_rewriter",
            "//xla/service/gpu/transforms:gemm_rewriter",
            "//xla/service/gpu/transforms:priority_fusion",
        ],
    }),
)

xla_test(
    name = "cublas_test",
    srcs = ["cublas_test.cc"],
    backends = [
        "gpu",
    ],
    deps = [
        ":cublas",
        "//xla/backends/autotuner:codegen_backend",
        "//xla/hlo/ir:hlo",
        "//xla/service:compiler",
        "//xla/service:hlo_runner",
        "//xla/service:platform_util",
        "//xla/service/gpu:nvptx_compiler_impl",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tests:hlo_runner_agnostic_test_base",
        "//xla/tests:test_utils",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
