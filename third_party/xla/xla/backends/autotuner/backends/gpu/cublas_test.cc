/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/autotuner/backends/gpu/cublas.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

const char kHlo[] = R"(
    HloModule module
  
    computation {
      p0 = bf16[1024,1024]{1,0} parameter(0)
      convert0 = f32[1024,1024]{1,0} convert(p0)
      p1 = bf16[1024,1024]{1,0} parameter(1)
      convert1 = f32[1024,1024]{1,0} convert(p1)
      ROOT dot = f32[1024,1024]{1,0} dot(convert0, convert1),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  
    ENTRY main {
      p0 = bf16[1024,1024]{1,0} parameter(0)
      p1 = bf16[1024,1024]{1,0} parameter(1)
      ROOT fusion = f32[1024,1024]{1,0} fusion(p0, p1),
        kind=kCustom, calls=computation,
        backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
    })";

const char kGpuSpec[] = R"(gpu_device_info {
      threads_per_block_limit: 1024
      threads_per_warp: 32
      shared_memory_per_block: 49152
      shared_memory_per_core: 233472
      threads_per_core_limit: 2048
      core_count: 132
      fpus_per_core: 128
      block_dim_limit_x: 2147483647
      block_dim_limit_y: 65535
      block_dim_limit_z: 65535
      memory_bandwidth: 3352320000000
      l2_cache_size: 52428800
      clock_rate_ghz: 1.98
      device_memory_size: 84978434048
      shared_memory_per_block_optin: 232448
      cuda_compute_capability {
        major: 9
      }
      registers_per_core_limit: 65536
      registers_per_block_limit: 65536
    }
    platform_name: "CUDA"
    dnn_version_info {
      major: 8
      minor: 9
      patch: 4
    }
    device_description_str: "NVIDIA H100 80GB HBM3")";

class CublasBackendTest : public HloRunnerAgnosticTestBase {
 public:
  CublasBackendTest()
      : HloRunnerAgnosticTestBase(std::make_unique<HloRunner>(
            PlatformUtil::GetDefaultPlatform().value())) {}

 protected:
  Compiler::TargetConfig target_config_ = Compiler::TargetConfig(
      ParseTextProto<stream_executor::GpuTargetConfigProto>(kGpuSpec).value());
  DebugOptions debug_options_;
  stream_executor::StreamExecutor* stream_executor_ =
      PlatformUtil::GetDefaultPlatform().value()->ExecutorForDevice(0).value();
  NVPTXCompiler compiler_;
  CublasBackend backend_{&target_config_, &debug_options_, stream_executor_,
                         &compiler_};
};

TEST_F(CublasBackendTest, CanCreateCublasBackend) {
  ASSERT_NE(nullptr, &backend_);
}

TEST_F(CublasBackendTest, GetSupportedConfigs) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));

  auto configs = backend_.GetSupportedConfigs(
      *(module->entry_computation()->root_instruction()));
  EXPECT_GT(configs.size(), 0);
}

TEST_F(CublasBackendTest, GetDefaultConfig) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));

  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(
          (*module->entry_computation()->root_instruction()));
  EXPECT_TRUE(config.ok());
}

TEST_F(CublasBackendTest, GetDefaultConfigFailsWithoutAFusion) {
  std::string hlo = R"(
    HloModule module
  
    ENTRY main {
      p0 = f32[1024,1024]{1,0} parameter(0)
      p1 = f32[1024,1024]{1,0} parameter(1)
      ROOT dot = f32[1024,1024]{1,0} dot(p0, p1),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(
          (*module->entry_computation()->root_instruction()));
  EXPECT_FALSE(config.ok());
}

TEST_F(CublasBackendTest, GetDefaultConfigFailsWithoutAGemm) {
  std::string hlo = R"(
    HloModule module
  
    computation {
      p0 = bf16[1024,1024]{1,0} parameter(0)
      p1 = bf16[1024,1024]{1,0} parameter(1)
      ROOT sum = f32[1024,1024]{1,0} add(p0, p1)
    }
  
    ENTRY main {
      p0 = bf16[1024,1024]{1,0} parameter(0)
      p1 = bf16[1024,1024]{1,0} parameter(1)
      ROOT fusion = f32[1024,1024]{1,0} fusion(p0, p1),
        kind=kCustom, calls=computation
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(
          (*module->entry_computation()->root_instruction()));
  EXPECT_FALSE(config.ok());
}

TEST_F(CublasBackendTest, Compile) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  auto config = backend_.GetDefaultConfig(
      (*module->entry_computation()->root_instruction()));
  EXPECT_TRUE(config.ok());
  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      backend_.Compile(*(module->entry_computation()->root_instruction()),
                       *(config.value())));
}

}  // namespace gpu
}  // namespace xla
