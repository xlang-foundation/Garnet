// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <NvInferRuntime.h>
#include <vector>
namespace Garnet {
// root width, hidden width, final hidden write, alias root, part count,
// followed by (root offset, part width) pairs. Every tensor is rank2 FP32.
bool ValidQwen35PackedStoreConfig(const std::vector<int32_t>&) noexcept;
nvinfer1::IPluginV3* CreateQwen35PackedStorePlugin(const std::vector<int32_t>&);
bool EnsureQwen35PackedStorePluginRegistered();
}
