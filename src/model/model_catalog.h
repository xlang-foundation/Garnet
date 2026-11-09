// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>

namespace Garnet
{
    std::filesystem::path RuntimeModuleFolder();

    std::filesystem::path ResolveModelCatalogRoot(
        const std::string& requestedRoot,
        const std::string& runtimeBaseFolder);

    std::string EnumerateAvailableModelsJson(
        const std::filesystem::path& catalogRoot);

    struct GenerationRoute {
        std::filesystem::path file;
        std::string module;
        std::string function;
        bool allowEnginePool = true;
    };

    GenerationRoute ResolveGenerationRoute(
        const std::filesystem::path& packageRoot,
        const std::string& model,
        const std::string& mode);
}
