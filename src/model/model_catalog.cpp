// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "model_catalog.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <set>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace Garnet
{
    namespace
    {
        using json = nlohmann::json;
        namespace fs = std::filesystem;

        fs::path AbsoluteNormalized(const fs::path& path)
        {
            std::error_code error;
            fs::path absolute = fs::absolute(path, error);
            return (error ? path : absolute).lexically_normal();
        }

        bool IsSafePackagePath(const fs::path& relative)
        {
            if (relative.empty() || relative.is_absolute()) return false;
            for (const auto& component : relative) {
                if (component == "..") return false;
            }
            return true;
        }

        bool ReadJsonFile(const fs::path& path, json& value, std::string& error)
        {
            try {
                std::ifstream stream(path, std::ios::binary);
                if (!stream) {
                    error = "cannot open " + path.string();
                    return false;
                }
                stream >> value;
                return true;
            }
            catch (const std::exception& exception) {
                error = exception.what();
                return false;
            }
        }

        bool ValidateReferencedFile(
            const fs::path& packageRoot,
            const std::string& relativeText,
            std::vector<std::string>& issues)
        {
            const fs::path relative(relativeText);
            if (!IsSafePackagePath(relative)) {
                issues.push_back("unsafe package-relative path: " + relativeText);
                return false;
            }
            if (!fs::is_regular_file(packageRoot / relative)) {
                issues.push_back("missing package file: " + relativeText);
                return false;
            }
            return true;
        }

        void ValidateGenerationEntry(const fs::path& root, const json& entry,
            const std::string& label, std::vector<std::string>& issues)
        {
            if (!entry.is_object() || !entry.contains("file") || !entry["file"].is_string() ||
                !entry.contains("function") || !entry["function"].is_string() ||
                entry["function"].get_ref<const std::string&>().empty() ||
                !entry.contains("runtime") || entry["runtime"] != "xlang3" ||
                (entry.contains("allow_engine_pool") && !entry["allow_engine_pool"].is_boolean())) {
                issues.push_back(label + " requires a file, function and xlang3 runtime");
                return;
            }
            ValidateReferencedFile(root, entry["file"].get<std::string>(), issues);
        }

        json InspectManifest(const fs::path& manifestPath)
        {
            const fs::path packageRoot = manifestPath.parent_path();
            std::vector<std::string> issues;
            json manifest;
            std::string readError;
            if (!ReadJsonFile(manifestPath, manifest, readError) ||
                !manifest.is_object()) {
                return {
                    {"id", ""},
                    {"available", false},
                    {"manifest_path", manifestPath.string()},
                    {"package_root", packageRoot.string()},
                    {"issues", json::array({"invalid manifest: " + readError})}
                };
            }

            if (manifest.value("schema_version", 0) != 1) {
                issues.push_back("unsupported or missing schema_version");
            }
            if (!manifest.contains("id") || !manifest["id"].is_string() ||
                manifest["id"].get<std::string>().empty()) {
                issues.push_back("id must be a non-empty string");
            }
            if (!manifest.contains("task") || !manifest["task"].is_string()) {
                issues.push_back("task must be a string");
            }
            if (manifest.contains("pipeline"))
                ValidateGenerationEntry(packageRoot, manifest["pipeline"], "pipeline", issues);
            if (manifest.contains("modes")) {
                if (!manifest.contains("pipeline") || !manifest["modes"].is_object()) {
                    issues.push_back("generation modes require a pipeline and a mode object");
                } else {
                    for (const auto& mode : manifest["modes"].items())
                        ValidateGenerationEntry(packageRoot, mode.value(), "mode " + mode.key(), issues);
                }
            }
            if (!manifest.contains("entrypoints") ||
                !manifest["entrypoints"].is_object()) {
                issues.push_back("entrypoints must be an object");
            }
            else {
                for (const auto& item : manifest["entrypoints"].items()) {
                    const json& entrypoint = item.value();
                    if (!entrypoint.is_object() ||
                        !entrypoint.contains("file") ||
                        !entrypoint["file"].is_string() ||
                        !entrypoint.contains("function") ||
                        !entrypoint["function"].is_string()) {
                        issues.push_back("invalid entrypoint: " + item.key());
                        continue;
                    }
                    ValidateReferencedFile(
                        packageRoot,
                        entrypoint["file"].get<std::string>(),
                        issues);
                }
            }

            json resolvedProfiles = json::array();
            if (!manifest.contains("profile_files") ||
                !manifest["profile_files"].is_array()) {
                issues.push_back("profile_files must be an array");
            }
            else {
                for (const json& profileValue : manifest["profile_files"]) {
                    if (!profileValue.is_string()) {
                        issues.push_back("profile file path must be a string");
                        continue;
                    }
                    const std::string relativeText = profileValue.get<std::string>();
                    if (!ValidateReferencedFile(packageRoot, relativeText, issues)) {
                        continue;
                    }
                    json profile;
                    std::string profileError;
                    if (!ReadJsonFile(packageRoot / relativeText, profile, profileError) ||
                        !profile.is_object()) {
                        issues.push_back(
                            "invalid profile " + relativeText + ": " + profileError);
                        continue;
                    }
                    if (profile.value("schema_version", 0) != 1 ||
                        !profile.contains("id") || !profile["id"].is_string() ||
                        !profile.contains("backend") ||
                        !profile["backend"].is_string() ||
                        !profile.contains("precision") ||
                        !profile["precision"].is_string()) {
                        issues.push_back("profile has an invalid contract: " + relativeText);
                        continue;
                    }
                    profile["profile_path"] = (packageRoot / relativeText).string();
                    resolvedProfiles.push_back(std::move(profile));
                }
            }

            manifest["manifest_path"] = manifestPath.string();
            manifest["package_root"] = packageRoot.string();
            manifest["profiles"] = std::move(resolvedProfiles);
            manifest["available"] = issues.empty();
            manifest["issues"] = issues;
            return manifest;
        }
    }

    std::filesystem::path RuntimeModuleFolder()
    {
#if defined(_WIN32)
        HMODULE module = nullptr;
        if (!GetModuleHandleExA(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                reinterpret_cast<LPCSTR>(&RuntimeModuleFolder),
                &module)) {
            return {};
        }
        std::vector<char> path(1024);
        for (;;) {
            const DWORD length = GetModuleFileNameA(
                module, path.data(), static_cast<DWORD>(path.size()));
            if (length == 0) return {};
            if (length < path.size() - 1) {
                return fs::path(std::string(path.data(), length)).parent_path();
            }
            path.resize(path.size() * 2);
        }
#else
        Dl_info info{};
        if (dladdr(reinterpret_cast<void*>(&RuntimeModuleFolder), &info) == 0 ||
            !info.dli_fname) {
            return {};
        }
        return fs::path(info.dli_fname).parent_path();
#endif
    }

    std::filesystem::path ResolveModelCatalogRoot(
        const std::string& requestedRoot,
        const std::string& runtimeBaseFolder)
    {
        namespace fs = std::filesystem;
        if (!requestedRoot.empty()) return AbsoluteNormalized(requestedRoot);
        if (const char* configured = std::getenv("GARNET_MODEL_CATALOG");
            configured && *configured) {
            return AbsoluteNormalized(configured);
        }

        fs::path base = runtimeBaseFolder.empty()
            ? RuntimeModuleFolder()
            : fs::path(runtimeBaseFolder);
        if (base.empty()) base = fs::current_path();
        const fs::path sibling = AbsoluteNormalized(base / ".." / "models");
        if (fs::is_directory(sibling)) return sibling;
        const fs::path adjacent = AbsoluteNormalized(base / "models");
        if (fs::is_directory(adjacent)) return adjacent;
        return sibling;
    }

    std::string EnumerateAvailableModelsJson(
        const std::filesystem::path& requestedCatalogRoot)
    {
        namespace fs = std::filesystem;
        const fs::path catalogRoot = AbsoluteNormalized(requestedCatalogRoot);
        json response = {
            {"schema_version", 1},
            {"catalog_root", catalogRoot.string()},
            {"models", json::array()},
            {"errors", json::array()}
        };

        if (!fs::is_directory(catalogRoot)) {
            response["errors"].push_back({
                {"code", "catalog_not_found"},
                {"message", "model catalog directory was not found"}
            });
            return response.dump();
        }

        std::vector<fs::path> manifests;
        std::error_code walkError;
        for (fs::recursive_directory_iterator iterator(
                 catalogRoot,
                 fs::directory_options::skip_permission_denied,
                 walkError), end;
             !walkError && iterator != end;
             iterator.increment(walkError)) {
            if (iterator->is_regular_file() &&
                iterator->path().filename() == "model.json") {
                manifests.push_back(AbsoluteNormalized(iterator->path()));
            }
        }
        if (walkError) {
            response["errors"].push_back({
                {"code", "catalog_scan_failed"},
                {"message", walkError.message()}
            });
        }
        std::sort(manifests.begin(), manifests.end());

        std::set<std::string> ids;
        for (const fs::path& manifestPath : manifests) {
            json model = InspectManifest(manifestPath);
            const std::string id = model.value("id", "");
            if (!id.empty() && !ids.insert(id).second) {
                model["available"] = false;
                model["issues"].push_back("duplicate model id: " + id);
            }
            response["models"].push_back(std::move(model));
        }
        std::sort(
            response["models"].begin(),
            response["models"].end(),
            [](const json& left, const json& right) {
                return left.value("id", "") < right.value("id", "");
            });
        return response.dump();
    }

    GenerationRoute ResolveGenerationRoute(const std::filesystem::path& packageRoot,
        const std::string& model, const std::string& mode)
    {
        const auto catalog = json::parse(EnumerateAvailableModelsJson(packageRoot));
        const json* match = nullptr;
        for (const auto& entry : catalog["models"]) {
            if (entry.value("id", "") != model &&
                (!entry.contains("slug") || entry["slug"] != model)) continue;
            if (match) throw std::runtime_error("ambiguous generation model: " + model);
            match = &entry;
        }
        if (!match || !match->value("available", false) || !match->contains("pipeline"))
            throw std::runtime_error("generation model is unavailable: " + model);

        const json* route = &match->at("pipeline");
        if (!mode.empty()) {
            if (!match->contains("modes") || !match->at("modes").contains(mode))
                throw std::runtime_error("unsupported generation mode for " + model + ": " + mode);
            route = &match->at("modes").at(mode);
        }
        const auto root = fs::canonical(packageRoot);
        auto file = fs::canonical(fs::u8path(match->at("package_root").get<std::string>()) /
            fs::u8path(route->at("file").get<std::string>()));
        auto relative = file.lexically_relative(root);
        if (!IsSafePackagePath(relative) || relative.extension() != ".py")
            throw std::runtime_error("generation entry must be a Python source inside the xModel package");
        relative.replace_extension();
        const auto identifier = [](const std::string& value) {
            static const std::string letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_";
            static const std::string characters = letters + "0123456789";
            return !value.empty() && letters.find(value[0]) != std::string::npos &&
                value.find_first_not_of(characters) == std::string::npos;
        };
        std::string module = "xModel";
        for (const auto& component : relative) {
            const auto name = component.u8string();
            if (!identifier(name)) throw std::runtime_error("invalid generation module name: " + name);
            module += "." + name;
        }
        const auto function = route->at("function").get<std::string>();
        if (!identifier(function)) throw std::runtime_error("invalid generation entry function");
        return {std::move(file), std::move(module), function, route->value("allow_engine_pool", true)};
    }
}
