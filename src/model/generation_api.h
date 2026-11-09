// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "xlang3/xlang3.h"
#include "model_catalog.h"
#include <map>
#include <string>
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_WITH_METAL)
#include "cuda_engine_pool.h"
#endif
namespace Garnet {
inline X::Value GenerateMedia(X3PackageHost* host,const X::ARGS& args,const X::KWARGS& kwargs){
    std::map<std::string,X::Value> values;
    for(const auto& item:kwargs)if(!values.emplace(item.first,item.second).second)throw X::Error("duplicate generation option");
    if(args.size()>2)throw X::Error("generate accepts positional model and prompt only");
    for(size_t i=0;i<args.size();++i)if(!values.emplace(i?"prompt":"model",args[i]).second)throw X::Error("duplicate model/prompt");
    for(const auto* name:{"model","prompt","checkpoint","cache","output"})if(!values.count(name)||!values.at(name).IsString())throw X::Error(std::string("generation requires string ")+name);
    const auto id=values.at("model").ToString();
    std::string mode;
    if(auto entry=values.find("mode");entry!=values.end()){
        if(!entry->second.IsString() || entry->second.ToString().empty())
            throw X::Error("generation mode must be a non-empty string");
        mode=entry->second.ToString();
        values.erase(entry);
    }
    X::Module package(host,"xModel");
    const auto packageFile=std::filesystem::u8path(package["__file__"].ToString());
    const auto route=ResolveGenerationRoute(packageFile.parent_path(),id,mode);
    if(!route.allowEnginePool && values.count("engine_pool"))
        throw X::Error("Use model_session for this generation mode");
    X::ARGS positional{values.at("checkpoint"),values.at("cache"),values.at("prompt"),values.at("output")};
    for(const auto* name:{"model","prompt","checkpoint","cache","output"})values.erase(name);
    X::Value poolValue;
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_WITH_METAL)
    struct PoolScope {CUDAEnginePool* pool=nullptr;~PoolScope(){if(pool)try{pool->Leave();}catch(...){}}} scope;
    if(auto entry=values.find("engine_pool");entry!=values.end()){
        poolValue=entry->second;values.erase(entry);
        if(poolValue.runtime()!=host->runtime)throw X::Error("Engine pool belongs to another runtime");
        scope.pool=static_cast<CUDAEnginePool*>(host->instance_get_native_data(poolValue.raw(),X::detail::native_type_name<CUDAEnginePool>()));
        if(!scope.pool)throw X::Error("engine_pool requires a native CUDA engine pool");
        scope.pool->Enter();
    }
#endif
    X::KWARGS options;for(const auto& item:values)options.push_back(item);
    X::Module pipeline(host,route.module.c_str());
    std::error_code error;
    if(!std::filesystem::equivalent(route.file,std::filesystem::u8path(pipeline["__file__"].ToString()),error))
        throw X::Error("generation module does not match its installed manifest");
    X::Value result;
    if(!pipeline[route.function].Call(positional,options,result))throw X::Error(host->runtime_last_error(host->runtime));return result;
}
}
