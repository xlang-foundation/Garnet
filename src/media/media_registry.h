// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "frame.h"
#include "nlohmann/json.hpp"
#include <memory>
#include <string>
namespace Garnet::Media {
const GarnetMediaProviderV1& Provider(const std::string& id);
const GarnetMediaInputProviderV1& InputProvider();
const GarnetMediaAudioRangeProviderV1& AudioRangeProvider();
const GarnetMediaAudioBytesProviderV1& AudioBytesProvider();
const GarnetMediaImageBytesProviderV1& ImageBytesProvider();
void EncodeImage(const GarnetMediaFrame&,const std::string& path,const nlohmann::json& options);
class VideoWriter {
    struct Impl;std::unique_ptr<Impl> impl;
public:
    explicit VideoWriter(const nlohmann::json& options);
    ~VideoWriter();
    void Submit(std::vector<uint8_t> rgb,uint32_t width,uint32_t height);
    void SubmitAudio(const GarnetMediaAudio& audio);
    void SubmitAudioChunk(const GarnetMediaAudio& audio);
    nlohmann::json Finish(const GarnetMediaAudio* audio=nullptr);
    nlohmann::json Status() const;
};
// Disk-backed tile-major assembly: keep one tile's causal state on the GPU,
// one pixel chunk in RAM, and never allocate a complete float video canvas.
class VideoAssembler {
    struct Impl;std::unique_ptr<Impl> impl;
public:
    explicit VideoAssembler(const nlohmann::json& options);
    VideoAssembler(const nlohmann::json& options,VideoWriter& movie,uint32_t skipFrames);
    ~VideoAssembler();
    void Add(size_t tile,uint32_t first,const float* data,uint32_t count,uint32_t h,uint32_t w);
    void SubmitAudio(const GarnetMediaAudio& audio);
    nlohmann::json Finish(const GarnetMediaAudio* audio=nullptr);
};
}
