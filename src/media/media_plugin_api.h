// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#include <stddef.h>
#if defined(_WIN32)
#define GARNET_MEDIA_EXPORT __declspec(dllexport)
#else
#define GARNET_MEDIA_EXPORT __attribute__((visibility("default")))
#endif
// Stable C boundary. No SDK, CUDA, FFmpeg, STL or allocator-owned types cross it.
// submit_video consumes input synchronously; providers copy/retain their own
// storage before returning. GPU providers must complete source reads on return.
enum GarnetMediaPixelFormat { GM_RGB8=1, GM_BGR8=2, GM_RGBA8=3, GM_RGB_F32_PLANAR=4 };
enum GarnetMediaDevice { GM_CPU=0, GM_CUDA=1 };
typedef struct GarnetMediaFrame {
    uint32_t size, format, device;
    int32_t device_id;
    uint32_t width, height;
    const void* planes[3];
    uint64_t strides[3];
    int64_t pts;
} GarnetMediaFrame;
typedef struct GarnetMediaDecoded {GarnetMediaFrame frame;void* owner;} GarnetMediaDecoded;
typedef struct GarnetMediaProviderV1 {
    uint32_t abi, size;
    const char* id;
    const char* (*capabilities)();
    int (*encode_image)(const GarnetMediaFrame*,const char* path,const char* options,char* error,size_t capacity);
    void* (*open_video)(const char* options,char* error,size_t capacity);
    int (*submit_video)(void*,const GarnetMediaFrame*,char* error,size_t capacity);
    int (*finish_video)(void*,char* error,size_t capacity);
    const char* (*video_info)(void*);
    void (*close_video)(void*);
    int (*decode_image)(const char* path,GarnetMediaDecoded*,char* error,size_t capacity);
    void (*release_image)(void* owner);
} GarnetMediaProviderV1;
typedef const GarnetMediaProviderV1* (*GarnetMediaGetProviderFn)();

// Optional audio extension, discovered through GarnetMediaGetAudioProviderV1.
// Keeps the original video/image ABI intact for existing providers.
typedef struct GarnetMediaAudio {
    uint32_t size,channels,sample_rate,samples;
    const float* planes[2]; // CPU planar FP32 PCM, consumed synchronously.
} GarnetMediaAudio;
typedef struct GarnetMediaAudioProviderV1 {
    uint32_t abi,size;
    int (*submit_audio)(void* video_session,const GarnetMediaAudio*,char* error,size_t capacity);
} GarnetMediaAudioProviderV1;
typedef const GarnetMediaAudioProviderV1* (*GarnetMediaGetAudioProviderFn)();

// Optional streaming audio for a single movie. Each submission owns the
// next contiguous PCM interval and is consumed synchronously.
typedef struct GarnetMediaStreamingAudioProviderV1 {
    uint32_t abi,size;
    int (*submit_audio_chunk)(void* video_session,const GarnetMediaAudio*,char* error,size_t capacity);
} GarnetMediaStreamingAudioProviderV1;
typedef const GarnetMediaStreamingAudioProviderV1* (*GarnetMediaGetStreamingAudioProviderFn)();

// Optional input extension. Decoder owns PCM until release_audio is called.
typedef struct GarnetMediaDecodedAudio {GarnetMediaAudio audio;void* owner;} GarnetMediaDecodedAudio;
typedef struct GarnetMediaInputProviderV1 {
    uint32_t abi,size;
    int (*decode_audio)(const char* path,uint32_t sample_rate,GarnetMediaDecodedAudio*,char* error,size_t capacity);
    void (*release_audio)(void* owner);
} GarnetMediaInputProviderV1;
typedef const GarnetMediaInputProviderV1* (*GarnetMediaGetInputProviderFn)();

typedef struct GarnetMediaAudioRangeProviderV1 {
    uint32_t abi,size;
    int (*decode_audio_range)(const char* path,uint32_t sample_rate,uint64_t offset,uint32_t count,
        GarnetMediaDecodedAudio*,char* error,size_t capacity);
    void (*release_audio)(void* owner);
} GarnetMediaAudioRangeProviderV1;
typedef const GarnetMediaAudioRangeProviderV1* (*GarnetMediaGetAudioRangeProviderFn)();

typedef struct GarnetMediaImageBytesProviderV1 {
    uint32_t abi,size;
    int (*decode_image)(const void* data,size_t length,GarnetMediaDecoded*,char* error,size_t capacity);
    void (*release_image)(void* owner);
    int (*encode_png)(const GarnetMediaFrame*,const void** data,size_t* length,void** owner,char* error,size_t capacity);
    void (*release_png)(void* owner);
} GarnetMediaImageBytesProviderV1;

typedef struct GarnetMediaAudioBytesProviderV1 {
    uint32_t abi,size;
    int (*decode_audio)(const void* data,size_t length,uint32_t sample_rate,uint64_t offset,uint32_t count,
        GarnetMediaDecodedAudio*,char* error,size_t capacity);
    void (*release_audio)(void* owner);
} GarnetMediaAudioBytesProviderV1;
