// SPDX-License-Identifier: Apache-2.0
// Adapted from xWorld/src/image_proc/HwImageProcessor.cpp::Encode.
#include "plugin_util.h"
#include <cuda_runtime.h>
#include <nvjpeg.h>
namespace {
using namespace GMPlugin;
void CUDA(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void JPEG(nvjpegStatus_t e){if(e!=NVJPEG_STATUS_SUCCESS)throw std::runtime_error("nvJPEG error "+std::to_string(e));}
__global__ void Convert(GarnetMediaFrame f,unsigned char* out,int* invalid){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(f.width)*f.height)return;
    size_t y=i/f.width,x=i%f.width;for(size_t c=0;c<3;++c){float value=*reinterpret_cast<const float*>(static_cast<const char*>(f.planes[c])+y*f.strides[c]+x*4);
        if(!isfinite(value)){atomicExch(invalid,1);value=0;}out[i*3+c]=static_cast<unsigned char>(__double2int_rn(fmin(fmax((double(value)+1.0)*127.5,0.0),255.0)));}}
int Encode(const GarnetMediaFrame* f,const char* path,const char* options,char* error,size_t n){return Guard(error,n,[&]{
    if(!f||f->size<sizeof(*f)||f->device!=GM_CUDA||!f->planes[0])throw std::runtime_error("CUDA image tensor required");Garnet::Media::Dimensions(f->width,f->height);
    auto o=Json::parse(options);const auto format=o.value("format","jpeg");if(format!="jpg"&&format!="jpeg")throw std::runtime_error("nvJPEG supports JPEG output");int quality=int(Integer(o,"quality",90,1,100));
    int previous=0;CUDA(cudaGetDevice(&previous));CUDA(cudaSetDevice(f->device_id));struct Restore{int device;~Restore(){cudaSetDevice(device);}} restore{previous};
    nvjpegHandle_t handle=nullptr;nvjpegEncoderState_t state=nullptr;nvjpegEncoderParams_t params=nullptr;unsigned char* converted=nullptr;int* invalid=nullptr;
    try{JPEG(nvjpegCreateSimple(&handle));JPEG(nvjpegEncoderStateCreate(handle,&state,cudaStreamPerThread));JPEG(nvjpegEncoderParamsCreate(handle,&params,cudaStreamPerThread));
        JPEG(nvjpegEncoderParamsSetQuality(params,quality,cudaStreamPerThread));JPEG(nvjpegEncoderParamsSetOptimizedHuffman(params,1,cudaStreamPerThread));JPEG(nvjpegEncoderParamsSetSamplingFactors(params,NVJPEG_CSS_420,cudaStreamPerThread));
        nvjpegImage_t image{};nvjpegInputFormat_t input=NVJPEG_INPUT_RGBI;
        if(f->format==GM_RGB_F32_PLANAR){if(!f->planes[1]||!f->planes[2])throw std::runtime_error("three float planes required");for(int c=0;c<3;++c)if(f->strides[c]<uint64_t(f->width)*4)throw std::runtime_error("invalid float image stride");
            CUDA(cudaMalloc(&converted,size_t(f->width)*f->height*3));CUDA(cudaMalloc(&invalid,4));CUDA(cudaMemsetAsync(invalid,0,4,cudaStreamPerThread));Convert<<<(size_t(f->width)*f->height+255)/256,256,0,cudaStreamPerThread>>>(*f,converted,invalid);CUDA(cudaGetLastError());
            int bad=0;CUDA(cudaMemcpyAsync(&bad,invalid,4,cudaMemcpyDeviceToHost,cudaStreamPerThread));CUDA(cudaStreamSynchronize(cudaStreamPerThread));if(bad)throw std::runtime_error("nonfinite CUDA image pixels");image.channel[0]=converted;image.pitch[0]=f->width*3;}
        else if(f->format==GM_RGB8||f->format==GM_BGR8){if(f->strides[0]<f->width*3)throw std::runtime_error("invalid JPEG image stride");image.channel[0]=const_cast<unsigned char*>(static_cast<const unsigned char*>(f->planes[0]));image.pitch[0]=f->strides[0];if(f->format==GM_BGR8)input=NVJPEG_INPUT_BGRI;}
        else throw std::runtime_error("unsupported CUDA JPEG pixel format");
        JPEG(nvjpegEncodeImage(handle,state,params,&image,input,f->width,f->height,cudaStreamPerThread));size_t length=0;JPEG(nvjpegEncodeRetrieveBitstream(handle,state,nullptr,&length,cudaStreamPerThread));CUDA(cudaStreamSynchronize(cudaStreamPerThread));
        std::vector<uint8_t> bytes(length);JPEG(nvjpegEncodeRetrieveBitstream(handle,state,bytes.data(),&length,cudaStreamPerThread));CUDA(cudaStreamSynchronize(cudaStreamPerThread));Write(path,bytes.data(),length);
    }catch(...){cudaStreamSynchronize(cudaStreamPerThread);if(converted)cudaFree(converted);if(invalid)cudaFree(invalid);if(params)nvjpegEncoderParamsDestroy(params);if(state)nvjpegEncoderStateDestroy(state);if(handle)nvjpegDestroy(handle);throw;}
    if(converted)CUDA(cudaFree(converted));if(invalid)CUDA(cudaFree(invalid));JPEG(nvjpegEncoderParamsDestroy(params));JPEG(nvjpegEncoderStateDestroy(state));JPEG(nvjpegDestroy(handle));
});}
const char* Caps(){return R"({"formats":["jpeg"],"devices":["cuda"],"pixel_formats":["rgb8","bgr8","rgb_f32_planar"]})";}
GarnetMediaProviderV1 api{1,sizeof(api),"nvjpeg",Caps,Encode,nullptr,nullptr,nullptr,nullptr,nullptr};
}
extern "C" GARNET_MEDIA_EXPORT const GarnetMediaProviderV1* GarnetMediaGetProviderV1(){return &api;}
