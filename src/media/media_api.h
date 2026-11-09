// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "media_registry.h"
#include "audio_features.h"
#include "xlang3/xlang3.h"
#include <cstring>
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_WITH_METAL)
#include "tensor_helper.h"
#endif
namespace Garnet {
namespace MediaBindings {
inline void Storage(const X3TensorInfo& i){
    if(i.symbolic||!i.data||!i.shape||!i.strides||i.rank<1||i.rank>5)throw X::Error("materialized image storage required");
    uint64_t span=i.dtype==X3_TENSOR_FLOAT32?4:i.dtype==X3_TENSOR_UINT8?1:0;if(!span)throw X::Error("unsupported image dtype");
    for(uint32_t a=0;a<i.rank;++a){if(i.shape[a]<1||i.strides[a]<0||uint64_t(i.shape[a]-1)>UINT64_MAX/(uint64_t(i.strides[a])+1))throw X::Error("invalid image shape/strides");
        auto bytes=uint64_t(i.shape[a]-1)*uint64_t(i.strides[a]);if(bytes>UINT64_MAX-span)throw X::Error("image storage span overflow");span+=bytes;}
    if(span>i.byte_size)throw X::Error("image storage is too small");
}
inline X::Tensor CPU(X::Tensor value){if(value.Info().device_type==0)return value;
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_WITH_METAL)
    return TensorHelper::CopyToCPU(value);
#else
    throw X::Error("download this device's frame before CPU encoding");
#endif
}
inline X3TensorInfo FloatVideo(const X::Tensor& t){const auto i=t.Info();
    if(i.symbolic||i.device_type!=0||i.dtype!=X3_TENSOR_FLOAT32||(i.rank!=4&&i.rank!=5)||!i.data||i.shape[0]!=1||i.shape[1]!=3)throw X::Error("media requires materialized CPU FP32 [1,3,F,H,W] or [1,3,H,W]");
    int64_t stride=4;for(uint32_t axis=i.rank;axis-->0;){if(i.shape[axis]<1||stride>INT64_MAX/i.shape[axis]||(i.shape[axis]>1&&i.strides[axis]!=stride))throw X::Error("media requires bounded contiguous pixels");stride*=i.shape[axis];}
    if(i.shape[i.rank-1]>16384||i.shape[i.rank-2]>16384||(i.rank==5&&i.shape[2]>INT32_MAX))throw X::Error("media shape exceeds admission limits");
    Media::Dimensions(uint32_t(i.shape[i.rank-1]),uint32_t(i.shape[i.rank-2]));
    if(i.byte_size<uint64_t(stride))throw X::Error("media tensor storage is too small");return i;
}
inline GarnetMediaAudio FloatAudio(const X::Tensor& tensor,int64_t rate){
    const auto i=tensor.Info();Storage(i);
    if(i.device_type!=0||i.dtype!=X3_TENSOR_FLOAT32||(i.rank!=2&&i.rank!=3)||(i.rank==3&&i.shape[0]!=1)||rate<8000||rate>192000)
        throw X::Error("audio requires CPU FP32 [C,N] or [1,C,N] and a supported sample rate");
    const auto channels=i.shape[i.rank-2],samples=i.shape[i.rank-1];
    if(channels<1||channels>2||samples>rate*600||i.strides[i.rank-1]!=4||i.strides[i.rank-2]!=samples*4)throw X::Error("audio must be contiguous mono/stereo PCM, at most 600 seconds");
    GarnetMediaAudio audio{};audio.size=sizeof(audio);audio.channels=uint32_t(channels);audio.sample_rate=uint32_t(rate);audio.samples=uint32_t(samples);
    for(uint32_t c=0;c<audio.channels;++c)audio.planes[c]=static_cast<const float*>(i.data)+c*samples;return audio;
}
}
class NativeMediaAPI {
    X::Value DecodeAudioBytes(const X::Value& input,int64_t rate,int64_t offset,int64_t count){
        const auto& provider=Media::AudioBytesProvider();
        uint64_t length=0;const auto* bytes=input.BytesData(&length);
        if(!bytes||!length||length>SIZE_MAX)throw X::Error("audio bytes required");
        GarnetMediaDecodedAudio result{};char error[2048]{};
        if(provider.decode_audio(bytes,static_cast<size_t>(length),uint32_t(rate),uint64_t(offset),uint32_t(count),&result,error,sizeof(error)))throw X::Error(error);
        return ImportAudio(result,rate,provider.release_audio);
    }
    X::Value ImportAudio(const GarnetMediaDecodedAudio& result,int64_t rate,void (*release)(void*)){
        struct Owner{void (*release)(void*);void* value;~Owner(){release(value);}} owner{release,result.owner};
        const auto& a=result.audio;
        if(a.size<sizeof(a)||a.channels!=2||a.sample_rate!=rate||!a.samples||a.samples>uint64_t(rate)*600||!a.planes[0]||!a.planes[1])throw X::Error("invalid decoded PCM");
        std::vector<float> values(size_t(a.samples)*2);for(size_t c=0;c<2;++c)std::memcpy(values.data()+c*a.samples,a.planes[c],a.samples*4);
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,{1,2,a.samples},values.data(),values.size()*4);
    }
public:
    X::Value AudioSegment(X::Value input,int64_t offset,int64_t count){
        X::Tensor value(input);const auto audio=MediaBindings::FloatAudio(value,192000);
        if(offset<0||offset>audio.samples||count<1||count>192000LL*600)throw X::Error("invalid audio segment");auto lease=value.Acquire();
        std::vector<float> data(size_t(count)*2,0);size_t copied=size_t(std::min<int64_t>(count,audio.samples-offset));
        for(size_t c=0;c<2;++c)std::memcpy(data.data()+c*count,audio.planes[c]+offset,copied*4);
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,{1,2,count},data.data(),data.size()*4);
    }
    X::Value DecodeAudio(X::Value input,int64_t rate){
        if(rate<8000||rate>192000)throw X::Error("invalid decoded sample rate");
        if(input.IsBin())return DecodeAudioBytes(input,rate,0,0);
        if(!input.IsString())throw X::Error("audio input must be a file path or encoded bytes");
        const auto path=input.ToString();const auto& provider=Media::InputProvider();GarnetMediaDecodedAudio result{};char error[2048]{};
        if(provider.decode_audio(path.c_str(),uint32_t(rate),&result,error,sizeof(error)))throw X::Error(error);
        return ImportAudio(result,rate,provider.release_audio);
    }
    X::Value DecodeAudioRange(X::Value input,int64_t rate,int64_t offset,int64_t count){
        if(rate<8000||rate>192000||offset<0||count<1||count>rate*600)throw X::Error("invalid decoded audio range");
        if(input.IsBin())return DecodeAudioBytes(input,rate,offset,count);
        if(!input.IsString())throw X::Error("audio input must be a file path or encoded bytes");
        const auto path=input.ToString();
        const auto& provider=Media::AudioRangeProvider();GarnetMediaDecodedAudio result{};char error[2048]{};
        if(provider.decode_audio_range(path.c_str(),uint32_t(rate),uint64_t(offset),uint32_t(count),&result,error,sizeof(error)))throw X::Error(error);
        return ImportAudio(result,rate,provider.release_audio);
    }
    X::Value AudioLogMel(X::Value input,int64_t rate,int64_t fft,int64_t hop,int64_t bins){
        X::Tensor value(input);const auto audio=MediaBindings::FloatAudio(value,rate);if(audio.channels!=2)throw X::Error("log-mel requires stereo PCM");
        if(fft<16||fft>8192||hop<1||hop>fft||bins<1||bins>512)throw X::Error("invalid log-mel parameters");auto lease=value.Acquire();
        auto features=Media::LogMel(audio.planes[0],audio.samples,int(rate),int(fft),int(hop),int(bins));
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,{1,2,audio.samples/hop+1,bins},features.data(),features.size()*4);
    }
    X::Value ResizeImage(X::Value input,int64_t height,int64_t width){
        X::Tensor value(input);auto info=MediaBindings::FloatVideo(value);if(info.rank!=4||height<1||width<1||height>16384||width>16384)throw X::Error("invalid resized image shape");
        Media::Dimensions(uint32_t(width),uint32_t(height));auto lease=value.Acquire();const int64_t sh=info.shape[2],sw=info.shape[3];auto* source=static_cast<const float*>(info.data);
        const double scale=std::max(double(height)/sh,double(width)/sw),cropH=height/scale,cropW=width/scale;
        std::vector<float> pixels(size_t(3)*height*width);for(int c=0;c<3;++c)for(int64_t y=0;y<height;++y)for(int64_t x=0;x<width;++x){
            double fy=std::clamp((y+.5)/scale+(sh-cropH)/2-.5,0.0,double(sh-1)),fx=std::clamp((x+.5)/scale+(sw-cropW)/2-.5,0.0,double(sw-1));
            int64_t y0=int64_t(fy),x0=int64_t(fx),y1=std::min(y0+1,sh-1),x1=std::min(x0+1,sw-1);double dy=fy-y0,dx=fx-x0;
            pixels[(size_t(c)*height+y)*width+x]=float((1-dy)*((1-dx)*source[(c*sh+y0)*sw+x0]+dx*source[(c*sh+y0)*sw+x1])+dy*((1-dx)*source[(c*sh+y1)*sw+x0]+dx*source[(c*sh+y1)*sw+x1]));}
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,{1,3,height,width},pixels.data(),pixels.size()*4);
    }
    std::string RequireProvider(const std::string& id){return Media::Provider(id).capabilities();}
    X::Value EncodeImage(X::Value source,const std::string& path,const std::string& text){
        auto options=nlohmann::json::parse(text);X::Tensor tensor(source);auto i=tensor.Info();MediaBindings::Storage(i);
        const auto provider=options.value("provider",options.value("format","png")=="png"?"image":i.device_type==1?"nvjpeg":"image");options["provider"]=provider;
        if(path.empty()&&(provider!="image"||options.value("format","png")!="png"))throw X::Error("in-memory image encoding requires PNG");
        auto encode=[&](const GarnetMediaFrame& frame)->X::Value{
            if(!path.empty()){
                Media::EncodeImage(frame,path,options);
                return X::Value::String(Host(),path);
            }
            const auto& api=Media::ImageBytesProvider();
            if(!api.encode_png||!api.release_png)throw X::Error("in-memory PNG encoder is unavailable");
            const void* data=nullptr;size_t length=0;void* owner=nullptr;char error[2048]{};
            if(api.encode_png(&frame,&data,&length,&owner,error,sizeof(error)))throw X::Error(error);
            struct Owner{void (*release)(void*);void* value;~Owner(){release(value);}} release{api.release_png,owner};
            if(!data||!length)throw X::Error("PNG encoder returned no bytes");
            return X::Value::Bytes(Host(),data,length);
        };
        if(provider!="nvjpeg")tensor=MediaBindings::CPU(tensor);i=tensor.Info();GarnetMediaFrame frame{};frame.size=sizeof(frame);frame.device=i.device_type;frame.device_id=i.device_id;
        std::vector<uint8_t> pixels;
        if(i.device_type==0&&i.dtype==X3_TENSOR_UINT8&&i.rank==3&&(i.shape[2]==3||i.shape[2]==4)&&i.strides[1]==i.shape[2]&&i.strides[2]==1){
            if(i.shape[0]>16384||i.shape[1]>16384)throw X::Error("image shape exceeds admission limits");
            Media::Dimensions(uint32_t(i.shape[1]),uint32_t(i.shape[0]));if(i.strides[0]<i.shape[1]*i.shape[2]||i.byte_size<uint64_t((i.shape[0]-1)*i.strides[0]+i.shape[1]*i.shape[2]))throw X::Error("invalid image byte strides/storage");
            frame=Media::RGBFrame(static_cast<const uint8_t*>(i.data),uint32_t(i.shape[1]),uint32_t(i.shape[0]));frame.strides[0]=i.strides[0];
            frame.format=i.shape[2]==4?GM_RGBA8:options.value("input_format","rgb8")=="bgr8"?GM_BGR8:GM_RGB8;auto lease=tensor.Acquire();return encode(frame);
        }else if(i.device_type==0){i=MediaBindings::FloatVideo(tensor);auto lease=tensor.Acquire();uint32_t count=i.rank==5?uint32_t(i.shape[2]):1;
            pixels=Media::PlanarRGB(static_cast<const float*>(i.data),uint32_t(i.shape[i.rank-1]),uint32_t(i.shape[i.rank-2]),count,0);
            frame=Media::RGBFrame(pixels.data(),uint32_t(i.shape[i.rank-1]),uint32_t(i.shape[i.rank-2]));return encode(frame);
        }else{if(i.symbolic||!i.data||i.device_type!=GM_CUDA)throw X::Error("materialized CUDA image required");
            if((i.rank==3&&(i.shape[0]>16384||i.shape[1]>16384))||(i.rank==4&&(i.shape[2]>16384||i.shape[3]>16384)))throw X::Error("image shape exceeds admission limits");
            if(i.dtype==X3_TENSOR_UINT8&&i.rank==3&&i.shape[2]==3&&i.strides[1]==3&&i.strides[2]==1){frame.format=options.value("input_format","rgb8")=="bgr8"?GM_BGR8:GM_RGB8;
                frame.width=uint32_t(i.shape[1]);frame.height=uint32_t(i.shape[0]);frame.planes[0]=i.data;frame.strides[0]=i.strides[0];}
            else if(i.dtype==X3_TENSOR_FLOAT32&&i.rank==4&&i.shape[0]==1&&i.shape[1]==3&&i.strides[3]==4){frame.format=GM_RGB_F32_PLANAR;frame.width=uint32_t(i.shape[3]);frame.height=uint32_t(i.shape[2]);
                for(size_t c=0;c<3;++c){frame.planes[c]=static_cast<const char*>(i.data)+c*i.strides[1];frame.strides[c]=i.strides[2];}}
            else throw X::Error("GPU image requires HWC uint8 or NCHW FP32 RGB");
            Media::Dimensions(frame.width,frame.height);auto lease=tensor.Acquire();return encode(frame);
        }
    }
    X::Value DecodeImage(X::Value input){
        GarnetMediaDecoded output{};char error[2048]{};
        void (*release)(void*)=nullptr;
        if(input.IsBin()){
            const auto& provider=Media::ImageBytesProvider();
            uint64_t length=0;const auto* bytes=input.BytesData(&length);
            if(!bytes||!length||length>SIZE_MAX)throw X::Error("image bytes required");
            if(provider.decode_image(bytes,static_cast<size_t>(length),&output,error,sizeof(error)))throw X::Error(error);
            release=provider.release_image;
        }else{
            if(!input.IsString())throw X::Error("image input must be a file path or encoded bytes");
            const auto& provider=Media::Provider("image");
            if(!provider.decode_image||!provider.release_image)throw X::Error("image decoder unavailable");
            if(provider.decode_image(input.ToString().c_str(),&output,error,sizeof(error)))throw X::Error(error);
            release=provider.release_image;
        }
        struct Owner{void (*release)(void*);void* data;~Owner(){release(data);}} owner{release,output.owner};const auto& f=output.frame;
        Media::Dimensions(f.width,f.height);if(f.device!=GM_CPU||f.format!=GM_RGB8||!f.planes[0])throw X::Error("invalid decoded image");
        size_t plane=size_t(f.width)*f.height;std::vector<float> data(plane*3);auto* bytes=static_cast<const uint8_t*>(f.planes[0]);
        for(size_t y=0;y<f.height;++y)for(size_t x=0;x<f.width;++x)for(size_t c=0;c<3;++c)data[c*plane+y*f.width+x]=float(double(bytes[y*f.strides[0]+x*3+c])/127.5-1.0);
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,{1,3,f.height,f.width},data.data(),data.size()*4);
    }
    BEGIN_PACKAGE(NativeMediaAPI)
        APISET().AddFunc<3>("audio_segment",&NativeMediaAPI::AudioSegment);
        APISET().AddFunc<2>("decode_audio",&NativeMediaAPI::DecodeAudio);
        APISET().AddFunc<4>("decode_audio_range",&NativeMediaAPI::DecodeAudioRange);
        APISET().AddFunc<5>("audio_log_mel",&NativeMediaAPI::AudioLogMel);
        APISET().AddFunc<3>("resize_image",&NativeMediaAPI::ResizeImage);
        APISET().AddFunc<1>("require_provider",&NativeMediaAPI::RequireProvider);
        APISET().AddFunc<3>("encode_image",&NativeMediaAPI::EncodeImage);
        APISET().AddFunc<1>("decode_image",&NativeMediaAPI::DecodeImage);
    END_PACKAGE
};
namespace MediaBindings {
inline void AddCompleteTile(Media::VideoAssembler& assembler,int64_t tile,X::Value value){
    X::Tensor tensor(value);const auto info=FloatVideo(tensor);
    if(info.rank!=5||tile<0)throw X::Error("invalid complete video tile");
    const uint32_t frames=uint32_t(info.shape[2]),h=uint32_t(info.shape[3]),w=uint32_t(info.shape[4]);const size_t plane=size_t(h)*w;
    auto use=tensor.Acquire();const auto* input=static_cast<const float*>(info.data);
    for(uint32_t first=0;first<frames;){const uint32_t count=std::min(first?4U:1U,frames-first);std::vector<float> chunk(3*count*plane);
        for(size_t c=0;c<3;++c)std::memcpy(chunk.data()+c*count*plane,input+(c*frames+first)*plane,count*plane*sizeof(float));
        assembler.Add(size_t(tile),first,chunk.data(),count,h,w);first+=count;}
}
}
class NativeVideoWriter {
    std::unique_ptr<Media::VideoWriter> writer;
    std::unique_ptr<Media::VideoAssembler> window;
    Media::VideoWriter& Writer(){if(!writer)throw X::Error("video writer is closed");return *writer;}
public:
    explicit NativeVideoWriter(X::Value options):writer(std::make_unique<Media::VideoWriter>(nlohmann::json::parse(options.ToString()))){}
    void WriteAudio(X::Value value,int64_t rate){auto tensor=MediaBindings::CPU(X::Tensor(value));const auto audio=MediaBindings::FloatAudio(tensor,rate);auto use=tensor.Acquire();Writer().SubmitAudio(audio);}
    void WriteAudioChunk(X::Value value,int64_t rate){
        if(window)throw X::Error("submit movie audio before opening its video window");
        auto tensor=MediaBindings::CPU(X::Tensor(value));const auto audio=MediaBindings::FloatAudio(tensor,rate);
        auto use=tensor.Acquire();Writer().SubmitAudioChunk(audio);
    }
    void WriteChunk(X::Value value){auto tensor=MediaBindings::CPU(X::Tensor(value));auto info=MediaBindings::FloatVideo(tensor);auto use=tensor.Acquire();uint32_t count=info.rank==5?uint32_t(info.shape[2]):1;
        if(window)throw X::Error("a tile window is active");
        uint32_t h=uint32_t(info.shape[info.rank-2]),w=uint32_t(info.shape[info.rank-1]);for(uint32_t f=0;f<count;++f)Writer().Submit(Media::PlanarRGB(static_cast<const float*>(info.data),w,h,count,f),w,h);}
    void BeginWindow(X::Value options,int64_t skip){
        if(window||skip<0||skip>INT32_MAX)throw X::Error("invalid movie window");
        window=std::make_unique<Media::VideoAssembler>(nlohmann::json::parse(options.ToString()),Writer(),uint32_t(skip));
    }
    void AddWindowTile(int64_t tile,X::Value value){
        if(!window)throw X::Error("no movie window is active");
        MediaBindings::AddCompleteTile(*window,tile,value);
    }
    std::string FinishWindow(){
        if(!window)throw X::Error("no movie window is active");
        auto result=window->Finish().dump();window.reset();return result;
    }
    std::string Finish(){if(window)throw X::Error("finish the active movie window first");return Writer().Finish().dump();}
    std::string Status(){return Writer().Status().dump();}
    void Abort(){window.reset();writer.reset();}
    BEGIN_PACKAGE(NativeVideoWriter)
        APISET().AddFunc<1>("write_chunk",&NativeVideoWriter::WriteChunk);
        APISET().AddFunc<2>("write_audio",&NativeVideoWriter::WriteAudio);
        APISET().AddFunc<2>("write_audio_chunk",&NativeVideoWriter::WriteAudioChunk);
        APISET().AddFunc<2>("begin_window",&NativeVideoWriter::BeginWindow);
        APISET().AddFunc<2>("add_window_tile",&NativeVideoWriter::AddWindowTile);
        APISET().AddFunc<0>("finish_window",&NativeVideoWriter::FinishWindow);
        APISET().AddFunc<0>("abort",&NativeVideoWriter::Abort);
        APISET().AddFunc<0>("finish",&NativeVideoWriter::Finish);
        APISET().AddFunc<0>("status_json",&NativeVideoWriter::Status);
    END_PACKAGE
};
class NativeVideoAssembler {
    std::unique_ptr<Media::VideoAssembler> assembler;
public:
    explicit NativeVideoAssembler(X::Value options):assembler(std::make_unique<Media::VideoAssembler>(nlohmann::json::parse(options.ToString()))){}
    void WriteAudio(X::Value value,int64_t rate){if(!assembler)throw X::Error("video assembler is closed");auto tensor=MediaBindings::CPU(X::Tensor(value));const auto audio=MediaBindings::FloatAudio(tensor,rate);auto use=tensor.Acquire();assembler->SubmitAudio(audio);}
    void Add(int64_t tile,int64_t first,X::Value value){X::Tensor tensor(value);auto info=MediaBindings::FloatVideo(tensor);if(info.rank!=5||tile<0||first<0||first>INT32_MAX)throw X::Error("invalid tile chunk");
        if(!assembler)throw X::Error("video assembler is closed");auto use=tensor.Acquire();assembler->Add(size_t(tile),uint32_t(first),static_cast<const float*>(info.data),uint32_t(info.shape[2]),uint32_t(info.shape[3]),uint32_t(info.shape[4]));}
    void AddTile(int64_t tile,X::Value value){
        if(!assembler)throw X::Error("video assembler is closed");
        MediaBindings::AddCompleteTile(*assembler,tile,value);
    }
    std::string Finish(){if(!assembler)throw X::Error("video assembler is closed");auto result=assembler->Finish().dump();assembler.reset();return result;}
    std::string FinishWithAudio(X::Value value,int64_t rate){
        if(!assembler)throw X::Error("video assembler is closed");
        auto tensor=MediaBindings::CPU(X::Tensor(value));
        const auto audio=MediaBindings::FloatAudio(tensor,rate);
        auto use=tensor.Acquire();
        auto result=assembler->Finish(&audio).dump();
        assembler.reset();
        return result;
    }
    void Abort(){assembler.reset();}
    BEGIN_PACKAGE(NativeVideoAssembler)
        APISET().AddFunc<3>("add_tile_chunk",&NativeVideoAssembler::Add);
        APISET().AddFunc<2>("add_tile",&NativeVideoAssembler::AddTile);
        APISET().AddFunc<2>("write_audio",&NativeVideoAssembler::WriteAudio);
        APISET().AddFunc<0>("finish",&NativeVideoAssembler::Finish);
        APISET().AddFunc<2>("finish_with_audio",&NativeVideoAssembler::FinishWithAudio);
        APISET().AddFunc<0>("abort",&NativeVideoAssembler::Abort);
    END_PACKAGE
};
}
