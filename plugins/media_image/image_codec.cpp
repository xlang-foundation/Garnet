// SPDX-License-Identifier: Apache-2.0
#include "plugin_util.h"
#include <limits>
#include <zlib.h>
#if defined(_WIN32)
#include <windows.h>
#include <wincodec.h>
#include <wrl/client.h>
#else
#include <jpeglib.h>
#include <png.h>
#include <setjmp.h>
#endif
namespace {
using namespace GMPlugin;
void BE(std::vector<uint8_t>& v,uint32_t n){for(int s=24;s>=0;s-=8)v.push_back(uint8_t(n>>s));}
void Chunk(std::vector<uint8_t>& v,const char* type,const std::vector<uint8_t>& data){BE(v,uint32_t(data.size()));size_t start=v.size();v.insert(v.end(),type,type+4);v.insert(v.end(),data.begin(),data.end());BE(v,uint32_t(crc32(0,v.data()+start,uInt(4+data.size()))));}
std::vector<uint8_t> PNG(const GarnetMediaFrame& f,const std::vector<uint8_t>& rgb){
    const size_t stride=size_t(f.width)*3;std::vector<uint8_t> raw((stride+1)*f.height);
    for(size_t y=0;y<f.height;++y)std::memcpy(raw.data()+y*(stride+1)+1,rgb.data()+y*stride,stride);
    uLongf length=compressBound(uLong(raw.size()));std::vector<uint8_t> compressed(length);
    if(compress2(compressed.data(),&length,raw.data(),uLong(raw.size()),Z_DEFAULT_COMPRESSION)!=Z_OK)throw std::runtime_error("PNG compression failed");compressed.resize(length);
    std::vector<uint8_t> bytes{137,80,78,71,13,10,26,10},header;BE(header,f.width);BE(header,f.height);header.insert(header.end(),{8,2,0,0,0});
    Chunk(bytes,"IHDR",header);Chunk(bytes,"IDAT",compressed);Chunk(bytes,"IEND",{});return bytes;
}
#if defined(_WIN32)
using Microsoft::WRL::ComPtr;
void Check(HRESULT status){if(FAILED(status))throw std::runtime_error("Windows image codec failed: "+std::to_string(uint32_t(status)));}
void JPEG(const GarnetMediaFrame& f,const char* path,const std::vector<uint8_t>& rgb,int quality){
    HRESULT initialized=CoInitializeEx(nullptr,COINIT_MULTITHREADED);if(FAILED(initialized)&&initialized!=RPC_E_CHANGED_MODE)Check(initialized);
    struct COM {bool own;~COM(){if(own)CoUninitialize();}} cleanup{SUCCEEDED(initialized)};
    ComPtr<IWICImagingFactory> factory;Check(CoCreateInstance(CLSID_WICImagingFactory,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&factory)));
    ComPtr<IWICStream> stream;Check(factory->CreateStream(&stream));auto p=std::filesystem::u8path(path);if(!p.parent_path().empty())std::filesystem::create_directories(p.parent_path());
    Check(stream->InitializeFromFilename(p.c_str(),GENERIC_WRITE));ComPtr<IWICBitmapEncoder> encoder;Check(factory->CreateEncoder(GUID_ContainerFormatJpeg,nullptr,&encoder));
    Check(encoder->Initialize(stream.Get(),WICBitmapEncoderNoCache));ComPtr<IWICBitmapFrameEncode> frame;ComPtr<IPropertyBag2> properties;
    Check(encoder->CreateNewFrame(&frame,&properties));PROPBAG2 property{};property.pstrName=const_cast<wchar_t*>(L"ImageQuality");VARIANT value{};value.vt=VT_R4;value.fltVal=quality/100.0f;
    Check(properties->Write(1,&property,&value));Check(frame->Initialize(properties.Get()));Check(frame->SetSize(f.width,f.height));WICPixelFormatGUID format=GUID_WICPixelFormat24bppBGR;Check(frame->SetPixelFormat(&format));
    if(!IsEqualGUID(format,GUID_WICPixelFormat24bppBGR))throw std::runtime_error("JPEG codec rejected BGR24");std::vector<uint8_t> bgr=rgb;
    for(size_t i=0;i<bgr.size();i+=3)std::swap(bgr[i],bgr[i+2]);Check(frame->WritePixels(f.height,f.width*3,UINT(bgr.size()),bgr.data()));Check(frame->Commit());Check(encoder->Commit());
}
#else
struct JPEGError {jpeg_error_mgr base;jmp_buf jump;char message[JMSG_LENGTH_MAX]{};};
void JPEGFailure(j_common_ptr state){auto* error=reinterpret_cast<JPEGError*>(state->err);state->err->format_message(state,error->message);longjmp(error->jump,1);}
struct JPEGCompress {
    jpeg_compress_struct codec{};JPEGError error{};FILE* file=nullptr;
    ~JPEGCompress(){if(codec.mem)jpeg_destroy_compress(&codec);if(file)std::fclose(file);}
};
struct JPEGDecompress {
    jpeg_decompress_struct codec{};JPEGError error{};FILE* file=nullptr;
    ~JPEGDecompress(){if(codec.mem)jpeg_destroy_decompress(&codec);if(file)std::fclose(file);}
};
void JPEG(const GarnetMediaFrame& f,const char* path,const std::vector<uint8_t>& rgb,int quality){
    auto output=std::filesystem::u8path(path);if(!output.parent_path().empty())std::filesystem::create_directories(output.parent_path());
    auto state=std::make_unique<JPEGCompress>();state->file=std::fopen(path,"wb");if(!state->file)throw std::runtime_error("cannot open JPEG output");
    auto& c=state->codec;c.err=jpeg_std_error(&state->error.base);state->error.base.error_exit=JPEGFailure;
    if(setjmp(state->error.jump))throw std::runtime_error(state->error.message);
    jpeg_create_compress(&c);jpeg_stdio_dest(&c,state->file);c.image_width=f.width;c.image_height=f.height;c.input_components=3;c.in_color_space=JCS_RGB;
    jpeg_set_defaults(&c);jpeg_set_quality(&c,quality,TRUE);jpeg_start_compress(&c,TRUE);while(c.next_scanline<c.image_height){JSAMPROW row=const_cast<JSAMPROW>(rgb.data()+size_t(c.next_scanline)*f.width*3);jpeg_write_scanlines(&c,&row,1);}jpeg_finish_compress(&c);
}
#endif
int Encode(const GarnetMediaFrame* f,const char* path,const char* options,char* error,size_t n){return Guard(error,n,[&]{auto rgb=PixelsRGB(f);auto o=Json::parse(options);auto format=o.value("format","png");
    if(format=="png"){auto bytes=PNG(*f,rgb);Write(path,bytes.data(),bytes.size());}else if(format=="jpeg"||format=="jpg"){int q=int(Integer(o,"quality",90,1,100));JPEG(*f,path,rgb,q);}else throw std::runtime_error("unsupported image format");});}
const char* Caps(){return R"({"formats":["png","jpeg"],"devices":["cpu"]})";}
struct Decoded {std::vector<uint8_t> rgb;};
int DecodeInput(const char* path,const void* data,size_t length,GarnetMediaDecoded* result,char* error,size_t n){return Guard(error,n,[&]{
    if (!path && (!data || !length || length > (std::numeric_limits<uint32_t>::max)()))
        throw std::runtime_error("invalid encoded image buffer");
    if(!result)throw std::runtime_error("decoded image output required");auto owned=std::make_unique<Decoded>();uint32_t width=0,height=0;
#if defined(_WIN32)
    HRESULT initialized=CoInitializeEx(nullptr,COINIT_MULTITHREADED);if(FAILED(initialized)&&initialized!=RPC_E_CHANGED_MODE)Check(initialized);
    struct COM {bool own;~COM(){if(own)CoUninitialize();}} cleanup{SUCCEEDED(initialized)};
    ComPtr<IWICImagingFactory> factory;Check(CoCreateInstance(CLSID_WICImagingFactory,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&factory)));
    ComPtr<IWICStream> stream;
    ComPtr<IWICBitmapDecoder> decoder;
    if (path) {
        auto p=std::filesystem::u8path(path);
        Check(factory->CreateDecoderFromFilename(p.c_str(),nullptr,GENERIC_READ,WICDecodeMetadataCacheOnLoad,&decoder));
    } else {
        Check(factory->CreateStream(&stream));
        Check(stream->InitializeFromMemory(const_cast<BYTE*>(static_cast<const BYTE*>(data)),static_cast<DWORD>(length)));
        Check(factory->CreateDecoderFromStream(stream.Get(),nullptr,WICDecodeMetadataCacheOnLoad,&decoder));
    }
    GUID container{};Check(decoder->GetContainerFormat(&container));if(!IsEqualGUID(container,GUID_ContainerFormatPng)&&!IsEqualGUID(container,GUID_ContainerFormatJpeg))throw std::runtime_error("image format must be PNG or JPEG");
    ComPtr<IWICBitmapFrameDecode> frame;Check(decoder->GetFrame(0,&frame));Check(frame->GetSize(&width,&height));Garnet::Media::Dimensions(width,height);
    ComPtr<IWICFormatConverter> converter;Check(factory->CreateFormatConverter(&converter));Check(converter->Initialize(frame.Get(),GUID_WICPixelFormat32bppRGBA,WICBitmapDitherTypeNone,nullptr,0,WICBitmapPaletteTypeCustom));
    std::vector<uint8_t> rgba(size_t(width)*height*4);Check(converter->CopyPixels(nullptr,width*4,UINT(rgba.size()),rgba.data()));owned->rgb.resize(size_t(width)*height*3);
    for(size_t i=0;i<size_t(width)*height;++i)for(size_t c=0;c<3;++c){unsigned alpha=rgba[i*4+3];owned->rgb[i*3+c]=uint8_t((rgba[i*4+c]*alpha+255*(255-alpha)+127)/255);}
#else
    // Portable PNG decode uses libpng's simplified API. JPEG decode is handled
    // by libjpeg; neither implementation depends on a model runtime.
    unsigned char signature[8]{};
    if (path) {
        std::ifstream file(path,std::ios::binary);
        file.read(reinterpret_cast<char*>(signature),sizeof(signature));
    } else {
        std::memcpy(signature,data,std::min(length,sizeof(signature)));
    }
    if(signature[0]==137&&signature[1]=='P'){
        png_image image{};struct PNGOwner {png_image* image;~PNGOwner(){png_image_free(image);}} cleanup{&image};
        image.version=PNG_IMAGE_VERSION;
        const auto ready=path ? png_image_begin_read_from_file(&image,path) : png_image_begin_read_from_memory(&image,data,length);
        if(!ready)throw std::runtime_error("PNG header decode failed");
        width=image.width;height=image.height;Garnet::Media::Dimensions(width,height);image.format=PNG_FORMAT_RGBA;std::vector<uint8_t> rgba(PNG_IMAGE_SIZE(image));
        if(!png_image_finish_read(&image,nullptr,rgba.data(),0,nullptr))throw std::runtime_error("PNG pixel decode failed");owned->rgb.resize(size_t(width)*height*3);
        for(size_t i=0;i<size_t(width)*height;++i)for(size_t c=0;c<3;++c){unsigned a=rgba[i*4+3];owned->rgb[i*3+c]=uint8_t((rgba[i*4+c]*a+255*(255-a)+127)/255);}
    }else{
        auto state=std::make_unique<JPEGDecompress>();
        if(path){state->file=std::fopen(path,"rb");if(!state->file)throw std::runtime_error("cannot open JPEG input");}
        auto& c=state->codec;c.err=jpeg_std_error(&state->error.base);state->error.base.error_exit=JPEGFailure;
        if(setjmp(state->error.jump))throw std::runtime_error(state->error.message);
        jpeg_create_decompress(&c);
        if(path)jpeg_stdio_src(&c,state->file);else jpeg_mem_src(&c,static_cast<const unsigned char*>(data),static_cast<unsigned long>(length));
        jpeg_read_header(&c,TRUE);c.out_color_space=JCS_RGB;
        Garnet::Media::Dimensions(c.image_width,c.image_height);jpeg_start_decompress(&c);
        width=c.output_width;height=c.output_height;Garnet::Media::Dimensions(width,height);owned->rgb.resize(size_t(width)*height*3);
        while(c.output_scanline<c.output_height){JSAMPROW row=owned->rgb.data()+size_t(c.output_scanline)*width*3;jpeg_read_scanlines(&c,&row,1);}jpeg_finish_decompress(&c);
    }
#endif
    result->frame=Garnet::Media::RGBFrame(owned->rgb.data(),width,height);result->owner=owned.release();
});}
int Decode(const char* path,GarnetMediaDecoded* result,char* error,size_t n){return DecodeInput(path,nullptr,0,result,error,n);}
int DecodeBytes(const void* data,size_t length,GarnetMediaDecoded* result,char* error,size_t n){return DecodeInput(nullptr,data,length,result,error,n);}
void Release(void* data){delete static_cast<Decoded*>(data);}
int EncodePNG(const GarnetMediaFrame* frame,const void** data,size_t* length,void** owner,char* error,size_t capacity){
    return Guard(error,capacity,[&]{
        if(!data||!length||!owner)throw std::runtime_error("encoded image output required");
        auto rgb=PixelsRGB(frame);
        auto bytes=std::make_unique<std::vector<uint8_t>>(PNG(*frame,rgb));
        *data=bytes->data();*length=bytes->size();*owner=bytes.release();
    });
}
void ReleasePNG(void* owner){delete static_cast<std::vector<uint8_t>*>(owner);}
GarnetMediaProviderV1 api{1,sizeof(api),"image",Caps,Encode,nullptr,nullptr,nullptr,nullptr,nullptr,Decode,Release};
GarnetMediaImageBytesProviderV1 bytesApi{1,sizeof(bytesApi),DecodeBytes,Release,EncodePNG,ReleasePNG};
}
extern "C" GARNET_MEDIA_EXPORT const GarnetMediaProviderV1* GarnetMediaGetProviderV1(){return &api;}
extern "C" GARNET_MEDIA_EXPORT const GarnetMediaImageBytesProviderV1* GarnetMediaGetImageBytesProviderV1(){return &bytesApi;}
