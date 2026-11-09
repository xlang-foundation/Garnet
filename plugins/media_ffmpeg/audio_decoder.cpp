// SPDX-License-Identifier: Apache-2.0
#include "plugin_util.h"
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
}
namespace {
using namespace GMPlugin;
void Check(int result){if(result<0){char message[256]{};av_strerror(result,message,sizeof(message));throw std::runtime_error(message);}}
struct PCM {std::vector<float> left,right;};
struct AudioBuffer {
    const uint8_t* bytes;
    int64_t size;
    int64_t position=0;
};
int ReadAudio(void* opaque,uint8_t* destination,int capacity) {
    auto& source=*static_cast<AudioBuffer*>(opaque);
    const auto count=std::min<int64_t>(source.size-source.position,capacity);
    if(!count)return AVERROR_EOF;
    std::memcpy(destination,source.bytes+source.position,static_cast<size_t>(count));
    source.position+=count;
    return static_cast<int>(count);
}
int64_t SeekAudio(void* opaque,int64_t offset,int whence) {
    auto& source=*static_cast<AudioBuffer*>(opaque);
    switch(whence & ~AVSEEK_FORCE) {
    case AVSEEK_SIZE:return source.size;
    case SEEK_SET:break;
    case SEEK_CUR:
        if(offset < -source.position || offset > source.size-source.position)return AVERROR(EINVAL);
        offset+=source.position;
        break;
    case SEEK_END:
        if(offset < -source.size || offset > 0)return AVERROR(EINVAL);
        offset+=source.size;
        break;
    default:return AVERROR(EINVAL);
    }
    if(offset<0 || offset>source.size)return AVERROR(EINVAL);
    return source.position=offset;
}
struct Decoder {
    AVFormatContext* format=nullptr;AVCodecContext* codec=nullptr;AVFrame* frame=nullptr;AVPacket* packet=nullptr;SwrContext* resampler=nullptr;
    AVIOContext* io=nullptr;
    ~Decoder(){swr_free(&resampler);av_packet_free(&packet);av_frame_free(&frame);avcodec_free_context(&codec);avformat_close_input(&format);
        if(io){av_freep(&io->buffer);avio_context_free(&io);}}
};
int DecodeInput(const char* path,const void* bytes,size_t length,uint32_t rate,uint64_t offset,uint32_t count,GarnetMediaDecodedAudio* output,char* error,size_t capacity){return Guard(error,capacity,[&]{
    if((!path&&(!bytes||!length||length>INT64_MAX))||!output||rate<8000||rate>192000)throw std::runtime_error("invalid audio input/rate");
    if(count>uint64_t(rate)*600)throw std::runtime_error("requested audio range exceeds 600 seconds");
    *output={};
    AudioBuffer source{static_cast<const uint8_t*>(bytes),path?0:static_cast<int64_t>(length)};
    Decoder d;
    if(!path){
        constexpr int bufferSize=32768;
        auto* buffer=static_cast<uint8_t*>(av_malloc(bufferSize));
        if(!buffer)throw std::bad_alloc();
        d.io=avio_alloc_context(buffer,bufferSize,0,&source,ReadAudio,nullptr,SeekAudio);
        if(!d.io){av_free(buffer);throw std::bad_alloc();}
        d.format=avformat_alloc_context();
        if(!d.format)throw std::bad_alloc();
        d.format->pb=d.io;
        d.format->flags|=AVFMT_FLAG_CUSTOM_IO;
        d.format->io_open=[](AVFormatContext*,AVIOContext**,const char*,int,AVDictionary**){return AVERROR(EACCES);};
    }
    Check(avformat_open_input(&d.format,path,nullptr,nullptr));Check(avformat_find_stream_info(d.format,nullptr));
    const AVCodec* decoder=nullptr;int stream=av_find_best_stream(d.format,AVMEDIA_TYPE_AUDIO,-1,-1,&decoder,0);Check(stream);
    auto* parameters=d.format->streams[stream]->codecpar;
    if(parameters->ch_layout.nb_channels<1||parameters->ch_layout.nb_channels>2)throw std::runtime_error("audio input must be mono or stereo");
    d.codec=avcodec_alloc_context3(decoder);if(!d.codec)throw std::bad_alloc();Check(avcodec_parameters_to_context(d.codec,parameters));Check(avcodec_open2(d.codec,decoder,nullptr));
    AVChannelLayout stereo=AV_CHANNEL_LAYOUT_STEREO;
    Check(swr_alloc_set_opts2(&d.resampler,&stereo,AV_SAMPLE_FMT_FLTP,int(rate),&d.codec->ch_layout,d.codec->sample_fmt,d.codec->sample_rate,0,nullptr));
    if(d.codec->ch_layout.nb_channels==1){const double matrix[2]={1,1};Check(swr_set_matrix(d.resampler,matrix,1));}
    Check(swr_init(d.resampler));d.frame=av_frame_alloc();d.packet=av_packet_alloc();if(!d.frame||!d.packet)throw std::bad_alloc();
    auto pcm=std::make_unique<PCM>();
    auto done=[&]{return count&&pcm->left.size()==count;};
    auto convert=[&](const uint8_t** input,int inputCount){
        int maximum=int(av_rescale_rnd(swr_get_delay(d.resampler,d.codec->sample_rate)+inputCount,rate,d.codec->sample_rate,AV_ROUND_UP))+32;
        std::vector<float> left(maximum),right(maximum);uint8_t* planes[]={reinterpret_cast<uint8_t*>(left.data()),reinterpret_cast<uint8_t*>(right.data())};
        int produced=swr_convert(d.resampler,planes,maximum,input,inputCount);Check(produced);
        const auto skip=std::min(offset,uint64_t(produced));offset-=skip;
        auto take=size_t(produced-skip);
        if(count)take=std::min(take,count-pcm->left.size());
        if(pcm->left.size()+take>uint64_t(rate)*600)throw std::runtime_error("audio input exceeds 600 seconds");
        for(size_t i=size_t(skip);i<skip+take;++i)if(!std::isfinite(left[i])||!std::isfinite(right[i]))throw std::runtime_error("nonfinite decoded audio");
        pcm->left.insert(pcm->left.end(),left.begin()+skip,left.begin()+skip+take);
        pcm->right.insert(pcm->right.end(),right.begin()+skip,right.begin()+skip+take);return produced;
    };
    auto receive=[&]{while(!done()){int result=avcodec_receive_frame(d.codec,d.frame);if(result==AVERROR(EAGAIN)||result==AVERROR_EOF)return;Check(result);
        convert(const_cast<const uint8_t**>(d.frame->extended_data),d.frame->nb_samples);av_frame_unref(d.frame);}};
    int result=0;while(!done()&&(result=av_read_frame(d.format,d.packet))>=0){if(d.packet->stream_index==stream){int sent=avcodec_send_packet(d.codec,d.packet);if(sent==AVERROR(EAGAIN)){receive();if(done())break;sent=avcodec_send_packet(d.codec,d.packet);}Check(sent);receive();}av_packet_unref(d.packet);}
    if(!done()){
        if(result!=AVERROR_EOF)Check(result);Check(avcodec_send_packet(d.codec,nullptr));receive();
        while(!done()&&convert(nullptr,0)>0){}
    }
    if(count&&!done())throw std::runtime_error("supplied recording is shorter than the requested audio range");
    if(pcm->left.empty())throw std::runtime_error("audio input has no samples");
    output->audio={sizeof(GarnetMediaAudio),2,rate,uint32_t(pcm->left.size()),{pcm->left.data(),pcm->right.data()}};output->owner=pcm.release();
});}
int DecodeRange(const char* path,uint32_t rate,uint64_t offset,uint32_t count,GarnetMediaDecodedAudio* output,char* error,size_t capacity){return DecodeInput(path,nullptr,0,rate,offset,count,output,error,capacity);}
int DecodeBytes(const void* data,size_t length,uint32_t rate,uint64_t offset,uint32_t count,GarnetMediaDecodedAudio* output,char* error,size_t capacity){return DecodeInput(nullptr,data,length,rate,offset,count,output,error,capacity);}
int Decode(const char* path,uint32_t rate,GarnetMediaDecodedAudio* output,char* error,size_t capacity){return DecodeRange(path,rate,0,0,output,error,capacity);}
void Release(void* owner){delete static_cast<PCM*>(owner);}
GarnetMediaInputProviderV1 api{1,sizeof(api),Decode,Release};
GarnetMediaAudioRangeProviderV1 rangeApi{1,sizeof(rangeApi),DecodeRange,Release};
GarnetMediaAudioBytesProviderV1 bytesApi{1,sizeof(bytesApi),DecodeBytes,Release};
}
extern "C" GARNET_MEDIA_EXPORT const GarnetMediaInputProviderV1* GarnetMediaGetInputProviderV1(){return &api;}
extern "C" GARNET_MEDIA_EXPORT const GarnetMediaAudioRangeProviderV1* GarnetMediaGetAudioRangeProviderV1(){return &rangeApi;}
extern "C" GARNET_MEDIA_EXPORT const GarnetMediaAudioBytesProviderV1* GarnetMediaGetAudioBytesProviderV1(){return &bytesApi;}
