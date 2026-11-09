// SPDX-License-Identifier: Apache-2.0
// Provider selection and software/hardware frame preparation adapted from
// Lumen/WebRTC/video_encoder.cpp. Preserve AVPackets for file muxing.
#include "plugin_util.h"
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}
namespace {
using namespace GMPlugin;
std::string Text(int status){char message[256]{};av_strerror(status,message,sizeof(message));return message;}
void Check(int status){if(status<0)throw std::runtime_error(Text(status));}
struct Encoder {
    AVFormatContext* format=nullptr;AVCodecContext* codec=nullptr;AVStream* stream=nullptr;
    AVCodecContext* audioCodec=nullptr;AVStream* audioStream=nullptr;AVFrame* audioFrame=nullptr;
    AVFrame* software=nullptr;AVFrame* hardware=nullptr;AVPacket* packet=nullptr;AVBufferRef* device=nullptr;AVBufferRef* frames=nullptr;SwsContext* convert=nullptr;
    uint32_t width=0,height=0;int64_t next=0,audioSamples=0,expectedFrames=0;
    bool finished=false,audioAtFinish=false,streamingAudio=false;
    int audioBuffered=0;
    std::string name,info;
    ~Encoder(){if(format&&format->pb)avio_closep(&format->pb);av_packet_free(&packet);av_frame_free(&software);av_frame_free(&hardware);
        sws_freeContext(convert);avcodec_free_context(&codec);avcodec_free_context(&audioCodec);av_frame_free(&audioFrame);av_buffer_unref(&frames);av_buffer_unref(&device);avformat_free_context(format);}
    void Open(const Json& o){
        width=uint32_t(Integer(o,"width",0,1,16384));height=uint32_t(Integer(o,"height",0,1,16384));Garnet::Media::Dimensions(width,height);int fps=int(Integer(o,"fps",24,1,240));if(width%2||height%2)throw std::runtime_error("invalid MP4 geometry");
        const int bframes=int(Integer(o,"b_frames",0,0,4)),quality=int(Integer(o,"quality",23,0,51)),deviceID=int(Integer(o,"device_id",0,0,INT32_MAX));
        const int64_t bitrate=Integer(o,"bitrate",4000000,1,1000000000);
        const std::string selected=o.value("encoder","auto"),codecName=o.value("codec","h264");std::vector<std::string> candidates;
        if(codecName!="h264"&&codecName!="hevc"&&codecName!="av1")throw std::runtime_error("unsupported video codec");
        if(selected!="auto")candidates={selected};else if(!o.value("allow_hardware",true)){
            candidates={codecName=="h264"?"libx264":codecName=="hevc"?"libx265":"libaom-av1"};
#if defined(_WIN32)
            if(codecName=="h264")candidates.push_back("h264_mf");
#endif
        }else{
            candidates={codecName+"_nvenc",codecName+"_qsv",codecName+"_amf"};
#if defined(__APPLE__)
            candidates.push_back(codecName+"_videotoolbox");
#elif defined(_WIN32)
            if(codecName=="h264")candidates.push_back("h264_mf");
#else
            candidates.push_back(codecName+"_vaapi");
#endif
            candidates.push_back(codecName=="h264"?"libx264":codecName=="hevc"?"libx265":"libaom-av1");}
        std::string failures;
        for(const auto& candidate:candidates){const AVCodec* implementation=avcodec_find_encoder_by_name(candidate.c_str());if(!implementation){failures+=candidate+":not compiled; ";continue;}
            if((codecName=="h264"&&implementation->id!=AV_CODEC_ID_H264)||(codecName=="hevc"&&implementation->id!=AV_CODEC_ID_HEVC)||(codecName=="av1"&&implementation->id!=AV_CODEC_ID_AV1))throw std::runtime_error("encoder and codec differ");
            codec=avcodec_alloc_context3(implementation);if(!codec)throw std::bad_alloc();codec->width=width;codec->height=height;codec->time_base={1,fps};codec->framerate={fps,1};
            codec->flags|=AV_CODEC_FLAG_GLOBAL_HEADER;codec->gop_size=fps*2;codec->max_b_frames=bframes;codec->thread_count=2;
            codec->color_range=AVCOL_RANGE_MPEG;codec->colorspace=AVCOL_SPC_BT709;codec->color_primaries=AVCOL_PRI_BT709;codec->color_trc=AVCOL_TRC_BT709;
            const bool nvenc=candidate.find("_nvenc")!=std::string::npos;
            const bool mf=candidate.find("_mf")!=std::string::npos;
            const bool hardwareName=nvenc||candidate.find("_qsv")!=std::string::npos||candidate.find("_amf")!=std::string::npos||candidate.find("_videotoolbox")!=std::string::npos||(mf&&o.value("hardware_encoding",false))||candidate.find("_vaapi")!=std::string::npos;
            if(hardwareName&&!o.value("allow_hardware",true))throw std::runtime_error("hardware encoder disabled by policy");
            codec->pix_fmt=hardwareName||mf?AV_PIX_FMT_NV12:AV_PIX_FMT_YUV420P;AVDictionary* settings=nullptr;
            if(mf)av_dict_set(&settings,"hw_encoding",o.value("hardware_encoding",false)?"1":"0",0);
            if(!hardwareName&&!mf){av_dict_set(&settings,"preset","medium",0);av_dict_set(&settings,"crf",std::to_string(quality).c_str(),0);}
            else {codec->bit_rate=bitrate;if(nvenc){av_dict_set(&settings,"preset","p4",0);av_dict_set(&settings,"rc","vbr",0);av_dict_set(&settings,"cq",std::to_string(quality).c_str(),0);}}
            int result=0;
            if(nvenc){ // Explicit logical CUDA device; caller's visibility mask is retained.
                result=av_hwdevice_ctx_create(&device,AV_HWDEVICE_TYPE_CUDA,std::to_string(deviceID).c_str(),nullptr,0);
                if(result>=0){frames=av_hwframe_ctx_alloc(device);if(!frames)result=AVERROR(ENOMEM);else{
                    auto* f=reinterpret_cast<AVHWFramesContext*>(frames->data);f->format=AV_PIX_FMT_CUDA;f->sw_format=AV_PIX_FMT_NV12;f->width=width;f->height=height;f->initial_pool_size=4;
                    result=av_hwframe_ctx_init(frames);if(result>=0){codec->pix_fmt=AV_PIX_FMT_CUDA;codec->hw_device_ctx=av_buffer_ref(device);codec->hw_frames_ctx=av_buffer_ref(frames);}}}
            }
            if(result>=0)result=avcodec_open2(codec,implementation,&settings);av_dict_free(&settings);
            if(result>=0){name=candidate;info=Json{{"name",name},{"hardware",hardwareName},{"device_id",deviceID},{"input","rgb8"},{"codec",codecName}}.dump();break;}
            failures+=candidate+":"+Text(result)+"; ";avcodec_free_context(&codec);av_buffer_unref(&frames);av_buffer_unref(&device);
        }
        if(!codec)throw std::runtime_error("no usable video encoder: "+failures);
        Check(avformat_alloc_output_context2(&format,nullptr,"mp4",o.at("path").get<std::string>().c_str()));stream=avformat_new_stream(format,nullptr);if(!stream)throw std::bad_alloc();
        stream->time_base=codec->time_base;Check(avcodec_parameters_from_context(stream->codecpar,codec));
        const int audioRate=int(Integer(o,"audio_sample_rate",0,0,192000));
        audioAtFinish=o.value("audio_at_finish",false);
        streamingAudio=o.value("streaming_audio",false);
        if(streamingAudio&&(!audioRate||audioAtFinish))throw std::runtime_error("streaming audio requires an enabled audio track");
        if(audioAtFinish){
            if(!audioRate)throw std::runtime_error("deferred audio requires an audio track");
            expectedFrames=Integer(o,"frames",0,1,int64_t(fps)*600);
            // The muxer may retain encoded packets up to this declared clip
            // length so it can interleave the late audio at its original PTS.
            format->max_interleave_delta=(expectedFrames+fps)*AV_TIME_BASE/fps;
        }
        if(audioRate){
            const int channels=int(Integer(o,"audio_channels",2,1,2));const auto* aac=avcodec_find_encoder(AV_CODEC_ID_AAC);
            if(!aac)throw std::runtime_error("AAC encoder is not compiled into this media provider");audioCodec=avcodec_alloc_context3(aac);if(!audioCodec)throw std::bad_alloc();
            audioCodec->sample_rate=audioRate;audioCodec->sample_fmt=AV_SAMPLE_FMT_FLTP;audioCodec->time_base={1,audioRate};audioCodec->bit_rate=128000;
            av_channel_layout_default(&audioCodec->ch_layout,channels);audioCodec->flags|=AV_CODEC_FLAG_GLOBAL_HEADER;Check(avcodec_open2(audioCodec,aac,nullptr));
            audioStream=avformat_new_stream(format,nullptr);if(!audioStream)throw std::bad_alloc();audioStream->time_base=audioCodec->time_base;Check(avcodec_parameters_from_context(audioStream->codecpar,audioCodec));
            audioFrame=av_frame_alloc();if(!audioFrame)throw std::bad_alloc();audioFrame->format=audioCodec->sample_fmt;audioFrame->sample_rate=audioRate;
            Check(av_channel_layout_copy(&audioFrame->ch_layout,&audioCodec->ch_layout));audioFrame->nb_samples=audioCodec->frame_size;Check(av_frame_get_buffer(audioFrame,0));
            auto details=Json::parse(info);details["audio"]={{"codec","aac"},{"sample_rate",audioRate},{"channels",channels}};info=details.dump();
        }
        const auto path=o.at("path").get<std::string>();
        const bool streaming=Garnet::Media::IsVideoStreamTarget(path);
        AVDictionary* io=nullptr;
        if(streaming)av_dict_set_int(&io,"rw_timeout",30000000,0);
        const auto opened=avio_open2(&format->pb,path.c_str(),AVIO_FLAG_WRITE,nullptr,&io);
        av_dict_free(&io);
        Check(opened);
        AVDictionary* mux=nullptr;
        // The delayed header can record AAC priming in an edit list even on a pipe.
        av_dict_set(&mux,"movflags",streaming?"+delay_moov+default_base_moof+frag_keyframe":"+faststart",0);
        if(streaming)av_dict_set_int(&mux,"frag_duration",1000000,0);
        if(streamingAudio)av_dict_set_int(&mux,"movie_timescale",audioRate,0);
        int result=avformat_write_header(format,&mux);av_dict_free(&mux);Check(result);
        software=av_frame_alloc();packet=av_packet_alloc();if(!software||!packet)throw std::bad_alloc();software->format=frames?AV_PIX_FMT_NV12:codec->pix_fmt;
        software->width=width;software->height=height;Check(av_frame_get_buffer(software,32));
        convert=sws_getContext(width,height,AV_PIX_FMT_RGB24,width,height,static_cast<AVPixelFormat>(software->format),SWS_BILINEAR,nullptr,nullptr,nullptr);if(!convert)throw std::runtime_error("RGB conversion unavailable");
        const int* coefficients=sws_getCoefficients(SWS_CS_ITU709);Check(sws_setColorspaceDetails(convert,coefficients,1,coefficients,0,0,1<<16,1<<16));
        if(frames){hardware=av_frame_alloc();if(!hardware)throw std::bad_alloc();}
    }
    void Packets(AVCodecContext* source=nullptr,AVStream* destination=nullptr){if(!source){source=codec;destination=stream;}while(true){int result=avcodec_receive_packet(source,packet);if(result==AVERROR(EAGAIN)||result==AVERROR_EOF)return;Check(result);
        av_packet_rescale_ts(packet,source->time_base,destination->time_base);packet->stream_index=destination->index;if(!packet->duration)packet->duration=av_rescale_q(source==codec?1:source->frame_size,source->time_base,destination->time_base);
        int written=av_interleaved_write_frame(format,packet);av_packet_unref(packet);Check(written);}}
    void Submit(const GarnetMediaFrame& f){if(finished||(audioAtFinish&&next>=expectedFrames)||f.device!=GM_CPU||f.format!=GM_RGB8||f.width!=width||f.height!=height||f.pts!=next||!f.planes[0]||f.strides[0]<width*3)throw std::runtime_error("invalid video frame or timestamp");
        Check(av_frame_make_writable(software));const uint8_t* source[]={static_cast<const uint8_t*>(f.planes[0])};int strides[]={int(f.strides[0])};
        if(sws_scale(convert,source,strides,0,height,software->data,software->linesize)!=int(height))throw std::runtime_error("frame conversion failed");software->pts=next++;
        software->color_range=codec->color_range;software->colorspace=codec->colorspace;software->color_primaries=codec->color_primaries;software->color_trc=codec->color_trc;
        AVFrame* input=software;if(frames){av_frame_unref(hardware);Check(av_hwframe_get_buffer(frames,hardware,0));Check(av_hwframe_transfer_data(hardware,software,0));Check(av_frame_copy_props(hardware,software));input=hardware;}
        int result=avcodec_send_frame(codec,input);if(result==AVERROR(EAGAIN)){Packets();result=avcodec_send_frame(codec,input);}Check(result);Packets();
    }
    void ValidateAudio(const GarnetMediaAudio& a){
        if(finished||!audioCodec||a.size<sizeof(a)||a.channels!=uint32_t(audioCodec->ch_layout.nb_channels)||a.sample_rate!=uint32_t(audioCodec->sample_rate)||!a.samples||a.samples>uint64_t(a.sample_rate)*600)
            throw std::runtime_error("invalid generated audio or audio/video configuration");
        for(uint32_t c=0;c<a.channels;++c){if(!a.planes[c])throw std::runtime_error("audio plane is missing");for(uint32_t i=0;i<a.samples;++i)if(!std::isfinite(a.planes[c][i]))throw std::runtime_error("nonfinite audio sample");}
    }
    void EncodeAudioFrame(int count,int64_t pts){
        audioFrame->nb_samples=count;audioFrame->pts=pts;
        int result=avcodec_send_frame(audioCodec,audioFrame);
        if(result==AVERROR(EAGAIN)){Packets(audioCodec,audioStream);result=avcodec_send_frame(audioCodec,audioFrame);}
        Check(result);Packets(audioCodec,audioStream);
    }
    void SubmitAudio(const GarnetMediaAudio& a){
        if(streamingAudio||(audioAtFinish?next!=expectedFrames:next!=0)||audioSamples)
            throw std::runtime_error("invalid generated audio or audio/video configuration");
        ValidateAudio(a);
        for(uint32_t first=0;first<a.samples;){const auto count=std::min(uint32_t(audioCodec->frame_size),a.samples-first);
            Check(av_frame_make_writable(audioFrame));
            for(uint32_t c=0;c<a.channels;++c)std::memcpy(audioFrame->data[c],a.planes[c]+first,count*sizeof(float));
            EncodeAudioFrame(int(count),first);first+=count;
        }audioSamples=a.samples;
    }
    void SubmitAudioChunk(const GarnetMediaAudio& a){
        if(!streamingAudio)throw std::runtime_error("streaming audio is not enabled");
        ValidateAudio(a);
        for(uint32_t first=0;first<a.samples;){
            if(!audioBuffered)Check(av_frame_make_writable(audioFrame));
            const auto count=std::min(uint32_t(audioCodec->frame_size-audioBuffered),a.samples-first);
            for(uint32_t c=0;c<a.channels;++c)
                std::memcpy(reinterpret_cast<float*>(audioFrame->data[c])+audioBuffered,a.planes[c]+first,count*sizeof(float));
            audioBuffered+=int(count);audioSamples+=count;first+=count;
            if(audioBuffered==audioCodec->frame_size){
                EncodeAudioFrame(audioBuffered,audioSamples-audioBuffered);audioBuffered=0;
            }
        }
    }
    void Finish(){if(finished)return;if(audioCodec&&!audioSamples)throw std::runtime_error("configured audio track has no generated samples");
        if(audioBuffered){EncodeAudioFrame(audioBuffered,audioSamples-audioBuffered);audioBuffered=0;}
        int result=avcodec_send_frame(codec,nullptr);if(result==AVERROR(EAGAIN)){Packets();result=avcodec_send_frame(codec,nullptr);}if(result!=AVERROR_EOF)Check(result);Packets();
        if(audioCodec){result=avcodec_send_frame(audioCodec,nullptr);if(result==AVERROR(EAGAIN)){Packets(audioCodec,audioStream);result=avcodec_send_frame(audioCodec,nullptr);}if(result!=AVERROR_EOF)Check(result);Packets(audioCodec,audioStream);}
        Check(av_write_trailer(format));Check(avio_closep(&format->pb));finished=true;}
};
void* Open(const char* options,char* error,size_t n){try{auto value=std::make_unique<Encoder>();value->Open(Json::parse(options));return value.release();}catch(const std::exception& e){Error(error,n,e.what());return nullptr;}}
int Submit(void* p,const GarnetMediaFrame* f,char* e,size_t n){return Guard(e,n,[&]{if(!p||!f||f->size<sizeof(*f))throw std::runtime_error("invalid video handle/frame");static_cast<Encoder*>(p)->Submit(*f);});}
int SubmitAudio(void* p,const GarnetMediaAudio* a,char* e,size_t n){return Guard(e,n,[&]{if(!p||!a)throw std::runtime_error("invalid audio handle/frame");static_cast<Encoder*>(p)->SubmitAudio(*a);});}
int SubmitAudioChunk(void* p,const GarnetMediaAudio* a,char* e,size_t n){return Guard(e,n,[&]{if(!p||!a)throw std::runtime_error("invalid audio handle/frame");static_cast<Encoder*>(p)->SubmitAudioChunk(*a);});}
int Finish(void* p,char* e,size_t n){return Guard(e,n,[&]{static_cast<Encoder*>(p)->Finish();});}
const char* Info(void* p){return static_cast<Encoder*>(p)->info.c_str();}
void Close(void* p){delete static_cast<Encoder*>(p);}
const char* Caps(){static const std::string value=[] {Json names=Json::array();void* state=nullptr;const AVCodec* codec=nullptr;
    while((codec=av_codec_iterate(&state)))if(av_codec_is_encoder(codec)&&(codec->id==AV_CODEC_ID_H264||codec->id==AV_CODEC_ID_HEVC||codec->id==AV_CODEC_ID_AV1))names.push_back(codec->name);
    Json audio=Json::array();if(avcodec_find_encoder(AV_CODEC_ID_AAC))audio.push_back("aac");
    return Json{{"formats",{"mp4"}},{"compiled_encoders",names},{"hardware","runtime encoder-open probe"},{"input_devices",{"cpu"}},
        {"audio_codecs",audio},{"audio_input","planar_fp32"},{"audio_max_channels",2}}.dump();}();return value.c_str();}
GarnetMediaProviderV1 api{1,sizeof(api),"ffmpeg",Caps,nullptr,Open,Submit,Finish,Info,Close};
GarnetMediaAudioProviderV1 audioApi{1,sizeof(audioApi),SubmitAudio};
GarnetMediaStreamingAudioProviderV1 streamingAudioApi{1,sizeof(streamingAudioApi),SubmitAudioChunk};
}
extern "C" GARNET_MEDIA_EXPORT const GarnetMediaProviderV1* GarnetMediaGetProviderV1(){return &api;}
extern "C" GARNET_MEDIA_EXPORT const GarnetMediaAudioProviderV1* GarnetMediaGetAudioProviderV1(){return &audioApi;}
extern "C" GARNET_MEDIA_EXPORT const GarnetMediaStreamingAudioProviderV1* GarnetMediaGetStreamingAudioProviderV1(){return &streamingAudioApi;}
