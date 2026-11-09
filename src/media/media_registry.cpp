// SPDX-License-Identifier: Apache-2.0
#include "media_registry.h"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <mutex>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <cstring>
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
namespace Garnet::Media {
namespace fs=std::filesystem;using Json=nlohmann::json;
namespace {
fs::path RuntimeDirectory(){
#if defined(_WIN32)
    HMODULE mod=nullptr;wchar_t path[32768]{};
    if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&RuntimeDirectory),&mod) || !GetModuleFileNameW(mod,path,32768))throw std::runtime_error("cannot locate media host");
    return fs::path(path).parent_path();
#else
    Dl_info info{};if(!dladdr(reinterpret_cast<void*>(&RuntimeDirectory),&info))throw std::runtime_error("cannot locate media host");
    return fs::path(info.dli_fname).parent_path();
#endif
}
struct Library {void* handle;const GarnetMediaProviderV1* api;};
std::mutex registryMutex;
// Providers stay loaded until process exit, including while encoder sessions exist.
auto& libraries=*new std::map<std::string,Library>;
uint32_t Positive(const Json& j,const char* name,uint32_t fallback=0){
    const auto value=j.value(name,Json(fallback));
    if(!value.is_number_integer() || value<1 || value>INT32_MAX)throw std::runtime_error(std::string("invalid integer ")+name);
    return value.get<uint32_t>();
}
uint32_t Nonnegative(const Json& value){
    if(!value.is_number_integer() || value<0 || value>INT32_MAX)throw std::runtime_error("invalid tile integer");
    return value.get<uint32_t>();
}
}
const GarnetMediaProviderV1& Provider(const std::string& id){
    std::lock_guard<std::mutex> lock(registryMutex);auto found=libraries.find(id);if(found!=libraries.end())return *found->second.api;
    if(id!="image" && id!="nvjpeg" && id!="ffmpeg")throw std::runtime_error("unknown media provider: "+id);
    const char* custom=std::getenv("GARNET_MEDIA_PLUGIN_DIR");fs::path root=custom&&*custom?fs::u8path(custom):RuntimeDirectory()/"plugins"/"media";
#if defined(_WIN32)
    auto path=fs::absolute(root/("garnet_media_"+id+".dll"));
    auto handle=LoadLibraryExW(path.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if(!handle)throw std::runtime_error("media provider unavailable: "+path.u8string()+" (Windows error "+std::to_string(GetLastError())+")");
    auto entry=reinterpret_cast<GarnetMediaGetProviderFn>(GetProcAddress(handle,"GarnetMediaGetProviderV1"));
#else
#if defined(__APPLE__)
    auto path=fs::absolute(root/("libgarnet_media_"+id+".dylib"));
#else
    auto path=fs::absolute(root/("libgarnet_media_"+id+".so"));
#endif
    auto handle=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if(!handle)throw std::runtime_error("media provider unavailable: "+path.string()+": "+dlerror());
    auto entry=reinterpret_cast<GarnetMediaGetProviderFn>(dlsym(handle,"GarnetMediaGetProviderV1"));
#endif
    const auto* api=entry?entry():nullptr;
    if(!api || api->abi!=1 || api->size<sizeof(GarnetMediaProviderV1) || !api->id || id!=api->id || !api->capabilities){
#if defined(_WIN32)
        FreeLibrary(handle);
#else
        dlclose(handle);
#endif
        throw std::runtime_error("media provider ABI/identity mismatch: "+id);
    }
    libraries.emplace(id,Library{reinterpret_cast<void*>(handle),api});return *api;
}
void EncodeImage(const GarnetMediaFrame& frame,const std::string& path,const Json& options){
    const auto& p=Provider(options.value("provider",frame.device==GM_CUDA?"nvjpeg":"image"));
    if(!p.encode_image)throw std::runtime_error("provider does not encode images");char error[2048]{};
    if(p.encode_image(&frame,path.c_str(),options.dump().c_str(),error,sizeof(error)))throw std::runtime_error(error);
}
const GarnetMediaAudioProviderV1& AudioProvider(){
    Provider("ffmpeg");std::lock_guard<std::mutex> lock(registryMutex);const auto handle=libraries.at("ffmpeg").handle;
#if defined(_WIN32)
    auto entry=reinterpret_cast<GarnetMediaGetAudioProviderFn>(GetProcAddress(static_cast<HMODULE>(handle),"GarnetMediaGetAudioProviderV1"));
#else
    auto entry=reinterpret_cast<GarnetMediaGetAudioProviderFn>(dlsym(handle,"GarnetMediaGetAudioProviderV1"));
#endif
    auto* api=entry?entry():nullptr;
    if(!api||api->abi!=1||api->size<sizeof(*api)||!api->submit_audio)throw std::runtime_error("FFmpeg provider audio extension is unavailable");return *api;
}
const GarnetMediaStreamingAudioProviderV1& StreamingAudioProvider(){
    Provider("ffmpeg");std::lock_guard<std::mutex> lock(registryMutex);const auto handle=libraries.at("ffmpeg").handle;
#if defined(_WIN32)
    auto entry=reinterpret_cast<GarnetMediaGetStreamingAudioProviderFn>(GetProcAddress(static_cast<HMODULE>(handle),"GarnetMediaGetStreamingAudioProviderV1"));
#else
    auto entry=reinterpret_cast<GarnetMediaGetStreamingAudioProviderFn>(dlsym(handle,"GarnetMediaGetStreamingAudioProviderV1"));
#endif
    auto* api=entry?entry():nullptr;
    if(!api||api->abi!=1||api->size<sizeof(*api)||!api->submit_audio_chunk)
        throw std::runtime_error("FFmpeg provider streaming audio extension is unavailable");
    return *api;
}
const GarnetMediaInputProviderV1& InputProvider(){
    Provider("ffmpeg");std::lock_guard<std::mutex> lock(registryMutex);const auto handle=libraries.at("ffmpeg").handle;
#if defined(_WIN32)
    auto entry=reinterpret_cast<GarnetMediaGetInputProviderFn>(GetProcAddress(static_cast<HMODULE>(handle),"GarnetMediaGetInputProviderV1"));
#else
    auto entry=reinterpret_cast<GarnetMediaGetInputProviderFn>(dlsym(handle,"GarnetMediaGetInputProviderV1"));
#endif
    auto* api=entry?entry():nullptr;
    if(!api||api->abi!=1||api->size<sizeof(*api)||!api->decode_audio||!api->release_audio)throw std::runtime_error("FFmpeg input extension is unavailable");return *api;
}
const GarnetMediaAudioRangeProviderV1& AudioRangeProvider(){
    Provider("ffmpeg");std::lock_guard<std::mutex> lock(registryMutex);const auto handle=libraries.at("ffmpeg").handle;
#if defined(_WIN32)
    auto entry=reinterpret_cast<GarnetMediaGetAudioRangeProviderFn>(GetProcAddress(static_cast<HMODULE>(handle),"GarnetMediaGetAudioRangeProviderV1"));
#else
    auto entry=reinterpret_cast<GarnetMediaGetAudioRangeProviderFn>(dlsym(handle,"GarnetMediaGetAudioRangeProviderV1"));
#endif
    auto* api=entry?entry():nullptr;
    if(!api||api->abi!=1||api->size<sizeof(*api)||!api->decode_audio_range||!api->release_audio)
        throw std::runtime_error("FFmpeg provider audio range extension is unavailable");
    return *api;
}
namespace {
template<class API>
const API& BytesProvider(const char* provider, const char* symbol) {
    Provider(provider);
    std::lock_guard<std::mutex> lock(registryMutex);
    const auto handle = libraries.at(provider).handle;
    using Entry = const API* (*)();
#if defined(_WIN32)
    auto entry = reinterpret_cast<Entry>(GetProcAddress(static_cast<HMODULE>(handle), symbol));
#else
    auto entry = reinterpret_cast<Entry>(dlsym(handle, symbol));
#endif
    auto* api = entry ? entry() : nullptr;
    if (!api || api->abi != 1 || api->size < sizeof(*api))
        throw std::runtime_error(std::string(provider) + " binary media extension is unavailable");
    return *api;
}
}
const GarnetMediaAudioBytesProviderV1& AudioBytesProvider() {
    const auto& api = BytesProvider<GarnetMediaAudioBytesProviderV1>("ffmpeg", "GarnetMediaGetAudioBytesProviderV1");
    if (!api.decode_audio || !api.release_audio) throw std::runtime_error("binary audio decoder is unavailable");
    return api;
}
const GarnetMediaImageBytesProviderV1& ImageBytesProvider() {
    const auto& api = BytesProvider<GarnetMediaImageBytesProviderV1>("image", "GarnetMediaGetImageBytesProviderV1");
    if (!api.decode_image || !api.release_image) throw std::runtime_error("binary image decoder is unavailable");
    return api;
}
struct VideoWriter::Impl {
    const GarnetMediaProviderV1& provider;void* encoder=nullptr;Json options;
    std::string path,frameDirectory,providerInfo,error;uint32_t width,height,fps,capacity,snapshotWorkers;
    bool streamingAudio;
    mutable std::mutex mutex;std::condition_variable ready,space;std::thread worker;
    struct Item {std::vector<uint8_t> bytes;int64_t pts;};std::deque<Item> queue;
    uint64_t submitted=0,encoded=0,peak=0,audioSamples=0;bool stopping=false,finished=false;
    Impl(const Json& o):provider(Provider("ffmpeg")),options(o),path(o.at("path")),width(Positive(o,"width")),height(Positive(o,"height")),
        fps(Positive(o,"fps",24)),capacity(Positive(o,"queue_capacity",2)),snapshotWorkers(Positive(o,"snapshot_workers",1)),
        streamingAudio(o.value("streaming_audio",false)){
        Dimensions(width,height);if(width%2 || height%2 || fps>240 || capacity>16 || snapshotWorkers>16)throw std::runtime_error("invalid video geometry, fps, queue capacity or snapshot workers");
        if(!provider.open_video || !provider.submit_video || !provider.finish_video || !provider.close_video || !provider.video_info)throw std::runtime_error("incomplete video provider");
        const bool streaming=IsVideoStreamTarget(path);
        if(o.value("save_frames",false)){
            if(streaming)throw std::runtime_error("streamed video cannot save frame files");
            frameDirectory=path+".frames";fs::create_directories(fs::u8path(frameDirectory));Provider("image");
        }
        if(!streaming){fs::path output=fs::u8path(path);if(!output.parent_path().empty())fs::create_directories(output.parent_path());}
        char message[2048]{};encoder=provider.open_video(o.dump().c_str(),message,sizeof(message));
        if(!encoder)throw std::runtime_error(message);
        try{providerInfo=provider.video_info(encoder);worker=std::thread([this]{Run();});}catch(...){provider.close_video(encoder);encoder=nullptr;throw;}
    }
    ~Impl(){ {std::lock_guard<std::mutex> lock(mutex);stopping=true;}ready.notify_all();space.notify_all();if(worker.joinable())worker.join();if(encoder)provider.close_video(encoder); }
    void Run() noexcept {
        try{
            std::deque<std::future<void>> snapshots;
            while(true){Item item;
                {std::unique_lock<std::mutex> lock(mutex);ready.wait(lock,[&]{return stopping||!queue.empty();});
                    if(queue.empty())break;item=std::move(queue.front());queue.pop_front();}
                space.notify_one();auto frame=RGBFrame(item.bytes.data(),width,height,item.pts);char message[2048]{};
                std::string snapshot;
                if(!frameDirectory.empty()){
                    auto number=std::to_string(item.pts);number=std::string(6>number.size()?6-number.size():0,'0')+number;
                    snapshot=(fs::u8path(frameDirectory)/("frame-"+number+".png")).u8string();
                    if(snapshotWorkers==1)EncodeImage(frame,snapshot,Json{{"format","png"}});
                }
                if(provider.submit_video(encoder,&frame,message,sizeof(message)))throw std::runtime_error(message);
                if(!snapshot.empty()&&snapshotWorkers>1){
                    if(snapshots.size()>=snapshotWorkers){snapshots.front().get();snapshots.pop_front();}
                    // Encoding retains this frame only. Video submission stays
                    // ordered; Finish joins every bounded snapshot task.
                    snapshots.push_back(std::async(std::launch::async,[bytes=std::move(item.bytes),snapshot,width=width,height=height,pts=item.pts]{
                        auto image=RGBFrame(bytes.data(),width,height,pts);
                        EncodeImage(image,snapshot,Json{{"format","png"}});
                    }));
                }
                {std::lock_guard<std::mutex> lock(mutex);++encoded;}
                if(streamingAudio)space.notify_all();
            }
            for(auto& snapshot:snapshots)snapshot.get();
        }catch(const std::exception& e){std::lock_guard<std::mutex> lock(mutex);error=e.what();stopping=true;queue.clear();}
        catch(...){std::lock_guard<std::mutex> lock(mutex);error="unknown media worker error";stopping=true;queue.clear();}
        space.notify_all();
    }
};
VideoWriter::VideoWriter(const Json& o):impl(std::make_unique<Impl>(o)){}
VideoWriter::~VideoWriter()=default;
void VideoWriter::SubmitAudio(const GarnetMediaAudio& audio){
    auto& p=*impl;const auto& provider=AudioProvider();std::lock_guard<std::mutex> lock(p.mutex);
    if(p.stopping||p.submitted||p.audioSamples)throw std::runtime_error("submit the complete generated audio once, before video frames");
    char error[2048]{};if(provider.submit_audio(p.encoder,&audio,error,sizeof(error)))throw std::runtime_error(error);p.audioSamples=audio.samples;
}
void VideoWriter::SubmitAudioChunk(const GarnetMediaAudio& audio){
    auto& p=*impl;const auto& provider=StreamingAudioProvider();
    std::unique_lock<std::mutex> lock(p.mutex);
    if(!p.streamingAudio)throw std::runtime_error("streaming audio is not enabled");
    p.space.wait(lock,[&]{return p.stopping||p.encoded==p.submitted;});
    if(!p.error.empty())throw std::runtime_error(p.error);
    if(p.stopping)throw std::runtime_error("video writer is closed");
    char error[2048]{};
    if(provider.submit_audio_chunk(p.encoder,&audio,error,sizeof(error)))throw std::runtime_error(error);
    p.audioSamples+=audio.samples;
}
void VideoWriter::Submit(std::vector<uint8_t> data,uint32_t w,uint32_t h){
    auto& p=*impl;if(w!=p.width||h!=p.height||data.size()!=size_t(w)*h*3)throw std::runtime_error("video frame dimensions differ");
    std::unique_lock<std::mutex> lock(p.mutex);p.space.wait(lock,[&]{return p.stopping||p.queue.size()<p.capacity;});
    if(!p.error.empty())throw std::runtime_error(p.error);if(p.stopping)throw std::runtime_error("video writer is closed");
    p.queue.push_back({std::move(data),int64_t(p.submitted++)});p.peak=std::max(p.peak,uint64_t(p.queue.size()));lock.unlock();p.ready.notify_one();
}
Json VideoWriter::Status() const {auto& p=*impl;std::lock_guard<std::mutex> lock(p.mutex);
    return Json{{"video",p.path},{"frame_directory",p.frameDirectory},{"width",p.width},{"height",p.height},{"fps",p.fps},
        {"frames",p.submitted},{"frames_consumed",p.encoded},{"audio_samples",p.audioSamples},{"queue_peak_frames",p.peak},{"queue_capacity",p.capacity},
        {"snapshot_workers",p.snapshotWorkers},
        {"queue_max_pixel_bytes",uint64_t(p.capacity+2+(!p.frameDirectory.empty()&&p.snapshotWorkers>1?p.snapshotWorkers:0))*p.width*p.height*3},{"encoder",Json::parse(p.providerInfo)}};
}
Json VideoWriter::Finish(const GarnetMediaAudio* audio){auto& p=*impl;
    {std::lock_guard<std::mutex> lock(p.mutex);if(!p.submitted)throw std::runtime_error("cannot finish an empty video");
        if(audio&&(!p.options.value("audio_at_finish",false)||p.audioSamples))
            throw std::runtime_error("deferred audio requires an enabled, empty audio track");
        p.stopping=true;}
    p.ready.notify_all();p.space.notify_all();if(p.worker.joinable())p.worker.join();
    if(!p.error.empty())throw std::runtime_error(p.error);
    if(audio){
        char message[2048]{};
        if(AudioProvider().submit_audio(p.encoder,audio,message,sizeof(message)))throw std::runtime_error(message);
        std::lock_guard<std::mutex> lock(p.mutex);p.audioSamples=audio->samples;
    }
    if(!p.finished){char message[2048]{};if(p.provider.finish_video(p.encoder,message,sizeof(message)))throw std::runtime_error(message);p.finished=true;}
    return Status();
}
struct VideoAssembler::Impl {
    std::unique_ptr<VideoWriter> ownedWriter;
    VideoWriter* writer;
    fs::path folder;std::vector<fs::path> paths;
    bool memoryOnly=false;
    struct Record {
        uint64_t offset;
        uint32_t count;
        std::vector<float> pixels;
    };
    std::vector<std::vector<Record>> records;
    std::vector<float> coveragePlane;
    struct Tile{uint32_t x,y,w,h;uint32_t feather[4];};std::vector<Tile> tiles;
    uint32_t width,height,frames,skipFrames;size_t current=0;uint32_t next=0;uint64_t submittedChunks=0,peakAssembly=0,spoolBytes=0;
    bool done=false,ownsFolder=false,directFullCanvas=false;uint64_t encodedBeforeLastChunk=0,peakBlendWorkspace=0;
    uint64_t memorySpoolLimit=0,memorySpoolBytes=0,peakMemorySpoolBytes=0;
    Impl(const Json& o,VideoWriter* movie=nullptr,uint32_t skip=0):writer(movie),width(Positive(o,"width")),
        height(Positive(o,"height")),frames(Positive(o,"frames")),skipFrames(skip){
        try{
            Dimensions(width,height);
            if(skipFrames>=frames)throw std::runtime_error("movie window must own at least one frame");
            if(writer){
                const auto info=writer->Status();
                if(info.at("width")!=width||info.at("height")!=height||info.at("fps")!=Positive(o,"fps",24))
                    throw std::runtime_error("movie window geometry differs from the writer");
                memoryOnly=IsVideoStreamTarget(info.at("video").get<std::string>());
            }else{
                memoryOnly=IsVideoStreamTarget(o.at("path").get<std::string>());
            }
            if(const auto limit=o.find("tile_spool_memory_mib");limit!=o.end())
                memorySpoolLimit=uint64_t(Nonnegative(*limit))<<20;
            else if(memoryOnly)memorySpoolLimit=8192ULL<<20;
            if(!o.at("tiles").is_array() || o.at("tiles").size()>4096)throw std::runtime_error("invalid tile count");
            for(const auto& t:o.at("tiles")){Tile a{Nonnegative(t.at("x")),Nonnegative(t.at("y")),Positive(t,"width"),Positive(t,"height"),{}};
                Dimensions(a.w,a.h);if(a.x>width||a.w>width-a.x||a.y>height||a.h>height-a.y)throw std::runtime_error("tile exceeds canvas");
                if(!t.at("feather").is_array() || t.at("feather").size()!=4)throw std::runtime_error("four feather widths required");
                for(size_t i=0;i<4;++i){a.feather[i]=Nonnegative(t.at("feather").at(i));if(a.feather[i]>(i%2?a.w:a.h))throw std::runtime_error("invalid feather width");}tiles.push_back(a);}
            if(tiles.empty()||tiles.size()>4096)throw std::runtime_error("invalid tile count");records.resize(tiles.size());
            const auto& first=tiles.front();
            directFullCanvas=tiles.size()==1 && first.x==0 && first.y==0 && first.w==width && first.h==height &&
                std::all_of(first.feather,first.feather+4,[](uint32_t value){return value==0;});
            if(memoryOnly){
                auto remaining=memorySpoolLimit;
                for(size_t i=0;i+1<tiles.size();++i){
                    const auto bytes=uint64_t(tiles[i].w)*tiles[i].h*frames*3*sizeof(float);
                    if(bytes>remaining)throw std::runtime_error("video tile buffers exceed the configured memory limit");
                    remaining-=bytes;
                }
            }else{
                auto root=fs::u8path(o.at("spool_directory").get<std::string>());fs::create_directories(root);
                static std::atomic<uint64_t> serial{0};
                for(unsigned attempt=0;attempt<100;++attempt){folder=root/("media-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+"-"+std::to_string(serial++));if(fs::create_directory(folder)){ownsFolder=true;break;}}
                if(!ownsFolder)throw std::runtime_error("cannot create tile spool");
                for(size_t i=0;i+1<tiles.size();++i)paths.push_back(folder/("tile-"+std::to_string(i)+".f32"));
            }
            if(!writer){ownedWriter=std::make_unique<VideoWriter>(o);writer=ownedWriter.get();}
        }catch(...){Cleanup();throw;}
    }
    void Cleanup() noexcept {if(!ownsFolder)return;for(const auto& path:paths){std::error_code ec;fs::remove(path,ec);}std::error_code ec;if(fs::remove(folder,ec))ownsFolder=false;}
    ~Impl(){Cleanup();}
    void Spool(size_t tile,const float* data,uint32_t count,uint64_t bytes){
        for(size_t i=0;i<bytes/sizeof(float);++i)
            if(!std::isfinite(data[i]))throw std::runtime_error("nonfinite tile pixel");
        if(bytes<=memorySpoolLimit-memorySpoolBytes){
            records[tile].push_back({0,count,std::vector<float>(data,data+bytes/sizeof(float))});
            memorySpoolBytes+=bytes;
            peakMemorySpoolBytes=std::max(peakMemorySpoolBytes,memorySpoolBytes);
            return;
        }
        if(memoryOnly)throw std::runtime_error("video tile buffers exceed the configured memory limit");
        const uint64_t offset=fs::exists(paths[tile])?fs::file_size(paths[tile]):0;
        std::ofstream out(paths[tile],std::ios::binary|std::ios::app);
        out.write(reinterpret_cast<const char*>(data),bytes);
        if(!out)throw std::runtime_error("tile spool write failed");
        records[tile].push_back({offset,count,{}});
        spoolBytes+=bytes;
    }
    static std::vector<float> FeatherWeights(uint32_t length,uint32_t leading,uint32_t trailing){
        std::vector<float> weights(length,1);
        for(uint32_t i=0;i<leading;++i)weights[i]=float(i+1)/float(leading+1);
        for(uint32_t i=length-trailing;i<length;++i)weights[i]=std::min(weights[i],float(length-i)/float(trailing+1));
        return weights;
    }
    void Blend(const float* input,const Tile& t,uint32_t count,std::vector<float>& canvas,std::vector<float>& weights,bool validated=false){
        const size_t plane=size_t(width)*height,tilePlane=size_t(t.w)*t.h;
        std::vector<float> alpha(tilePlane);
        {
            const auto horizontal=FeatherWeights(t.w,t.feather[1],t.feather[3]);
            const auto vertical=FeatherWeights(t.h,t.feather[0],t.feather[2]);
            peakBlendWorkspace=std::max(peakBlendWorkspace,uint64_t(alpha.size()+horizontal.size()+vertical.size())*sizeof(float));
            for(uint32_t y=0;y<t.h;++y)for(uint32_t x=0;x<t.w;++x)
                alpha[size_t(y)*t.w+x]=horizontal[x]*vertical[y];
        }
        // Validate separately, then traverse contiguous rows so the compiler
        // can vectorize. Each pixel keeps the original tile/addition order.
        if(!validated)for(size_t i=0;i<size_t(3)*count*tilePlane;++i)
            if(!std::isfinite(input[i]))throw std::runtime_error("nonfinite tile pixel");
        if(!weights.empty())for(uint32_t y=0;y<t.h;++y){
            float* coverageRow=weights.data()+size_t(t.y+y)*width+t.x;
            const float* blend=alpha.data()+size_t(y)*t.w;
            for(uint32_t x=0;x<t.w;++x)coverageRow[x]+=blend[x];
        }
        for(size_t c=0;c<3;++c)for(uint32_t f=0;f<count;++f)for(uint32_t y=0;y<t.h;++y){
            float* output=canvas.data()+(c*count+f)*plane+size_t(t.y+y)*width+t.x;
            const float* source=input+(c*count+f)*tilePlane+size_t(y)*t.w;
            const float* blend=alpha.data()+size_t(y)*t.w;
            for(uint32_t x=0;x<t.w;++x)output[x]+=source[x]*blend[x];
        }
    }
    void BlendSpool(size_t tile,uint32_t count,std::vector<float>& canvas,std::vector<float>& coverage){
        const auto& geometry=tiles[tile];
        auto& record=records[tile].at(submittedChunks);
        if(record.count!=count)throw std::runtime_error("spatial tiles use inconsistent temporal chunks");
        if(!record.pixels.empty()){
            Blend(record.pixels.data(),geometry,count,canvas,coverage,true);
            memorySpoolBytes-=record.pixels.size()*sizeof(float);
            std::vector<float>().swap(record.pixels);
            return;
        }
        std::vector<float> pixels(size_t(3)*count*geometry.h*geometry.w);
        std::ifstream in(paths[tile],std::ios::binary);
        in.seekg(record.offset);
        in.read(reinterpret_cast<char*>(pixels.data()),pixels.size()*sizeof(float));
        if(!in)throw std::runtime_error("tile spool read failed");
        Blend(pixels.data(),geometry,count,canvas,coverage);
    }
    template<class Convert>
    void SubmitFrames(uint32_t first,uint32_t count,Convert convert){
        const uint32_t skip=first<skipFrames?std::min(count,skipFrames-first):0;
        count-=skip;
        std::vector<std::vector<uint8_t>> rgb(count);
        if(count>1&&std::thread::hardware_concurrency()>1){
            std::vector<std::future<std::vector<uint8_t>>> conversions;
            conversions.reserve(count);
            for(uint32_t f=0;f<count;++f)
                conversions.push_back(std::async(std::launch::async|std::launch::deferred,convert,f+skip));
            // Complete the chunk before submitting frames in order. Futures
            // join on failure before any worker can outlive the source pixels.
            for(uint32_t f=0;f<count;++f)rgb[f]=conversions[f].get();
        }else for(uint32_t f=0;f<count;++f)rgb[f]=convert(f+skip);
        for(auto& frame:rgb)writer->Submit(std::move(frame),width,height);
    }
};
VideoAssembler::VideoAssembler(const Json& o):impl(std::make_unique<Impl>(o)){}
VideoAssembler::VideoAssembler(const Json& o,VideoWriter& movie,uint32_t skip):impl(std::make_unique<Impl>(o,&movie,skip)){}
VideoAssembler::~VideoAssembler()=default;
void VideoAssembler::SubmitAudio(const GarnetMediaAudio& audio){impl->writer->SubmitAudio(audio);}
void VideoAssembler::Add(size_t tile,uint32_t first,const float* data,uint32_t count,uint32_t h,uint32_t w){
    auto& p=*impl;if(p.done||tile!=p.current||first!=p.next||!count||count>4||first>p.frames||count>p.frames-first||h!=p.tiles.at(tile).h||w!=p.tiles[tile].w)
        throw std::runtime_error("tile chunks must arrive in tile-major, consecutive frame order, with at most four frames");
    const uint64_t bytes=uint64_t(3)*count*h*w*4;
    if(tile+1<p.tiles.size()){
        p.Spool(tile,data,count,bytes);
    }else if(p.directFullCanvas){
        // A single unfeathered full canvas needs neither blending nor coverage
        // normalization. Keep the exact quantizer and at most four RGB frames.
        p.peakAssembly=std::max(p.peakAssembly,uint64_t(3)*count*p.width*p.height);
        p.SubmitFrames(first,count,[&](uint32_t f){return PlanarRGB(data,p.width,p.height,count,f);});
        if(first+count<p.frames)p.encodedBeforeLastChunk=p.writer->Status().at("frames_consumed").get<uint64_t>();
        ++p.submittedChunks;
    }else{
        const size_t plane=size_t(p.width)*p.height;
        std::vector<float> canvas(3*count*plane,0),coverage(p.coveragePlane.empty()?plane:0,0);
        p.peakAssembly=std::max(p.peakAssembly,uint64_t(canvas.size()+plane)*sizeof(float));
        for(size_t previous=0;previous<tile;++previous)p.BlendSpool(previous,count,canvas,coverage);
        p.Blend(data,p.tiles[tile],count,canvas,coverage);
        if(!coverage.empty()){
            for(float weight:coverage)if(!std::isfinite(weight)||weight<=0)throw std::runtime_error("uncovered video pixels");
            p.coveragePlane=std::move(coverage);
        }
        p.SubmitFrames(first,count,[&](uint32_t f){
            const float* weights=p.coveragePlane.data();
            std::vector<uint8_t> rgb(plane*3);
            for(size_t i=0;i<plane;++i)for(size_t c=0;c<3;++c)
                rgb[i*3+c]=Quantize(canvas[(c*count+f)*plane+i]/weights[i]);
            return rgb;
        });
        if(first+count<p.frames)p.encodedBeforeLastChunk=p.writer->Status().at("frames_consumed").get<uint64_t>();
        ++p.submittedChunks;
    }
    p.next+=count;if(p.next==p.frames){++p.current;p.next=0;}
}
Json VideoAssembler::Finish(const GarnetMediaAudio* audio){auto& p=*impl;if(p.current!=p.tiles.size())throw std::runtime_error("video tile assembly incomplete");
    if(audio&&!p.ownedWriter)throw std::runtime_error("movie window audio must be submitted through the movie writer");
    auto out=p.ownedWriter?p.writer->Finish(audio):p.writer->Status();p.done=true;p.Cleanup();out["assembly_peak_pixel_bytes"]=p.peakAssembly;out["spool_bytes"]=p.spoolBytes;
    if(p.memorySpoolLimit)out["memory_spool_peak_bytes"]=p.peakMemorySpoolBytes;
    out["blend_workspace_peak_bytes"]=p.peakBlendWorkspace;out["assembly_direct_full_canvas"]=p.directFullCanvas;
    out["frames_consumed_before_final_chunk"]=p.encodedBeforeLastChunk;out["assembly"]="bounded_chunk_tile_spool";return out;
}
}
