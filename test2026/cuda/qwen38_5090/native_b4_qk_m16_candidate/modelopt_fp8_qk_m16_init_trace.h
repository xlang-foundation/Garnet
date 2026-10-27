// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
namespace Garnet {
// Observation only: no SDK calls, allocations of device resources, or policy changes.
class ModelOptFP8QKM16InitTrace {
    struct Shared { std::mutex mutex; unsigned long long next=0; };
    static Shared& shared(){static Shared value;return value;}
    bool enabled_=false;
    unsigned long long id_=0,sequence_=0;
    int exceptions_=0;
    const char* outcome_="unreturned";
public:
    ModelOptFP8QKM16InitTrace() noexcept {
        const char* value=std::getenv("GARNET_QKM16_FACTORY_TRACE");
        enabled_=value&&std::strcmp(value,"1")==0;
        if(!enabled_)return;
        exceptions_=std::uncaught_exceptions();
        try{auto& state=shared();std::lock_guard<std::mutex> lock(state.mutex);id_=++state.next;}
        catch(...){enabled_=false;return;}
        note("begin","factory");
    }
    ~ModelOptFP8QKM16InitTrace() noexcept {
        note("end",std::uncaught_exceptions()>exceptions_?"exception":outcome_);
    }
    ModelOptFP8QKM16InitTrace(const ModelOptFP8QKM16InitTrace&)=delete;
    ModelOptFP8QKM16InitTrace& operator=(const ModelOptFP8QKM16InitTrace&)=delete;
    void note(const char* event,const char* name,long long a=0,long long b=0,
              long long c=0,const void* opaque=nullptr,size_t bytes=0) noexcept {
        if(!enabled_)return;
        try{
            auto& state=shared();std::lock_guard<std::mutex> lock(state.mutex);
            char hex[129]{};
            if(opaque&&bytes<=64){
                const auto* raw=static_cast<const unsigned char*>(opaque);
                constexpr char digits[]="0123456789abcdef";
                for(size_t i=0;i<bytes;++i){hex[2*i]=digits[raw[i]>>4];hex[2*i+1]=digits[raw[i]&15];}
            }
            std::fprintf(stderr,"GARNET_QKM16_FACTORY {\"id\":%llu,\"seq\":%llu,\"event\":\"%s\",\"name\":\"%s\",\"a\":%lld,\"b\":%lld,\"c\":%lld,\"opaque\":\"%s\"}\n",
                         id_,++sequence_,event,name,a,b,c,hex);
            std::fflush(stderr);
        }catch(...){}
    }
    template<class Call> auto api(const char* name,Call call)->decltype(call()) {
        // Exactly one forwarding call, even when tracing is disabled. SDK exceptions propagate.
        const auto result=call();note("api",name,static_cast<long long>(result));return result;
    }
    template<class Context> std::shared_ptr<Context> empty(const char* guard,
                                                        long long a=0,long long b=0,long long c=0) noexcept {
        outcome_="empty";note("reject",guard,a,b,c);return {};
    }
    void success(bool cached) noexcept {outcome_=cached?"cached":"created";note("accept",outcome_);}
};
}
