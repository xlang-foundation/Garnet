// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
namespace Garnet {
inline bool ModelOptPluginInitTraceEnabled() noexcept {
    const char* value=std::getenv("GARNET_MODELOPT_INIT_TRACE");
    return value&&std::strcmp(value,"1")==0;
}
struct ModelOptPluginInitTraceState {std::mutex mutex;unsigned long long next=0;};
inline ModelOptPluginInitTraceState& ModelOptPluginInitTraceShared(){
    static ModelOptPluginInitTraceState state;return state;
}
inline unsigned long long ModelOptPluginInitTraceEvent(unsigned long long id,const char* phase,
    const char* route,const void* instance,int rows,int columns,int policy,bool context,
    bool factoryCalled,int result,bool exception) noexcept {
    try{
        auto& state=ModelOptPluginInitTraceShared();std::lock_guard<std::mutex> lock(state.mutex);
        if(!id)id=++state.next;
        std::fprintf(stderr,"GARNET_MODELOPT_INIT {\"id\":%llu,\"phase\":\"%s\",\"route\":\"%s\",\"instance\":\"%p\",\"rows\":%d,\"columns\":%d,\"policy\":%d,\"context_present\":%s,\"factory_called\":%s,\"result\":%d,\"exception\":%s}\n",
            id,phase,route,instance,rows,columns,policy,context?"true":"false",factoryCalled?"true":"false",result,exception?"true":"false");
        std::fflush(stderr);return id;
    }catch(...){return 0;}
}
template<class Context,class Factory>
int InitializeModelOptPluginContext(std::shared_ptr<Context>& context,Factory factory,
    const char* route,const void* instance,int rows,int columns,int policy) noexcept {
    // The disabled path retains the original factory/return/catch behavior.
    if(!ModelOptPluginInitTraceEnabled()){
        try{if(!context)context=factory();return context?0:1;}catch(...){return 1;}
    }
    const auto id=ModelOptPluginInitTraceEvent(0,"begin",route,instance,rows,columns,policy,bool(context),false,-1,false);
    bool called=false;
    try{
        if(!context){called=true;context=factory();}
        const int result=context?0:1;
        if(id)ModelOptPluginInitTraceEvent(id,"end",route,instance,rows,columns,policy,bool(context),called,result,false);
        return result;
    }catch(...){
        if(id)ModelOptPluginInitTraceEvent(id,"end",route,instance,rows,columns,policy,bool(context),called,1,true);
        return 1;
    }
}
}
