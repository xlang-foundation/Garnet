#include "metal_context.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
using namespace Garnet;namespace fs=std::filesystem;
void need(bool b,const char* why){if(!b)throw std::runtime_error(why);}
int main(){auto root=fs::temp_directory_path()/("garnet-capture-exception-"+std::to_string(getpid()));try{
 need(!fs::exists(root),"Fresh test root required");fs::create_directory(root);
 for(int mode=0;mode<2;++mode){
  {MetalContext gpu;gpu.DiagnosticAttentionCaptureArm((root/std::to_string(mode)).string());
   if(mode){std::vector<MetalMatrix> args;const size_t rows[]={512,512,512,4,4,4,4,512},cols[]={4096,1024,1024,4194304,4194304,128,128,4096};
    for(size_t i=0;i<8;++i){std::vector<float> data(rows[i]*cols[i],0);if(i==5)for(size_t b=0;b<4;++b)for(size_t t=0;t<128;++t)data[b*128+t]=1920+t;if(i==6)std::fill(data.begin(),data.end(),1);args.push_back(gpu.Upload(data,rows[i],cols[i]));}
    auto output=gpu.Allocate(512,4096);gpu.Begin();bool failed=false;try{gpu.EncodeAttentionChunk(args,output,32,8,128,4096,128,false);}catch(const std::logic_error& e){failed=true;std::cout<<"Expected original failure: "<<e.what()<<'\n';}need(failed,"Unprepared MPS route unexpectedly accepted");gpu.Abort();
    bool finishRejected=false;try{gpu.DiagnosticAttentionCaptureFinish();}catch(const std::logic_error&){finishRejected=true;}need(finishRejected,"Failed capture exported");
   }
   // Deliberately do not cancel: destructor must release an armed/failed plan.
  }
  auto o=MetalContext::ProfileOwnerStats();auto bytes=MetalContext::MemoryStats().allocated;std::cout<<"Closed mode "<<mode<<" allocated "<<bytes<<'\n';need(!o.matrices&&!o.affineWeights&&!o.bf16Weights&&!o.pipelineSets,"Native owners retained");need(bytes<=4194304,"Armed/exceptional capture retained device buffers");
 }
 fs::remove_all(root);std::cout<<"Armed and exception-path destructor cleanup PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"; retained root "<<root<<'\n';return 1;}}
