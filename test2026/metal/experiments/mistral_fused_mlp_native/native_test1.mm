// Exercise the actual runtime method, native parent guards and async ownership.
#include "metal_context.h"
#include "float16_codec.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace Garnet;namespace fs=std::filesystem;
void Need(bool v,const char* s){if(!v)throw std::runtime_error(s);}
template<class T>std::vector<T> Read(fs::path p,size_t count){std::ifstream f(p,std::ios::binary|std::ios::ate);Need(bool(f)&&size_t(f.tellg())==count*sizeof(T),"Native unit fixture size");f.seekg(0);std::vector<T> v(count);f.read((char*)v.data(),count*sizeof(T));Need(bool(f),"Read native unit fixture");return v;}
void Same(const std::vector<float>& a,const std::vector<float>& b){Need(a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*4),"Actual native fused output/guard changed");}
int main(int argc,char**argv){try{
 Need(argc==3,"ASSETS CAPTURES required");size_t cases=0,words=0,rejected=0,halfWords=0;
 auto reject=[&](auto action){bool failed=false;try{action();}catch(const std::exception&){failed=true;}Need(failed,"Invalid fused runtime operation accepted");++rejected;};
 for(size_t layer:{0ul,15ul,31ul}){
  MetalContext gpu;size_t k=4096,n=14336;auto assets=fs::path(argv[1])/("layer-"+std::to_string(layer)),capture=fs::path(argv[2])/("layer-"+std::to_string(layer));
  auto norm=Read<float>(capture/"norm-input.f32",512*k),gold=Read<float>(capture/"down.f32",512*k),activated=Read<float>(capture/"silu.f32",512*n);
  std::vector<float> input(514*k,-71.f),poison(514*k,-91.f),unused(514*n,17.f);std::copy(norm.begin(),norm.end(),input.begin()+k);
  auto root=gpu.Upload(input,514,k),outputRoot=gpu.Upload(poison,514,k),gateRoot=gpu.Upload(unused,514,n),upRoot=gpu.Upload(unused,514,n),siluRoot=gpu.Upload(unused,514,n);
  auto a=gpu.PrepareDiagnosticDenseView(root,k,512,k),out=gpu.PrepareDiagnosticDenseView(outputRoot,k,512,k),g=gpu.PrepareDiagnosticDenseView(gateRoot,n,512,n),u=gpu.PrepareDiagnosticDenseView(upRoot,n,512,n),s=gpu.PrepareDiagnosticDenseView(siluRoot,n,512,n);
  std::array<MetalAffineInt4Weight,3> weights;
  for(size_t stream=0;stream<3;stream++){std::string name=stream==0?"gate":stream==1?"up":"down";size_t inner=stream==2?n:k,columns=stream==2?k:n;auto w=Read<uint32_t>(assets/(name+"-weight.u32"),columns*inner/8);auto scale=Read<uint16_t>(assets/(name+"-scales.f16"),columns*inner/64),bias=Read<uint16_t>(assets/(name+"-biases.f16"),columns*inner/64);weights[stream]=gpu.UploadAffineInt4(w.data(),w.size()*4,scale.data(),scale.size()*2,bias.data(),bias.size()*2,columns,inner,true);}
  Need(gpu.PreparePackedAffineProfile(a,weights[0],g,512)&&gpu.PreparePackedAffineProfile(a,weights[1],u,512)&&gpu.PreparePackedAffineProfile(s,weights[2],out,512),"Cannot prepare actual native MLP profiles");
  auto arena=gpu.FusedMistralMlpScratch();Need(MetalContext::MatrixBufferInfo(arena).lengthBytes==14680064,"Wrong private half arena bytes");
  auto call=[&](const MetalMatrix& source,const MetalMatrix& destination,const std::vector<size_t>* rows){return gpu.TryEncodeFusedMistralMlp(source,weights[0],weights[1],weights[2],g,u,s,destination,rows);};
  std::vector<size_t> all(512);for(size_t row=0;row<512;row++)all[row]=row;
  reject([&]{call(a,out,nullptr);});
  for(size_t count:{127ul,128ul,129ul,255ul,256ul,257ul,383ul,384ul,511ul,512ul})for(bool alias:{false,true}){
   gpu.Update(root,input);gpu.Update(outputRoot,poison);std::vector<size_t> selected;
   // A noncontiguous, reversed selection covers arbitrary active owner order.
   for(size_t i=0;i<count;i++)selected.push_back((511-i*257)%512);
   gpu.Begin();bool took=call(a,alias?a:out,&selected);
   Need(took==(count>=128),"Native minimum-row routing changed");
   if(!took){gpu.Abort();Same(gpu.Download(root),input);Same(gpu.Download(outputRoot),poison);continue;}
   reject([&]{gpu.Download(alias?root:outputRoot);});gpu.SubmitAndWait();
   std::vector<float> expected=alias?input:poison;std::fill(expected.begin()+k,expected.begin()+513*k,0.f);
   for(auto row:selected)std::copy_n(gold.data()+row*k,k,expected.begin()+(row+1)*k);
   Same(gpu.Download(alias?root:outputRoot),expected);if(!alias)Same(gpu.Download(root),input);
   Same(gpu.Download(gateRoot),unused);Same(gpu.Download(upRoot),unused);Same(gpu.Download(siluRoot),unused);
   Need(gpu.LastFusedMistralMlpBlocks()==1&&gpu.LastPackedAffineDispatches()==3&&gpu.LastAffineProfileDispatches()==3,"Wrong native fused logical/physical route counters");
   auto carrier=gpu.Download(arena);auto half=reinterpret_cast<const uint16_t*>(carrier.data());
   for(size_t at=0;at<count;at++)for(size_t col=0;col<n;col++)Need(half[at*n+col]==Float32ToFloat16RNE(activated[selected[at]*n+col]),"Actual native compact half bits differ");
   for(size_t i=count*n;i<512*n;i++)Need(half[i]==0,"Native inactive half arena not initialized");
   ++cases;words+=expected.size();halfWords+=count*n;
  }
  gpu.Update(root,input);gpu.Update(outputRoot,poison);
  auto decline=[&](const std::vector<size_t>& selected){gpu.Begin();Need(!call(a,out,&selected),"Unsupported selection did not decline");gpu.Abort();Same(gpu.Download(outputRoot),poison);};decline({});
  auto bad=all;bad[1]=bad[0];gpu.Begin();reject([&]{call(a,out,&bad);});gpu.Abort();Same(gpu.Download(outputRoot),poison);
  bad=all;bad.back()=512;gpu.Begin();reject([&]{call(a,out,&bad);});gpu.Abort();Same(gpu.Download(outputRoot),poison);
  gpu.Begin();Need(call(a,out,&all),"Cannot encode Abort case");gpu.Abort();reject([&]{gpu.Download(outputRoot);});reject([&]{gpu.Download(arena);});
  gpu.Update(outputRoot,poison);gpu.Begin();Need(call(a,out,&all),"Cannot recover fused arena after Abort");gpu.Flush();reject([&]{gpu.Download(outputRoot);});gpu.Abort();reject([&]{gpu.Download(outputRoot);});
  gpu.Update(outputRoot,poison);gpu.Begin();Need(call(a,out,&all),"Cannot recover after partial submit");gpu.SubmitAndWait();auto expected=poison;std::copy(gold.begin(),gold.end(),expected.begin()+k);Same(gpu.Download(outputRoot),expected);
  // A different context may not write our private arena while its batch owns it.
  MetalContext other;auto x=other.Upload(std::vector<float>(512*7168,1.f),512,7168);
  other.Begin();other.EncodeAdd(x,x,arena);gpu.Begin();reject([&]{call(a,out,&all);});gpu.Abort();other.Abort();
  gpu.Update(arena,std::vector<float>(512*7168,0.f));auto privateView=gpu.PrepareDiagnosticDenseView(arena,0,512,k);
  gpu.Begin();reject([&]{call(privateView,out,&all);});gpu.Abort();
  MetalContext unprepared;unprepared.Begin();Need(!unprepared.TryEncodeFusedMistralMlp(a,weights[0],weights[1],weights[2],g,u,s,out,&all),"Unprepared native context lazily ran fusion");unprepared.Abort();
  gpu.SetPackedAffineEnabled(false);gpu.Begin();Need(!call(a,out,&all),"Packed-disabled control did not decline");gpu.Abort();
 }
 auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Native fused runtime retained owners");
 std::cout<<"{\"passed\":true,\"trained_layers\":3,\"cases\":"<<cases<<",\"complete_parent_word_comparisons\":"<<words<<",\"half_word_comparisons\":"<<halfWords<<",\"invalid_operations_rejected\":"<<rejected<<",\"input_output_aliases_tested\":true,\"partial_submit_abort_recovery\":true,\"all_native_owners_zero\":true}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
