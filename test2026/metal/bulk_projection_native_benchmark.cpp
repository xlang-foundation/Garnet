// SPDX-License-Identifier: Apache-2.0
// Actual native chunk/bulk projection feasibility, no model/serving admission.
#include "../../src/backends/metal/metal_context.h"
#include "../../src/model/affine_int4_codec.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <vector>
using namespace Garnet;using JSON=nlohmann::json;
void Need(bool v,const char* m){if(!v)throw std::runtime_error(m);}
template<class F>void Reject(F f){bool no=false;try{f();}catch(const std::exception&){no=true;}Need(no,"Missing native guard");}
template<class T>void Save(const std::filesystem::path& p,const std::vector<T>& v){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));Need(bool(f),"Cannot save full payload");}
int main(int argc,char** argv){try{Need(argc==2,"Fresh output directory required");std::filesystem::path root=argv[1];Need(std::filesystem::create_directory(root),"Fresh output required");JSON report={{"scope","Actual native projection-only amortization screen. Chunk512 current packed path versus one dense bulk MPS product, all input packing/weight decoding in timing. No layer-major runtime, causal attention, TTFT or quality qualification."},{"cases",JSON::array()}};size_t index=0;
for(auto shape:std::vector<std::array<size_t,2>>{{14336,4096},{4096,14336},{4096,4096}})for(size_t chunks:{1,4,16}){
 const size_t n=shape[0],k=shape[1],m=512*chunks;auto dir=root/("case-"+std::to_string(index));Need(std::filesystem::create_directory(dir),"Fresh case required");
 MetalContext gpu;auto layout=AffineInt4Layout::Create(n,k,AffineInt4CoefficientType::FP16);std::vector<float> input(m*k);std::vector<uint32_t> words(layout.weightBytes/4);std::vector<uint16_t> scales(layout.coefficientBytes/2),biases(scales.size());uint32_t seed=37;
 for(auto& x:input){seed=1664525u*seed+1013904223u;x=float(int((seed>>8)%65521)-32760)*0x1p-18f;}
 for(auto& x:words){seed=1664525u*seed+1013904223u;x=seed;}for(size_t i=0;i<scales.size();++i){scales[i]=Float32ToFloat16RNE(float(int(i%13)-6)*.00171f);biases[i]=Float32ToFloat16RNE(float(int(i%17)-8)*.00313f);}
 auto weight=gpu.UploadAffineInt4(words.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes,n,k,true);std::vector<MetalMatrix> ins,outs;std::vector<float> poison(512*n,std::numeric_limits<float>::quiet_NaN());
 for(size_t c=0;c<chunks;++c){std::vector<float> part(input.begin()+c*512*k,input.begin()+(c+1)*512*k);ins.push_back(gpu.Upload(part,512,k));outs.push_back(gpu.Upload(poison,512,n));Need(gpu.PrepareAffineInt4Profile(ins.back(),weight,outs.back(),512)&&gpu.PreparePackedAffineProfile(ins.back(),weight,outs.back(),512),"Cannot prepare native control");}
 auto all=gpu.Upload(input,m,k),output=gpu.Upload(std::vector<float>(m*n,std::numeric_limits<float>::quiet_NaN()),m,n);Need(gpu.PrepareDiagnosticBulkAffine(all,weight,output),"Cannot prepare bulk plan");auto owners=MetalContext::ProfileOwnerStats();size_t scratchBytes=0;std::set<uintptr_t> seen;
 for(const auto& x:gpu.DiagnosticBulkScratch()){auto b=MetalContext::MatrixBufferInfo(x);Need(b.identity&&seen.insert(b.identity).second,"Missing/aliased bulk workspace");scratchBytes+=b.lengthBytes;}Need(scratchBytes==352321536,"Bulk workspace not bounded");
 const auto plans=gpu.PackedAffinePlanCount(),pipes=gpu.AffineProfilePipelineCount();std::array<std::vector<float>,2> first;std::array<std::vector<double>,2> times;JSON execution=JSON::array();size_t checked=0,exactComparisons=0;
 for(int rep=-3;rep<9;++rep)for(size_t order=0;order<2;++order){size_t mode=(order+size_t(rep+3))%2;
  if(mode)gpu.Update(output,std::vector<float>(m*n,std::numeric_limits<float>::quiet_NaN()));else for(auto& x:outs)gpu.Update(x,poison);
  gpu.Begin();if(mode)gpu.EncodeDiagnosticBulkAffine(all,weight,output);else for(size_t c=0;c<chunks;++c)gpu.EncodeAffineInt4MatMul(ins[c],weight,outs[c]);gpu.SubmitAndWait();
  Need(gpu.LastPackedAffineDispatches()==(mode?0:chunks)&&gpu.LastDiagnosticBulkDispatches()==mode,"Wrong actual native route");Need(gpu.LastAffineProfileMisses()==0&&gpu.LastAffineProfileDispatches()==(mode?0:chunks),"Native profile observations differ");
  std::vector<float> values;if(mode)values=gpu.Download(output);else{values.resize(m*n);for(size_t c=0;c<chunks;++c){auto p=gpu.Download(outs[c]);std::copy(p.begin(),p.end(),values.begin()+c*512*n);}}
  for(float v:values)Need(std::isfinite(v),"Missing/nonfinite writer after poison");if(first[mode].empty())first[mode]=values;else Need(std::memcmp(first[mode].data(),values.data(),values.size()*4)==0,"Unstable native output");checked+=values.size();if(rep>=0)times[mode].push_back(gpu.LastGPUMs());
  execution.push_back(JSON{{"rep",rep},{"mode",mode},{"packed",gpu.LastPackedAffineDispatches()},{"bulk",gpu.LastDiagnosticBulkDispatches()}});auto now=MetalContext::ProfileOwnerStats();Need(now.matrices==owners.matrices&&now.pipelineSets==owners.pipelineSets&&now.affineWeights==owners.affineWeights&&gpu.PackedAffinePlanCount()==plans&&gpu.AffineProfilePipelineCount()==pipes,"Request-time allocation/preparation");
 }
 double maxDiff=0;for(size_t i=0;i<first[0].size();++i){if(std::memcmp(first[0].data()+i,first[1].data()+i,4)==0)++exactComparisons;maxDiff=std::max(maxDiff,std::abs(double(first[0][i])-first[1][i]));}
 AffineInt4View view(layout,words.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes);std::vector<float> decoded(k);JSON oracle=JSON::array();double fraction=0.;
 for(size_t col:std::set<size_t>{0,1,15,16,31,32,63,64,n/2,n-1}){view.DecodeRow(col,decoded.data(),k);for(size_t row:std::set<size_t>{0,511,m/2,m-1}){double total=0,absolute=0;for(size_t j=0;j<k;++j){double p=double(Float16ToFloat32(Float32ToFloat16RNE(input[row*k+j])))*Float16ToFloat32(Float32ToFloat16RNE(decoded[j]));total+=p;absolute+=std::abs(p);}for(size_t mode=0;mode<2;++mode){double f=std::abs(double(first[mode][row*n+col])-total)/(2e-6+4e-6*absolute);Need(f<=1,"Original FP64 sample bound failed");fraction=std::max(fraction,f);}oracle.push_back(JSON{{"row",row},{"column",col},{"sum",total},{"sum_abs",absolute}});}}
 // Guard checks run outside collection and after complete outputs are saved.
 Save(dir/"input.f32",input);Save(dir/"weight.u32",words);Save(dir/"scales.f16",scales);Save(dir/"biases.f16",biases);Save(dir/"chunked.f32",first[0]);Save(dir/"bulk.f32",first[1]);
 gpu.Begin();Reject([&]{gpu.PrepareDiagnosticBulkAffine(all,weight,output);});gpu.Abort();auto unready=gpu.Allocate(m,k);gpu.Begin();Reject([&]{gpu.EncodeDiagnosticBulkAffine(unready,weight,output);});gpu.Abort();
 gpu.Begin();gpu.EncodeDiagnosticBulkAffine(all,weight,output);gpu.Abort();Need(gpu.LastDiagnosticBulkDispatches()==0,"Abort counter not reset");Reject([&]{gpu.Download(output);});gpu.Begin();gpu.EncodeDiagnosticBulkAffine(all,weight,output);gpu.SubmitAndWait();Need(gpu.LastDiagnosticBulkDispatches()==1,"Recovery route differs");
 JSON cell={{"index",index},{"rows",m},{"chunks",chunks},{"outputs",n},{"inner",k},{"gpu_ms",times},{"execution",execution},{"warmups",3},{"repetitions",9},{"scratch_bytes",scratchBytes},{"checked_runtime_words",checked},{"equal_output_words",exactComparisons},{"total_output_words",m*n},{"maximum_absolute_difference",maxDiff},{"maximum_FP64_sample_fraction",fraction},{"oracle",oracle}};report["cases"].push_back(cell);std::cout<<"bulk_projection "<<index++<<" PASS\n"<<std::flush;
}
auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Native owners retained");report["passed"]=true;Need(index==9,"Missing bulk profile cells");std::ofstream f(root/"results.json");f<<report.dump(2)<<'\n';Need(bool(f),"Cannot save raw report");return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
