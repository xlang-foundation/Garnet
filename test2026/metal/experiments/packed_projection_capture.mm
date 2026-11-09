// SPDX-License-Identifier: Apache-2.0
// Native isolated packed projection capture. No model/serving timing claim.
#include "metal_context.h"
#include "affine_int4_codec.h"
#include "nlohmann/json.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <vector>
using namespace Garnet;using JSON=nlohmann::json;
void Need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
template<class T>void Save(const std::filesystem::path& p,const std::vector<T>& a){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(a.data()),a.size()*sizeof(T));Need(bool(f),"Cannot save native capture witness");}
int main(int argc,char** argv){try{@autoreleasepool{
 Need(argc==6,"NEW_CAPTURE.gputrace NEW_WITNESS_DIRECTORY M N K");std::filesystem::path trace=std::filesystem::absolute(argv[1]),root=std::filesystem::absolute(argv[2]);size_t m=std::stoull(argv[3]),n=std::stoull(argv[4]),k=std::stoull(argv[5]);Need(!std::filesystem::exists(trace),"Capture path must be fresh");Need(std::filesystem::create_directory(root),"Witness directory must be fresh");Need((m==128||m==256||m==512)&&((k==4096&&(n==4096||n==14336))||(k==14336&&n==4096)),"Unqualified native capture geometry");
 auto device=MTLCreateSystemDefaultDevice();Need(device!=nil,"No native Metal capture device");auto manager=[MTLCaptureManager sharedCaptureManager];Need([manager supportsDestination:MTLCaptureDestinationGPUTraceDocument],"GPU trace document destination unsupported; check documented MTL_CAPTURE_ENABLED=1 process flag");Need(!manager.isCapturing,"Refuse to replace an active capture");JSON report={{"scope","One actual native original packed projection, sharedF16 gather/vectorQ4/strictF32result, same qualified library. Complete control/guard/FP64 witnesses. Capture overhead and replay are not benchmark results."},{"rows",m},{"outputs",n},{"inner",k},{"trace",trace.string()}};
 {
 MetalContext gpu;uint32_t seed=37;auto next=[&](){seed=1664525u*seed+1013904223u;return seed;};std::vector<float> input((m+2)*k,42.5f);for(size_t i=0;i<m*k;++i)input[k+i]=float(int((next()>>8)%65521)-32760)*0x1p-18f;std::fill(input.end()-k,input.end(),-37.25f);
 auto layout=AffineInt4Layout::Create(n,k,AffineInt4CoefficientType::FP16);std::vector<uint32_t> packed(layout.weightBytes/4);for(auto& x:packed)x=next();std::vector<uint16_t> scales(layout.coefficientBytes/2),biases(scales.size());for(size_t i=0;i<scales.size();++i){scales[i]=Float32ToFloat16RNE(float(int(i%13)-6)*.00171f);biases[i]=Float32ToFloat16RNE(float(int(i%17)-8)*.00313f);}
 auto weight=gpu.UploadAffineInt4(packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes,n,k,true);auto inputOwner=gpu.Upload(input,m+2,k),a=gpu.PrepareDiagnosticDenseView(inputOwner,k,m,k);std::vector<float> poison((m+2)*n,std::numeric_limits<float>::quiet_NaN());std::fill_n(poison.begin(),n,42.5f);std::fill(poison.end()-n,poison.end(),-37.25f);auto outputOwner=gpu.Upload(poison,m+2,n),output=gpu.PrepareDiagnosticDenseView(outputOwner,n,m,n);
 Need(gpu.PrepareAffineInt4Profile(a,weight,output,m)&&gpu.PreparePackedAffineProfile(a,weight,output,m),"Cannot prepare original native projection");gpu.SetDiagnosticPackedFullTileEnabled(false);gpu.SetPackedAffineEnabled(false);gpu.Begin();gpu.EncodeAffineInt4MatMul(a,weight,output);gpu.SubmitAndWait();Need(!gpu.LastPackedAffineDispatches(),"Control unexpectedly packed");auto reference=gpu.Download(outputOwner);gpu.SetPackedAffineEnabled(true);
 auto execute=[&](){gpu.Update(outputOwner,poison);gpu.Begin();gpu.EncodeAffineInt4MatMul(a,weight,output);gpu.SubmitAndWait();Need(gpu.LastPackedAffineDispatches()==1&&!gpu.LastDiagnosticPackedFullTileDispatches()&&!gpu.LastAffineProfileMisses(),"Captured native route differs");};
 for(int i=0;i<3;++i){execute();auto warm=gpu.Download(outputOwner);Need(std::memcmp(warm.data(),reference.data(),reference.size()*4)==0,"Uncaptured original packed control differs");}
 const auto owners=MetalContext::ProfileOwnerStats();const size_t plans=gpu.PackedAffinePlanCount(),pipelines=gpu.AffineProfilePipelineCount();gpu.Update(outputOwner,poison);
 // MetalContext and this descriptor use the same documented default-device
 // factory. Capture is process-local and begins only after profile warmup.
 auto descriptor=[MTLCaptureDescriptor new];descriptor.captureObject=device;descriptor.destination=MTLCaptureDestinationGPUTraceDocument;descriptor.outputURL=[NSURL fileURLWithPath:[NSString stringWithUTF8String:trace.c_str()]];NSError* error=nil;
 if(![manager startCaptureWithDescriptor:descriptor error:&error])throw std::runtime_error(std::string("Cannot start native GPU capture: ")+error.localizedDescription.UTF8String);Need(manager.isCapturing,"Capture manager did not enter capture state");
 try{gpu.Begin();gpu.EncodeAffineInt4MatMul(a,weight,output);gpu.SubmitAndWait();}catch(...){[manager stopCapture];gpu.Abort();throw;}[manager stopCapture];Need(!manager.isCapturing,"Native GPU capture did not stop");
 Need(gpu.LastPackedAffineDispatches()==1&&!gpu.LastDiagnosticPackedFullTileDispatches()&&!gpu.LastAffineProfileMisses(),"Captured command selected wrong kernel policy");auto actual=gpu.Download(outputOwner);Need(actual.size()==reference.size()&&std::memcmp(actual.data(),reference.data(),actual.size()*4)==0,"GPU capture changed any control/guard word");auto unchanged=gpu.Download(inputOwner);Need(std::memcmp(input.data(),unchanged.data(),input.size()*4)==0,"Capture changed input/guards");for(size_t i=n;i<n+m*n;++i)Need(std::isfinite(actual[i]),"Unwritten/nonfinite captured output");for(size_t i=0;i<n;++i)Need(actual[i]==42.5f&&actual[actual.size()-n+i]==-37.25f,"Capture overwrote parent guards");
 AffineInt4View view(layout,packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes);std::vector<float> decoded(k);JSON samples=JSON::array();double maximum=0;
 for(size_t row:std::set<size_t>{0,m/2,m-1})for(size_t col:std::set<size_t>{0,1,15,16,31,32,n/2,n-1}){view.DecodeRow(col,decoded.data(),k);double sum=0,absolute=0;for(size_t j=0;j<k;++j){double product=double(Float16ToFloat32(Float32ToFloat16RNE(input[k+row*k+j])))*Float16ToFloat32(Float32ToFloat16RNE(decoded[j]));sum+=product;absolute+=std::abs(product);}double fraction=std::abs(double(actual[n+row*n+col])-sum)/(2e-6+4e-6*absolute);Need(fraction<=1,"Captured projection fails original FP64 bound");maximum=std::max(maximum,fraction);samples.push_back({{"row",row},{"column",col},{"sum",sum},{"sum_abs",absolute}});}
 auto after=MetalContext::ProfileOwnerStats();Need(after.matrices==owners.matrices&&after.affineWeights==owners.affineWeights&&after.pipelineSets==owners.pipelineSets&&gpu.PackedAffinePlanCount()==plans&&gpu.AffineProfilePipelineCount()==pipelines,"Capture changed native owners/plans/pipelines");
 Save(root/"input-parent.f32",input);Save(root/"weight.u32",packed);Save(root/"scales.f16",scales);Save(root/"biases.f16",biases);Save(root/"reference-parent.f32",reference);Save(root/"captured-parent.f32",actual);
 report["complete_output_and_guards_bit_identical"]=true;report["logical_output_words"]=m*n;report["input_and_guards_unchanged"]=true;report["original_packed_dispatches"]=gpu.LastPackedAffineDispatches();report["fulltile_dispatches"]=gpu.LastDiagnosticPackedFullTileDispatches();report["profile_misses"]=gpu.LastAffineProfileMisses();report["original_FP64_samples"]=samples;report["maximum_original_FP64_bound_fraction"]=maximum;report["native_owners_and_plans_stable"]=true;
 }
 auto closed=MetalContext::ProfileOwnerStats();Need(!closed.matrices&&!closed.affineWeights&&!closed.bf16Weights&&!closed.pipelineSets,"Native capture wrapper retained owners");Need(std::filesystem::exists(trace),"Native capture did not produce a trace document");size_t bytes=0,files=0;if(std::filesystem::is_regular_file(trace)){bytes=std::filesystem::file_size(trace);files=1;}else for(const auto& e:std::filesystem::recursive_directory_iterator(trace))if(e.is_regular_file()){bytes+=e.file_size();++files;}Need(bytes>4096,"Native capture document empty");report["trace_file_count"]=files;report["trace_document_bytes"]=bytes;report["all_native_owners_closed"]=true;report["post_capture_metal_allocated_snapshot"]=MetalContext::MemoryStats().allocated;report["passed"]=true;std::ofstream f(root/"result.json");f<<report.dump(2)<<"\n";Need(bool(f),"Cannot write capture result");std::cout<<"Native original packedGPUcapture control/guard/FP64 PASS, document bytes="<<bytes<<"\n";return 0;
}}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
