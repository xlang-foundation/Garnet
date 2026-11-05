// SPDX-License-Identifier: Apache-2.0
// One real-weight MLP block, not full-model coding/serving qualification.
#include "../../src/backends/coreml/coreml_session.h"
#include "../../src/backends/metal/metal_context.h"
#include "../../src/model/mlx_qwen35_checkpoint.h"
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/resource.h>
using namespace Garnet;
using JSON=nlohmann::json;
namespace {
void Require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
JSON ReadJSON(const std::filesystem::path& path){std::ifstream stream(path);Require(bool(stream),"Cannot open MLP JSON");JSON value;stream>>value;return value;}
std::vector<float> ReadInput(const std::filesystem::path& path,size_t words){std::ifstream stream(path,std::ios::binary|std::ios::ate);Require(bool(stream)&&stream.tellg()==std::streamoff(words*4),"MLP input byte count differs");stream.seekg(0);std::vector<float> result(words);stream.read(reinterpret_cast<char*>(result.data()),std::streamsize(words*4));Require(bool(stream),"Cannot read MLP input");for(float value:result)Require(std::isfinite(value),"Nonfinite MLP input");return result;}
void Save(const std::filesystem::path& path,const std::vector<float>& values){std::ofstream stream(path,std::ios::binary);stream.write(reinterpret_cast<const char*>(values.data()),std::streamsize(values.size()*4));Require(bool(stream),"Cannot save MLP output");}
double Milliseconds(std::chrono::steady_clock::time_point start){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();}
float Half(float value){return Float16ToFloat32(Float32ToFloat16RNE(value));}
void Finite(const std::vector<float>& values){for(float value:values)Require(std::isfinite(value),"Nonfinite MLP output");}
JSON CheckDot(const AffineInt4View& weight,const std::vector<float>& input,const std::vector<float>& output,size_t m,size_t n,size_t k){
    std::vector<float> decoded(k);size_t checks=0;double maximum=0,ratio=0;
    for(size_t col:{size_t(0),n/3,n/2,n-1}){weight.DecodeRow(col,decoded.data(),k);
        for(size_t row:{size_t(0),m/2,m-1}){double expected=0,absolute=0;for(size_t inner=0;inner<k;++inner){float left=input[row*k+inner],right=decoded[inner];if(m>4){left=Half(left);right=Half(right);}double product=double(left)*right;expected+=product;absolute+=std::abs(product);}double error=std::abs(double(output[row*n+col])-expected),limit=2e-6+4e-6*absolute;Require(error<=limit,"Trained MLP projection differs from FP64 sample oracle");maximum=std::max(maximum,error);ratio=std::max(ratio,error/limit);++checks;}}
    return JSON{{"samples",checks},{"max_abs_error",maximum},{"max_error_fraction",ratio}};
}
JSON Metal(const std::filesystem::path& checkpoint,const std::vector<float>& input,size_t m,std::vector<float>& output){
    SafeTensorsIndex index;std::string error;if(!index.Open(checkpoint,error))throw std::runtime_error(error);MlxQwen35Checkpoint adapter;adapter.Open(index);Require(adapter.DenseMistral(),"Mistral checkpoint required");
    MetalContext gpu;Require(gpu.HasMPPAffine(),"FP16 Metal tensor pipeline required");
    std::array<AffineInt4Metadata,3> metadata;std::array<AffineInt4Mapping,3> mapping;std::array<MetalAffineInt4Weight,3> weights;
    const std::array<std::string,3> names{"gate_proj","up_proj","down_proj"};
    for(size_t i=0;i<3;++i){metadata[i]=AffineInt4Metadata::Resolve(index,adapter,"model.layers.0.mlp."+names[i]+".weight");mapping[i].Open(metadata[i]);const auto& l=metadata[i].layout;Require(l.coefficientType==AffineInt4CoefficientType::FP16,"FP16 model coefficients required");weights[i]=gpu.UploadAffineInt4(mapping[i].Data(0),l.weightBytes,mapping[i].Data(1),l.coefficientBytes,mapping[i].Data(2),l.coefficientBytes,l.rows,l.columns,true);}
    auto a=gpu.Upload(input,m,4096),gate=gpu.Allocate(m,14336),up=gpu.Allocate(m,14336),active=gpu.Allocate(m,14336),out=gpu.Allocate(m,4096);
    Require(gpu.PrepareAffineInt4Profile(a,weights[0],gate,m)&&gpu.PrepareAffineInt4Profile(a,weights[1],up,m)&&gpu.PrepareAffineInt4Profile(active,weights[2],out,m),"Cannot prepare trained MLP profile");const auto count=gpu.AffineProfilePipelineCount();
    std::vector<double> host,gpuMs,commands;
    for(int repeat=-3;repeat<9;++repeat){auto began=std::chrono::steady_clock::now();gpu.Update(a,input);auto command=std::chrono::steady_clock::now();gpu.Begin();gpu.EncodeAffineInt4MatMul(a,weights[0],gate);gpu.EncodeAffineInt4MatMul(a,weights[1],up);gpu.EncodeSiluMul(gate,up,active);gpu.EncodeAffineInt4MatMul(active,weights[2],out);gpu.SubmitAndWait();double elapsed=Milliseconds(command);output=gpu.Download(out);double total=Milliseconds(began);
        Require(gpu.AffineProfilePipelineCount()==count&&gpu.LastAffineProfileDispatches()==3&&gpu.LastAffineProfileMisses()==0,"MLP profile compiled/missed at runtime");Require(gpu.LastMPPAffineDispatches()==size_t(m>4?3:0),"MLP tensor dispatch count mismatch");
        if(repeat>=0){host.push_back(total);gpuMs.push_back(gpu.LastGPUMs());commands.push_back(elapsed);}}
    auto gates=gpu.Download(gate),ups=gpu.Download(up),acts=gpu.Download(active);Finite(gates);Finite(ups);Finite(acts);Finite(output);
    double maxActivation=0;for(size_t i=0;i<acts.size();++i){float expected=(gates[i]/(1.f+std::exp(-gates[i])))*ups[i];double difference=std::abs(double(acts[i])-expected);Require(difference<=2e-6+4e-6*std::abs(double(expected)),"MLP activation differs from independent float32 formula");maxActivation=std::max(maxActivation,difference);}
    JSON oracle={{"gate",CheckDot(mapping[0].View(),input,gates,m,14336,4096)},{"up",CheckDot(mapping[1].View(),input,ups,m,14336,4096)},{"down",CheckDot(mapping[2].View(),acts,output,m,4096,14336)},{"activation_words",acts.size()},{"activation_max_abs_error",maxActivation}};
    return JSON{{"engine","metal"},{"device",gpu.DeviceName()},{"operand_dtype",m>4?"float16":"float32"},{"intermediate_dtype","float32"},{"accumulator_dtype","float32"},{"output_dtype","float32"},{"host_to_host_ms",host},{"gpu_ms",gpuMs},{"command_encode_wait_ms",commands},{"oracle",oracle},{"profile_misses",0},{"runtime_pipeline_compiles",0},{"timing_scope","Update shared input + encode/submit/wait + host output download; excludes loading, compilation and post-run oracle"}};
}
JSON CoreML(const std::filesystem::path& root,const std::vector<float>& input,size_t m,std::vector<float>& output,const std::string& policy){
    auto manifest=ReadJSON(root/"manifest.json");auto arithmetic=manifest.value("arithmetic",std::string("fp16"));
    Require(arithmetic=="fp16"||arithmetic=="fp32_prefill_control","Unknown Core ML MLP arithmetic policy");
    auto profile=ReadJSON(root/"profile.json");profile["compute_units"]=policy;CoreMLSession session;auto opened=std::chrono::steady_clock::now();session.Open((root/"mistral_mlp.mlmodelc").string(),profile.dump());double openMs=Milliseconds(opened);
    CoreMLTensor x{"x","float16",{1,4096,1,int64_t(m)},std::vector<uint8_t>(m*4096*2)};std::vector<double> host,predict,inner;output.resize(m*4096);
    for(int repeat=-3;repeat<9;++repeat){auto began=std::chrono::steady_clock::now();
        for(size_t row=0;row<m;++row)for(size_t h=0;h<4096;++h){uint16_t value=Float32ToFloat16RNE(input[row*4096+h]);std::memcpy(x.bytes.data()+(h*m+row)*2,&value,2);}
        auto predicted=std::chrono::steady_clock::now();auto results=session.Predict({x});double nativeMs=Milliseconds(predicted);Require(results.size()==1&&results[0].name=="y"&&results[0].dtype=="float16"&&results[0].shape==x.shape,"Core ML MLP output shape/type differs");
        for(size_t row=0;row<m;++row)for(size_t h=0;h<4096;++h){uint16_t value;std::memcpy(&value,results[0].bytes.data()+(h*m+row)*2,2);output[row*4096+h]=Float16ToFloat32(value);}double total=Milliseconds(began);
        if(repeat>=0){host.push_back(total);predict.push_back(nativeMs);inner.push_back(JSON::parse(session.ReportJSON()).at("last_prediction_and_output_copy_ms").get<double>());}}
    Finite(output);auto report=JSON::parse(session.ReportJSON());session.Close();
    return JSON{{"engine","coreml"},{"policy",policy},{"arithmetic",arithmetic},{"operand_dtype","float16_transport"},{"intermediate_dtype",arithmetic=="fp16"?"float16":"float32_with_fp16_down_operand"},{"accumulator_dtype","framework_selected"},{"output_dtype","float16_expanded_to_float32"},{"host_to_host_ms",host},{"native_predict_ms",predict},{"framework_prediction_and_output_copy_ms",inner},{"open_and_compute_plan_ms",openMs},{"native_report",report},{"timing_scope","FP32 row-major to FP16 channel-first conversion + native prediction and I/O + FP16 channel-first to FP32 row-major conversion; excludes model open/plan"}};
}
}
int main(int argc,char** argv){try{
    Require(argc==6,"Usage: probe metal|coreml_cpu|coreml_cpu_ane CHECKPOINT EXPORT_ROOT NEW_OUTPUT_ROOT 0");
    // The final argument is the explicit model layer, currently only0.
    Require(std::string(argv[5])=="0","Only actual Mistral layer0 is admitted");std::string mode=argv[1];Require(mode=="metal"||mode=="coreml_cpu"||mode=="coreml_cpu_ane","Unknown MLP engine");std::filesystem::path root=argv[3],out=argv[4];Require(std::filesystem::create_directory(out),"New MLP output directory required");
    auto manifest=ReadJSON(root/"manifest.json");size_t m=manifest.at("rows").get<size_t>();Require((m==1||m==128||m==512)&&manifest.at("layer")==0&&manifest.at("hidden")==4096&&manifest.at("intermediate")==14336,"Unexpected Mistral MLP profile");auto input=ReadInput(root/"input.f32",m*4096);std::vector<float> output;JSON report;
    if(mode=="metal")report=Metal(argv[2],input,m,output);else report=CoreML(root,input,m,output,mode=="coreml_cpu"?"cpu":"cpu_ane");
    Save(out/"output.f32",output);rusage usage{};Require(getrusage(RUSAGE_SELF,&usage)==0,"Cannot sample probe RSS");report["maximum_process_rss_bytes"]=usage.ru_maxrss;report["memory_scope"]="Process maximum RSS only, not unique GPU allocation, compressed memory or total CoreML service footprint";report["rows"]=m;report["warmups"]=3;report["repetitions"]=9;report["output_words"]=output.size();report["scope"]="One actual-weight Mistral MLP block with embedding-driven input. Different intermediate precision; no whole-model coding/TTFT/throughput or observed ANE placement claim.";
    auto owners=MetalContext::ProfileOwnerStats();Require(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"MLP probe leaked Metal owners");report["metal_native_owners_after_close"]=0;std::ofstream stream(out/"report.json");stream<<report.dump(2)<<'\n';Require(bool(stream),"Cannot save MLP report");std::cout<<report.dump()<<'\n';return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
