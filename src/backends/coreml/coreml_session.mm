// SPDX-License-Identifier: Apache-2.0
#include "coreml_session.h"
#include "nlohmann/json.hpp"
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
namespace Garnet {
namespace {
using JSON=nlohmann::json;
void Require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
NSString* Name(const std::string& value){auto s=[NSString stringWithUTF8String:value.c_str()];Require(s!=nil,"Invalid Core ML UTF8 name/path");return s;}
size_t Multiply(size_t a,size_t b){Require(b==0||a<=SIZE_MAX/b,"Core ML shape/stride overflow");return a*b;}
size_t Width(const std::string& dtype){if(dtype=="float16")return 2;if(dtype=="float32"||dtype=="int32")return 4;throw std::runtime_error("Unsupported Core ML dtype");}
MLMultiArrayDataType Type(const std::string& dtype){Width(dtype);return dtype=="float16"?MLMultiArrayDataTypeFloat16:dtype=="float32"?MLMultiArrayDataTypeFloat32:MLMultiArrayDataTypeInt32;}
std::string TypeName(MLMultiArrayDataType dtype){if(dtype==MLMultiArrayDataTypeFloat16)return "float16";if(dtype==MLMultiArrayDataTypeFloat32)return "float32";if(dtype==MLMultiArrayDataTypeInt32)return "int32";throw std::runtime_error("Unsupported Core ML model dtype");}
size_t Elements(const std::vector<int64_t>& shape){Require(!shape.empty()&&shape.size()<=8,"Invalid Core ML tensor rank");size_t count=1;for(auto d:shape){Require(d>0,"Invalid Core ML dimension");count=Multiply(count,size_t(d));}return count;}
std::vector<int64_t> Shape(NSArray<NSNumber*>* shape){std::vector<int64_t> result;for(NSNumber* d in shape)result.push_back(d.longLongValue);Elements(result);return result;}
std::string Failure(NSError* error){return error?std::string(error.localizedDescription.UTF8String):"Unknown Core ML failure";}
using Specs=std::map<std::string,CoreMLTensor>;
Specs Bounds(const JSON& object,size_t& total){
    Require(object.is_object()&&!object.empty(),"Core ML I/O profile must name every feature");Specs specs;
    for(auto i=object.begin();i!=object.end();++i){CoreMLTensor t;t.name=i.key();t.dtype=i.value().at("dtype").get<std::string>();t.shape=i.value().at("maximum_shape").get<std::vector<int64_t>>();auto bytes=Multiply(Elements(t.shape),Width(t.dtype));Require(total<=SIZE_MAX-bytes,"Core ML I/O byte overflow");total+=bytes;specs.emplace(t.name,std::move(t));}return specs;
}
void Check(const CoreMLTensor& tensor,const CoreMLTensor& bound,bool exact=false,bool payload=true){
    Require(tensor.name==bound.name&&tensor.dtype==bound.dtype&&tensor.shape.size()==bound.shape.size(),"Core ML tensor identity/type/rank differs from profile");
    for(size_t i=0;i<tensor.shape.size();++i)Require(tensor.shape[i]>0&&(exact?tensor.shape[i]==bound.shape[i]:tensor.shape[i]<=bound.shape[i]),"Core ML actual shape exceeds compiled profile");
    if(payload)Require(tensor.bytes.size()==Multiply(Elements(tensor.shape),Width(tensor.dtype)),"Core ML tensor payload size mismatch");
}
// Core ML may expose padded/non-contiguous arrays. Never assume linear layout.
void Copy(MLMultiArray* array,CoreMLTensor& tensor,bool toArray){
    Require(array.dataType==Type(tensor.dtype)&&Shape(array.shape)==tensor.shape,"Core ML array differs from tensor");
    const size_t count=Elements(tensor.shape),width=Width(tensor.dtype);Require(array.strides.count==tensor.shape.size(),"Invalid Core ML strides");
    std::vector<size_t> strides;size_t maximumOffset=0;
    for(size_t i=0;i<tensor.shape.size();++i){auto stride=array.strides[i].longLongValue;Require(stride>0,"Invalid Core ML stride");strides.push_back(size_t(stride));auto tail=Multiply(size_t(tensor.shape[i]-1),size_t(stride));Require(maximumOffset<=SIZE_MAX-tail,"Core ML stride sum overflow");maximumOffset+=tail;}
    Require(maximumOffset<SIZE_MAX/width&&array.dataPointer!=nullptr,"Invalid Core ML array storage");
    auto raw=static_cast<uint8_t*>(array.dataPointer);
    bool contiguous=true;size_t expectedStride=1;
    for(size_t axis=tensor.shape.size();axis>0;--axis){
        if(tensor.shape[axis-1]>1&&strides[axis-1]!=expectedStride)contiguous=false;
        expectedStride=Multiply(expectedStride,size_t(tensor.shape[axis-1]));
    }
    if(contiguous){
        const size_t bytes=Multiply(count,width);
        if(toArray)std::memcpy(raw,tensor.bytes.data(),bytes);
        else std::memcpy(tensor.bytes.data(),raw,bytes);
        return;
    }
    for(size_t index=0;index<count;++index){size_t linear=index,offset=0;for(size_t axis=tensor.shape.size();axis>0;--axis){offset+=(linear%size_t(tensor.shape[axis-1]))*strides[axis-1];linear/=size_t(tensor.shape[axis-1]);}if(toArray)std::memcpy(raw+offset*width,tensor.bytes.data()+index*width,width);else std::memcpy(tensor.bytes.data()+index*width,raw+offset*width,width);}
}
void Descriptions(NSDictionary<NSString*,MLFeatureDescription*>* descriptions,const Specs& specs){
    Require(descriptions.count==specs.size(),"Core ML feature inventory differs from profile");
    for(const auto& item:specs){MLFeatureDescription* d=descriptions[Name(item.first)];Require(d!=nil&&d.type==MLFeatureTypeMultiArray&&!d.optional&&d.multiArrayConstraint.dataType==Type(item.second.dtype),"Core ML feature dtype/name/optionality differs from profile");}
}
std::string DeviceName(id<MLComputeDeviceProtocol> device){return [device isKindOfClass:MLNeuralEngineComputeDevice.class]?"neural_engine":[device isKindOfClass:MLGPUComputeDevice.class]?"gpu":[device isKindOfClass:MLCPUComputeDevice.class]?"cpu":"unknown";}
void Placements(MLModelStructureProgramBlock* block,MLComputePlan* plan,JSON& counts,JSON& operations){
    for(MLModelStructureProgramOperation* op in block.operations){auto usage=[plan computeDeviceUsageForMLProgramOperation:op];auto name=DeviceName(usage.preferredComputeDevice);counts[name]=counts.value(name,size_t(0))+1;
        JSON supported=JSON::array(),outputs=JSON::array();for(id<MLComputeDeviceProtocol> device in usage.supportedComputeDevices)supported.push_back(DeviceName(device));for(MLModelStructureProgramNamedValueType* value in op.outputs)outputs.push_back(std::string(value.name.UTF8String));
        operations.push_back(JSON{{"operator",std::string(op.operatorName.UTF8String)},{"outputs",outputs},{"preferred",name},{"supported",supported}});
        for(MLModelStructureProgramBlock* nested in op.blocks)Placements(nested,plan,counts,operations);}
}
}
struct CoreMLSession::Impl {
    mutable std::mutex mutex;
    MLModel* model=nil;
    MLState* state=nil;
    Specs inputs,outputs,states;
    JSON report=JSON::object();
    bool poisoned=false;
    void Ready(){Require(model!=nil&&!poisoned,"Core ML session closed or needs reset after failed prediction");}
};
CoreMLSession::CoreMLSession():impl_(std::make_unique<Impl>()){}
CoreMLSession::~CoreMLSession()=default;
void CoreMLSession::Open(const std::string& path,const std::string& profileJSON){
    std::lock_guard<std::mutex> lock(impl_->mutex);@autoreleasepool {
        Require(impl_->model==nil,"Core ML session already open");
        if(@available(macOS 15.0,*)){}else throw std::runtime_error("Stateful Core ML requires macOS15+");
        auto profile=JSON::parse(profileJSON);auto units=profile.at("compute_units").get<std::string>();Require(units=="cpu"||units=="cpu_ane","Explicit Core ML cpu/cpu_ane compute policy required");
        size_t io=0;auto inputs=Bounds(profile.at("inputs"),io),outputs=Bounds(profile.at("outputs"),io);
        Require(profile.at("maximum_io_bytes").is_number_unsigned()&&io<=profile.at("maximum_io_bytes").get<size_t>(),"Core ML maximum I/O byte budget exceeded");
        Require(profile.at("maximum_state_bytes").is_number_unsigned(),"Core ML state byte budget required");
        std::filesystem::path disk(path);Require(disk.extension()==".mlmodelc"&&std::filesystem::is_directory(disk),"Precompiled Core ML .mlmodelc directory required");
        const auto function=profile.value("function",std::string("main"));
        auto config=[MLModelConfiguration new];config.computeUnits=units=="cpu"?MLComputeUnitsCPUOnly:MLComputeUnitsCPUAndNeuralEngine;config.functionName=Name(function);
        NSURL* url=[NSURL fileURLWithPath:Name(std::filesystem::canonical(disk).string()) isDirectory:YES];NSError* error=nil;
        MLModel* model=[MLModel modelWithContentsOfURL:url configuration:config error:&error];if(!model)throw std::runtime_error(Failure(error));
        Descriptions(model.modelDescription.inputDescriptionsByName,inputs);Descriptions(model.modelDescription.outputDescriptionsByName,outputs);
        Specs states;size_t stateBytes=0;
        for(NSString* name in model.modelDescription.stateDescriptionsByName){auto constraint=model.modelDescription.stateDescriptionsByName[name].stateConstraint;Require(constraint!=nil,"Unsupported Core ML state description");CoreMLTensor t;t.name=name.UTF8String;t.dtype=TypeName(constraint.dataType);t.shape=Shape(constraint.bufferShape);size_t bytes=Multiply(Elements(t.shape),Width(t.dtype));Require(stateBytes<=SIZE_MAX-bytes,"Core ML state byte overflow");stateBytes+=bytes;states.emplace(t.name,std::move(t));}
        Require(stateBytes<=profile.at("maximum_state_bytes").get<size_t>(),"Core ML persistent state budget exceeded");
        auto done=dispatch_semaphore_create(0);__block MLComputePlan* plan=nil;__block NSError* planError=nil;
        [MLComputePlan loadContentsOfURL:url configuration:config completionHandler:^(MLComputePlan* loaded,NSError* failure){plan=loaded;planError=failure;dispatch_semaphore_signal(done);}];
        Require(dispatch_semaphore_wait(done,dispatch_time(DISPATCH_TIME_NOW,30*NSEC_PER_SEC))==0,"Core ML compute-plan inspection timeout");
        if(!plan)throw std::runtime_error(Failure(planError));Require(plan.modelStructure.program!=nil,"Core ML plugin currently requires an ML Program");
        auto selected=plan.modelStructure.program.functions[Name(function)];Require(selected!=nil,"Core ML compiled function missing");
        JSON placement=JSON::object(),operations=JSON::array();Placements(selected.block,plan,placement,operations);
        Require(!profile.value("require_ane_plan",false)||placement.value("neural_engine",size_t(0))>0,"Core ML plan contains no preferred ANE operation");
        impl_->model=model;impl_->state=[model newState];impl_->inputs=std::move(inputs);impl_->outputs=std::move(outputs);impl_->states=std::move(states);impl_->poisoned=false;
        JSON stateSpecs=JSON::object();for(const auto& item:impl_->states)stateSpecs[item.first]=JSON{{"dtype",item.second.dtype},{"maximum_shape",item.second.shape}};
        impl_->report=JSON{{"backend","coreml"},{"compute_units",units},{"function",function},{"path",disk.string()},{"declared_io_maximum_bytes",io},{"profile_io_budget_bytes",profile.at("maximum_io_bytes")},{"declared_state_bytes",stateBytes},{"state_tensors",stateSpecs},{"anticipated_operation_devices",placement},{"anticipated_operations",operations},{"placement_scope","MLComputePlan anticipated/preferred devices, not observed hardware execution"},{"memory_scope","Declared tensor bounds only; excludes model weights/compiler/driver allocations"},{"predictions",0},{"stateful",!impl_->states.empty()}};
    }
}
std::vector<CoreMLTensor> CoreMLSession::Predict(const std::vector<CoreMLTensor>& inputs){
    std::lock_guard<std::mutex> lock(impl_->mutex);@autoreleasepool {
        impl_->Ready();Require(inputs.size()==impl_->inputs.size(),"Core ML input count mismatch");std::set<std::string> names;
        // Validate the complete call before touching any state.
        for(const auto& t:inputs){Require(names.insert(t.name).second&&impl_->inputs.count(t.name),"Duplicate/unknown Core ML input");Check(t,impl_->inputs.at(t.name));}
        auto features=[NSMutableDictionary<NSString*,MLFeatureValue*> dictionary];NSError* error=nil;
        for(const auto& original:inputs){auto t=original;auto shape=[NSMutableArray<NSNumber*> array];for(auto d:t.shape)[shape addObject:@(d)];auto array=[[MLMultiArray alloc] initWithShape:shape dataType:Type(t.dtype) error:&error];if(!array)throw std::runtime_error(Failure(error));Copy(array,t,true);features[Name(t.name)]=[MLFeatureValue featureValueWithMultiArray:array];}
        auto provider=[[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:&error];if(!provider)throw std::runtime_error(Failure(error));
        const auto began=std::chrono::steady_clock::now();impl_->poisoned=true;
        auto result=[impl_->model predictionFromFeatures:provider usingState:impl_->state error:&error];if(!result)throw std::runtime_error(Failure(error));
        std::vector<CoreMLTensor> outputs;
        Require(result.featureNames.count==impl_->outputs.size(),"Core ML output inventory changed");
        for(const auto& item:impl_->outputs){auto feature=[result featureValueForName:Name(item.first)];Require(feature!=nil&&feature.type==MLFeatureTypeMultiArray,"Missing Core ML output");auto array=feature.multiArrayValue;CoreMLTensor t;t.name=item.first;t.dtype=TypeName(array.dataType);t.shape=Shape(array.shape);Check(t,item.second,false,false);t.bytes.resize(Multiply(Elements(t.shape),Width(t.dtype)));Copy(array,t,false);outputs.push_back(std::move(t));}
        impl_->poisoned=false;impl_->report["predictions"]=impl_->report["predictions"].get<size_t>()+1;impl_->report["last_prediction_and_output_copy_ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count();return outputs;
    }
}
std::vector<CoreMLTensor> CoreMLSession::SnapshotState(){
    std::lock_guard<std::mutex> lock(impl_->mutex);@autoreleasepool {impl_->Ready();std::vector<CoreMLTensor> result;
        for(const auto& item:impl_->states){CoreMLTensor t=item.second;t.bytes.resize(Multiply(Elements(t.shape),Width(t.dtype)));auto* destination=&t;[impl_->state getMultiArrayForStateNamed:Name(t.name) handler:^(MLMultiArray* array){Copy(array,*destination,false);}];result.push_back(std::move(t));}return result;}
}
void CoreMLSession::RestoreState(const std::vector<CoreMLTensor>& state){
    std::lock_guard<std::mutex> lock(impl_->mutex);@autoreleasepool {impl_->Ready();Require(state.size()==impl_->states.size(),"Core ML state inventory mismatch");std::set<std::string> names;
        for(const auto& t:state){Require(names.insert(t.name).second&&impl_->states.count(t.name),"Duplicate/unknown Core ML state");Check(t,impl_->states.at(t.name),true);}
        // Restore into a new state; publish only after every copy succeeds.
        auto fresh=[impl_->model newState];for(const auto& original:state){auto t=original;auto* source=&t;[fresh getMultiArrayForStateNamed:Name(t.name) handler:^(MLMultiArray* array){Copy(array,*source,true);}];}impl_->state=fresh;}
}
void CoreMLSession::Reset(){std::lock_guard<std::mutex> lock(impl_->mutex);@autoreleasepool {Require(impl_->model!=nil,"Core ML session closed");impl_->state=[impl_->model newState];impl_->poisoned=false;}}
void CoreMLSession::Close(){std::lock_guard<std::mutex> lock(impl_->mutex);impl_->state=nil;impl_->model=nil;impl_->inputs.clear();impl_->outputs.clear();impl_->states.clear();impl_->report=JSON::object();impl_->poisoned=false;}
std::string CoreMLSession::ReportJSON()const{std::lock_guard<std::mutex> lock(impl_->mutex);return impl_->report.dump();}
}
