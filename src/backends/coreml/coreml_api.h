// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "coreml_session.h"
#include "../../model/float16_codec.h"
#include "xlang3/xlang3.h"
#include "nlohmann/json.hpp"
#include <cstring>
#include <mutex>
namespace Garnet {
class CoreMLAPI {
    CoreMLSession session_;
    nlohmann::json profile_,states_;
    mutable std::mutex mutex_;
    static X3TensorDType Type(const std::string& dtype){if(dtype=="float16")return X3_TENSOR_FLOAT16;if(dtype=="float32")return X3_TENSOR_FLOAT32;if(dtype=="int32")return X3_TENSOR_INT32;throw X::Error("Unsupported Core ML transport dtype");}
    static std::vector<CoreMLTensor> Read(X::Value values,const nlohmann::json& bounds,bool float32Transport=false){
        if(!values.IsDict()||values.Size()!=bounds.size())throw X::Error("Core ML named tensor inventory mismatch");std::vector<CoreMLTensor> result;
        for(auto item=bounds.begin();item!=bounds.end();++item){
            auto value=values[item.key().c_str()];X::Tensor tensor(value);auto lease=tensor.Acquire();auto info=tensor.Info();CoreMLTensor t;t.name=item.key();t.dtype=item.value().at("dtype").get<std::string>();auto maximum=item.value().at("maximum_shape").get<std::vector<int64_t>>();
            const bool convert=float32Transport&&t.dtype=="float16"&&info.dtype==X3_TENSOR_FLOAT32;
            if(info.symbolic||info.device_type!=0||(!convert&&info.dtype!=Type(t.dtype))||!info.data||info.rank!=maximum.size())throw X::Error("Core ML requires CPU tensors with the profiled type/rank");
            size_t width=convert?4:(t.dtype=="float16"?2:4),bytes=width;
            for(size_t axis=info.rank;axis>0;--axis){auto d=info.shape[axis-1];if(d<=0||d>maximum[axis-1]||size_t(d)>SIZE_MAX/bytes||(d>1&&info.strides&&info.strides[axis-1]!=int64_t(bytes)))throw X::Error("Core ML tensor exceeds profile or is noncontiguous");bytes*=size_t(d);}
            if(info.byte_size<bytes)throw X::Error("Core ML tensor payload is too short");t.shape.assign(info.shape,info.shape+info.rank);
            t.bytes.resize(convert?bytes/2:bytes);
            if(convert){for(size_t i=0;i<bytes/4;++i){float value;std::memcpy(&value,static_cast<const uint8_t*>(info.data)+i*4,4);auto half=Float32ToFloat16RNE(value);std::memcpy(t.bytes.data()+i*2,&half,2);}}
            else std::memcpy(t.bytes.data(),info.data,bytes);result.push_back(std::move(t));
        }return result;
    }
    X::Value Write(const std::vector<CoreMLTensor>& tensors,bool float32Transport=false){auto result=X::Value::Dict(Host());for(const auto& t:tensors){
        if(float32Transport&&t.dtype=="float16"){std::vector<float> values(t.bytes.size()/2);for(size_t i=0;i<values.size();++i){uint16_t half;std::memcpy(&half,t.bytes.data()+i*2,2);values[i]=Float16ToFloat32(half);}result.Set(t.name.c_str(),X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,t.shape,values.data(),values.size()*4));}
        else result.Set(t.name.c_str(),X::Tensor::Create(Host(),Type(t.dtype),t.shape,t.bytes.data(),t.bytes.size()));}return result;}
public:
    bool Open(const std::string& path,const std::string& profile){std::lock_guard<std::mutex> lock(mutex_);auto parsed=nlohmann::json::parse(profile);session_.Open(path,profile);profile_=std::move(parsed);states_=nlohmann::json::parse(session_.ReportJSON()).at("state_tensors");return true;}
    X::Value Predict(X::Value inputs){std::lock_guard<std::mutex> lock(mutex_);return Write(session_.Predict(Read(inputs,profile_.at("inputs"))));}
    // Explicit host transport conversion; the model still computes with its
    // declared types. INT32 features remain INT32. State snapshots stay raw.
    X::Value PredictFloat32(X::Value inputs){std::lock_guard<std::mutex> lock(mutex_);return Write(session_.Predict(Read(inputs,profile_.at("inputs"),true)),true);}
    X::Value Snapshot(){std::lock_guard<std::mutex> lock(mutex_);return Write(session_.SnapshotState());}
    bool Restore(X::Value state){std::lock_guard<std::mutex> lock(mutex_);session_.RestoreState(Read(state,states_));return true;}
    bool Reset(){std::lock_guard<std::mutex> lock(mutex_);session_.Reset();return true;}
    bool Close(){std::lock_guard<std::mutex> lock(mutex_);session_.Close();profile_=nullptr;states_=nullptr;return true;}
    std::string Report()const{std::lock_guard<std::mutex> lock(mutex_);return session_.ReportJSON();}
    BEGIN_PACKAGE(CoreMLAPI)
        APISET().AddFunc<2>("open",&CoreMLAPI::Open);
        APISET().AddFunc<1>("predict",&CoreMLAPI::Predict);
        APISET().AddFunc<1>("predict_float32",&CoreMLAPI::PredictFloat32);
        APISET().AddFunc<0>("snapshot_state",&CoreMLAPI::Snapshot);
        APISET().AddFunc<1>("restore_state",&CoreMLAPI::Restore);
        APISET().AddFunc<0>("reset",&CoreMLAPI::Reset);
        APISET().AddFunc<0>("close",&CoreMLAPI::Close);
        APISET().AddFunc<0>("report_json",&CoreMLAPI::Report);
    END_PACKAGE
};
}
