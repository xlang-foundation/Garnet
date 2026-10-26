// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "modelopt_fp8_z128_profile.h"
namespace Garnet {
inline bool ModelOptFP8Z128ReconstructedGDNZSelected(const char* reconstruct,const char* canonical,int batch,int tokens,int64_t inputRows,int outputRows,int columns,bool gdnZ){
    if(reconstruct&&*reconstruct&&std::strcmp(reconstruct,"0")&&std::strcmp(reconstruct,"1"))
        throw std::invalid_argument("GARNET_FP8_GDN_Z128_RECONSTRUCT must be0 or1");
    const bool enabled=reconstruct&&std::strcmp(reconstruct,"1")==0;
    if(enabled&&(!canonical||std::strcmp(canonical,"1")))
        throw std::invalid_argument("Z128 reconstruction requires explicit canonical Z128 policy");
    return enabled&&ModelOptFP8Z128GDNZSelected(canonical,batch,tokens,inputRows,outputRows,columns,gdnZ);
}
}
