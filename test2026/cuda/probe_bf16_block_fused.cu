// SPDX-License-Identifier: Apache-2.0
// Unwired follow-up: packed BF16 -> shared tile -> tensor-core FP32 accumulator.
#include "bf16_block_projection_utils.cuh"
#include <mma.h>

template<bool Streaming> __global__ void packedMMA(const __nv_bfloat16* input,const unsigned char* blob,
    __nv_bfloat16* output,int rows,int inner,int columns){
    namespace wmma=nvcuda::wmma;
    constexpr int tile=16,kstep=128;
    __align__(32) __shared__ __nv_bfloat16 x[tile*kstep],w[tile*kstep];
    __align__(32) __shared__ float c[tile*tile];
    int m0=int(blockIdx.y)*tile,n0=int(blockIdx.x)*tile,tid=threadIdx.x;
    size_t blocks=(size_t(inner)*columns+255)/256;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    if(tid<32)wmma::fill_fragment(acc,0.f);
    for(int k0=0;k0<inner;k0+=kstep){
        for(int pair=tid;pair<tile*kstep/2;pair+=128){
            int row=pair/(kstep/2),k=(pair%(kstep/2))*2;
            reinterpret_cast<unsigned*>(x)[pair]=m0+row<rows?*reinterpret_cast<const unsigned*>(input+size_t(m0+row)*inner+k0+k):0;
        }
        for(int pair=tid;pair<tile*kstep/2;pair+=128){
            int column=pair/(kstep/2),k=(pair%(kstep/2))*2;
            size_t word=size_t(n0+column)*inner+k0+k,block=word/256,within=word%256;
            const auto* header=blob+block*8;unsigned offset=*reinterpret_cast<const unsigned*>(header),base=header[4],mode=header[5];
            const auto* payload=blob+blocks*8+offset;unsigned short low,high;
            if(mode==0){low=loadWord<Streaming>(payload+within*2);high=loadWord<Streaming>(payload+(within+1)*2);}
            else{
                const auto* p=payload+(within/2)*3;
                unsigned bits=unsigned(loadByte<Streaming>(p))|(unsigned(loadByte<Streaming>(p+1))<<8)|(unsigned(loadByte<Streaming>(p+2))<<16);
                low=unpack<Streaming>(bits&4095,base,payload);high=unpack<Streaming>(bits>>12,base,payload);
            }
            reinterpret_cast<unsigned*>(w)[pair]=unsigned(low)|(unsigned(high)<<16);
        }
        __syncthreads();
        if(tid<32){
            wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
            #pragma unroll
            for(int k=0;k<kstep;k+=16){
                wmma::load_matrix_sync(a,x+k,kstep);wmma::load_matrix_sync(b,w+k,kstep);
                wmma::mma_sync(acc,a,b,acc);
            }
        }
        __syncthreads();
    }
    if(tid<32)wmma::store_matrix_sync(c,acc,tile,wmma::mem_row_major);
    __syncthreads();
    for(int index=tid;index<tile*tile;index+=128){int row=m0+index/tile,column=n0+index%tile;
        if(row<rows&&column<columns)output[size_t(row)*columns+column]=__float2bfloat16_rn(c[index]);}
}
void fused(Buffer& input,Buffer& encoded,Buffer& output,int rows,int inner,int columns,cudaStream_t stream,bool streaming){
    require(inner%256==0&&columns%16==0,"Unsupported fused component shape");
    dim3 grid(unsigned(columns/16),unsigned((rows+15)/16));
    if(streaming)packedMMA<true><<<grid,128,0,stream>>>(reinterpret_cast<const __nv_bfloat16*>(input.data),encoded.data,reinterpret_cast<__nv_bfloat16*>(output.data),rows,inner,columns);
    else packedMMA<false><<<grid,128,0,stream>>>(reinterpret_cast<const __nv_bfloat16*>(input.data),encoded.data,reinterpret_cast<__nv_bfloat16*>(output.data),rows,inner,columns);
    check(cudaGetLastError());
}

int main(int argc,char** argv){try{
    require(argc==3||(argc==4&&std::string(argv[3])=="--quick"),"Require fixture manifest and unused outputJSONL path");bool quick=argc==4;require(!std::ifstream(argv[2]).good(),"Preserve fused component output");
    std::ifstream source(argv[1]);json manifest;source>>manifest;std::ofstream log(argv[2]);require(bool(log),"Fused output unavailable");
    check(cudaSetDevice(0));cudaDeviceProp property{};check(cudaGetDeviceProperties(&property,0));
    cudaStream_t stream=nullptr;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto context=Garnet::CreateBF16ProjectionContext();require(bool(context),"GEMM context failed");
    log<<json{{"event","metadata"},{"device",property.name},{"l2_bytes",property.l2CacheSize},{"quick",quick},
        {"scope","Unwired fused shared-tile decoder/MMA. Every BF16 output must equal ordinary production GEMM before timing. No model integration or2x claim."}}.dump()<<std::endl;
    size_t profiles=0;
    for(const auto& fixture:manifest.at("fixtures")){
        int inner=fixture.at("inner"),columns=fixture.at("columns");if(!inner)continue;
        std::string name=fixture.at("name");auto raw=bytes(fixture.at("raw_file")),packed=bytes(fixture.at("packed_file"));
        Buffer weight(raw.size()),encoded(packed.size()),input(size_t(512)*inner*2),eviction(size_t(256)<<20);
        check(cudaMemcpyAsync(weight.data,raw.data(),raw.size(),cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(encoded.data,packed.data(),packed.size(),cudaMemcpyHostToDevice,stream));
        check(cudaMemsetAsync(eviction.data,0,eviction.size,stream));
        initialize<<<unsigned((input.size/2+255)/256),256,0,stream>>>(reinterpret_cast<__nv_bfloat16*>(input.data),input.size/2);check(cudaStreamSynchronize(stream));
        std::vector<int> shapes=quick?std::vector<int>{4,32,512}:std::vector<int>{4,8,16,32,64,128,256,512};
        for(int rows:shapes){
            Buffer ordinary(size_t(rows)*columns*2),candidate(size_t(rows)*columns*2);
            auto baseline=[&]{require(Garnet::RunBF16Projection(context,input.data,weight.data,ordinary.data,rows,inner,columns,stream,nullptr,0)==0,"Ordinary GEMM failed");};
            baseline();check(cudaStreamSynchronize(stream));auto expected=download(ordinary);
            std::array<std::unique_ptr<Captured>,3> graphs;
            graphs[0]=capture(stream,baseline);
            for(int mode=1;mode<3;++mode){
                fused(input,encoded,candidate,rows,inner,columns,stream,mode==2);check(cudaStreamSynchronize(stream));
                auto got=download(candidate);size_t mismatch=differentWords(expected,got);
                bool guards=ordinary.guards()&&candidate.guards()&&weight.guards()&&encoded.guards()&&input.guards();
                json first=json::array();if(mismatch)for(size_t k=0;k<got.size()&&first.size()<8;k+=2)if(expected[k]!=got[k]||expected[k+1]!=got[k+1])
                    first.push_back({{"word",k/2},{"ordinary_bits",unsigned(expected[k])|(unsigned(expected[k+1])<<8)},{"candidate_bits",unsigned(got[k])|(unsigned(got[k+1])<<8)}});
                log<<json{{"event","correctness"},{"fixture",name},{"rows",rows},{"variant",mode==1?"fused":"fused_streaming"},{"different_bf16_words",mismatch},{"first_mismatches",first},{"guards_passed",guards}}.dump()<<std::endl;
                require(!mismatch&&guards,"Fused output/guard gate failed; no timing admitted");
                graphs[mode]=capture(stream,[&,mode]{fused(input,encoded,candidate,rows,inner,columns,stream,mode==2);});
            }
            Timer timer;std::array<std::vector<double>,3> cold,warm,evictionTimes;
            int repeats=quick?3:31;
            for(int round=0;round<repeats;++round)for(int pass=0;pass<3;++pass){int mode=round%2?2-pass:pass;auto t=timer.run(*graphs[mode],stream,eviction,true,unsigned(round));cold[mode].push_back(t[0]);evictionTimes[mode].push_back(t[1]);}
            for(int mode=0;mode<3;++mode){
                for(int i=0;i<5;++i)check(cudaGraphLaunch(graphs[mode]->executable,stream));check(cudaStreamSynchronize(stream));
                for(int i=0;i<repeats;++i)warm[mode].push_back(timer.run(*graphs[mode],stream,eviction,false,unsigned(i))[0]);
                require(differentWords(expected,download(mode?candidate:ordinary))==0,"Fused replay output changed");
                require(candidate.guards()&&ordinary.guards()&&input.guards()&&encoded.guards()&&weight.guards()&&eviction.guards(),"Fused replay canary changed");
                log<<json{{"event","projection"},{"fixture",name},{"rows",rows},{"variant",mode==0?"ordinary":mode==1?"fused":"fused_streaming"},
                    {"cold_median_us",median(cold[mode])},{"warm_median_us",median(warm[mode])},{"eviction_median_us",median(evictionTimes[mode])},
                    {"cold_samples_us",cold[mode]},{"warm_samples_us",warm[mode]},{"global_stage_bytes",0},{"shared_bytes_per_block",9216},{"all_output_words_bit_exact",true}}.dump()<<std::endl;
            }
            ++profiles;
        }
    }
    check(cudaStreamSynchronize(stream));context.reset();check(cudaStreamDestroy(stream));log<<json{{"event","complete"},{"profiles",profiles},{"all_passed",true}}.dump()<<std::endl;
    std::cout<<"BF16_BLOCK_FUSED_COMPLETE "<<profiles<<std::endl;return 0;
}catch(const std::exception& error){std::cerr<<"BF16_BLOCK_FUSED_FAILED "<<error.what()<<std::endl;return 1;}}
