// CPU-only source-extracted lifecycle test. Mapping/allocator/CUDA sync are simulated.
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
namespace X {
struct Error:std::runtime_error{using std::runtime_error::runtime_error;};
struct Tensor{int id=0;bool IsValid()const{return id!=0;}};
}
constexpr int X3_TENSOR_BFLOAT16=1;
static bool failMap=false,failSpan=false,failSync=false,failAllocation=false;
static int mappings=0,opens=0,allocations=0,checks=0;
static std::vector<unsigned char> bytes(8,0);
struct SafeTensorMetadata{std::string dataType="BF16",filePath="simulated";std::vector<long long> shape{2,2};uint64_t dataSize=8;uint64_t dataOffset=0;};
struct SafeTensorsIndex{SafeTensorMetadata entry;const SafeTensorMetadata* Find(const std::string&)const{return &entry;}};
struct SafeTensorsMappedFile{
    SafeTensorsMappedFile(){++mappings;}~SafeTensorsMappedFile(){--mappings;}
    bool Open(const std::string&,std::string& error){++opens;if(failMap){error="injected mapping failure";return false;}return true;}
    const void* DataAt(uint64_t,uint64_t)const{return failSpan?nullptr:bytes.data();}
};
int cudaDeviceSynchronize(){return failSync?1:0;}
void Check(int code){if(code)throw X::Error("injected synchronization failure");}
struct TensorHelper{static X::Tensor CreateGPU(int,int,const std::vector<int64_t>&,const void*){
    ++allocations;if(failAllocation)throw X::Error("injected allocation failure");return X::Tensor{77};}};
class CUDATensorIO{
public:
    bool modelOpen_=false,nativeNVFP4Head_=false;int vocab_=0,width_=0;
    SafeTensorsIndex index_;X::Tensor table_,packedHead_;
    std::unique_ptr<SafeTensorsMappedFile> hostEmbedding_;
    const unsigned char* hostEmbeddingBytes_=nullptr;uint64_t hostEmbeddingSize_=0;
    std::mutex mutex_;int Host(){return 0;}
    void EnableHostEmbedding(bool enabled);void HostEmbeddingRow(int64_t id,float* destination)const;
};
// INSERT_ACTUAL_METHODS
static void require(bool condition,const char* message){++checks;if(!condition)throw std::runtime_error(message);}
static void ready(CUDATensorIO& m){m.modelOpen_=true;m.nativeNVFP4Head_=true;m.vocab_=2;m.width_=2;m.table_.id=42;m.packedHead_.id=7;}
static void reset(){failMap=failSpan=failSync=failAllocation=false;opens=allocations=0;require(mappings==0,"mapping leak between cases");}
static void rejected(const std::function<void()>& f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}require(caught,"expected rejection");}
int main(){try{
    // Every BF16 bit pattern, including signed zero/subnormals/nonfinite values,
    // is observed as bits. This does not claim CUDA nonfinite arithmetic parity.
    std::vector<unsigned char> raw(1+65536*2);std::vector<float> output(4096);
    for(uint32_t code=0;code<65536;++code){raw[1+2*code]=code&255;raw[2+2*code]=code>>8;}
    CUDATensorIO row;row.width_=4096;row.hostEmbeddingBytes_=raw.data()+1;
    for(int part=0;part<16;++part){row.HostEmbeddingRow(part,output.data());for(int col=0;col<4096;++col){
        uint32_t bits;std::memcpy(&bits,&output[col],4);require(bits==uint32_t(part*4096+col)*65536u,"BF16 row bits differ");}}
    {reset();CUDATensorIO m;rejected([&]{m.EnableHostEmbedding(true);});require(!m.table_.IsValid()&&!m.hostEmbedding_,"closed model mutated");}
    {reset();CUDATensorIO m;ready(m);m.EnableHostEmbedding(false);require(opens==0&&allocations==0&&m.table_.id==42,"default mode performed work");}
    for(int kind=0;kind<3;++kind){reset();CUDATensorIO m;ready(m);
        if(kind==0)m.nativeNVFP4Head_=false;if(kind==1)m.packedHead_.id=0;if(kind==2)m.index_.entry.dataSize=7;
        rejected([&]{m.EnableHostEmbedding(true);});require(opens==0&&m.table_.id==42&&!m.hostEmbedding_,"admission failure mutated model");}
    for(int kind=0;kind<3;++kind){reset();CUDATensorIO m;ready(m);
        if(kind==0)failMap=true;if(kind==1)failSpan=true;if(kind==2)failSync=true;
        rejected([&]{m.EnableHostEmbedding(true);});require(m.table_.id==42&&!m.hostEmbedding_&&m.hostEmbeddingBytes_==nullptr&&m.hostEmbeddingSize_==0&&mappings==0,"enable failure was not atomic");}
    {reset();CUDATensorIO m;ready(m);m.EnableHostEmbedding(true);
        require(!m.table_.IsValid()&&m.hostEmbedding_&&m.hostEmbeddingBytes_==bytes.data()&&m.hostEmbeddingSize_==8&&mappings==1,"host enable state invalid");
        const auto* original=m.hostEmbedding_.get();int oldOpens=opens;
        m.EnableHostEmbedding(true);require(opens==oldOpens&&m.hostEmbedding_.get()==original,"host enable is not idempotent");
        failAllocation=true;rejected([&]{m.EnableHostEmbedding(false);});
        require(m.hostEmbedding_.get()==original&&m.hostEmbeddingBytes_==bytes.data()&&m.hostEmbeddingSize_==8&&!m.table_.IsValid()&&mappings==1,"allocation failure lost working host mode");
        failAllocation=false;m.EnableHostEmbedding(false);
        require(m.table_.id==77&&!m.hostEmbedding_&&m.hostEmbeddingBytes_==nullptr&&m.hostEmbeddingSize_==0&&mappings==0,"device restoration invalid");
        int oldAllocations=allocations;m.EnableHostEmbedding(false);require(allocations==oldAllocations,"device disable is not idempotent");}
    require(mappings==0,"final mapping leak");
    std::cout<<"CPU_EXTRACTED_HOST_EMBEDDING_PASS checks="<<checks<<" bf16_patterns=65536 simulated_failure_classes=4\n";return 0;
}catch(const std::exception& error){std::cerr<<"CPU_EXTRACTED_HOST_EMBEDDING_FAIL "<<error.what()<<"\n";return 1;}}
