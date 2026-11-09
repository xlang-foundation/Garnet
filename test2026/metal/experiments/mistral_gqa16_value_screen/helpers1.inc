// Fixed-profile GQA score sharing screen. No inference-runtime modifications.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "nlohmann/json.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
using J=nlohmann::json;namespace fs=std::filesystem;
void Need(bool x,const char* why){if(!x)throw std::runtime_error(why);}
J Read(fs::path p){std::ifstream f(p);J j;f>>j;return j;}
void Save(fs::path p,const J& j){std::ofstream f(p);f<<j.dump(2)<<'\n';Need(bool(f),"JSON write failed");}
void Binary(fs::path p,const void* v,size_t n){std::ofstream f(p,std::ios::binary);f.write((const char*)v,n);Need(bool(f),"Binary write failed");}
std::string Sha(const void* ptr,size_t n){Need(n<=UINT32_MAX,"Hash bound");unsigned char d[32];CC_SHA256(ptr,CC_LONG(n),d);std::ostringstream s;for(auto b:d)s<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(b);return s.str();}
uint64_t Now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct Params{uint64_t batch,chunk,heads,kvHeads,dim,capacity,qStride,kStride,vStride,kcStride,vcStride,sStride,aStride,gStride,pStride,oStride,gated;};
struct Clock{double start,end;uint64_t hostStart,hostEnd;};
struct Case{std::vector<size_t> firsts;std::vector<unsigned> masks;};
