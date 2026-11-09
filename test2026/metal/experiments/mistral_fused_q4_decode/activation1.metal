#include <metal_stdlib>
using namespace metal;
struct Params{ulong rows,columns,aStride,bStride,cStride;float epsilon,offset;};
kernel void silu_mul(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],constant Params& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.rows*p.columns)return;
    ulong row=id/p.columns,col=id%p.columns;
    float gate=a[row*p.aStride+col];
    c[row*p.cStride+col]=(gate/(1.0f+exp(-gate)))*b[row*p.bStride+col];
}
kernel void decode_pressure(device uint* data [[buffer(0)]],uint i [[thread_position_in_grid]]){data[i]=data[i]*1664525u+1013904223u;}
