// CPU-only interface fixture: no CUDA runtime or device is used.
#pragma once
#ifndef CUDART_VERSION
#define CUDART_VERSION 13020
#endif
using cudaError_t=int;
constexpr int cudaSuccess=0;
struct cudaDeviceProp{int major=0,minor=0;};
cudaError_t cudaGetDevice(int*);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp*,int);
cudaError_t cudaSetDevice(int);
