// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>
namespace Garnet::Media {
// Generic Slaney log-mel frontend: periodic Hann, centered reflect padding,
// magnitude STFT and area-normalized triangular mel filters.
inline std::vector<float> LogMel(const float* samples,size_t count,int rate,int fft,int hop,int bins){
    if(!samples||count<2||rate<8000||rate>192000||fft<16||fft>8192||(fft&(fft-1))||hop<1||hop>fft||bins<1||bins>512)
        throw std::runtime_error("invalid log-mel geometry");
    const size_t frames=count/hop+1;const int frequencies=fft/2+1;constexpr double pi=3.14159265358979323846;
    auto mel=[](double hz){return hz<1000?hz/(200.0/3.0):15+std::log(hz/1000)/(std::log(6.4)/27);};
    auto hz=[](double m){return m<15?m*(200.0/3.0):1000*std::exp((m-15)*(std::log(6.4)/27));};
    std::vector<double> edges(bins+2),window(fft);std::vector<float> filters(size_t(bins)*frequencies),result(size_t(2)*frames*bins);
    for(int i=0;i<bins+2;++i)edges[i]=hz(mel(rate/2.0)*i/(bins+1));
    for(int i=0;i<fft;++i)window[i]=.5-.5*std::cos(2*pi*i/fft);
    for(int bin=0;bin<bins;++bin)for(int k=0;k<frequencies;++k){double f=double(k)*rate/fft;
        filters[size_t(bin)*frequencies+k]=float(std::max(0.0,std::min((f-edges[bin])/(edges[bin+1]-edges[bin]),(edges[bin+2]-f)/(edges[bin+2]-edges[bin+1])))*2/(edges[bin+2]-edges[bin]));}
    std::vector<std::complex<double>> spectrum(fft);std::vector<double> magnitude(frequencies);
    for(int channel=0;channel<2;++channel)for(size_t frame=0;frame<frames;++frame){
        for(int j=0;j<fft;++j){int64_t index=int64_t(frame*hop)+j-fft/2;const int64_t period=int64_t(count-1)*2;index=((index%period)+period)%period;if(index>=int64_t(count))index=period-index;
            const float x=samples[size_t(channel)*count+size_t(index)];if(!std::isfinite(x))throw std::runtime_error("nonfinite log-mel input");spectrum[j]=double(x)*window[j];}
        for(int i=1,j=0;i<fft;++i){int bit=fft>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j)std::swap(spectrum[i],spectrum[j]);}
        for(int length=2;length<=fft;length*=2){std::complex<double> root=std::polar(1.0,-2*pi/length);
            for(int first=0;first<fft;first+=length){std::complex<double> w=1;for(int j=0;j<length/2;++j){auto a=spectrum[first+j],b=spectrum[first+j+length/2]*w;spectrum[first+j]=a+b;spectrum[first+j+length/2]=a-b;w*=root;}}}
        for(int k=0;k<frequencies;++k)magnitude[k]=std::abs(spectrum[k]);
        for(int bin=0;bin<bins;++bin){double energy=0;for(int k=0;k<frequencies;++k)energy+=magnitude[k]*filters[size_t(bin)*frequencies+k];result[(size_t(channel)*frames+frame)*bins+bin]=float(std::log(std::max(energy,1e-5)));}
    }return result;
}
}
