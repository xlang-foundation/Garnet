// SPDX-License-Identifier: Apache-2.0
#include "trt_builder.h"
#include "../common/tensor_contracts.h"
#include <algorithm>
#include <cmath>
#include <climits>
#include <optional>
#include <limits>
#include <stdexcept>

namespace Garnet {
namespace {
using namespace nvinfer1;
// Standard tensor operations shared by model families. Model architectures and
// weight names are deliberately absent from this hardware lowering layer.
class TensorPrimitives {
    INetworkDefinition& network;
    std::deque<std::vector<float>>& constants;
    std::deque<std::vector<long long>>& shapeConstants;
public:
    TensorPrimitives(INetworkDefinition& n, std::deque<std::vector<float>>& storage,
        std::deque<std::vector<long long>>& shapeStorage)
        : network(n), constants(storage),shapeConstants(shapeStorage) {}
    static ITensor* output(ILayer* layer) {
        if (!layer || !layer->getOutput(0)) throw std::runtime_error("tensor primitive layer construction failed");
        return layer->getOutput(0);
    }
    static std::vector<int64_t> shape(ITensor* tensor,bool dynamic=false) {
        if (!tensor) throw std::invalid_argument("tensor primitive input is missing");
        const auto dimensions = tensor->getDimensions();
        std::vector<int64_t> result;
        for (int axis = 0; axis < dimensions.nbDims; ++axis) {
            if ((dimensions.d[axis] < 1 && !(dynamic&&dimensions.d[axis]==-1)) || dimensions.d[axis] > INT_MAX)
                throw std::invalid_argument("tensor primitive requires static positive dimensions");
            result.push_back(dimensions.d[axis]);
        }
        if (result.empty()) throw std::invalid_argument("tensor primitive requires a non-scalar input");
        return result;
    }
    static Dims dims(const std::vector<int64_t>& shape) {
        if (shape.size() > Dims::MAX_DIMS) throw std::invalid_argument("tensor rank exceeds TensorRT limit");
        Dims result{}; result.nbDims = int(shape.size());
        for (int axis = 0; axis < result.nbDims; ++axis) result.d[axis] = shape[axis];
        return result;
    }
    ITensor* cast(ITensor* tensor) {
        if (!tensor) throw std::invalid_argument("tensor primitive input is missing");
        return tensor->getType() == DataType::kFLOAT ? tensor : output(network.addCast(*tensor, DataType::kFLOAT));
    }
    ITensor* constant(std::vector<float> values, const std::vector<int64_t>& shape) {
        constants.push_back(std::move(values)); const auto& data = constants.back();
        return output(network.addConstant(dims(shape), Weights{DataType::kFLOAT, data.data(), int64_t(data.size())}));
    }
    ITensor* scalar(float value, size_t rank) { return constant({value}, std::vector<int64_t>(rank, 1)); }
    ITensor* shapeConstant(std::vector<long long> values){
        shapeConstants.push_back(std::move(values));const auto& data=shapeConstants.back();
        return output(network.addConstant(Dims{1,{int64_t(data.size())}},Weights{DataType::kINT64,data.data(),int64_t(data.size())}));
    }
    ITensor* shapePart(ITensor* value,int axis){
        return output(network.addSlice(*value,Dims{1,{axis}},Dims{1,{1}},Dims{1,{1}}));
    }
    ITensor* reshape(ITensor* tensor, const std::vector<int64_t>& target) {
        auto* layer = network.addShuffle(*tensor);
        if (!layer) throw std::runtime_error("tensor primitive reshape failed");
        layer->setZeroIsPlaceholder(false); layer->setReshapeDimensions(dims(target));
        return output(layer);
    }
    ITensor* transpose(ITensor* tensor, std::initializer_list<int> axes) {
        auto* layer = network.addShuffle(*tensor);
        if (!layer) throw std::runtime_error("tensor primitive transpose failed");
        Permutation order{}; int index = 0;
        for (int axis : axes) order.order[index++] = axis;
        layer->setFirstTranspose(order); return output(layer);
    }
    ITensor* element(ITensor* left, ITensor* right, ElementWiseOperation operation) {
        auto a = shape(left,true), b = shape(right,true);
        const auto rank = std::max(a.size(), b.size());
        auto expand=[&](ITensor* tensor,std::vector<int64_t>& size){
            const auto padding=rank-size.size();size.insert(size.begin(),padding,1);
            if(std::find(size.begin(),size.end(),-1)==size.end())return reshape(tensor,size);
            ITensor* parts[]{shapeConstant(std::vector<long long>(padding,1)),output(network.addShape(*tensor))};
            auto* joined=network.addConcatenation(parts,2);if(!joined)throw std::runtime_error("dynamic broadcast shape failed");joined->setAxis(0);
            auto* view=network.addShuffle(*tensor);if(!view)throw std::runtime_error("dynamic broadcast reshape failed");
            view->setInput(1,*output(joined));return output(view);
        };
        if (a.size() < rank) left=expand(left,a);
        if (b.size() < rank) right=expand(right,b);
        for (size_t axis = 0; axis < rank; ++axis)
            if (a[axis] != b[axis] && a[axis] != 1 && b[axis] != 1 && a[axis]!=-1 && b[axis]!=-1)
                throw std::invalid_argument("tensor primitive broadcast dimensions differ");
        return output(network.addElementWise(*left, *right, operation));
    }
    ITensor* add(ITensor* a, ITensor* b) { return element(a,b,ElementWiseOperation::kSUM); }
    ITensor* mul(ITensor* a, ITensor* b) { return element(a,b,ElementWiseOperation::kPROD); }
    ITensor* slice(ITensor* tensor, int axis, int64_t start, int64_t count) {
        auto size = shape(tensor,true); const int rank = int(size.size());
        axis=TensorContracts::Axis(axis,rank);
        if(size[axis]<0)throw std::invalid_argument("tensor_slice requires a static sliced extent");
        axis = TensorContracts::Slice(size,axis,start,count);
        std::vector<int64_t> begin(rank, 0), stride(rank, 1);
        begin[axis] = start; size[axis] = count;
        const bool dynamic=std::find(size.begin(),size.end(),-1)!=size.end();
        auto concrete=size;if(dynamic)for(auto& extent:concrete)if(extent<0)extent=1;
        auto* layer=network.addSlice(*tensor,dims(begin),dims(concrete),dims(stride));
        if(!layer)throw std::runtime_error("tensor_slice construction failed");
        if(dynamic){
            auto* runtimeShape=output(network.addShape(*tensor));std::vector<ITensor*> parts;
            for(int i=0;i<rank;++i)parts.push_back(i==axis?shapeConstant({count}):shapePart(runtimeShape,i));
            auto* joined=network.addConcatenation(parts.data(),rank);if(!joined)throw std::runtime_error("dynamic slice shape failed");
            joined->setAxis(0);layer->setInput(2,*output(joined));
        }
        return output(layer);
    }
    ITensor* concat(ITensor* left, ITensor* right, int axis) {
        const auto a = shape(left,true), b = shape(right,true);
        if(a.size()!=b.size())throw std::invalid_argument("tensor_concat ranks differ");
        axis=TensorContracts::Axis(axis,a.size());
        for(size_t i=0;i<a.size();++i)if(int(i)!=axis&&a[i]!=b[i]&&a[i]!=-1&&b[i]!=-1)
            throw std::invalid_argument("tensor_concat extents differ");
        if (left->getType() != right->getType()) throw std::invalid_argument("tensor_concat types differ");
        if (a[axis] > INT_MAX - b[axis]) throw std::invalid_argument("tensor_concat extent overflows");
        ITensor* inputs[]{left, right}; auto* layer = network.addConcatenation(inputs, 2);
        if (!layer) throw std::runtime_error("tensor_concat construction failed");
        layer->setAxis(axis); return output(layer);
    }
    ITensor* layerNorm(ITensor* tensor, float epsilon) {
        TensorContracts::PositiveEpsilon(epsilon);
        tensor = cast(tensor); const auto rank = shape(tensor,true).size();
        const auto axes = 1U << (rank - 1);
        auto* mean = output(network.addReduce(*tensor, ReduceOperation::kAVG, axes, true));
        auto* centered = element(tensor, mean, ElementWiseOperation::kSUB);
        auto* square = mul(centered, centered);
        auto* variance = output(network.addReduce(*square, ReduceOperation::kAVG, axes, true));
        auto* denominator = output(network.addUnary(*add(variance, scalar(epsilon, rank)), UnaryOperation::kSQRT));
        return element(centered, denominator, ElementWiseOperation::kDIV);
    }
    ITensor* l2Normalize(ITensor* tensor,float epsilon,int axis) {
        TensorContracts::PositiveEpsilon(epsilon);
        tensor=cast(tensor);const auto rank=shape(tensor,true).size();
        axis=TensorContracts::Axis(axis,rank);
        auto* square=mul(tensor,tensor);
        auto* sum=output(network.addReduce(*square,ReduceOperation::kSUM,1U<<axis,true));
        auto* length=output(network.addUnary(*sum,UnaryOperation::kSQRT));
        auto* denominator=element(length,scalar(epsilon,rank),ElementWiseOperation::kMAX);
        return element(tensor,denominator,ElementWiseOperation::kDIV);
    }
    ITensor* reduceMean(ITensor* tensor,int axis) {
        tensor=cast(tensor);axis=TensorContracts::Axis(axis,shape(tensor,true).size());
        return output(network.addReduce(*tensor,ReduceOperation::kAVG,1U<<axis,true));
    }
    ITensor* rmsNorm(ITensor* tensor,float epsilon,int axis){
        TensorContracts::PositiveEpsilon(epsilon);tensor=cast(tensor);const auto rank=shape(tensor,true).size();axis=TensorContracts::Axis(axis,rank);
        auto* mean=output(network.addReduce(*mul(tensor,tensor),ReduceOperation::kAVG,1U<<axis,true));
        auto* denominator=output(network.addUnary(*add(mean,scalar(epsilon,rank)),UnaryOperation::kSQRT));
        return element(tensor,denominator,ElementWiseOperation::kDIV);
    }
    ITensor* reduce(ITensor* tensor,int axis,ReduceOperation operation){
        tensor=cast(tensor);axis=TensorContracts::Axis(axis,shape(tensor,true).size());return output(network.addReduce(*tensor,operation,1U<<axis,true));
    }
    ITensor* fourierRotaryGrid(ITensor* positions,int width,int heads,float theta){
        const auto s=shape(positions);
        if(s.size()!=3||s[2]<1||s[2]>3||width<2||width>65536||heads<1||width%heads||(width/heads)%2||!std::isfinite(theta)||theta<=1)
            throw std::invalid_argument("invalid Fourier rotary grid dimensions");
        const int frequencies=width/(2*int(s[2]));if(frequencies<1)throw std::invalid_argument("rotary width is too small for coordinate axes");
        std::vector<float> values(frequencies);
        for(int i=0;i<frequencies;++i)values[i]=float(std::pow(double(theta),frequencies==1?0.0:double(i)/(frequencies-1))*1.57079632679489661923);
        auto* normalized=element(mul(cast(positions),scalar(2,3)),scalar(1,3),ElementWiseOperation::kSUB);
        auto* angles=mul(reshape(normalized,{s[0],s[1],1,s[2]}),constant(std::move(values),{1,1,frequencies,1}));
        angles=reshape(angles,{s[0],s[1],frequencies*s[2]});
        const int padding=width/2-frequencies*int(s[2]);
        if(padding)angles=concat(constant(std::vector<float>(size_t(s[0]*s[1]*padding),0),{s[0],s[1],padding}),angles,2);
        auto* half=reshape(angles,{s[0],s[1],heads,width/heads/2});angles=concat(half,half,3);
        const std::vector<int64_t> outputShape{1,s[0],s[1],heads,width/heads};
        auto* cosine=reshape(output(network.addUnary(*angles,UnaryOperation::kCOS)),outputShape);
        auto* sine=reshape(output(network.addUnary(*angles,UnaryOperation::kSIN)),outputShape);return concat(cosine,sine,0);
    }
    ITensor* convolution1D(ITensor* source,ITensor* weight,ITensor* bias,int stride,int before,int after,int dilation,int groups,bool transposed,bool fuseBias=false){
        const auto input=shape(source),filter=shape(weight);
        if(input.size()!=3||filter.size()!=3||stride<1||before<0||after<0||dilation<1||groups<1||
            input[1]!=(transposed?filter[0]:filter[1]*groups)||filter[0]%groups)throw std::invalid_argument("invalid 1D convolution geometry");
        auto* x=reshape(source,{input[0],input[1],1,input[2]});auto* w=reshape(weight,{filter[0],filter[1],1,filter[2]});ITensor* y=nullptr;
        if(!transposed)y=convolution(x,w,bias,2,groups,{1,stride},{0,before},{0,after},{1,dilation},fuseBias);
        else{
            if(dilation!=1)throw std::invalid_argument("transposed 1D convolution currently requires dilation one");
            const auto storedType=weight->getType();const auto type=storedType==DataType::kBF16?DataType::kFLOAT:storedType;
            // TensorRT's deconvolution tactics do not support BF16 on this CUDA
            // profile. Accumulate exactly represented BF16 operands in FP32,
            // then restore the checkpoint's output rounding.
            x=x->getType()==type?x:output(network.addCast(*x,type));w=w->getType()==type?w:output(network.addCast(*w,type));Weights empty{type,nullptr,0};
            auto* layer=network.addDeconvolutionNd(*x,int(filter[1]*groups),DimsHW{1,int(filter[2])},empty,empty);
            if(!layer)throw std::runtime_error("transposed convolution construction failed");layer->setInput(1,*w);layer->setNbGroups(groups);layer->setStrideNd(DimsHW{1,stride});layer->setPrePadding(DimsHW{0,before});layer->setPostPadding(DimsHW{0,after});y=cast(output(layer));
            if(bias){if(shape(bias)!=std::vector<int64_t>{filter[1]*groups})throw std::invalid_argument("transposed convolution bias differs");
                if(fuseBias){auto* typed=bias->getType()==type?bias:output(network.addCast(*bias,type));layer->setInput(2,*typed);}
                else y=add(y,reshape(cast(bias),{1,filter[1]*groups,1,1}));}
            if(storedType==DataType::kBF16)y=cast(output(network.addCast(*y,DataType::kBF16)));
        }
        const auto size=shape(y);return reshape(y,{size[0],size[1],size[3]});
    }
    ITensor* swiglu(ITensor* tensor) {
        tensor = cast(tensor); const auto size = shape(tensor,true);
        if (size.back() % 2) throw std::invalid_argument("swiglu requires an even packed width");
        auto* gate = slice(tensor, -1, 0, size.back()/2);
        auto* linear = slice(tensor, -1, size.back()/2, size.back()/2);
        return mul(mul(gate, output(network.addActivation(*gate, ActivationType::kSIGMOID))), linear);
    }
    ITensor* modulate(ITensor* tensor, ITensor* scale, ITensor* shift) {
        tensor = cast(tensor); scale = cast(scale); shift = cast(shift);
        return add(mul(tensor, add(scale, scalar(1, shape(scale,true).size()))), shift);
    }
    ITensor* rotary(ITensor* tensor, ITensor* cosine, ITensor* sine, bool interleaved) {
        tensor = cast(tensor); cosine = cast(cosine); sine = cast(sine);
        auto size = shape(tensor,true);
        TensorContracts::RotaryInterleaved(size,shape(cosine,true),shape(sine,true));
        ITensor* rotated=nullptr;
        if(interleaved){
            auto* pairs = reshape(tensor, {size[0],size[1],size[2],size[3]/2,2});
            auto* even = slice(pairs, 4, 0, 1); auto* odd = slice(pairs, 4, 1, 1);
            rotated = reshape(concat(output(network.addUnary(*odd, UnaryOperation::kNEG)), even, 4), size);
        }else{
            auto* first=slice(tensor,3,0,size[3]/2);
            auto* second=slice(tensor,3,size[3]/2,size[3]/2);
            rotated=concat(output(network.addUnary(*second,UnaryOperation::kNEG)),first,3);
        }
        return add(mul(tensor, cosine), mul(rotated, sine));
    }
    ITensor* attention(ITensor* query, ITensor* key, ITensor* value, ITensor* mask,
        std::optional<float> scale,bool bf16Scores,bool bf16Probabilities,bool bf16Compute,bool fp16Compute=false) {
        query = cast(query); key = cast(key); value = cast(value);
        const auto q = shape(query,true), k = shape(key,true), v = shape(value,true);
        if(q.size()!=4||k.size()!=4||v.size()!=4||q[2]<1||q[3]<1||v[3]<1)
            throw std::invalid_argument("attention requires static head counts and widths");
        TensorContracts::Attention(q,k,v);
        const float scoreScale=scale.value_or(1.0F/std::sqrt(float(q[3])));
        if(!std::isfinite(scoreScale))throw std::invalid_argument("attention scale must be finite");
        auto roundBF16=[&](ITensor* source){return cast(output(network.addCast(*source,DataType::kBF16)));};
        if(mask){
            const auto maskShape=shape(mask,true);
            if(maskShape.size()!=4 || (maskShape[0]!=1 && maskShape[0]!=q[0]) ||
                (maskShape[1]!=1 && maskShape[1]!=q[2]) || maskShape[2]!=q[1] || maskShape[3]!=k[1])
                throw std::invalid_argument("attention mask requires broadcastable [B,H,Q,K]");
        }
        if(bf16Compute || fp16Compute){
            if(bf16Scores || bf16Probabilities)
                throw std::invalid_argument("Fused attention compute cannot request separate score/probability rounding");
            const auto operandType=fp16Compute?DataType::kHALF:DataType::kBF16;
            auto roundOperand=[&](ITensor* x){return cast(output(network.addCast(*x,operandType)));};
            query=roundOperand(query);key=roundOperand(key);value=roundOperand(value);
            ITensor* valueMean=nullptr;
            if(fp16Compute){
                // Softmax weights sum to one: attention(V-mean(V))+mean(V)
                // is the same operation. Centering avoids accumulating a large
                // common value offset in the half-precision fused kernel.
                valueMean=output(network.addReduce(*value,ReduceOperation::kAVG,1U<<1,true));
                value=element(value,valueMean,ElementWiseOperation::kSUB);
            }
#if NV_TENSORRT_MAJOR > 10 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 16)
            // IAttention does not apply the SDPA scale. Scale Q explicitly in
            // FP32, then use explicit 16-bit operands in the fused
            // kernel. Do not permit decomposition into quadratic score buffers.
            auto* queries=transpose(output(network.addCast(*mul(query,scalar(scoreScale,4)),operandType)),{0,2,1,3});
            auto* keys=transpose(output(network.addCast(*key,operandType)),{0,2,1,3});
            auto* values=transpose(output(network.addCast(*value,operandType)),{0,2,1,3});
            auto* layer=network.addAttention(*queries,*keys,*values,AttentionNormalizationOp::kSOFTMAX,false);
            if(!layer)throw std::runtime_error("fused attention construction failed");
            if(!layer->setDecomposable(false))throw std::runtime_error("fused attention decomposition policy rejected");
            if(mask && !layer->setMask(*output(network.addCast(*mask,operandType))))
                throw std::runtime_error("fused attention mask rejected");
            auto* result=layer->getOutput(0);
            if(!result)throw std::runtime_error("fused attention output is missing");
            auto* restored=cast(transpose(result,{0,2,1,3}));
            if(valueMean){
                if(q[2]!=v[2]){
                    auto* expanded=reshape(valueMean,{v[0],1,v[2],1,v[3]});
                    std::vector<ITensor*> copies(size_t(q[2]/v[2]),expanded);
                    auto* repeated=network.addConcatenation(copies.data(),int(copies.size()));
                    if(!repeated)throw std::runtime_error("grouped value mean repetition failed");
                    repeated->setAxis(3);valueMean=reshape(output(repeated),{v[0],1,q[2],v[3]});
                }
                restored=add(restored,valueMean);
                if(mask){
                    auto* typedMask=cast(output(network.addCast(*mask,DataType::kHALF)));
                    auto* maximum=output(network.addReduce(*typedMask,ReduceOperation::kMAX,1U<<3,true));
                    auto* valid=element(maximum,scalar(-std::numeric_limits<float>::infinity(),4),ElementWiseOperation::kGREATER);
                    valid=transpose(valid,{0,2,1,3});
                    restored=output(network.addSelect(*valid,*restored,*scalar(0,4)));
                }
            }
            return restored;
#else
            if(fp16Compute)throw std::runtime_error("fused FP16 attention requires TensorRT 10.16 or newer");
#endif
        }
        if(q[2]!=k[2]){
            const auto repeats=q[2]/k[2];
            auto repeatHeads=[&](ITensor* source,const std::vector<int64_t>& size){
                auto* expanded=reshape(source,{size[0],size[1],size[2],1,size[3]});
                std::vector<ITensor*> copies(size_t(repeats),expanded);
                auto* layer=network.addConcatenation(copies.data(),int(copies.size()));
                if(!layer)throw std::runtime_error("grouped-query head repetition failed");
                layer->setAxis(3);
                return reshape(output(layer),{size[0],size[1],q[2],size[3]});
            };
            key=repeatHeads(key,k);value=repeatHeads(value,v);
        }
        auto* queries = transpose(query,{0,2,1,3});
        auto* keys = transpose(key,{0,2,1,3});
        auto* values = transpose(value,{0,2,1,3});
        auto* scores = output(network.addMatrixMultiply(*queries,MatrixOperation::kNONE,*keys,MatrixOperation::kTRANSPOSE));
        if(scoreScale!=1.0F)scores=mul(scores,scalar(scoreScale,4));
        if(bf16Scores)scores=roundBF16(scores);
        if(mask){
            scores=add(scores,cast(mask));
            if(bf16Scores)scores=roundBF16(scores);
        }
        auto* softmax = network.addSoftMax(*scores);
        if (!softmax) throw std::runtime_error("attention softmax construction failed");
        softmax->setAxes(1U<<3);
        auto* probabilities=output(softmax);
        if(bf16Probabilities)probabilities=roundBF16(probabilities);
        auto* result = output(network.addMatrixMultiply(*probabilities,MatrixOperation::kNONE,*values,MatrixOperation::kNONE));
        result=transpose(result,{0,2,1,3});
        return bf16Compute?roundBF16(result):result;
    }
    ITensor* sinusoidal(ITensor* timestep, int width, float scale, float period) {
        const auto size = shape(timestep,true);
        if (size.size()!=1 || width<2 || width%2 || width>65536 || !std::isfinite(scale) ||
            !std::isfinite(period) || period<=1) throw std::invalid_argument("sinusoidal_embedding dimensions or frequency settings invalid");
        const int half=width/2; std::vector<float> frequencies(half);
        for (int index=0;index<half;++index) frequencies[index]=scale*std::exp(-std::log(period)*float(index)/half);
        auto* angles=mul(reshape(cast(timestep),{size[0],1}),constant(std::move(frequencies),{1,half}));
        return concat(output(network.addUnary(*angles,UnaryOperation::kCOS)),output(network.addUnary(*angles,UnaryOperation::kSIN)),1);
    }
    ITensor* groupNorm(ITensor* source, ITensor* scale, ITensor* bias, int groups, float epsilon) {
        TensorContracts::PositiveEpsilon(epsilon);
        source=cast(source); scale=cast(scale); bias=cast(bias);
        const auto size=shape(source);
        if(size.size()<3 || groups<1 || size[1]%groups || shape(scale)!=std::vector<int64_t>{size[1]} || shape(bias)!=shape(scale))
            throw std::invalid_argument("group_norm requires [B,C,...], channel scale/bias and groups dividing C");
        int64_t width=size[1]/groups;
        for(size_t axis=2;axis<size.size();++axis){
            if(width>INT_MAX/size[axis])throw std::invalid_argument("group_norm reduction size overflows");
            width*=size[axis];
        }
        auto* grouped=reshape(source,{size[0],groups,width});
        auto* normalized=reshape(layerNorm(grouped,epsilon),size);
        std::vector<int64_t> channelShape(size.size(),1);channelShape[1]=size[1];
        return add(mul(normalized,reshape(scale,channelShape)),reshape(bias,channelShape));
    }
    ITensor* convolution(ITensor* source, ITensor* weight, ITensor* bias, int spatial, int groups,
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
        const std::vector<int64_t>& after, const std::vector<int64_t>& dilation,bool fuseBias=false) {
        const auto size=shape(source), filter=shape(weight);
        TensorContracts::Convolution(size,filter,spatial,groups,stride,padding,after,dilation);
        std::vector<int64_t> kernel(filter.begin()+2,filter.end());
        const auto type=weight->getType();
        if(type!=DataType::kFLOAT && type!=DataType::kBF16 && type!=DataType::kHALF)
            throw std::invalid_argument("convolution requires floating checkpoint weights");
        auto* input=source->getType()==type?source:output(network.addCast(*source,type));
        const Weights empty{type,nullptr,0};
        auto* layer=network.addConvolutionNd(*input,int(filter[0]),dims(kernel),empty,empty);
        if(!layer)throw std::runtime_error("convolution layer construction failed");
        layer->setInput(1,*weight);layer->setNbGroups(groups);
        if(bias&&fuseBias){
            if(shape(bias)!=std::vector<int64_t>{filter[0]})throw std::invalid_argument("convolution bias channel count differs");
            // Bias input 2 participates in convolution accumulation, before
            // BF16 output rounding. A later elementwise add rounds twice.
            auto* typed=bias->getType()==type?bias:output(network.addCast(*bias,type));layer->setInput(2,*typed);
        }
        layer->setStrideNd(dims(stride));layer->setPrePadding(dims(padding));
        layer->setPostPadding(dims(after));layer->setDilationNd(dims(dilation));
        auto* result=cast(output(layer));
        if(bias&&!fuseBias){
            if(shape(bias)!=std::vector<int64_t>{filter[0]})throw std::invalid_argument("convolution bias channel count differs");
            std::vector<int64_t> channelShape(size.size(),1);channelShape[1]=filter[0];
            result=add(result,reshape(cast(bias),channelShape));
        }
        return result;
    }
    ITensor* upsampleNearest2D(ITensor* source, int factor) {
        const auto size=shape(source);
        if(size.size()!=4 || factor<1 || factor>32 || size[2]>INT_MAX/factor || size[3]>INT_MAX/factor)
            throw std::invalid_argument("upsample_nearest2d requires bounded [B,C,H,W] and factor 1..32");
        auto* layer=network.addResize(*source);
        if(!layer)throw std::runtime_error("nearest upsampling construction failed");
        layer->setResizeMode(InterpolationMode::kNEAREST);
        layer->setCoordinateTransformation(ResizeCoordinateTransformation::kASYMMETRIC);
        layer->setNearestRounding(ResizeRoundMode::kFLOOR);
        layer->setOutputDimensions(dims({size[0],size[1],size[2]*factor,size[3]*factor}));
        return output(layer);
    }
};
}

bool TRTBuilder::LowerTensorPrimitive(const std::string& name, nvinfer1::ITensor* source,
    nvinfer1::ITensor* right, X::KWARGS& options, nvinfer1::ITensor*& result) {
    const bool supported=name=="tensor_slice" || name=="tensor_concat" || name=="layer_norm_unaffine" ||
        name=="swiglu" || name=="modulate" || name=="rotary_interleaved" || name=="rotary_split_half" || name=="cast" ||
        name=="scaled_dot_product_attention" || name=="sinusoidal_embedding" ||
        name=="group_norm" || name=="conv2d" || name=="conv3d" || name=="upsample_nearest2d" ||
        name=="l2_normalize" || name=="reduce_mean" || name=="reduce_min" || name=="reduce_max" || name=="reduce_sum" ||
        name=="rms_norm_unaffine" || name=="conv1d" || name=="conv_transpose1d" || name=="leaky_relu" || name=="tensor_gather" || name=="fourier_rotary_grid";
    if (!supported) return false;
    result=nullptr;
    try {
        TensorPrimitives ops(*network,vectorWeights,integer64VectorWeights);
        auto integer=[&](const char* key,int64_t fallback) {
            auto* item=FindKeyword(options,key);
            if (item && !item->second.IsInt64()) throw std::invalid_argument(std::string(key)+" must be an integer");
            return item?item->second.ToLongLong():fallback;
        };
        auto axis=[&]() {
            const auto value=integer("axis",-1);
            if(value<INT_MIN || value>INT_MAX) throw std::invalid_argument("tensor axis overflows");
            return int(value);
        };
        auto number=[&](const char* key,double fallback) { auto* item=FindKeyword(options,key);return float(item?item->second.ToDouble():fallback); };
        auto tensor=[&](const char* key) {
            auto* item=FindKeyword(options,key);
            if(!item || !IsTensor(item->second)) throw std::invalid_argument(std::string("missing tensor attribute: ")+key);
            auto* value=GetOrCreateTRTTensor(item->second);
            if(!value) throw std::invalid_argument(std::string("tensor attribute lowering failed: ")+key);
            return value;
        };
        auto weight=[&](const char* key,bool required) -> ITensor* {
            const std::string nameKey=std::string(key)+"_name";
            if(auto* item=FindKeyword(options,nameKey.c_str())){
                if(!item->second.IsString())throw std::invalid_argument(nameKey+" must be a checkpoint tensor name");
                auto* value=GetOrCreateTRTWeight(item->second.ToString());
                if(!value)throw std::invalid_argument("checkpoint tensor unavailable: "+item->second.ToString());
                return value;
            }
            if(FindKeyword(options,key))return tensor(key);
            if(required)throw std::invalid_argument(std::string("missing checkpoint tensor: ")+key);
            return nullptr;
        };
        auto boundedInteger=[&](const char* key,int64_t fallback,int minimum,int maximum){
            const auto value=integer(key,fallback);
            if(value<minimum || value>maximum)throw std::invalid_argument(std::string(key)+" out of range");
            return int(value);
        };
        if(name=="tensor_slice") result=ops.slice(source,axis(),integer("start",-1),integer("count",0));
        else if(name=="tensor_gather"){
            auto* indices=tensor("indices");if(indices->getType()!=DataType::kINT32||indices->getDimensions().nbDims!=1)throw std::invalid_argument("gather indices require an INT32 vector");
            const int a=TensorContracts::Axis(axis(),source->getDimensions().nbDims);result=TensorPrimitives::output(network->addGather(*source,*indices,a));
        }
        else if(name=="tensor_concat") result=ops.concat(source,right,axis());
        else if(name=="layer_norm_unaffine") result=ops.layerNorm(source,number("eps",1e-6));
        else if(name=="l2_normalize") result=ops.l2Normalize(source,number("eps",1e-12),axis());
        else if(name=="reduce_mean") result=ops.reduceMean(source,axis());
        else if(name=="reduce_min"||name=="reduce_max"||name=="reduce_sum")result=ops.reduce(source,axis(),name=="reduce_min"?ReduceOperation::kMIN:name=="reduce_max"?ReduceOperation::kMAX:ReduceOperation::kSUM);
        else if(name=="rms_norm_unaffine")result=ops.rmsNorm(source,number("eps",1e-6),axis());
        else if(name=="fourier_rotary_grid")result=ops.fourierRotaryGrid(source,boundedInteger("width",0,2,65536),boundedInteger("heads",1,1,65536),number("theta",10000));
        else if(name=="leaky_relu"){
            const float slope=number("negative_slope",0.01);if(!std::isfinite(slope)||slope<0)throw std::invalid_argument("invalid leaky ReLU slope");
            auto* layer=network->addActivation(*source,ActivationType::kLEAKY_RELU);if(!layer)throw std::runtime_error("activation failed");layer->setAlpha(slope);result=TensorPrimitives::output(layer);
        }
        else if(name=="swiglu") result=ops.swiglu(source);
        else if(name=="modulate") result=ops.modulate(source,tensor("scale"),tensor("shift"));
        else if(name=="rotary_interleaved" || name=="rotary_split_half")
            result=ops.rotary(source,tensor("cos"),tensor("sin"),name=="rotary_interleaved");
        else if(name=="scaled_dot_product_attention") {
            auto bf16=[&](const char* key){
                auto* item=FindKeyword(options,key);
                if(!item)return false;
                if(!item->second.IsString())throw std::invalid_argument(std::string(key)+" requires float32 or bfloat16");
                const auto dtype=item->second.ToString();
                if(dtype!="float32" && dtype!="bfloat16")throw std::invalid_argument(std::string(key)+" requires float32 or bfloat16");
                return dtype=="bfloat16";
            };
            const auto scale=FindKeyword(options,"scale")?std::optional<float>(number("scale",1)):std::nullopt;
            const auto* compute=FindKeyword(options,"compute_dtype");
            const bool fp16=compute && compute->second.IsString() && compute->second.ToString()=="float16";
            result=ops.attention(source,tensor("key"),tensor("value"),
                FindKeyword(options,"mask")?tensor("mask"):nullptr,scale,bf16("score_dtype"),bf16("probability_dtype"),fp16?false:bf16("compute_dtype"),fp16);
        }
        else if(name=="cast") {
            auto* dtype=FindKeyword(options,"dtype");
            if(!dtype || !dtype->second.IsString())throw std::invalid_argument("cast requires a dtype string");
            const auto value=dtype->second.ToString();
            nvinfer1::DataType target;
            if(value=="float32")target=nvinfer1::DataType::kFLOAT;
            else if(value=="bfloat16")target=nvinfer1::DataType::kBF16;
            else if(value=="float16")target=nvinfer1::DataType::kHALF;
            else if(value=="int32")target=nvinfer1::DataType::kINT32;
            else if(value=="int64")target=nvinfer1::DataType::kINT64;
            else throw std::invalid_argument("unsupported cast dtype: "+value);
            result=source->getType()==target?source:TensorPrimitives::output(network->addCast(*source,target));
        }
        else if(name=="sinusoidal_embedding") {
            const auto width=integer("width",256);
            if(width<2 || width>65536) throw std::invalid_argument("sinusoidal_embedding width out of range");
            result=ops.sinusoidal(source,int(width),number("scale",1),number("max_period",10000));
        }
        else if(name=="group_norm")result=ops.groupNorm(source,weight("weight",true),weight("bias",true),
            boundedInteger("groups",32,1,INT_MAX),number("eps",1e-6));
        else if(name=="conv1d"||name=="conv_transpose1d")result=ops.convolution1D(source,weight("weight",true),weight("bias",false),
            boundedInteger("stride",1,1,INT_MAX),boundedInteger("padding",0,0,INT_MAX),boundedInteger("padding_after",integer("padding",0),0,INT_MAX),
            boundedInteger("dilation",1,1,INT_MAX),boundedInteger("groups",1,1,INT_MAX),name=="conv_transpose1d",boundedInteger("fuse_bias",0,0,1)!=0);
        else if(name=="conv2d" || name=="conv3d") {
            const int spatial=name=="conv2d"?2:3;
            auto spatialAttribute=[&](const char* key,int fallback){
                return FindKeyword(options,key)?IntegerAttribute(options,key):std::vector<int64_t>(spatial,fallback);
            };
            const auto before=spatialAttribute("padding",0);
            const auto after=FindKeyword(options,"padding_after")?IntegerAttribute(options,"padding_after"):before;
            result=ops.convolution(source,weight("weight",true),weight("bias",false),spatial,
                boundedInteger("groups",1,1,INT_MAX),spatialAttribute("strides",1),before,after,spatialAttribute("dilations",1),boundedInteger("fuse_bias",0,0,1)!=0);
        }
        else if(name=="upsample_nearest2d")result=ops.upsampleNearest2D(source,boundedInteger("factor",2,1,32));
    } catch(const std::exception& error) { loweringError=name+": "+error.what(); }
    return true;
}
}
