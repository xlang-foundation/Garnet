// SPDX-License-Identifier: Apache-2.0
#include "../../src/backends/metal/lookahead_layout.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using Layout=Garnet::LookaheadLayout;
static size_t checks=0,rejections=0,numerical=0,noninterference=0,profiles=0;
static void Need(bool value){++checks;if(!value)throw std::runtime_error("lookahead layout check failed");}
template<class F>static void Reject(F f){try{f();}catch(const std::invalid_argument&){++rejections;return;}throw std::runtime_error("invalid lookahead control accepted");}
using Vector=std::array<double,3>;
using Bank=std::array<Vector,Layout::LocalRows+3>;
// Independent path oracle: build the actual causal input sequence for one
// window column or verification branch, rather than using the runtime mask.
static std::vector<size_t> Path(const Layout& p,size_t q) {
    std::vector<size_t> keys{0,1,2};
    if(q<p.LookaheadRows()) {
        const auto level=q/p.Window(),column=q%p.Window();
        for(size_t i=0;i<=column;++i)keys.push_back(3+i);
        for(size_t l=1;l<=level;++l)keys.push_back(3+l*p.Window()+column);
    }else {
        const auto relative=q-p.LookaheadRows(),branch=relative/(p.Levels()-1),depth=relative%(p.Levels()-1);
        keys.push_back(3);
        for(size_t t=0;t<=depth;++t)keys.push_back(3+p.LookaheadRows()+branch*(p.Levels()-1)+t);
    }
    return keys;
}
static Vector Attention(const Vector& query,const Bank& keys,const Bank& values,const std::vector<size_t>& selected) {
    std::vector<double> scores;double maximum=-std::numeric_limits<double>::infinity();
    for(auto k:selected){double score=0;for(size_t d=0;d<3;++d)score+=query[d]*keys[k][d];scores.push_back(score);maximum=std::max(maximum,score);}
    double total=0;Vector result{};
    for(size_t i=0;i<selected.size();++i){const double weight=std::exp(scores[i]-maximum);total+=weight;for(size_t d=0;d<3;++d)result[d]+=weight*values[selected[i]][d];}
    for(auto& value:result)value/=total;
    return result;
}
static void Numerical(const Layout& p) {
    auto view=p.Request(0,17,3,64,p.Guesses(),p.Levels()-1);
    Bank keys{},values{};
    for(size_t i=0;i<keys.size();++i)for(size_t d=0;d<3;++d){keys[i][d]=double(int((i*7+d*11)%19)-9)/17;values[i][d]=double(int((i*13+d*5)%23)-11)/13;}
    for(size_t q=0;q<p.UsedRows();++q) {
        const auto oracle=Path(p,q);std::vector<size_t> actual{0,1,2};
        for(size_t k=0;k<Layout::LocalRows;++k)if(p.Sees(view,q,k))actual.push_back(k+3);
        Need(actual==oracle);
        const Vector query{double(q+1)/31,-.25,.375};
        const auto reference=Attention(query,keys,values,oracle),output=Attention(query,keys,values,actual);
        for(size_t d=0;d<3;++d){Need(std::isfinite(output[d])&&std::abs(reference[d]-output[d])<1e-13);++numerical;}
        Bank poisonedKeys=keys,poisonedValues=values;
        for(size_t k=0;k<Layout::LocalRows;++k)if(!p.Sees(view,q,k))for(size_t d=0;d<3;++d)
            poisonedKeys[k+3][d]=poisonedValues[k+3][d]=std::numeric_limits<double>::quiet_NaN();
        const auto isolated=Attention(query,poisonedKeys,poisonedValues,actual);
        for(size_t d=0;d<3;++d){Need(isolated[d]==output[d]);++noninterference;}
        // A caller change to visible prefix state must remain observable.
        auto patched=values;patched[0][0]+=1;
        Need(Attention(query,keys,patched,actual)[0]!=output[0]);
    }
}
static void Views(const Layout& p) {
    for(size_t row=0;row<p.Batch();++row)for(size_t fill=0;fill<p.Levels();++fill)for(size_t guesses=0;guesses<=p.Guesses();++guesses) {
        for(auto edge:std::array<std::array<size_t,2>,5>{{{0,1},{3,2},{3,64},{4094,4},{4095,4096}}}) {
            const size_t slot=edge[0],budget=edge[1];const auto v=p.Request(row,31,slot,budget,guesses,fill);
            Need(v.Row()==row&&v.Epoch()==31&&v.Slot()==slot&&v.Remaining()==budget);
            for(size_t q=0;q<Layout::LocalRows;++q) {
                bool enabled=false;size_t position=0;
                if(q<p.LookaheadRows()) {
                    const size_t level=q/p.Window(),column=q%p.Window();position=level+column;
                    enabled=q==0||level<fill;
                }else if(q<p.UsedRows()) {
                    const size_t index=q-p.LookaheadRows();position=index%(p.Levels()-1)+1;enabled=index/(p.Levels()-1)<guesses;
                }
                const bool active=enabled&&position<budget&&position<4096-slot;
                Need(bool(v.Active()&(uint16_t(1)<<q))==active);
                if(active) {
                    Need(p.Position(v,q)==slot+position);
                    for(auto k:Path(p,q))if(k>=3)Need(p.Sees(v,q,k-3));
                    for(size_t k=0;k<Layout::LocalRows;++k)if(p.Sees(v,q,k))Need(p.Position(v,k)<=p.Position(v,q));
                }else Reject([&]{p.Position(v,q);});
            }
            for(size_t branch=0;branch<p.Guesses();++branch)for(size_t count=1;count<=p.Levels();++count) {
                const bool valid=count<=budget&&count<=4096-slot&&(count==1||branch<guesses);
                if(!valid){Reject([&]{p.SelectCommit(v,branch,count);});continue;}
                const auto selected=p.SelectCommit(v,branch,count);Need(selected.count==count&&selected.local[0]==0);
                std::array<int,12> canonical;canonical.fill(-77);
                for(size_t i=0;i<count;++i) {
                    const auto node=selected.local[i];Need(node==0||p.At(node).role==Layout::Role::Verification);
                    Need(p.Position(v,node)==slot+i);canonical[i+3]=int(node);
                    if(i)Need(p.At(node).branch==branch);
                }
                for(size_t i=0;i<canonical.size();++i)if(i<3||i>=3+count)Need(canonical[i]==-77);
            }
        }
    }
}
int main() {
    for(size_t batch=1;batch<=4;++batch)for(size_t window=1;window<=4;++window)for(size_t levels=2;levels<=4;++levels)for(size_t guesses=1;guesses<=4;++guesses) {
        if((levels-1)*(window+guesses)>16){Reject([&]{Layout p(batch,4096,window,levels,guesses);});continue;}
        Layout p(batch,4096,window,levels,guesses);++profiles;
        Numerical(p);Views(p);Need(p.StageBytes(32,8,128)==batch*4194304);
    }
    Layout p(4,4096,4,3,4);size_t ordinal=0;
    for(size_t l=0;l<32;++l)for(size_t bank=0;bank<2;++bank)for(size_t row=0;row<4;++row)for(size_t node=0;node<16;++node)for(size_t h=0;h<8;++h)for(size_t d=0;d<128;++d)
        Need(p.StageOffset(32,8,128,l,bank,row,h,node,d)==(ordinal++)*sizeof(float));
    Need(ordinal*sizeof(float)==p.StageBytes(32,8,128));
    Need(p.StageOffset(32,8,128,0,0,0,0,1,0)-p.StageOffset(32,8,128,0,0,0,0,0,0)==8*128*sizeof(float));
    Reject([]{Layout q(0,4096,4,4,1);});Reject([]{Layout q(5,4096,4,4,1);});
    Reject([]{Layout q(4,0,4,4,1);});Reject([]{Layout q(4,4097,4,4,1);});
    Reject([]{Layout q(4,4096,0,4,1);});Reject([]{Layout q(4,4096,5,4,1);});
    Reject([]{Layout q(4,4096,4,1,1);});Reject([]{Layout q(4,4096,4,5,1);});
    Reject([]{Layout q(4,4096,4,4,0);});Reject([]{Layout q(4,4096,4,4,5);});
    Reject([&]{p.Request(4,1,0,4,1,2);});Reject([&]{p.Request(0,0,0,4,1,2);});
    Reject([&]{p.Request(0,1,4096,4,1,2);});Reject([&]{p.Request(0,1,0,0,1,2);});
    Reject([&]{p.Request(0,1,0,4097,1,2);});Reject([&]{p.Request(0,1,0,4,5,2);});
    Reject([&]{p.Request(0,1,0,4,1,3);});
    const auto v=p.Request(0,1,0,4,1,2);
    Reject([&]{p.SelectCommit(v,0,0);});Reject([&]{p.SelectCommit(v,4,1);});Reject([&]{p.SelectCommit(v,0,4);});
    Reject([&]{p.At(16);});Reject([&]{p.Sees(v,16,0);});Reject([&]{p.Sees(v,0,16);});
    Reject([&]{p.StageBytes(33,8,128);});Reject([&]{p.StageBytes(32,9,128);});Reject([&]{p.StageBytes(32,8,129);});
    Reject([&]{p.StageOffset(32,8,128,32,0,0,0,0,0);});Reject([&]{p.StageOffset(32,8,128,0,2,0,0,0,0);});
    Reject([&]{p.StageOffset(32,8,128,0,0,4,0,0,0);});Reject([&]{p.StageOffset(32,8,128,0,0,0,8,0,0);});
    Reject([&]{p.StageOffset(32,8,128,0,0,0,0,16,0);});Reject([&]{p.StageOffset(32,8,128,0,0,0,0,0,128);});
    Layout other(3,4096,4,3,4);Reject([&]{other.Position(v,0);});Reject([&]{other.SelectCommit(v,0,1);});
    Layout shortProfile(1,1,4,4,1);const auto tail=shortProfile.Request(0,1,0,1,1,3);Need(tail.Active()==1);
    std::cout<<"lookahead layout PASS profiles="<<profiles<<" checks="<<checks<<" rejections="<<rejections
        <<" FP64_components="<<numerical<<" forbidden_read_checks="<<noninterference
        <<" stage_bytes_B4="<<p.StageBytes(32,8,128)<<" GPU_calls=0\n";
}
