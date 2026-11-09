// SPDX-License-Identifier: Apache-2.0
#include "lookahead_row_owner.h"
#include <iostream>
#include <set>
#include <string>
using namespace Garnet;
namespace {
size_t checks=0,rejections=0,trajectories=0,steps=0,outputs=0,windowChecks=0;
std::set<size_t> consumedCounts;
void Need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);++checks;}
template<class F>void Reject(F f){bool rejected=false;try{f();}catch(const std::invalid_argument&){rejected=true;}catch(const std::logic_error&){rejected=true;}Need(rejected,"Invalid owner action accepted");++rejections;}
int32_t Oracle(const std::vector<int32_t>& prefix,size_t mode) {
    if(mode==0)return 7;
    if(mode==1)return (prefix.back()+1)%32;
    uint32_t state=13;for(auto token:prefix)state=(state^uint32_t(token+71))*16777619u;
    return int32_t((state^uint32_t(prefix.size()*19))%32);
}
std::array<int32_t,16> Predict(const LookaheadLayout& layout,const LookaheadRowOwner::Plan& plan,
    const std::vector<int32_t>& canonical,size_t mode) {
    std::array<int32_t,16> result;result.fill(-1);
    for(size_t query=0;query<16;++query)if(plan.View().Active()&(uint16_t(1)<<query)) {
        auto branch=canonical;
        for(size_t key=0;key<16;++key)if(layout.Sees(plan.View(),query,key))branch.push_back(plan.Inputs()[key]);
        Need(branch.size()==layout.Position(plan.View(),query)+1,"Toy branch causal length differs");result[query]=Oracle(branch,mode);
    }
    return result;
}
void SamePlan(const LookaheadRowOwner::Plan& a,const LookaheadRowOwner::Plan& b) {
    Need(a.Inputs()==b.Inputs()&&a.Guesses()==b.Guesses()&&a.View().Active()==b.View().Active()&&
        a.View().Slot()==b.View().Slot()&&a.View().Epoch()==b.View().Epoch(),"Unpublished/rejected transition mutated owner");
}
void Window(const LookaheadLayout& layout,const LookaheadRowOwner::Plan& before,const std::array<int32_t,16>& predictions,
            const LookaheadRowOwner::Plan& after,int32_t root) {
    const size_t top=layout.Levels()-2;
    for(size_t node=0;node<layout.LookaheadRows();++node) {
        if(!(after.View().Active()&(uint16_t(1)<<node))){Need(after.Inputs()[node]==-1,"Inactive next window not sentinel");continue;}
        if(!node){Need(after.Inputs()[node]==root,"Next root differs");continue;}
        const size_t level=layout.At(node).level;const bool prediction=level==top;
        const size_t sourceLevel=prediction?level:level+1,targetPosition=layout.Position(after.View(),node);
        int32_t expected=root;bool found=false;
        // Independent provenance check by absolute positions, not the native
        // window's indexing/shift formula. This also covers W<consumed.
        for(size_t old=0;old<layout.LookaheadRows();++old)if((before.View().Active()&(uint16_t(1)<<old))&&layout.At(old).level==sourceLevel&&
            layout.Position(before.View(),old)+size_t(prediction)==targetPosition) {
            Need(!found,"Ambiguous window provenance");found=true;expected=prediction?predictions[old]:before.Inputs()[old];
        }
        Need(after.Inputs()[node]==expected,"Next window position/source provenance differs");++windowChecks;
    }
}
void Trajectory(size_t n,size_t w,size_t g,size_t row,size_t mode,size_t budget,size_t capacity) {
    LookaheadLayout layout(4,capacity,w,n,g);LookaheadRowOwner owner(layout,32);std::vector<int32_t> canonical;
    for(size_t i=0;i<7;++i){if(mode==0)canonical.push_back(7);else if(mode==1)canonical.push_back(int32_t((i+26)%32));else canonical.push_back(int32_t((i*17+row)%32));}
    const int32_t initial=Oracle(canonical,mode);owner.Bind(row,100+row,77,canonical.size(),budget,initial,canonical);size_t remaining=budget;
    while(owner.Live()) {
        auto plan=owner.BuildPlan();Need(owner.Slot()==canonical.size()&&owner.KnownRoot()==plan.Inputs()[0],"Owner/canonical root mismatch");
        auto predictions=Predict(layout,plan,canonical,mode);const auto version=owner.Version();auto staged=owner.Stage(plan,predictions);
        SamePlan(plan,owner.BuildPlan());Need(owner.Version()==version,"Staging published owner version");
        const auto count=staged.Count();Need(count>=1&&count<=n&&count<=remaining&&count<=capacity-canonical.size(),"Verified length exceeds profile");consumedCounts.insert(count);
        auto ordinary=canonical;int32_t root=owner.KnownRoot();
        for(size_t i=0;i<count;++i) {
            ordinary.push_back(root);root=Oracle(ordinary,mode);Need(root==staged.Emitted()[i],"Emitted stream differs from ordinary toy decoding");++outputs;
        }
        auto actual=canonical;for(size_t i=0;i<count;++i)actual.push_back(plan.Inputs()[staged.SelectedPath().local[i]]);
        Need(actual==ordinary&&staged.NextKnownRoot()==root,"Selected canonical branch differs from ordinary state");
        for(size_t at=count;at<4;++at)Need(staged.Emitted()[at]==-1,"Unused emitted slots not sentinel");
        for(size_t node=0;node<16;++node) {
            auto corrupted=predictions;corrupted[node]=(plan.View().Active()&(uint16_t(1)<<node))?32:0;
            Reject([&]{owner.Stage(plan,corrupted);});SamePlan(plan,owner.BuildPlan());
        }
        Reject([&]{owner.Stage(plan,predictions,{32});});SamePlan(plan,owner.BuildPlan());
        owner.Publish(staged);Need(owner.Version()==version+1,"Publish version differs");canonical=ordinary;remaining-=count;
        Need(owner.Remaining()==remaining&&owner.Slot()==canonical.size()&&owner.KnownRoot()==root,"Published metadata differs");
        Reject([&]{owner.Publish(staged);});Reject([&]{owner.Stage(plan,predictions);});
        if(owner.Live())Window(layout,plan,predictions,owner.BuildPlan(),root);
        ++steps;
    }
    Need(!remaining||canonical.size()==capacity,"Owner ended before budget/capacity");Reject([&]{owner.BuildPlan();});++trajectories;
}
void Boundaries() {
    LookaheadLayout layout(4,64,4,4,1);LookaheadRowOwner owner(layout,32),foreign(layout,32);
    std::vector<int32_t> prompt;for(int32_t i=0;i<32;++i)prompt.push_back(i);
    owner.Bind(3,19,71,32,13,0,prompt);foreign.Bind(3,19,71,32,13,0,prompt);
    auto plan=owner.BuildPlan();auto prefix=prompt;auto predictions=Predict(layout,plan,prefix,1);
    Reject([&]{foreign.Stage(plan,predictions);});
    auto full=owner.Stage(plan,predictions);Need(full.Count()==4,"Fully matching gram did not consume all N target positions");
    auto eos=owner.Stage(plan,predictions,{3,31});Need(eos.Count()==3&&eos.EndedByEOS()&&eos.NextKnownRoot()==3,"EOS did not truncate accepted branch");
    SamePlan(plan,owner.BuildPlan());Reject([&]{owner.Bind(3,20,71,32,13,0,prompt);});SamePlan(plan,owner.BuildPlan());
    owner.Invalidate();Reject([&]{owner.Publish(full);});Reject([&]{owner.Stage(plan,predictions);});
    owner.Bind(3,19,71,32,13,0,prompt);Reject([&]{owner.Publish(full);});
    auto fresh=owner.BuildPlan();auto freshPred=Predict(layout,fresh,prefix,1);auto eosFresh=owner.Stage(fresh,freshPred,{3});owner.Publish(eosFresh);
    Need(!owner.Live()&&owner.Slot()==35&&owner.Remaining()==10&&owner.KnownRoot()==3,"EOS metadata/canonical progress differs");
    auto version=owner.Version();auto bad=prompt;bad[0]=-1;Reject([&]{owner.Bind(0,20,72,32,1,0,bad);});Need(owner.Version()==version&&!owner.Live(),"Invalid bind changed inactive state");
    Reject([&]{owner.Bind(0,20,0,32,1,0,prompt);});Reject([&]{owner.Bind(4,20,72,32,1,0,prompt);});Reject([&]{owner.Bind(0,0,72,32,1,0,prompt);});
    Reject([&]{owner.Bind(0,20,72,32,65,0,prompt);});
    Reject([&]{owner.Bind(0,20,72,0,1,0,prompt);});
    Reject([&]{owner.Bind(0,20,72,0,1,3,{2});});
    owner.Bind(0,20,72,0,1,3,{3});Need(owner.BuildPlan().Inputs()[0]==3,"Known current root seed rejected");owner.Invalidate();
}
}
int main(){try{
    for(size_t n=2;n<=4;++n)for(size_t w=1;w<=4;++w)for(size_t g=1;g<=4;++g) {
        if((n-1)*(w+g)>16)continue;
        for(size_t row=0;row<4;++row)for(size_t mode=0;mode<3;++mode) {
            Trajectory(n,w,g,row,mode,19,41);Trajectory(n,w,g,row,mode,1,41);Trajectory(n,w,g,row,mode,13,13);
        }
    }
    Boundaries();Need(consumedCounts==std::set<size_t>({1,2,3,4}),"Missing actual accepted widths");
    std::cout<<"native lookahead owner CPU PASS trajectories="<<trajectories<<" steps="<<steps<<" outputs="<<outputs<<" checks="<<checks<<" rejections="<<rejections<<" window_position_checks="<<windowChecks
        <<" owner_bytes="<<sizeof(LookaheadRowOwner)<<" plan_bytes="<<sizeof(LookaheadRowOwner::Plan)<<" transition_bytes="<<sizeof(LookaheadRowOwner::Transition)
        <<" accepted_widths=1..4 ordinary_token_and_canonical_history_exact=true unpublished_stale_foreign_EOS_budget_capacity=true GPU_calls=0 runtime_integration=false\n";return 0;
}catch(const std::exception& e){std::cerr<<"native lookahead owner CPU FAILED: "<<e.what()<<'\n';return 1;}}
