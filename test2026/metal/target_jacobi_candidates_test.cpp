// SPDX-License-Identifier: Apache-2.0
#include "../../src/backends/metal/target_jacobi_candidates.h"
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>
using Core=Garnet::TargetJacobiCandidates;
namespace {
size_t checks=0,rejections=0,trajectories=0,multi=0;
void Need(bool value,const char* why){++checks;if(!value)throw std::runtime_error(why);}
template<class F>void Reject(F function){bool threw=false;try{function();}catch(const std::exception&){threw=true;}Need(threw,"Expected bounded owner rejection");++rejections;}
Core::Observation Example(size_t slot=3){Core::Observation o;o.epoch=7;o.slot=slot;o.length=4;o.consumed=2;o.remaining=20;o.input={10,11,12,13};o.prediction={11,70,80,90};return o;}
void Publish(Core& owner,size_t row,const Core::Observation& observed){std::array<Core::Observation,4> rows{};rows[row]=observed;Need(owner.Commit(owner.Stage(rows)),"Fresh candidate publication rejected");}
void Geometry(){
 Core owner;
 for(size_t batch=1;batch<=4;++batch)for(size_t depth=2;depth<=4;++depth)
 for(size_t vocabulary:{size_t(1),size_t(2),size_t(32768)})for(size_t capacity:{size_t(1),size_t(2),size_t(4),size_t(8),size_t(4096)}){
  owner.Configure(batch,depth,vocabulary,capacity);
  for(size_t row=0;row<batch;++row)for(size_t slot:{size_t(0),capacity-1})for(size_t budget:{size_t(1),size_t(2),size_t(20)}){
   auto draft=owner.Propose(row,1,slot,int64_t(vocabulary-1),budget);
   Need(draft.count==std::min({depth-1,budget-1,capacity-slot-1})&&!draft.reused,"Geometry/budget/seed mismatch");
   for(size_t t=0;t<draft.count;++t)Need(draft.following[t]==int64_t(vocabulary-1),"Seed read noncausal state");
  }
 }
 Reject([&]{owner.Configure(0,4,32768,4096);});Reject([&]{owner.Configure(5,4,32768,4096);});
 Reject([&]{owner.Configure(4,1,32768,4096);});Reject([&]{owner.Configure(4,5,32768,4096);});
 Reject([&]{owner.Configure(4,4,0,4096);});Reject([&]{owner.Configure(4,4,32768,0);});
 Reject([&]{owner.Configure(4,4,std::numeric_limits<size_t>::max(),4096);});
 Reject([&]{owner.Configure(4,4,32768,std::numeric_limits<size_t>::max());});
 owner.Configure(4,4,32768,32);
 Reject([&]{owner.Propose(4,1,0,10,3);});Reject([&]{owner.Propose(0,0,0,10,3);});
 Reject([&]{owner.Propose(0,1,std::numeric_limits<size_t>::max(),10,3);});
 Reject([&]{owner.Propose(0,1,0,-1,3);});Reject([&]{owner.Propose(0,1,0,32768,3);});Reject([&]{owner.Propose(0,1,0,10,0);});
}
void Lifecycle(){
 Core owner;owner.Configure(4,4,32768,32);Publish(owner,0,Example());
 auto d=owner.Propose(0,7,5,70,18);Need(d.count==3&&d.reused==2&&d.following==std::array<int64_t,3>{80,90,70},"Shifted prediction positions wrong");
 Need(!owner.Propose(0,8,5,70,18).reused,"Cross-epoch reuse");Need(!owner.Propose(0,7,6,70,18).reused,"Cross-slot reuse");Need(!owner.Propose(0,7,5,71,18).reused,"Cross-known-token reuse");
 Need(owner.Propose(0,7,5,70,1).count==0,"Output tail not clipped");Need(owner.Propose(0,7,31,70,18).count==0,"Capacity tail not clipped");
 auto other=Example();other.epoch=9;Publish(owner,1,other);Need(owner.Propose(0,7,5,70,18).reused==2,"Inactive owner changed");
 std::array<Core::Observation,4> rows{};rows[0]=Example(5);rows[1]=other;rows[1].prediction[3]=-1;
 Reject([&]{owner.Stage(rows);});Need(owner.Propose(0,7,5,70,18).reused==2,"Failed batch partially published candidates");
 for(int kind=0;kind<6;++kind){rows={};rows[0]=Example();auto& o=rows[0];if(kind==0)o.consumed=0;if(kind==1)o.consumed=5;if(kind==2)o.length=5;if(kind==3)o.remaining=0;if(kind==4)o.prediction[0]=12;if(kind==5)o.slot=31;Reject([&]{owner.Stage(rows);});}
 auto finished=Example();finished.finished=true;Publish(owner,0,finished);Need(!owner.Propose(0,7,5,70,18).reused,"Terminal owner retained candidates");
 auto all=Example();all.consumed=4;all.prediction={11,12,13,14};Publish(owner,0,all);Need(!owner.Propose(0,7,7,14,16).reused,"Fully consumed window retained stale guesses");
 Publish(owner,0,Example());rows={};rows[0]=Example();auto staged=owner.Stage(rows);owner.ResetRow(1);Need(!owner.Commit(staged),"Reset accepted a stale staged batch");
 Need(owner.Propose(0,7,5,70,18).reused==2,"Reset of another row lost its valid entry");
 owner.Invalidate();Need(!owner.Propose(0,7,5,70,18).reused,"Ordinary/prefix replacement retained candidates");
 staged=owner.Stage(rows);owner.Configure(4,4,32768,32);Need(!owner.Commit(staged),"Replacement profile accepted stale staged batch");
 owner.Reset();Need(owner.Counters().queries==0,"Full reset retained statistics");owner.Clear();Reject([&]{owner.Propose(0,7,5,70,18);});
 Need(owner.ResidentBytes()==sizeof(Core)&&sizeof(Core)<=1024,"Proposal owner exceeds fixed host bound");
}
int64_t Toy(const std::vector<int64_t>& prefix,int kind){if(!kind)return int64_t(prefix.size()%97);uint64_t sum=prefix.size();for(auto token:prefix)sum=(sum*33+uint64_t(token)+1)%97;return int64_t(sum);}
void Trajectory(int kind,int seed){
 Core owner;owner.Configure(1,4,97,256);std::vector<int64_t> processed{1,int64_t(seed),3};int64_t known=Toy(processed,kind);size_t generated=1;
 for(size_t step=0;generated<65;++step){Need(step<65,"Candidate policy stalled");const size_t remaining=65-generated,slot=processed.size();auto draft=owner.Propose(0,7,slot,known,remaining);auto path=processed;Core::Observation o;o.epoch=7;o.slot=slot;o.remaining=remaining;o.length=1+draft.count;o.input[0]=known;
  for(size_t t=0;t<draft.count;++t)o.input[t+1]=draft.following[t];
  for(size_t t=0;t<o.length;++t){path.push_back(o.input[t]);o.prediction[t]=Toy(path,kind);}
  for(size_t t=0;t<o.length;++t){processed.push_back(o.input[t]);Need(Toy(processed,kind)==o.prediction[t],"Committed toy trajectory differs from sequential target");known=o.prediction[t];++generated;++o.consumed;
   if(!known){o.finished=true;break;}if(t+1==o.length||known!=o.input[t+1])break;
  }
  o.finished=o.finished||generated==65;multi+=size_t(o.consumed>1);Publish(owner,0,o);if(o.finished)break;
 }
 ++trajectories;
}
}
int main(){try{
 static_assert(std::is_trivially_copyable<Core::Row>::value&&std::is_trivially_copyable<Core::Update>::value,"Publication must remain fixed-size and allocation-free");
 Geometry();Lifecycle();for(int kind=0;kind<2;++kind)for(int seed=1;seed<=16;++seed)Trajectory(kind,seed);Need(multi>0,"No multi-token candidate reuse exercised");
 std::cout<<"target Jacobi candidates PASS checks="<<checks<<" rejections="<<rejections<<" trajectories="<<trajectories<<" multi_token_steps="<<multi<<" owner_bytes="<<sizeof(Core)<<" GPU_calls=0\n";return 0;
}catch(const std::exception& e){std::cerr<<"target Jacobi candidates FAILED: "<<e.what()<<'\n';return 1;}}
