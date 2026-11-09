// SPDX-License-Identifier: Apache-2.0
#include "../../src/backends/metal/bulk_append_policy.h"
#include <functional>
#include <iostream>
using namespace Garnet;
void Need(bool v){if(!v)throw std::runtime_error("Bulk append policy assertion");}
void Reject(std::function<void()> fn){bool bad=false;try{fn();}catch(const std::invalid_argument&){bad=true;}Need(bad);}
BulkAppendValidator Fresh(){return {{0,9,3,4095},{true,true,true,true},{1,0,3,4},4096,32768};}
int main(){try{auto v=Fresh();float z[]{0,0,0},three[]{3,3,3},last[]{4095,4095,4095};v.Step(0,1,0,z,9);v.Step(1,0,0,z,0);v.Step(2,1,3,three,19);v.Step(3,1,4095,last,29);v.Finish(true);Need(v.Counts()==std::array<size_t,4>{1,0,1,1});
 for(size_t kind=0;kind<13;++kind)Reject([&]{auto p=Fresh();float zero[]{0,0,0},wrong[]{1,0,0};switch(kind){case 0:p.Step(4,1,0,zero,0);break;case 1:p.Step(0,.5f,0,zero,0);break;case 2:p.Step(0,1,0,zero,-1);break;case 3:p.Step(0,1,0,zero,32768);break;case 4:p.Step(1,1,0,zero,0);break;case 5:p.Step(0,1,0,wrong,0);break;case 6:p.Step(0,0,1,zero,0);break;case 7:p.Step(0,0,0,wrong,0);break;case 8:p.Step(0,0,0,zero,0);p.Step(0,1,0,zero,0);break;case 9:p.Step(3,1,4095,last,0);p.Step(3,1,4095,last,0);break;case 10:p.Finish(true);break;case 11:p.Step(0,1,0,zero,0);p.Finish(false);break;default:{BulkAppendValidator bad({0,0,0,0},{false,true,true,true},{1,2,3,4},4096,32768);bad.Step(0,1,0,zero,0);}}});
 std::cout<<"Bulk tracked append policy PASS: independent starts, ragged end, exact capacity, 13 invalid cases\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
