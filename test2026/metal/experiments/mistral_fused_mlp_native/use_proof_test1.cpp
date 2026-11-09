#include "fused_mlp_use_proof.h"
#include <iostream>
#include <stdexcept>
using namespace Garnet;
void Need(bool v){if(!v)throw std::runtime_error("Fused MLP logical-use proof accepted invalid graph");}
int main(){try{
 std::vector<FusedMlpUseNode> nodes={{{1,2},{10}},{{1,3},{11}},{{10,11},{12}},{{12,4},{13}},{{13,5},{14}}};
 const std::array<size_t,3> ids{10,11,12};std::vector<size_t> outputs{1,14,13};
 auto valid=[&](const auto& n,const auto& d,const auto& o){return ProveFusedMlpUses(n,0,d,o);};Need(valid(nodes,ids,outputs));size_t rejected=0;
 auto reject=[&](auto mutate){auto n=nodes;auto d=ids;auto o=outputs;mutate(n,d,o);Need(!valid(n,d,o));++rejected;};
 reject([](auto&,auto&,auto& o){o.push_back(10);});reject([](auto&,auto&,auto& o){o.push_back(11);});reject([](auto&,auto&,auto& o){o.push_back(12);});
 reject([](auto& n,auto&,auto&){n.push_back({{10},{90}});});reject([](auto& n,auto&,auto&){n.push_back({{11},{90}});});reject([](auto& n,auto&,auto&){n.push_back({{12},{90}});});
 reject([](auto& n,auto&,auto&){n[4].writes.push_back(10);});reject([](auto& n,auto&,auto&){n[2].reads.push_back(10);});reject([](auto& n,auto&,auto&){n[0].writes.clear();});
 reject([](auto&,auto& d,auto&){d[1]=d[0];});reject([](auto& n,auto&,auto&){n[3].reads={10};});reject([](auto& n,auto&,auto&){n.resize(3);});
 Need(!ProveFusedMlpUses(nodes,SIZE_MAX,ids,outputs));++rejected;
 std::cout<<"{\"passed\":true,\"invalid_graphs_rejected\":"<<rejected<<",\"observable_alias_roots_protected\":true}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
