"""Derive a timing-only host; rejected three-term arithmetic is never dispatched."""
from pathlib import Path
import hashlib
import json

p = Path(__file__).resolve().parent
source = (p / 'compensated_attention1.mm').read_text()
assert hashlib.sha256(source.encode()).hexdigest() == '76b665c60a2e3ef8859ef25b999ebb30d78247c29ffa9f79031a79cb5c25c3fd'
prefix, old_main = source.split('int main(int argc,char** argv)', 1)
assert old_main.count('trained') > 5
main = r'''int main(int argc,char** argv){@autoreleasepool{try{
 need(argc==5,"Usage: timing-helper combined.metal audited-correctness fresh-output forward|reverse");
 fs::path fixture=argv[2],out=argv[3];std::string mode=argv[4];bool reverse=mode=="reverse";
 need(reverse||mode=="forward","Explicit timing order required");need(!fs::exists(out),"Fresh output required");
 auto correctness=readJson(fixture/"correctness1.json"),audit=readJson(fixture/"audit1.json");
 need(correctness["collected"]==true&&audit["passed"]==true,"Completed independently audited collection required");
 const std::vector<int> admitted={0,1,3};
 need(audit["qualified_variants"].size()==4,"Wrong variant gate count");
 need(correctness["variant_names"]==json({"original4mps","committed2mps","compensated_half3_fp32","compensated_half4_fp32"}),"Wrong numerical policies");
 need(audit["original_bound"]=="1e-5 + 2e-4*abs(FP64 expected)","Changed numerical bound");
 need(correctness["probability_scale"]==32768&&correctness["qkv_scale"]==16&&correctness["relaxed_precision"]==false&&correctness["math_mode"]=="Safe","Wrong precision policy");
 need(correctness["cases"].size()==audit["cases"].size(),"Case audit count differs");
 for(int v:admitted){need(audit["qualified_variants"][v]==true,"Unqualified numerical variant");
  for(auto& a:audit["cases"]){need(a["qualified_variants"][v]==true&&a["complete_control_bound_pass"][v]==true,"Unqualified case");need(a["fp64_bound_fraction_max"][v].get<double>()<=1,"FP64 bound violated");}}
 fs::create_directories(out);Pipelines gpu(argv[1]);std::vector<std::unique_ptr<Cell>> cells;
 for(size_t i=0;i<correctness["cases"].size();++i)cells.push_back(std::make_unique<Cell>(gpu,i,correctness["cases"][i],fixture,true));
 // Complete all compilation, allocation, input preparation and warmup before measurement.
 for(auto& c:cells)for(int warm=0;warm<3;++warm)for(int v:admitted){c->run(v);c->check(true,v);}
 json report={{"scope","Same trained/synthetic attention operands and committed-cache inputs; isolated component timing, not serving or quality qualification."},{"cases",json::array()},{"device",gpu.device.name.UTF8String},{"actual_trained_operands",correctness["actual_trained_operands"]},{"admitted_variants",admitted},{"excluded_variants",{2}},{"all_prepared_before_measurement",true},{"warmups_per_cell_variant",3},{"measured_rounds",9},{"reverse",reverse},{"compensated_half_layout_capacities",gpu.layoutWords},{"compensated_half_coordinate_parent_words",gpu.coordinateWords},{"register_layout_status",gpu.registerLayoutWords},{"probability_scale",32768},{"qkv_scale",16},{"relaxed_precision",false},{"math_mode","Safe"},{"variant_names",correctness["variant_names"]},{"native_integration",false},{"serving_speed_claim",false},{"goal_complete",false}};
 for(size_t i=0;i<cells.size();++i){auto& c=*cells[i];auto meta=c.metadata(correctness["cases"][i]);meta["samples"]=json::array();
  for(int round=0;round<9;++round)for(int order=0;order<3;++order){int slot=(round+order)%3,v=admitted[reverse?2-slot:slot];auto t=c.run(v);
   const auto& expected=c.expected[v];need(expected.size()==c.blank.size()&&!std::memcmp(c.out.contents,expected.data(),expected.size()*4),"Timed result changed");
   meta["samples"].push_back({{"round",round},{"order",order},{"variant",v},{"gpu_ms",t[0]},{"wall_ms",t[1]}});}
  // Full readonly parents and immutable committed cache checks stay outside timing.
  c.check(false,3);meta["checked_parent_words"]=c.checkedWords;report["cases"].push_back(meta);
  std::cout<<"Completed timing cell "<<i<<'\n'<<std::flush;
 }
 report["collected"]=true;std::ofstream(out/"screen1.json")<<report.dump(2)<<'\n';return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}}
'''
destination = p / 'compensated_attention2.mm'
assert not destination.exists()
destination.write_text(prefix + main)
(p / 'timing-host-bindings2.json').write_text(json.dumps({
    'base_host_sha256': hashlib.sha256(source.encode()).hexdigest(),
    'timing_host_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
    'candidate_shader_sha256': hashlib.sha256((p / 'combined2.metal').read_bytes()).hexdigest(),
    'admitted_variants': [0, 1, 3], 'excluded_variants': [2],
    'kernel_implementation_unchanged': True,
}, indent=2) + '\n')
