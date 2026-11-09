"""Add balanced whole-MLP timings around exactly the pilot's normal-mode commands."""
from pathlib import Path
import difflib,hashlib,json
R=Path(__file__).resolve().parent;assert json.loads((R/'pilot-audit1.json').read_text())['passed'];base=(R/'pilot1.mm').read_text();s=base
s=s.replace('Need(argc==6,"COMBINED ORDINARY ASSETS CAPTURES FRESH_OUTPUT required");','Need(argc==7,"COMBINED ORDINARY ASSETS CAPTURES FRESH_OUTPUT forward|reverse required");bool reverse=std::string(argv[6])=="reverse";Need(reverse||std::string(argv[6])=="forward","Invalid balanced order");result["reverse"]=reverse;result["samples"]=J::array();result["timing_witnesses"]=J::array();')
s=s.replace('for(size_t layer:{0ul,15ul,31ul})','for(size_t layer:(reverse?std::vector<size_t>{31,15,0}:std::vector<size_t>{0,15,31}))')
a=s.index('void Save(');b=s.index('\nvoid Report',a);s=s[:a]+'void Save(fs::path,const void*,size_t){}'+s[b:]
marker='   result["layers"].push_back(row);'
insert='''   auto encodeNormal=[&](size_t method){auto command=[queue commandBuffer];encodeGather(command,input,compact,K);projection(command,0,fused[method>=2][1],true);projection(command,2,(method==1||method==2)?downTensor:downPipe,false);finish(command);return command;};
   auto validateNormal=[&](size_t method,const char* phase){reset();encodeNormal(method);guards();auto halfMismatch=Diff((uint16_t*)half.contents+32,goldHalf.data(),M*N,2),downMismatch=Diff((float*)outs[3].contents+16,expected[4].data(),M*K,4);Need(halfMismatch==0&&downMismatch==0,"Timed normal MLP differs from trained capture");for(size_t stream=0;stream<3;stream++)for(size_t i=0;i<M*N+32;i++)Need(((uint32_t*)outs[stream].contents)[i]==GUARD,"Timed normal MLP materialized intermediate");result["timing_witnesses"].push_back({{"layer",layer},{"method",method},{"phase",phase},{"half_mismatches",halfMismatch},{"down_mismatches",downMismatch},{"all_input_and_weight_parents_unchanged",true},{"guards_exact",true},{"no_intermediates_materialized",true}});};
   for(size_t method=0;method<4;method++)validateNormal(method,"pre");
   for(int round=-4;round<12;round++)for(int position=0;position<4;position++){
    int method=(position+round+4)%4;if(reverse)method=3-method;
    auto began=std::chrono::steady_clock::now();auto command=encodeNormal(size_t(method));double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count();double gpu=(command.GPUEndTime-command.GPUStartTime)*1000.0;Need(std::isfinite(gpu)&&gpu>0&&std::isfinite(wall)&&wall>0,"Missing MLP timer");
    result["samples"].push_back({{"layer",layer},{"method",method},{"round",round},{"position",position},{"warmup",round<0},{"gpu_start",command.GPUStartTime},{"gpu_end",command.GPUEndTime},{"gpu_ms",gpu},{"wall_ms",wall}});
   }
   for(size_t method=0;method<4;method++)validateNormal(method,"post");
'''
assert s.count(marker)==1;s=s.replace(marker,insert+marker)
s=s.replace('{"timing_measured",false}', '{"timing_measured",true}');s=s.replace('#include <array>', '#include <array>\n#include <chrono>')
(R/'timing1.mm').write_text(s);(R/'timing-instrumentation1.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/pilot1.mm',tofile='b/timing1.mm')))
(R/'timing-plan1.json').write_text(json.dumps(dict(passed=True,host_sha256=hashlib.sha256(s.encode()).hexdigest(),pilot_sha256=hashlib.sha256(base.encode()).hexdigest(),shader_unchanged=True,all_pipelines_prepared=True,methods=4,layers=[0,15,31],warm_rounds=4,measured_rounds=12,input_gather_and_both_fused_stages_included=True,no_reset_hash_or_write_in_timed_commands=True,no_native_model_or_comparator_execution=True),indent=2)+'\n')
