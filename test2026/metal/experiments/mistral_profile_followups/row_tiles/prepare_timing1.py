"""Derive balanced timing around exactly the pilot's gather and projection commands."""
from pathlib import Path
import difflib,hashlib,json
R=Path(__file__).resolve().parent;assert json.loads((R/'pilot-audit1.json').read_text())['passed'];base=(R/'pilot1.mm').read_text();s=base
s=s.replace('Need(argc==5,"SHADER ASSETS CAPTURES FRESH_OUTPUT");','Need(argc==6,"SHADER ASSETS CAPTURES FRESH_OUTPUT forward|reverse");bool reverse=std::string(argv[5])=="reverse";Need(reverse||std::string(argv[5])=="forward","Invalid method order");')
s=s.replace('for(bool down:{false,true})','for(bool down:(reverse?std::vector<bool>{true,false}:std::vector<bool>{false,true}))')
a=s.index('  for(size_t mode=0;mode<6;++mode){\n   std::fill_n');b=s.index('\n  Save(dest/(shape+"-compact.u16")',a);old=s[a:b]
ea=old.index('   auto cb=[queue commandBuffer]');eb=old.index('   finish(cb);')+len('   finish(cb);');encode=old[ea:eb].replace('   finish(cb);','   finish(cb);return cb;')
va=old.index('size_t exact=0;');vb=old.index('   std::string payload=');validate=old[va:vb]
new='''  auto encode=[&](size_t mode){
'''+encode+'''
  };
  auto validate=[&](size_t mode,const char* phase){
   auto cb=encode(mode);
   '''+validate+'''
   report["cases"].push_back({{"shape",shape},{"phase",phase},{"mode",names[mode]},{"exact_words",exact},{"total_words",M*n},{"maximum_absolute_difference",maximum},{"parent_sha256",Sha(out.contents,out.length)},{"all_parents_unchanged",true}});
  };
  std::fill_n((uint32_t*)out.contents,M*n+32,G);std::fill_n((uint16_t*)compact.contents,M*k+64,H);
  for(size_t mode=0;mode<6;++mode)validate(mode,"pre");
  for(int round=-6;round<12;++round)for(int position=0;position<6;++position){
   int index=(position+round+6)%6;if(reverse)index=5-index;
   auto began=std::chrono::steady_clock::now();auto cb=encode(size_t(index));double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count();double gpu=(cb.GPUEndTime-cb.GPUStartTime)*1000.0;Need(std::isfinite(gpu)&&gpu>0&&std::isfinite(wall)&&wall>0,"Missing timer");
   report["samples"].push_back({{"shape",shape},{"round",round},{"position",position},{"method_index",index},{"mode",names[size_t(index)]},{"gpu_start",cb.GPUStartTime},{"gpu_end",cb.GPUEndTime},{"gpu_ms",gpu},{"wall_ms",wall},{"warmup",round<0}});
  }
  for(size_t mode=0;mode<6;++mode)validate(mode,"post");
  Report(dest/"result1.json",report);std::cout<<J{{"timed_shape",shape},{"reverse",reverse},{"samples",report["samples"].size()}}.dump()<<'\\n'<<std::flush;
'''
s=s[:a]+new+s[b:]
s=s.replace('{"prepared",J::array()}', '{"prepared",J::array()},{"samples",J::array()}')
s=s.replace('{"pilot_only",true}', '{"operator_timing_only",true},{"reverse",reverse}')
(R/'timing1.mm').write_text(s);(R/'timing-instrumentation1.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/pilot1.mm',tofile='b/timing1.mm')))
(R/'timing-preparation1.json').write_text(json.dumps(dict(passed=True,pilot_sha256=hashlib.sha256(base.encode()).hexdigest(),host_sha256=hashlib.sha256(s.encode()).hexdigest(),measured_rounds=12,warm_rounds=6,methods=6,shapes=2,all_executed_methods_use_bounded_q4_tile=True,no_expansion_or_scatter=True,shader_unchanged=True,no_model_or_comparator_execution=True),indent=2)+'\n')
