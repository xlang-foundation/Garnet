"""Derive a six-method balanced operator timer after the independent pilot gate."""
from pathlib import Path
import difflib,hashlib,json
R=Path(__file__).resolve().parent
assert json.loads((R/'pilot4-audit.json').read_text())['passed']
base=(R/'pilot4.mm').read_text();s=base
s=s.replace('Need(argc==5,"SHADER ASSETS CAPTURES FRESH_OUTPUT");','Need(argc==6,"SHADER ASSETS CAPTURES FRESH_OUTPUT forward|reverse");bool reverse=std::string(argv[5])=="reverse";Need(reverse||std::string(argv[5])=="forward","Invalid balanced order");')
s=s.replace('for(bool down:{false,true})', 'for(bool down:(reverse?std::vector<bool>{true,false}:std::vector<bool>{false,true}))')
start=s.index('  for(size_t mode=0;mode<10;++mode){\n   std::fill_n')
end=s.index('\n }}\n report.update',start)
original=s[start:end]
encode_start=original.index('   cb=[queue commandBuffer];e=')
encode_end=original.index('   finish(cb);')+len('   finish(cb);')
encode=original[encode_start:encode_end]
encode=encode.replace('   cb=[queue commandBuffer];e=', '''   auto cb=[queue commandBuffer];
   if(expand){auto prep=[cb computeCommandEncoder];uint32_t ds[]={uint32_t(k),uint32_t(n),0,0};[prep setComputePipelineState:prepare];[prep setBuffer:w offset:0 atIndex:0];[prep setBuffer:sc offset:0 atIndex:1];[prep setBuffer:bi offset:0 atIndex:2];[prep setBuffer:prepared offset:64 atIndex:3];[prep setBytes:ds length:sizeof(ds) atIndex:4];[prep setBuffer:writers offset:0 atIndex:5];[prep dispatchThreads:MTLSizeMake(n*k/4,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[prep endEncoding];}
   auto e=''',1)
encode=encode.replace('   finish(cb);','   finish(cb);return cb;')
validation_start=original.index('size_t exact=0;')
validation_end=original.index('   std::string payload=')
validation=original[validation_start:validation_end]
new='''  struct Method{size_t mode;bool expand;const char* label;};
  const std::array<Method,6> methods{{{0,false,"inline"},{6,false,"inline-rowfast"},{8,false,"inline-pair"},{5,false,"shared-tensor"},{7,false,"shared-rowfast"},{9,false,"shared-pair"}}};
  auto encode=[&](size_t mode,bool expand){
'''+encode+'''
  };
  auto validate=[&](const Method& method,const char* phase){
   size_t mode=method.mode;auto cb=encode(mode,method.expand);
   '''+validation+'''
   report["cases"].push_back({{"shape",shape},{"phase",phase},{"method",method.label},{"mode",mode},{"expand",method.expand},{"exact_words",exact},{"total_words",M*n},{"maximum_absolute_difference",maximum},{"parent_sha256",Sha(out.contents,out.length)},{"all_parents_unchanged",true}});
  };
  std::fill_n((uint32_t*)out.contents,M*n+32,G);std::fill_n((uint32_t*)temporary.contents,M*n+32,G);std::fill_n((uint16_t*)compact.contents,M*k+64,H);
  for(const auto& method:methods)validate(method,"pre");
  for(int round=-6;round<12;++round)for(int position=0;position<6;++position){
   int index=(position+round+6)%6;if(reverse)index=5-index;const auto& method=methods[size_t(index)];
   auto began=std::chrono::steady_clock::now();auto cb=encode(method.mode,method.expand);double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count();double gpu=(cb.GPUEndTime-cb.GPUStartTime)*1000.0;Need(std::isfinite(gpu)&&gpu>0&&std::isfinite(wall)&&wall>0,"Missing timer");
   report["samples"].push_back({{"shape",shape},{"round",round},{"position",position},{"method_index",index},{"method",method.label},{"mode",method.mode},{"expand",method.expand},{"gpu_start",cb.GPUStartTime},{"gpu_end",cb.GPUEndTime},{"gpu_ms",gpu},{"wall_ms",wall},{"warmup",round<0}});
  }
  for(const auto& method:methods)validate(method,"post");
  Report(dest/"result1.json",report);std::cout<<J{{"timed_shape",shape},{"reverse",reverse},{"samples",report["samples"].size()}}.dump()<<'\\n'<<std::flush;
'''
s=s[:start]+new+s[end:]
s=s.replace('{"prepared",J::array()}', '{"prepared",J::array()},{"samples",J::array()}')
s=s.replace('{"pilot_only",true}', '{"operator_timing_only",true},{"reverse",reverse}')
p=R/'timing2.mm';assert not p.exists();p.write_text(s)
(R/'timing-instrumentation2.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/pilot4.mm',tofile='b/timing2.mm')))
(R/'timing-preparation2.json').write_text(json.dumps(dict(passed=True,base_sha256=hashlib.sha256(base.encode()).hexdigest(),host_sha256=hashlib.sha256(s.encode()).hexdigest(),shader_unchanged=True,measured_rounds=12,warm_rounds=6,methods=6,shapes=2,weight_expansion_methods=[],cached_weight_methods=[],no_model_integration=True,no_comparator_execution=True),indent=2)+'\n')
print('Prepared balanced six-method timer with explicit expansion costs.')
