"""Derive a seven-method balanced operator timer after the independent pilot gate."""
from pathlib import Path
import difflib,hashlib,json
R=Path(__file__).resolve().parent
assert json.loads((R/'pilot2-audit.json').read_text())['passed']
base=(R/'pilot2.mm').read_text();s=base
s=s.replace('Need(argc==5,"SHADER ASSETS CAPTURES FRESH_OUTPUT");','Need(argc==6,"SHADER ASSETS CAPTURES FRESH_OUTPUT forward|reverse");bool reverse=std::string(argv[5])=="reverse";Need(reverse||std::string(argv[5])=="forward","Invalid balanced order");')
s=s.replace('for(bool down:{false,true})', 'for(bool down:(reverse?std::vector<bool>{true,false}:std::vector<bool>{false,true}))')
start=s.index('  for(size_t mode=0;mode<5;++mode){\n   std::fill_n')
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
  const std::array<Method,7> methods{{{0,false,"inline"},{1,false,"manual-cache"},{3,false,"tensor64m-cache"},{4,false,"tensor64n-cache"},{1,true,"manual-stream"},{3,true,"tensor64m-stream"},{4,true,"tensor64n-stream"}}};
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
  for(int round=-7;round<14;++round)for(int position=0;position<7;++position){
   int index=(position+round+7)%7;if((round+7)&1)index=6-index;if(reverse)index=6-index;const auto& method=methods[size_t(index)];
   auto began=std::chrono::steady_clock::now();auto cb=encode(method.mode,method.expand);double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count();double gpu=(cb.GPUEndTime-cb.GPUStartTime)*1000.0;Need(std::isfinite(gpu)&&gpu>0&&std::isfinite(wall)&&wall>0,"Missing timer");
   report["samples"].push_back({{"shape",shape},{"round",round},{"position",position},{"method_index",index},{"method",method.label},{"mode",method.mode},{"expand",method.expand},{"gpu_start",cb.GPUStartTime},{"gpu_end",cb.GPUEndTime},{"gpu_ms",gpu},{"wall_ms",wall},{"warmup",round<0}});
  }
  for(const auto& method:methods)validate(method,"post");
  Report(dest/"result1.json",report);std::cout<<J{{"timed_shape",shape},{"reverse",reverse},{"samples",report["samples"].size()}}.dump()<<'\\n'<<std::flush;
'''
s=s[:start]+new+s[end:]
s=s.replace('{"prepared",J::array()}', '{"prepared",J::array()},{"samples",J::array()}')
s=s.replace('{"pilot_only",true}', '{"operator_timing_only",true},{"reverse",reverse}')
p=R/'timing1.mm';assert not p.exists();p.write_text(s)
(R/'timing-instrumentation1.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/pilot2.mm',tofile='b/timing1.mm')))
(R/'timing-preparation1.json').write_text(json.dumps(dict(passed=True,base_sha256=hashlib.sha256(base.encode()).hexdigest(),host_sha256=hashlib.sha256(s.encode()).hexdigest(),shader_unchanged=True,measured_rounds=14,warm_rounds=7,methods=7,shapes=2,weight_expansion_methods=['manual-stream','tensor64m-stream','tensor64n-stream'],cached_weight_methods=['manual-cache','tensor64m-cache','tensor64n-cache'],no_model_integration=True,no_comparator_execution=True),indent=2)+'\n')
print('Prepared balanced seven-method timer with explicit expansion costs.')
