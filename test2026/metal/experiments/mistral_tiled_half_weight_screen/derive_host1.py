"""Reuse the audited host and add bounded, independently inspectable caches."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent;F=ROOT.parent/'mistral-packed-fragment-schedule1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(F/'provenance1.json')=='35165e34967e1a558a8e424f1ec17e3f5722b9352355b3f2d76c3c0bc542808d'
text=(F/'screen1.mm').read_text()
def edit(a,b):
    global text
    assert text.count(a)==1,(a[:100],text.count(a));text=text.replace(a,b)
edit('#include <cmath>','#include <cmath>\n#include <chrono>')
edit('// Trained-data fragment scheduling factors; no native/model/serving selection.','// Prepared half-weight layouts with strict native MPP arithmetic; operator only.')
edit('{"fp64",J::array()}','{"fp64",J::array()},{"prepared_weights",J::array()}')
edit('"fragment_mode"+std::to_string(mode)','"cached_mode"+std::to_string(mode)')
edit(' auto gather=pipe(@"packed_affine_gather");',''' auto prepPipe=[&](bool counted){auto c=[MTLFunctionConstantValues new];[c setConstantValue:&counted type:MTLDataTypeBool atIndex:23];return pipe(@"prepare_half_weights",c);};
 auto prepareNormal=prepPipe(false),prepareCounted=prepPipe(true);
 auto gather=pipe(@"packed_affine_gather");''')
anchor='  auto encode=[&](size_t rows,'
insert='''  std::array<id<MTLBuffer>,2> prepared;
  size_t rowElements=n*k,tiledElements=n*k/128*136;
  size_t extra=(rowElements+tiledElements)*2+256+(timing?256:(rowElements+tiledElements+64)*4);
  Need(device.recommendedMaxWorkingSetSize>extra+256*1024*1024&&device.currentAllocatedSize<=device.recommendedMaxWorkingSetSize-extra-256*1024*1024,"Prepared cache exceeds device allocation envelope");
  for(size_t layout=0;layout<2;layout++){@autoreleasepool{
   auto began=std::chrono::steady_clock::now();size_t elements=layout?tiledElements:rowElements;
   prepared[layout]=buffer((elements+64)*2);auto target=prepared[layout];
   if(!timing)std::fill_n((uint16_t*)target.contents,elements+64,H);
   else{std::fill_n((uint16_t*)target.contents,32,H);std::fill_n((uint16_t*)target.contents+32+elements,32,H);}
   auto writers=buffer(timing?128:(elements+32)*4);
   if(!timing){std::fill_n((uint32_t*)writers.contents,elements+32,G);std::fill_n((uint32_t*)writers.contents+16,elements,0u);}
   auto command=[queue commandBuffer];auto enc=[command computeCommandEncoder];uint32_t dims[]={uint32_t(k),uint32_t(n),uint32_t(layout),0};
   [enc setComputePipelineState:timing?prepareNormal:prepareCounted];[enc setBuffer:w offset:0 atIndex:0];[enc setBuffer:scale offset:0 atIndex:1];[enc setBuffer:bias offset:0 atIndex:2];[enc setBuffer:target offset:64 atIndex:3];[enc setBytes:dims length:sizeof(dims) atIndex:4];[enc setBuffer:writers offset:64 atIndex:5];[enc dispatchThreads:MTLSizeMake(elements/4,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[enc endEncoding];finish(command);
   double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count(),gpu=(command.GPUEndTime-command.GPUStartTime)*1000;
   Need(std::isfinite(gpu)&&gpu>0,"Missing preparation GPU clock");
   J metadata={{"layer",layer},{"shape",shape.name},{"inner",k},{"columns",n},{"layout",layout},{"elements",elements},{"payload_bytes",elements*2},{"allocation_bytes",target.length},{"counted",!timing},{"gpu_start_seconds",command.GPUStartTime},{"gpu_end_seconds",command.GPUEndTime},{"gpu_ms",gpu},{"allocation_encode_wait_wall_ms",wall}};
   if(!timing){
    for(size_t i=0;i<32;i++)Need(((uint16_t*)target.contents)[i]==H&&((uint16_t*)target.contents)[32+elements+i]==H,"Prepared weight guard changed");
    for(size_t i=0;i<16;i++)Need(((uint32_t*)writers.contents)[i]==G&&((uint32_t*)writers.contents)[16+elements+i]==G,"Preparation writer guard changed");
    for(size_t i=0;i<elements;i++)Need(((uint32_t*)writers.contents)[16+i]==1u,"Preparation did not write exactly once");
    metadata["half_sha256"]=Sha(target.contents,target.length);metadata["writers_sha256"]=Sha(writers.contents,writers.length);metadata["guards_exact"]=true;
    if(layer==0){std::string name="prepared-layer0-"+std::string(shape.name)+"-layout"+std::to_string(layout)+".u16";Save(destination/name,target.contents,target.length);metadata["raw_half"]=name;}
   }
   report["prepared_weights"].push_back(metadata);
  }}
'''
edit(anchor,insert+anchor)
edit('if(mode!=-1)[e setBuffer:counts offset:64 atIndex:7];','if(mode!=-1)[e setBuffer:counts offset:64 atIndex:7];if(mode>0)[e setBuffer:prepared[mode==1?0:1] offset:64 atIndex:8];')
edit('for(int round=-5;round<15;round++)','for(int round=-5;round<20;round++)')
edit('{"gpu_ms",ms}});','{"gpu_ms",ms},{"gpu_start_seconds",cb.GPUStartTime},{"gpu_end_seconds",cb.GPUEndTime}});')
dest=ROOT/'screen1.mm';assert not dest.exists();dest.write_text(text)
(ROOT/'host-derivation1.json').write_text(json.dumps(dict(passed=True,source_sha256=sha(F/'screen1.mm'),new_source_sha256=sha(dest),base_provenance_sha256=sha(F/'provenance1.json'),changes='Additional bounded prepared buffers, one-time half-weight decode with counted ownership/guards, cache bindings,20balanced timing rounds and absolute GPU timestamps. Original data/FP64/full parent checks retained.'),indent=2)+'\n')
print(json.dumps(dict(prepared_host=True,sha256=sha(dest))))
