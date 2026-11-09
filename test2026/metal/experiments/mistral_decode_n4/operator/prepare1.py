"""Derive a lossless N4-word layout and isolated strict decode replay."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parent;M=ROOT.parent;J=M/'mistral-prefill-completion-wide1';G=M/'mistral-decode-threadgroup-screen1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=read(ROOT/'plan1.json');assert sha(J/'provenance1.json')==plan['base_engine_provenance_sha256'];assert sha(G/'provenance1.json')==plan['previous_geometry_provenance_sha256']
S=J/'source2/src/backends/metal';bindings=read(J/'source-bindings2.json')['files_sha256'];inputs={}
def shader(name):
    p=S/name;assert sha(p)==bindings['src/backends/metal/'+name];inputs[str(p)]=sha(p)
    return re.search(r'R"METAL\((.*?)\)METAL"',p.read_text(),re.S).group(1)
prefix='#define GARNET_AFFINE_FP16_COEFFICIENTS 1\n#define GARNET_AFFINE_PROFILE 1\n#define GARNET_AFFINE_INDEX32 1\n'+shader('affine_int4_profile_shader.h')
parts={}
for indexed in [False,True]:
    legacy=shader('affine_int4_indexed_decode_shader.h' if indexed else 'affine_int4_shader.h')
    if not indexed:legacy=legacy[:legacy.index('kernel void affine_q4_gemm')]
    text=prefix+'\n'+legacy+'\n#define GARNET_BLOCKED_COLUMNS 4\n#define GARNET_BLOCKED_ROWS '+('selected_decode_rows' if indexed else 'affine_decode_rows')+'\n'
    if indexed:text+='#define GARNET_BLOCKED_INDEXED 1\n'
    text+=shader('affine_int4_blocked_decode_shader.h')+'\n'+(ROOT/'interleaved1.inc').read_text()+'\n'
    text+='kernel void geometry_pressure(device uint* payload [[buffer(0)]],uint tid [[thread_position_in_grid]]) {payload[tid]=payload[tid]*1664525u+1013904223u;}\n'
    name='indexed1.metal' if indexed else 'ordinary1.metal';assert not(ROOT/name).exists();(ROOT/name).write_text(text);parts[name]=sha(ROOT/name)
host=(G/'screen3.mm').read_text();original=host;changes=[]
def edit(a,b):
    global host
    assert host.count(a)==1,(a[:100],host.count(a));host=host.replace(a,b);changes.append(dict(old=a,new=b))
edit('Need(argc==8,','Need(argc==9,')
edit('std::string mode=argv[5];bool correctness=', 'bool padded=std::string(argv[8])=="padded";Need(padded||std::string(argv[8])=="contiguous","Bad layout");std::string mode=argv[5];bool correctness=')
edit('for(int groups:{0,2,4,8})pipeline(indexed,shape.k,shape.n,shape.k,shape.n,active,groups);','for(int groups:{0,4,9})pipeline(indexed,shape.k,shape.n,shape.k+(padded?3:0),shape.n+(padded?5:0),active,groups);')
edit('aStride=k,oStride=n;', 'aStride=k+(padded?3:0),oStride=n+(padded?5:0);')
edit('for(int groups:{0,2,4,8})if(!ids.empty())', 'for(int groups:{0,4,9})if(!ids.empty())')
edit('for(int groups:{2,4,8}){','for(int groups:{4,9}){')
edit('at<3;at++','at<2;at++')
edit('int index=(at+(round+5)%3)%3;if(((round+5)+order)&1)index=2-index;int groups=std::array<int,3>{2,4,8}[index];','int index=(at+(round+5)%2)%2;if(((round+5)+order)&1)index=1-index;int groups=std::array<int,2>{4,9}[index];')
edit('groups==4?"affine_q4_blocked":"affine_q4_groups"+std::to_string(groups)', 'groups==4?"affine_q4_blocked":"affine_q4_interleaved4"')
edit('p.maxTotalThreadsPerThreadgroup>=size_t(groups?groups:4)*32','p.maxTotalThreadsPerThreadgroup>=128')
edit('size_t width=groups?size_t(groups)*4:4,threads=groups?size_t(groups)*32:128;', 'size_t width=groups?16:4,threads=128;')
edit('{"contiguous_layout",true}', '{"contiguous_layout",!padded}')
edit('auto upload=[&](const auto& v)', '''auto pack=[&](const auto& source,size_t width){
   using V=std::decay_t<decltype(source)>;V result(((n+3)/4)*width*4,0);
   for(size_t col=0;col<n;col++)for(size_t word=0;word<width;word++)result[((col/4)*width+word)*4+col%4]=source[col*width+word];
   for(size_t col=0;col<n;col++)for(size_t word=0;word<width;word++)Need(result[((col/4)*width+word)*4+col%4]==source[col*width+word],"Packed word roundtrip failed");
   for(size_t col=n;col<(n+3)/4*4;col++)for(size_t word=0;word<width;word++)Need(result[((col/4)*width+word)*4+col%4]==0,"Packed tail padding nonzero");
   return result;
  };
  auto packedW=pack(w,k/8);auto packedS=pack(sc,k/64),packedB=pack(bi,k/64);
  auto upload=[&](const auto& v)''')
edit('auto x=buffer(xWords*4),y=buffer(yWords*4);', '''auto x=buffer(xWords*4),y=buffer(yWords*4);
  auto guarded=[&](const auto& data){auto b=buffer(data.size()*sizeof(data[0])+128);std::fill_n((uint32_t*)b.contents,b.length/4,G);std::memcpy((char*)b.contents+64,data.data(),data.size()*sizeof(data[0]));return b;};
  auto packedWeights=guarded(packedW),packedScales=guarded(packedS),packedBiases=guarded(packedB);
  const std::string pwSha=correctness?Sha(packedWeights.contents,packedWeights.length):"",psSha=correctness?Sha(packedScales.contents,packedScales.length):"",pbSha=correctness?Sha(packedBiases.contents,packedBiases.length):"";
  if(correctness){
   auto metadata=[&](const auto& source,const auto& packed,id<MTLBuffer> b,const char* kind){
    std::string name="layer"+std::to_string(layer)+"-"+shape.name+"-n"+std::to_string(n)+"-"+kind+".bin";
    J row={{"layer",layer},{"shape",shape.name},{"inner",k},{"columns",n},{"padded_columns",(n+3)/4*4},{"kind",kind},{"element_bytes",sizeof(source[0])},{"source_elements",source.size()},{"packed_elements",packed.size()},{"source_sha256",Sha(source.data(),source.size()*sizeof(source[0]))},{"packed_sha256",Sha(packed.data(),packed.size()*sizeof(packed[0]))},{"allocated_bytes",b.length},{"roundtrip_exact",true},{"tail_padding_zero",true}};
    if(layer==0&&n==7){Save(dest/name,packed.data(),packed.size()*sizeof(packed[0]));row["raw_packed"]=name;}
    report["prepared_weights"].push_back(row);
   };metadata(w,packedW,packedWeights,"weight");metadata(sc,packedS,packedScales,"scale");metadata(bi,packedB,packedBiases,"bias");
  }''')
edit('[e setBuffer:weights offset:0 atIndex:1];[e setBuffer:scales offset:0 atIndex:2];[e setBuffer:biases offset:0 atIndex:3];', '[e setBuffer:(groups==9?packedWeights:weights) offset:(groups==9?64:0) atIndex:1];[e setBuffer:(groups==9?packedScales:scales) offset:(groups==9?64:0) atIndex:2];[e setBuffer:(groups==9?packedBiases:biases) offset:(groups==9?64:0) atIndex:3];')
edit('Need(Sha(x.contents,x.length)==xSha&&Sha(weights.contents,weights.length)==wSha&&Sha(scales.contents,scales.length)==sSha&&Sha(biases.contents,biases.length)==bSha,"Read-only input or weights changed");', 'Need(Sha(x.contents,x.length)==xSha&&Sha(weights.contents,weights.length)==wSha&&Sha(scales.contents,scales.length)==sSha&&Sha(biases.contents,biases.length)==bSha&&Sha(packedWeights.contents,packedWeights.length)==pwSha&&Sha(packedScales.contents,packedScales.length)==psSha&&Sha(packedBiases.contents,packedBiases.length)==pbSha,"Read-only input/weight parents or guards changed");')
edit('#include <tuple>','#include <tuple>\n#include <type_traits>')
host=re.sub(r'\bgroups\b','variant',host)
host=host.replace('// Isolated native decode geometry screen;', '// Lossless N4 packed-word layout screen;')
assert not(ROOT/'screen1.mm').exists();(ROOT/'screen1.mm').write_text(host)
record=dict(passed=True,source_headers_sha256=inputs,shader_sha256=parts,host_base_sha256=hashlib.sha256(original.encode()).hexdigest(),host_sha256=sha(ROOT/'screen1.mm'),changes=changes,final_identifier_rename='groups->variant; IDs0/4/9 identify legacy/native-blocked/N4, all use four SIMD groups.',all_pipelines_prepared_before_collection=True,no_native_runtime_integration=True,goal_complete=False)
(ROOT/'preparation1.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(dict(prepared=True,shader_sha256=parts,host_sha256=record['host_sha256'])))
