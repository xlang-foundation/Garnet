from pathlib import Path
import collections,csv,hashlib,json,re,sys
s=Path(sys.argv[1]).resolve()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
 return h.hexdigest()
cases={'up':(14336,4096,'half_m256k128',229,'capture-copy-before1.json'),'down':(4096,14336,'half_m256k128',229,'down512-1.gputrace-copy-before1.json'),'qo':(4096,4096,'half_m128k128',146,'qo512-1.gputrace-copy-before1.json')}
result={'scope':'Xcode27.1 UI profiling of copies of three previously qualified synthetic projection captures. Not trained-model or serving measurements; GPU neural accelerator labels do not mean Core ML ANE. Percentages are the exported Xcode counters, not independently measured peak hardware fractions. Replay conditions are uncontrolled.','cases':{},'production_selection_changed':False,'new_serving_speed_claim':False,'new_coding_quality_claim':False,'goal_complete':False}
selected=['Occupancy Manager Target','Kernel Occupancy','Neural Accelerator Limiter','Neural Accelerator Utilization','Instruction Throughput Limiter','Instruction Throughput Utilization','ALU Utilization','Last Level Cache Limiter','MMU Limiter','L1 Cache Limiter','Compute Shader Launch Limiter','Control Flow Limiter','Kernel GEMM Operations Issued','Kernel ALU Instructions','Buffer Device Memory Bytes Read','Buffer Device Memory Bytes Written']
for name,(n,k,shader,regs,copyfile) in cases.items():
 p=s/(name+'-encoder-counters1.csv'); table=list(csv.reader(p.open(newline=''))); headers=table[0]; rows=table[1:]; assert len(headers)==272 and len(rows)==2 and all(len(row)==len(headers) for row in rows)
 def get(row,label):
  assert headers.count(label)==1,label
  return row[headers.index(label)]
 assert [get(row,'Encoder FunctionIndex') for row in rows]==['4','13']
 assert all(float(v)>=0 for row in rows for v in row[5:])
 indexed={'source_csv_sha256':sha(p),'columns':[{'index':i,'label':h} for i,h in enumerate(headers)],'rows':rows,'note':'Repeated header labels retained by column index; no DictReader overwriting duplicate columns.'}
 (s/(name+'-counters-indexed1.json')).write_text(json.dumps(indexed,indent=2)+'\n')
 text=(s/(name+'-shaders1.txt')).read_text(); fields=re.findall(r'^\s+\d+ text (.*)$',text,re.M)
 # Only shader table has plain text elements (toolbar activity uses a text without this value pattern).
 start=fields.index(shader)-1; values=fields[start:start+8]; assert values[1:4]==[shader,'Compute','MTLComputePipelineState 2']; assert values[5:8]==[str(regs),str(regs),'0 bytes']
 gather=fields[fields.index('packed_affine_gather')-1:fields.index('packed_affine_gather')+7]; assert gather[5:8]==['14','14','0 bytes']
 source=json.loads((s/copyfile).read_text()); original=Path(source['source']); changes=[]
 assert all(sha(original/rel)==v for rel,v in source['files_sha256'].items())
 copied=Path(source['copy']); after={str(f.relative_to(copied)):sha(f) for f in copied.rglob('*') if f.is_file()}
 for rel in sorted(set(after)|set(source['files_sha256'])):
  if after.get(rel)!=source['files_sha256'].get(rel): changes.append(rel)
 (s/(name+'-capture-copy-after1.json')).write_text(json.dumps({'files_sha256':after,'changed_or_added_relative_to_copy_before':changes,'original_unchanged':True},indent=2)+'\n')
 result['cases'][name]={'M':512,'N':n,'K':k,'shader':shader,'shader_ui':dict(zip(['cost','name','type','pipeline','simd_groups','allocated_registers','high_register','spilled_bytes'],values)),'gather_shader_ui':dict(zip(['cost','name','type','pipeline','simd_groups','allocated_registers','high_register','spilled_bytes'],gather)),'encoder_counters':{'gather':{label:float(get(rows[0],label)) for label in selected},'projection':{label:float(get(rows[1],label)) for label in selected}},'export_columns':len(headers),'duplicate_labels':{key:value for key,value in collections.Counter(headers).items() if value>1},'csv_sha256':sha(p),'copy_mutation_file_count':len(changes)}
assert all(v['encoder_counters']['projection']['Neural Accelerator Utilization']>80 for v in result['cases'].values())
result['interpretation']=['All three isolated original packed projection encoders report >80% GPU Neural Accelerator Utilization, with instruction throughput limiter >75% and lower LLC limiter. This suggests matrix/instruction work is a stronger limit than external-memory bandwidth in these captured cases, but does not establish full-model bottlenecks.','Gate/up and down report229 registers, Q/O146, zero reported spills. Occupancy is21.51/19.58/30.61 percent. Register pressure is observed; a causal gain from reducing it is not established.','Prior paired row-tile screen did not justify a replacement tile. Do not use higher occupancy alone as a success criterion.','Next capture must cover actual trained-model native prefill, including MPS attention and ungrouped Q/K/V/O, before extrapolating these synthetic bulk projection counters to serving TTFT.']
(s/'counter-analysis1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({name:{'registers':v['shader_ui']['allocated_registers'],'spills':v['shader_ui']['spilled_bytes'],'NAX_util':v['encoder_counters']['projection']['Neural Accelerator Utilization'],'kernel_occupancy':v['encoder_counters']['projection']['Kernel Occupancy'],'original_unchanged':True} for name,v in result['cases'].items()},indent=2))
