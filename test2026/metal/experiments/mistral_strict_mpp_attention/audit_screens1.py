"""Audit both strict FP32 attention prototypes; retained-data only, no inference."""
from pathlib import Path
import copy,hashlib,json,math,statistics,sys
import numpy as np
r=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert load(r/'correctness-status1.json')['returncode']==1
assert load(r/'correctness-status2.json')['returncode']==0
assert load(r/'correctness-status3.json')['returncode']==1
assert 'register mapping differs' in (r/'correctness3.stderr').read_text()
assert load(r/'correctness-status4.json')['returncode']==0
assert load(r/'strict-map-audit1.json')['passed']
probe=load(r/'coordinate-dump1.json');assert probe['old_register_guard_flags']==[14]*32 and probe['parent_guards_intact'] and not probe['attention_kernel_launched']
for lane in range(32):
 row=((lane>>1)&3)|((lane>>4)&1)*4;col=(lane&1)*2+((lane>>3)&1)*4
 for tensor in [0,2,3,4,5]:
  for i in range(probe['capacities'][tensor]):
   actual=probe['coordinates'][tensor*1536+(lane*16+i)*3:tensor*1536+(lane*16+i)*3+3]
   expected=[col+(i%2)+((i//2)%(2 if tensor in [0,3] else 4))*8,row+(i//(4 if tensor in [0,3] else 8))*8,1]
   assert actual==expected,'Incorrect strict coordinate map'
source_refs=load(r/'source-references1.json')
for p,h in source_refs.items():assert sha(Path(p))==h,p
old=r.parent/'mistral-merged-kv-attention1'
original=load(old/'provenance1.json')['files_sha256']
for n in ['combined1.metal','merged_attention3.mm']:assert sha(old/n)==original[n]
assert (r/'combined2.metal').read_bytes()==(old/'combined1.metal').read_bytes()+b'\n'+(r/'strict-mpp2.metal').read_bytes()
assert (r/'combined3.metal').read_bytes()==(r/'combined2.metal').read_bytes()+(r/'coordinate-probe1.metal').read_bytes()
assert (r/'combined6.metal').read_bytes()==(r/'combined3.metal').read_bytes()+(r/'strict-register2.metal').read_bytes()
# Both MPS controls must retain the previous qualified arithmetic on exactly
# the same raw inputs. New control1 uses precommitted buffers, without copying.
control_pairs=0
for folder in ['correctness2','correctness4']:
 assert load(r/folder/'audit1.json')['passed']
 for i in range(7):
  for part in range(8):assert sha(r/folder/f'case{i}-input{part}.f32')==sha(old/'correctness2'/f'case{i}-input{part}.f32')
  for current,previous in [(0,0),(1,2)]:
   assert sha(r/folder/f'case{i}-output{current}.f32')==sha(old/'correctness2'/f'case{i}-output{previous}.f32')
   control_pairs+=np.fromfile(r/folder/f'case{i}-output{current}.f32',dtype='<u4').size
sets={}
for name,prefix,correctness,variant in [('shared','screen','correctness2','strict_mpp_online_fp32'),('register','register-screen','correctness4','strict_mpp_register_fp32')]:
 status=load(r/(prefix+'-status1.json'));pre=load(r/(prefix+'-preflight-hashes1.json'))
 for n,h in pre.items():assert sha(r/n)==h,n
 reports=[load(r/(prefix+str(c+1))/'screen1.json') for c in [0,1]]
 sets[name]=(prefix,correctness,variant,status,reports)
def verify(name,item):
 prefix,correctness,variant,status,reports=item;assert status['status']=='collected_pending_audit' and status['source_hashing_finished_before_collection']
 assert len(status['runs'])==2;last=0;comparisons=[]
 reference=load(r/correctness/'correctness1.json')
 for cohort,(run,report) in enumerate(zip(status['runs'],reports)):
  assert run['cohort']==cohort and run['reverse']==bool(cohort) and run['status']=='terminal' and run['returncode']==0
  assert last<=run['started_ns']<run['finished_ns'];last=run['finished_ns']
  assert report['collected'] and report['all_prepared_before_measurement'] and not report['native_integration'] and not report['serving_speed_claim']
  assert report['reverse']==bool(cohort) and report['relaxed_precision'] is False and report['math_mode']=='Safe'
  assert report['variant_names']==['original4mps','committed2mps',variant]
  assert report['strict_mpp_layout_capacities']==[8,16,16,8,16,16]*32
  coords=report['strict_mpp_coordinate_parent_words'];assert len(coords)==2593
  assert coords[:16]==coords[-16:]==[0x7fc12345]*16 and coords[16:2576]==[1]*2560 and coords[2576]==0
  if name=='register':assert report['register_layout_status']==[0]*32
  assert len(report['cases'])==7
  for i,(cell,ref) in enumerate(zip(report['cases'],reference['cases'])):
   for k,v in ref.items():
    if k!='checked_parent_words':assert cell[k]==v,(name,i,k)
   samples=cell['samples'];assert len(samples)==27
   for round in range(9):
    row=samples[round*3:(round+1)*3];assert [s['order'] for s in row]==[0,1,2]
    order=[(round+j)%3 for j in range(3)];order=[2-v for v in order] if cohort else order
    assert [s['variant'] for s in row]==order
    for s in row:
     assert s['round']==round and all(type(s[k]) in (int,float) and math.isfinite(s[k]) and s[k]>0 for k in ['gpu_ms','wall_ms'])
     assert s['gpu_ms']<=s['wall_ms']+1
   median=[statistics.median(s['gpu_ms'] for s in samples if s['variant']==v) for v in range(3)]
   ratios=[next(s['gpu_ms'] for s in samples if s['round']==j and s['variant']==1)/next(s['gpu_ms'] for s in samples if s['round']==j and s['variant']==2) for j in range(9)]
   comparisons.append(dict(prototype=name,cohort=cohort,case=i,batch=cell['batch'],chunk=cell['chunk'],capacity=cell['capacity'],extent=cell['extent'],first=cell['first'],counts=cell['counts'],gated=cell['gated'],original4mps_gpu_ms=median[0],committed2mps_gpu_ms=median[1],candidate_gpu_ms=median[2],paired_committed_over_candidate=statistics.median(ratios),faster_pairs=sum(v>1 for v in ratios),measured_calls=27))
 return comparisons
comparisons=[c for name,item in sets.items() for c in verify(name,item)]
negative=[]
mutations={
 'precision':lambda d:d[4][0].update(relaxed_precision=True),
 'layout_capacity':lambda d:d[4][0]['strict_mpp_layout_capacities'].__setitem__(0,16),
 'missing_coordinate':lambda d:d[4][0]['strict_mpp_coordinate_parent_words'].__setitem__(16,0),
 'duplicate_coordinate':lambda d:d[4][0]['strict_mpp_coordinate_parent_words'].__setitem__(16,2),
 'failed_register_guard':lambda d:d[4][0]['register_layout_status'].__setitem__(0,2),
 'native_claim':lambda d:d[4][0].update(native_integration=True),
 'missing_sample':lambda d:d[4][0]['cases'][0]['samples'].pop(),
 'bad_gpu_time':lambda d:d[4][0]['cases'][0]['samples'][0].update(gpu_ms=-1),
 'wrong_order':lambda d:d[4][0]['cases'][0]['samples'][0].update(variant=2),
 'nonserial':lambda d:d[3]['runs'][1].update(started_ns=0),
}
for label,mutate in mutations.items():
 changed=copy.deepcopy(sets['register']);mutate(changed)
 try:verify('register',changed)
 except (AssertionError,KeyError,ValueError,TypeError):negative.append(label)
 else:raise AssertionError('Invalid evidence admitted: '+label)
result=dict(passed=True,scope=__doc__,comparisons=comparisons,measured_gpu_observations=756,preserved_mps_control_word_pairs=control_pairs,numeric_values_per_variant_per_prototype=28672,fixed_FP64_bound='1e-5+2e-4*abs(reference), unchanged',negative_metadata_controls_rejected=negative,failed_mapping_rejected_before_attention=True,strict_fp32_only=True,assessment='Register adaptation removes the broad long-context regression of the first public-coordinate/shared-softmax prototype, but B4 full2K paired ratios0.979/0.937 do not beat no-copyMPS; full4K1.030/1.019 is small. Ragged/cold component gains do not establish native benefit: this prototype MPS control executes full physical owners/rows, whereas native already skips empty owners. Do not promote a broad native selector or claim2x model/serving gain.',native_integration=False,serving_speed_claim=False,quality_gate_passed=False,goal_complete=False)
(r/'screens-audit1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
