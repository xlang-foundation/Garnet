"""Independent full-parent, source-composition, CPU-probability and clock audit."""
from pathlib import Path
import copy,hashlib,json,math,statistics
import numpy as np
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def need(v,why):
 if not v:raise AssertionError(why)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
P=read(R/'plan1.json');GUARD=np.uint32(0x1937abef)
def params(batch,layout):
 pad=8 if layout else 0
 return [batch,4,32,8,128,4096,4096+pad,1024+pad,1024+pad,8*4096*128+pad,8*4096*128+pad,4+pad,4+pad,4096+pad,32*4096+pad,4096+pad,0]
def cases(batch):
 out=[]
 def valid(fs,ms):return all(not(ms[r]&(1<<t)) or fs[r]+t<4096 for r in range(batch) for t in range(4))
 for f in P['firsts']:
  for m in P['all_uniform_masks']:
   if valid([f]*batch,[m]*batch):out.append(([f]*batch,[m]*batch))
 if batch==4:
  for fs in P['mixed_firsts']:
   for ms in P['mixed_masks']:
    if valid(fs,ms):out.append((fs,ms))
 return out
EXPECTED={(b,l,i):c for b in [1,4] for l in [0,1] for i,c in enumerate(cases(b))}
def check_input(a,b,l,i):
 p=params(b,l);off=16 if l else 0;stride=p[[6,7,9][i]];rows=b*(4 if i<2 else 1);width=[4096,1024,8*4096*128][i]
 need(a.dtype==np.uint32 and a.size==off+rows*stride+16,'Input length')
 need(np.all(a[:off]==GUARD) and np.all(a[off+rows*stride:]==GUARD),'Input parent guard')
 v=a[off:off+rows*stride].reshape(rows,stride)
 need(np.all(v[:,width:]==GUARD),'Input stride guard')
 x=v[:,:width].copy().view('<f4');need(np.all(np.isfinite(x)) and np.all(np.abs(x)<=0.5),'Input values')
def check_cell(c,outputs,inputs,cpu=None):
 b,l,index=c['batch'],c['layout'],c['index'];p=params(b,l);off=16 if l else 0
 need((b,l,index) in EXPECTED,'Unknown cell')
 fs,ms=EXPECTED[b,l,index]
 need(c['firsts']==fs and c['masks']==ms and c['params']==p and c['offset_words']==off,'Cell profile/masks')
 need(c['id']==f'b{b}-l{l}-c{index}','Cell identity')
 size=off+b*4*p[14]+16
 need(len(outputs)==3 and sorted(x['method'] for x in c['outputs'])==[0,1,2],'Method coverage')
 for manifest in c['outputs']:
  method=manifest['method'];a=outputs[method]
  need(manifest['file']==c['id']+f'-m{method}.u32','Output path')
  need(a.dtype==np.uint32 and a.size==size==c['parent_words'],'Output parent bound')
  need(digest(a)==manifest['sha256'],'Output hash binding')
  need(np.array_equal(a,outputs[0]),'All probability parent bits exact')
 a=outputs[0]
 need(np.all(a[:off]==GUARD) and np.all(a[off+b*4*p[14]:]==GUARD),'Output boundary guard')
 z=a[off:off+b*4*p[14]].reshape(b,4,p[14]);need(np.all(z[:,:,32*4096:]==GUARD),'Output stride guard')
 q=inputs[b,l,0][off:off+b*4*p[6]].reshape(b,4,p[6]).copy().view('<f4')
 key=inputs[b,l,1][off:off+b*4*p[7]].reshape(b,4,p[7]).copy().view('<f4')
 kc=inputs[b,l,2][off:off+b*p[9]].reshape(b,p[9])[:,:8*4096*128].reshape(b,8,4096,128).view('<f4')
 for row in range(b):
  for t in range(4):
   words=z[row,t,:32*4096].reshape(32,4096);values=words.view('<f4')
   if not(ms[row]&(1<<t)):need(np.all(words==0),'Inactive positive zero');continue
   visible=fs[row]+t+1
   need(np.all(words[:,visible:]==0),'Causal suffix positive zero')
   x=values[:,:visible].astype('float64');need(np.all(np.isfinite(x)) and np.all(x>=0) and np.all(x<=1),'Probability bounds')
   need(np.all(np.abs(x.sum(axis=1)-1)<2e-5),'Independent row sums')
   if cpu is not None:
    for head in [0,3,4,31]:
     cachekey=b,l,row,fs[row],t,head
     if cachekey not in cpu:
      vectors=np.concatenate((kc[row,head//4,:fs[row]],key[row,:t+1,(head//4)*128:(head//4+1)*128])).astype('float64')
      query=q[row,t,head*128:(head+1)*128].astype('float64')
      scores=(vectors*query).sum(axis=1)/math.sqrt(128);exp=np.exp(scores-scores.max());cpu[cachekey]=exp/exp.sum()
     oracle=cpu[cachekey];need(np.all(np.abs(x[head]-oracle)<=2e-6+4e-6*np.abs(oracle)),'Independent FP64 probability samples')
 return size*2

def check_timing(timing,cohort):
 need(len(timing)==612,'Timing coverage');seen=set();by={};lastH=0;lastG=0
 for t in timing:
  b,f,m,method,rd,order=[t[k] for k in ['batch','first','mask','method','round','order']]
  need(b in [1,4] and f in P['timed_firsts'] and m in [1,15] and method in [0,1,2] and 0<=rd<17 and order in [0,1,2],'Timing cell')
  key=b,f,m,method,rd;need(key not in seen,'Duplicate timing');seen.add(key);by[key]=t
  want=(order+(rd+4)%3)%3;want=want if cohort==1 else 2-want;need(method==want,'Timing method order')
  start,end=t['gpu_start_seconds'],t['gpu_end_seconds'];hs,he=t['host_start_ns'],t['host_end_ns']
  need(math.isfinite(start) and math.isfinite(end) and end>start>=lastG,'Serial GPU clocks')
  need(isinstance(hs,int) and isinstance(he,int) and he>hs>=lastH,'Serial host clocks');lastG=end;lastH=he
  need(abs(t['gpu_ms']-(end-start)*1000)<1e-9 and t['gpu_ms']>0,'GPU duration arithmetic')
  need(t['gpu_ms']<=(he-hs)/1e6+0.001,'GPU inside host wait')
 result=[]
 for b in [1,4]:
  for f in P['timed_firsts']:
   for m in [1,15]:
    med=[statistics.median(by[b,f,m,method,rd]['gpu_ms'] for rd in range(17)) for method in [0,1,2]]
    result.append(dict(cohort=cohort,batch=b,first=f,mask=m,original_gpu_ms=med[0],current_gpu_ms=med[1],gqa16_gpu_ms=med[2],paired_ratio_vs_current=statistics.median(by[b,f,m,1,rd]['gpu_ms']/by[b,f,m,2,rd]['gpu_ms'] for rd in range(17)),paired_ratio_vs_original=statistics.median(by[b,f,m,0,rd]['gpu_ms']/by[b,f,m,2,rd]['gpu_ms'] for rd in range(17))))
 return result

def main():
 pre=read(R/'preflight1.json');post=read(R/'postflight1.json')
 need(pre['passed'] and post['passed'] and not pre['hashes_during_collection'] and post['execution_bindings_unchanged'],'Pre/postflight')
 need(post['base_source_sha256']==pre['base_source_sha256']==P['base_source_sha256'],'Base source binding')
 for name,h in P['base_source_sha256'].items():need(sha(Path(name))==h,'Native source changed')
 for name,h in read(R/'execution-bindings1.json').items():need(sha(R/name)==h,'Executed file changed: '+name)
 headers=[Path(n) for n in P['base_source_sha256']]
 raws=[p.read_text().split('R"METAL(',1)[1].split(')METAL"',1)[0] for p in headers]
 control=''.join(raws);need((R/'control1.metal').read_text()==control,'Original native control composition')
 soft=raws[1][raws[1].index('kernel void verification_split_score_softmax'):].replace('verification_split_score_softmax','verification_gqa16_softmax').replace('scratch[s]=output[base+s]','scratch[s]=output[base+s]*rsqrt(float(p.dim))')
 need((R/'softmax1.metal').read_text()==soft,'Unchanged full softmax except dot scale at load')
 need((R/'candidate1.metal').read_text()==control+(R/'dots1.metal').read_text()+soft,'Candidate composition')
 paircount=0;metrics=[];cpu={};cross={};previousEnd=0;negative=[];sample=None
 for cohort in [1,2]:
  folder=R/f'run1-cohort{cohort}';report=read(folder/'result1.json');status=read(R/f'run1-cohort{cohort}-status.json');pid=read(R/f'run1-cohort{cohort}-pid.json')
  need(status['terminal'] and status['exit_code']==0 and status['pid']==pid['pid'],'Terminal process')
  need(pid['started_unix']>=previousEnd,'Serial fresh processes');previousEnd=pid['started_unix']+status['elapsed_seconds']
  need(report['completed'] and report['cohort']==cohort and report['compiler_options'] is None and report['all_pipelines_prepared_before_collection'],'Compiler contract')
  need(report['all_probability_parent_bits_exact'] and report['readonly_guards_preserved'] and report['no_runtime_integration'] and not report['serving_speed_claim'] and not report['goal_complete'],'Scope and exactness')
  need(report['close_allocated_bytes']<=report['baseline_allocated_bytes']+4194304,'Resource release')
  resources=report['resources'];need(len(resources)==10 and len({(r['batch'],r['name']) for r in resources})==10,'PSO coverage')
  for r in resources:need(r['simd_width']==32 and r['max_threads']>=128,'PSO bound')
  need(len(report['inputs'])==12,'Input coverage');inputs={}
  for item in report['inputs']:
   b,l,i=item['batch'],item['layout'],item['input'];name=f'b{b}-l{l}-input{i}.u32';need(item['file']==name,'Input path')
   file=folder/name;a=np.fromfile(file,dtype='<u4');after=np.fromfile(file.with_name(file.stem+'-after.u32'),dtype='<u4')
   need(sha(file)==item['sha256'] and a.size==item['words'],'Input hash');check_input(a,b,l,i);need(np.array_equal(a,after),'Read-only before/after');inputs[b,l,i]=a
  need(len(inputs)==12,'Unique inputs')
  expectedOrder=[(b,l,i) for b in ([1,4] if cohort==1 else [4,1]) for l in [0,1] for i in range(len(cases(b)))]
  need([(c['batch'],c['layout'],c['index']) for c in report['cases']]==expectedOrder,'Case coverage and order')
  count=0
  for c in report['cases']:
   outputs=[np.fromfile(folder/(c['id']+f'-m{m}.u32'),dtype='<u4') for m in [0,1,2]]
   count+=check_cell(c,outputs,inputs,cpu if cohort==1 else None)
   h=[digest(a) for a in outputs]
   if cohort==1:cross[c['id']]=h
   else:need(cross[c['id']]==h,'Fresh process exactness')
   if cohort==1 and c['batch']==4 and c['layout']==1 and c['firsts']==[511]*4 and c['masks']==[1]*4:sample=copy.deepcopy(c),[a.copy() for a in outputs],inputs
  need(count==report['parent_word_pairs'],'Parent pair count');paircount+=count;metrics+=check_timing(report['timing'],cohort)
 c,outputs,inputs=sample
 def reject(name,fn):
  try:fn()
  except (AssertionError,ValueError,KeyError,IndexError):negative.append(name);return
  raise AssertionError('Corruption accepted: '+name)
 def mutate(index,word,both=False):
  cc=copy.deepcopy(c);aa=[a.copy() for a in outputs]
  for m in ([0,1,2] if both else [2]):aa[m][index]=np.uint32(word)
  for o in cc['outputs']:o['sha256']=digest(aa[o['method']])
  check_cell(cc,aa,inputs,{})
 off=c['offset_words'];stride=c['params'][14]
 reject('raw_probability_bit',lambda:mutate(off,int(outputs[2][off])^1))
 reject('shared_wrong_probability_cpu',lambda:mutate(off,int(np.array([0.25],dtype='<f4').view('<u4')[0]),True))
 reject('inactive_positive_zero',lambda:mutate(off+stride,1,True))
 reject('causal_suffix',lambda:mutate(off+512,1,True))
 reject('stride_padding',lambda:mutate(off+32*4096,0,True))
 reject('parent_guard',lambda:mutate(0,0,True))
 cc=copy.deepcopy(c);cc['masks'][1]=15;reject('wrong_owner_mask',lambda:check_cell(cc,outputs,inputs,{}))
 cc=copy.deepcopy(c);cc['params'][9]+=8;reject('wrong_cache_stride',lambda:check_cell(cc,outputs,inputs,{}))
 tt=copy.deepcopy(read(R/'run1-cohort1/result1.json')['timing']);tt[0]['gpu_ms']*=0.5;reject('fake_gpu_duration',lambda:check_timing(tt,1))
 tt=copy.deepcopy(read(R/'run1-cohort1/result1.json')['timing']);tt[1]=copy.deepcopy(tt[0]);reject('duplicate_timing',lambda:check_timing(tt,1))
 tt=copy.deepcopy(read(R/'run1-cohort1/result1.json')['timing']);tt.pop();reject('missing_timing',lambda:check_timing(tt,1))
 a=inputs[4,1,0].copy();a[0]=0;reject('input_guard',lambda:check_input(a,4,1,0))
 result=dict(passed=True,scope='Isolated synthetic operator screen; no serving or native admission',cohorts=2,cases_per_cohort=len(EXPECTED),parent_word_pairs=paircount,independent_fp64_head_samples=len(cpu),all_probability_parent_bits_exact=True,fresh_process_outputs_exact=True,readonly_input_parents_exact=True,negative_controls_rejected=negative,timing=metrics,native_integration_qualified=False,serving_speed_claim=False,no_comparator_execution=True,goal_complete=False,files_sha256={str(p):sha(p) for p in [R/'audit1.py',R/'plan1.json',R/'preflight1.json',R/'postflight1.json',R/'execution-bindings1.json']+[R/f'run1-cohort{cohort}/result1.json' for cohort in [1,2]]})
 need(not (R/'audit1.json').exists(),'Never overwrite audit')
 (R/'audit1.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ['timing','files_sha256']}))
 for m in metrics:print(json.dumps(m))
if __name__=='__main__':main()
