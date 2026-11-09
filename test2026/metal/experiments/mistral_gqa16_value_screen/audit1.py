"""Audit complete value parents, independent FP64 sums and four-method clocks."""
from pathlib import Path
import copy,hashlib,json,math,statistics
import numpy as np
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
P=read(R/'plan2.json');GUARD=np.uint32(0x1937abef);NAN=np.uint32(0x7fc01234)
def need(v,m):
 if not v:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
def params(b,l):
 pad=8 if l else 0
 return [b,4,32,8,128,4096,4096+pad,1024+pad,1024+pad,8*4096*128+pad,8*4096*128+pad,4+pad,4+pad,4096+pad,32*4096+pad,4096+pad,0]
def cases(b):
 out=[]
 def valid(fs,ms):return all(not(ms[r]&(1<<t)) or fs[r]+t<4096 for r in range(b) for t in range(4))
 for first in P['firsts']:
  for mask in P['all_uniform_masks']:
   if valid([first]*b,[mask]*b):out.append(([first]*b,[mask]*b))
 if b==4:
  for fs in P['mixed_firsts']:
   for ms in P['mixed_masks']:
    if valid(fs,ms):out.append((fs,ms))
 return out
EXPECTED={(b,l,i):c for b in [1,4] for l in [0,1] for i,c in enumerate(cases(b))}
def check_parent(a,off,rows,stride,width):
 need(a.dtype==np.uint32 and a.size==off+rows*stride+16,'Parent size')
 need(np.all(a[:off]==GUARD) and np.all(a[off+rows*stride:]==GUARD),'Parent boundary guard')
 view=a[off:off+rows*stride].reshape(rows,stride)
 need(np.all(view[:,width:]==GUARD),'Parent stride guard')
 return view[:,:width]
def check_input(a,b,l,i):
 p=params(b,l);off=16 if l else 0;stride=p[[6,7,9,8,10,13][i]];rows=b*(1 if i in [2,4] else 4);width=[4096,1024,8*4096*128,1024,8*4096*128,4096][i]
 values=check_parent(a,off,rows,stride,width).copy().view('<f4')
 need(np.all(np.isfinite(values)) and np.all(np.abs(values)<=0.5),'Input finite/range')
def check_cell(c,outputs,prob,inputs,cpu=None):
 b,l,i=c['batch'],c['layout'],c['index'];p=params(b,l);off=16 if l else 0
 need((b,l,i) in EXPECTED,'Unknown case');fs,ms=EXPECTED[b,l,i]
 need(c['id']==f'b{b}-l{l}-c{i}' and c['firsts']==fs and c['masks']==ms and c['params']==p and c['offset_words']==off,'Case identity/layout')
 need(len(outputs)==4 and sorted(o['method'] for o in c['outputs'])==[0,1,2,3],'Method coverage')
 for manifest in c['outputs']:
  method=manifest['method'];a=outputs[method];need(manifest['file']==c['id']+f'-m{method}.u32','Output path')
  need(a.size==c['parent_words'] and digest(a)==manifest['sha256'],'Output binding')
  check_parent(a,off,b*4,p[15],4096);need(np.array_equal(a,outputs[0]),'Context whole-parent bit equality')
 need(prob.size==c['probability_words'] and digest(prob)==c['probability_sha256'],'Probability binding')
 pv=check_parent(prob,off,b*4,p[14],32*4096).reshape(b,4,32,4096)
 context=outputs[0][off:off+b*4*p[15]].reshape(b,4,p[15])[:,:,:4096].reshape(b,4,32,128)
 value=inputs[b,l,3].copy();vv=value[off:off+b*4*p[8]].reshape(b,4,p[8])
 for row in range(b):
  last=max([t for t in range(4) if ms[row]&(1<<t)],default=-1)
  for t in range(last+1,4):vv[row,t,:1024]=NAN
 need(digest(value)==c['case_value_sha256'],'Inactive NaN tail mutation binding')
 v=vv[:,:,:1024].copy().reshape(b,4,8,128).view('<f4')
 vc=inputs[b,l,4][off:off+b*p[10]].reshape(b,p[10])[:,:8*4096*128].reshape(b,8,4096,128).view('<f4')
 for row in range(b):
  for t in range(4):
   if not(ms[row]&(1<<t)):
    need(np.all(pv[row,t]==0) and np.all(context[row,t]==0),'Inactive positive zero');continue
   visible=fs[row]+t+1;words=pv[row,t];x=words[:,:visible].copy().view('<f4').astype('float64')
   need(np.all(words[:,visible:]==0),'Probability causal suffix')
   need(np.all(np.isfinite(x)) and np.all(x>=0) and np.all(x<=1) and np.all(np.abs(x.sum(axis=1)-1)<2e-5),'Probability bounds/sum')
   y=context[row,t].copy().view('<f4');need(np.all(np.isfinite(y)),'Active context finite')
   if cpu is not None:
    for head in [0,3,4,31]:
     for col in [0,3,31,32,127]:
      key=b,l,row,fs[row],t,head,col
      if key not in cpu:
       values=np.concatenate((vc[row,head//4,:fs[row],col],v[row,:t+1,head//4,col])).astype('float64')
       need(len(values)==visible and np.all(np.isfinite(values)),'Oracle never consumes inactive tail NaNs')
       products=x[head]*values;cpu[key]=(float(products.sum()),float(2e-6+4e-6*np.abs(products).sum()))
      expected,bound=cpu[key];need(abs(float(y[head,col])-expected)<=bound,'Independent FP64 context sum')
 return (len(outputs)-1)*outputs[0].size

def check_timing(timing,cohort):
 need(len(timing)==816,'Timing coverage');by={};lastG=0;lastH=0
 for t in timing:
  b,f,m,method,rd,order=[t[k] for k in ['batch','first','mask','method','round','order']]
  need(b in [1,4] and f in P['timed_firsts'] and m in [1,15] and method in [0,1,2,3] and 0<=rd<17 and order in [0,1,2,3],'Timing key')
  key=b,f,m,method,rd;need(key not in by,'Duplicate timing');by[key]=t
  want=(order+(rd+4)%4)%4;want=want if cohort==1 else 3-want;need(method==want,'Method rotation/order')
  gs,ge,hs,he=[t[k] for k in ['gpu_start_seconds','gpu_end_seconds','host_start_ns','host_end_ns']]
  need(math.isfinite(gs) and math.isfinite(ge) and ge>gs>=lastG and isinstance(hs,int) and isinstance(he,int) and he>hs>=lastH,'Serial clock bounds');lastG=ge;lastH=he
  need(abs(t['gpu_ms']-(ge-gs)*1000)<1e-9 and 0<t['gpu_ms']<=(he-hs)/1e6+0.001,'GPU duration arithmetic')
 metrics=[]
 for b in [1,4]:
  for f in P['timed_firsts']:
   for mask in [1,15]:
    med=[statistics.median(by[b,f,mask,m,rd]['gpu_ms'] for rd in range(17)) for m in [0,1,2,3]]
    for method in [1,2,3]:
     metrics.append(dict(cohort=cohort,batch=b,first=f,mask=mask,method=P['methods'][method],current_gpu_ms=med[0],candidate_gpu_ms=med[method],plain_vector4_gpu_ms=med[3],paired_ratio_vs_current=statistics.median(by[b,f,mask,0,rd]['gpu_ms']/by[b,f,mask,method,rd]['gpu_ms'] for rd in range(17)),paired_ratio_vs_plain_vector4=statistics.median(by[b,f,mask,3,rd]['gpu_ms']/by[b,f,mask,method,rd]['gpu_ms'] for rd in range(17))))
 return metrics

def main():
 need(not (R/'audit1.json').exists(),'Audit overwrite');pre=read(R/'preflight2.json');post=read(R/'postflight2.json')
 need(pre['passed'] and post['passed'] and not pre['hashes_during_collection'] and post['execution_bindings_unchanged'],'Pre/postflight')
 need(post['base_source_sha256']==pre['base_source_sha256']==P['base_source_sha256'],'Native source binding')
 for name,h in P['base_source_sha256'].items():need(sha(Path(name))==h,'Native source changed')
 for name,h in read(R/'execution-bindings2.json').items():need(sha(R/name)==h,'Executed source changed: '+name)
 native=Path(next(iter(P['base_source_sha256']))).read_text().split('R"METAL(',1)[1].split(')METAL"',1)[0]
 need((R/'control1.metal').read_text()=='#define GARNET_VERIFY_VALUE_VECTOR4 1\n'+native,'Original control composition')
 need((R/'candidate1.metal').read_text()==(R/'control1.metal').read_text()+(R/'value1.metal').read_text(),'Candidate composition')
 need((R/'screen1.mm').read_text()==(R/'helpers1.inc').read_text()+(R/'main1.inc').read_text(),'Pilot harness composition')
 cpu={};pairs=0;metrics=[];cross={};sample=None;last=0;pids=set()
 for cohort in [1,2]:
  folder=R/f'run2-cohort{cohort}';d=read(folder/'result1.json');state=read(R/f'run2-cohort{cohort}-status.json');pid=read(R/f'run2-cohort{cohort}-pid.json')
  need(state['terminal'] and state['exit_code']==0 and state['pid']==pid['pid'] and state['pid'] not in pids and pid['started_unix']>=last,'Serial fresh terminal process');pids.add(state['pid']);last=pid['started_unix']+state['elapsed_seconds']
  need(d['completed'] and d['cohort']==cohort and d['compiler_options'] is None and d['all_pipelines_prepared_before_collection'] and d['all_context_parent_bits_exact'] and d['readonly_guards_preserved'],'Compiler/exactness contract')
  need(d['no_runtime_integration'] and not d['serving_speed_claim'] and not d['goal_complete'] and d['close_allocated_bytes']<=d['baseline_allocated_bytes']+4194304,'Scope/resource release')
  need(len(d['resources'])==10 and {(v['batch'],v['index']) for v in d['resources']}=={(b,i) for b in [1,4] for i in range(5)},'PSO coverage')
  for pso in d['resources']:
   i=pso['index'];need(pso['simd_width']==32 and pso['max_threads']>=128 and pso['dynamic_threadgroup_bytes']==[16384,0,6144,12288,0][i],'PSO limits/layout')
   want='verification_attention_scores' if i==0 else 'verification_attention_context_vector4' if i==4 or (i==1 and pso['batch']==1) else 'verification_attention_context' if i==1 else 'verification_gqa16_context';need(pso['function']==want,'PSO function routing')
  inputs={};need(len(d['inputs'])==24,'Input coverage')
  for manifest in d['inputs']:
   b,l,i=manifest['batch'],manifest['layout'],manifest['input'];name=f'b{b}-l{l}-input{i}.u32';need(manifest['file']==name,'Input path');file=folder/name
   a=np.fromfile(file,dtype='<u4');after=np.fromfile(file.with_name(file.stem+'-after.u32'),dtype='<u4')
   need(a.size==manifest['words'] and sha(file)==manifest['sha256'],'Input binding');check_input(a,b,l,i);need(np.array_equal(a,after),'Read-only before/after');inputs[b,l,i]=a
  need(len(inputs)==24,'Unique input coverage')
  expected=[(b,l,i) for b in ([1,4] if cohort==1 else [4,1]) for l in [0,1] for i in range(len(cases(b)))]
  need([(c['batch'],c['layout'],c['index']) for c in d['cases']]==expected,'Case/profile coverage')
  count=0
  for c in d['cases']:
   outputs=[np.fromfile(folder/(c['id']+f'-m{method}.u32'),dtype='<u4') for method in range(4)];prob=np.fromfile(folder/(c['id']+'-prob.u32'),dtype='<u4')
   count+=check_cell(c,outputs,prob,inputs,cpu if cohort==1 else None)
   fingerprints=[digest(a) for a in outputs]+[digest(prob),c['case_value_sha256']]
   if cohort==1:cross[c['id']]=fingerprints
   else:need(cross[c['id']]==fingerprints,'Fresh process repeatability')
   if cohort==1 and c['batch']==4 and c['layout']==1 and c['firsts']==[511]*4 and c['masks']==[1]*4:sample=copy.deepcopy(c),[a.copy() for a in outputs],prob.copy(),inputs
  need(count==d['parent_word_pairs'],'Word-pair count');pairs+=count;metrics+=check_timing(d['timing'],cohort)
 c,outputs,prob,inputs=sample;negative=[]
 def reject(name,fn):
  try:fn()
  except (ValueError,TypeError,KeyError,IndexError):negative.append(name);return
  raise AssertionError('Corruption admitted: '+name)
 def mutate(index,word,all_methods=False):
  cc=copy.deepcopy(c);aa=[a.copy() for a in outputs]
  for method in (range(4) if all_methods else [1]):aa[method][index]=np.uint32(word)
  for manifest in cc['outputs']:manifest['sha256']=digest(aa[manifest['method']])
  check_cell(cc,aa,prob,inputs,{})
 off=c['offset_words'];stride=c['params'][15]
 reject('context_bit',lambda:mutate(off,int(outputs[1][off])^1))
 reject('shared_wrong_context_fp64',lambda:mutate(off,int(np.array([0.25],dtype='<f4').view('<u4')[0]),True))
 reject('inactive_positive_zero',lambda:mutate(off+stride,1,True))
 reject('stride_guard',lambda:mutate(off+4096,0,True))
 reject('boundary_guard',lambda:mutate(0,0,True))
 cc=copy.deepcopy(c);cc['case_value_sha256']='0'*64;reject('inactive_nan_tail_binding',lambda:check_cell(cc,outputs,prob,inputs,{}))
 cc=copy.deepcopy(c);cc['masks'][1]=15;reject('wrong_owner_mask',lambda:check_cell(cc,outputs,prob,inputs,{}))
 cc=copy.deepcopy(c);cc['params'][10]+=8;reject('wrong_cache_stride',lambda:check_cell(cc,outputs,prob,inputs,{}))
 tt=copy.deepcopy(read(R/'run2-cohort1/result1.json')['timing']);tt[0]['gpu_ms']/=2;reject('fake_gpu_duration',lambda:check_timing(tt,1))
 tt=copy.deepcopy(read(R/'run2-cohort1/result1.json')['timing']);tt[1]=copy.deepcopy(tt[0]);reject('duplicate_timing',lambda:check_timing(tt,1))
 tt=copy.deepcopy(read(R/'run2-cohort1/result1.json')['timing']);tt.pop();reject('missing_timing',lambda:check_timing(tt,1))
 bad=inputs[4,1,4].copy();bad[0]=0;reject('input_boundary_guard',lambda:check_input(bad,4,1,4))
 result=dict(passed=True,scope=__doc__,cohorts=2,cases_per_cohort=len(EXPECTED),methods=4,parent_word_pairs=pairs,independent_fp64_context_samples=len(cpu),all_context_parent_bits_exact=True,probability_parents_preserved=True,inactive_nan_tail_bound=True,fresh_process_results_exact=True,negative_controls_rejected=negative,timing=metrics,native_integration_qualified=False,serving_speed_claim=False,no_comparator_execution=True,goal_complete=False,files_sha256={str(p):sha(p) for p in [R/'audit1.py',R/'preflight2.json',R/'postflight2.json',R/'execution-bindings2.json',R/'plan2.json']+[R/f'run2-cohort{c}/result1.json' for c in [1,2]]})
 (R/'audit1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['timing','files_sha256']}))
 for m in metrics:
  if m['first']==2044:print(json.dumps(m))
if __name__=='__main__':main()
