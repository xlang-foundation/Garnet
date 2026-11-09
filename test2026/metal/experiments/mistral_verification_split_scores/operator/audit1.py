"""Independently audit retained probability parents, CPU samples and clocks."""
from pathlib import Path
import copy,hashlib,json,math,statistics,sys
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
GUARD=np.uint32(0x1937abef)
FIRSTS=[0,1,127,511,2048,4092,4093,4094,4095]
EXPECTED={(l,f,m) for l in [0,1] for f in FIRSTS for m in range(16) if all(not(m&(1<<t)) or f+t<4096 for t in range(4))}
def parameters(l):
 pad=8 if l else 0
 return [1,4,32,8,128,4096,4096+pad,1024+pad,1024+pad,8*4096*128+pad,8*4096*128+pad,4+pad,4+pad,4096+pad,32*4096+pad,4096+pad,0]
def check_input(a,l,i):
 p=parameters(l);off=16 if l else 0;stride=p[[6,7,9][i]];n=4 if i<2 else 1;width=[4096,1024,8*4096*128][i]
 need(a.dtype==np.uint32 and a.size==off+n*stride+16,'input length')
 need(np.all(a[:off]==GUARD) and np.all(a[off+n*stride:]==GUARD),'input boundary')
 for row in range(n):
  v=a[off+row*stride:off+row*stride+width].view(np.float32)
  need(np.all(np.isfinite(v)) and np.all(np.abs(v)<=0.5),'input values')
  need(np.all(a[off+row*stride+width:off+(row+1)*stride]==GUARD),'input padding')
def check_cell(c,a,b,inputs,cpu_cache=None):
 l,f,m=c['layout'],c['first'],c['mask'];p=parameters(l);off=16 if l else 0
 need((l,f,m) in EXPECTED,'case outside declared profile')
 need(c['id']==f'l{l}-f{f}-m{m}' and c['params']==p and c['offset_words']==off,'cell identity/params')
 size=off+4*p[14]+16
 need(a.dtype==np.uint32 and b.dtype==np.uint32 and a.size==b.size==c['parent_words']==size,'parent length')
 need(digest(a)==c['control_sha256'] and digest(b)==c['split_sha256'],'saved parent SHA')
 need(c['nonidentical_words']==0 and np.array_equal(a,b),'all parent words exact')
 need(np.all(a[:off]==GUARD) and np.all(a[off+4*p[14]:]==GUARD),'output boundary')
 for t in range(4):
  row=a[off+t*p[14]:off+(t+1)*p[14]]
  need(np.all(row[32*4096:]==GUARD),'output padding')
  z=row[:32*4096].reshape(32,4096);v=z.view(np.float32)
  if not(m&(1<<t)):need(np.all(z==0),'inactive positive zero');continue
  visible=f+t+1
  need(np.all(z[:,visible:]==0),'causal suffix positive zero')
  x=v[:,:visible].astype(np.float64)
  need(np.all(np.isfinite(x)) and np.all(x>=0) and np.all(x<=1),'probability bounds')
  need(np.all(np.abs(x.sum(axis=1)-1)<2e-5),'CPU row sums')
  if cpu_cache is not None:
   # Independent FP64 sampled heads straddle GQA boundaries; every declared
   # layout/position/active token is sampled, irrespective of mask reuse.
   q=inputs[l,0][off:].view(np.float32);key=inputs[l,1][off:].view(np.float32);kc=inputs[l,2][off:].view(np.float32)[:8*4096*128].reshape(8,4096,128)
   for h in [0,3,4,31]:
    k=(l,f,t,h)
    if k not in cpu_cache:
     vectors=np.concatenate((kc[h//4,:f,:],np.stack([key[s*p[7]+(h//4)*128:s*p[7]+(h//4+1)*128] for s in range(t+1)])),axis=0).astype(np.float64)
     query=q[t*p[6]+h*128:t*p[6]+(h+1)*128].astype(np.float64)
     scores=(vectors*query).sum(axis=1)/math.sqrt(128)
     exp=np.exp(scores-scores.max());cpu_cache[k]=exp/exp.sum()
    oracle=cpu_cache[k]
    need(np.all(np.abs(x[h]-oracle)<=2e-6+4e-6*np.abs(oracle)),'independent FP64 probability samples')
 return size

def check_timing(rows,cohort):
 need(len(rows)==252,'timing coverage')
 seen=set();last=0;groups={}
 for r in rows:
  f,m,k,round_,position=(r[n] for n in ['first','mask','method','round','position'])
  need(f in [508,2044,4092] and m in [1,15] and k in ['control','split'] and 0<=round_<21 and position in [0,1],'timing cell')
  ident=(f,m,k,round_);need(ident not in seen,'duplicate timing');seen.add(ident)
  need(k==('split' if (round_+4+position+cohort-1)&1 else 'control'),'predeclared alternating order')
  begin,end=r['gpu_start_seconds'],r['gpu_end_seconds']
  need(math.isfinite(begin) and math.isfinite(end) and end>begin,'GPU endpoints')
  need(math.isclose(r['gpu_ms'],(end-begin)*1000,rel_tol=1e-12,abs_tol=1e-12),'GPU duration arithmetic')
  hb,he=r['host_begin_ns'],r['host_end_ns'];need(isinstance(hb,int) and isinstance(he,int) and he>hb>=last,'host clock/serial order');last=he
  need((he-hb)/1e6>=r['gpu_ms']-0.001,'GPU duration within host wait')
  groups.setdefault((f,m,k),[]).append(r['gpu_ms'])
 summary=[]
 for f in [508,2044,4092]:
  for m in [1,15]:
   c,s=(statistics.median(groups[f,m,k]) for k in ['control','split'])
   summary.append(dict(cohort=cohort,first=f,mask=m,control_ms=c,split_ms=s,ratio=c/s))
 return summary

def main():
 pre=read(R/'preflight1.json');post=read(R/'postflight1.json');need(pre['passed'] and post['passed'] and not pre['hashes_during_collection'],'collection provenance')
 for n,h in pre['files_sha256'].items():need(sha(Path(n))==h,'source/preflight changed: '+n)
 native=Path(read(R/'plan1.json')['base_header']).read_text();raw=native.split('R"METAL(',1)[1].split(')METAL"',1)[0]
 prefix='#define GARNET_VERIFY_VALUE_VECTOR4 1\n#define GARNET_VERIFY_MISTRAL_LAYOUT 1\n'
 control=(R/'control1.metal').read_text();split=(R/'split1.metal').read_text()
 need(control==prefix+raw,'unmodified native control composition')
 softmax=raw.split('    float maximum=-INFINITY;',1)[1].split('\n}\nkernel void verification_attention_context',1)[0]
 need('    float maximum=-INFINITY;'+softmax in split,'softmax arithmetic copied literally')
 dot=raw.split('    for(ulong s=simd;s<visible;s+=4) {',1)[1].split('    threadgroup_barrier(mem_flags::mem_threadgroup);',1)[0]
 need('    for(ulong s=begin+simd;s<end;s+=4) {'+dot.replace('scratch[s]=score','output[base+s]=score') in split,'dot arithmetic copied literally')
 need((R/'candidate1.metal').read_text()==control+split,'candidate composition')
 summaries=[];parent=0;probwords=0;cpusamples=0;negative=[];first_data=None;cross={};previous_end=0
 for cohort in [1,2]:
  root=R/f'cohort{cohort}';d=read(root/'result1.json');state=read(R/f'cohort{cohort}-status1.json')
  need(state['status']=='terminal' and state['returncode']==0 and state['finished_ns']>state['started_ns']>=previous_end,'serial terminal processes');previous_end=state['finished_ns']
  need(d['completed'] and d['cohort']==cohort and d['compile_options'] is None and d['same_native_default_compiler'] and d['all_pipelines_prepared_before_collection'],'compiler/process completion')
  need(d['close_allocated_bytes']<=d['baseline_allocated_bytes']+4194304,'resources released')
  need({x['name'] for x in d['resources']}=={'verification_attention_scores','verification_split_score_dots','verification_split_score_softmax'},'prepared PSOs')
  for resource in d['resources']:need(resource['simd_width']==32 and resource['max_threads']>=128,'PSO limits')
  inputs={}
  need(len(d['inputs'])==6,'input manifest coverage')
  for manifest in d['inputs']:
   l,i=manifest['layout'],manifest['input'];file=root/manifest['file'];a=np.fromfile(file,dtype='<u4');after=np.fromfile(file.with_name(file.stem+'-after.u32'),dtype='<u4')
   need(a.size==manifest['words'] and sha(file)==manifest['sha256'],'input binding');check_input(a,l,i);need(np.array_equal(a,after),'readonly before-after');inputs[l,i]=a
  need({(c['layout'],c['first'],c['mask']) for c in d['cells']}==EXPECTED and len(d['cells'])==len(EXPECTED)==220,'complete unique mask/position/layout coverage')
  cache={};count=0
  for c in d['cells']:
   a=np.fromfile(root/(c['id']+'-control.u32'),dtype='<u4');b=np.fromfile(root/(c['id']+'-split.u32'),dtype='<u4');count+=check_cell(c,a,b,inputs,cache);probwords+=2*4*32*4096
   if cohort==1:cross[c['id']]=c['split_sha256']
   else:need(cross[c['id']]==c['split_sha256'],'fresh-process parent bits repeat')
   if cohort==1 and c['layout']==1 and c['first']==511 and c['mask']==1:first_data=(copy.deepcopy(c),a.copy(),b.copy(),inputs)
  need(count==d['parent_words_checked'],'parent count');parent+=count;cpusamples+=len(cache);summaries.extend(check_timing(d['timing'],cohort))
 need(read(R/'cohort1-status1.json')['pid']!=read(R/'cohort2-status1.json')['pid'],'fresh PIDs')
 c,a,b,inputs=first_data
 def reject(name,fn):
  try:fn()
  except (AssertionError,ValueError,KeyError,IndexError):negative.append(name);return
  raise AssertionError('negative control accepted: '+name)
 def mutate(index,word=None,both=False):
  cc=copy.deepcopy(c);aa=a.copy();bb=b.copy();bb[index]=np.uint32(word) if word is not None else bb[index]^np.uint32(1)
  if both:aa[index]=bb[index]
  cc['control_sha256']=digest(aa);cc['split_sha256']=digest(bb)
  return check_cell(cc,aa,bb,inputs,{})
 off=c['offset_words'];stride=c['params'][14]
 reject('probability_word',lambda:mutate(off))
 reject('shared_wrong_probability_cpu',lambda:mutate(off,int(np.array([0.25],np.float32).view(np.uint32)[0]),True))
 reject('inactive_row',lambda:mutate(off+stride,1,True))
 reject('causal_suffix',lambda:mutate(off+512,1,True))
 reject('stride_padding',lambda:mutate(off+32*4096,0,True))
 reject('boundary_guard',lambda:mutate(0,0,True))
 bad=copy.deepcopy(c);bad['params'][6]+=8;reject('profile_stride',lambda:check_cell(bad,a,b,inputs,{}))
 rows=copy.deepcopy(read(R/'cohort1/result1.json')['timing']);rows[0]['gpu_ms']*=0.5;reject('fake_gpu_duration',lambda:check_timing(rows,1))
 rows=copy.deepcopy(read(R/'cohort1/result1.json')['timing']);rows[1]['host_begin_ns']=rows[0]['host_begin_ns'];reject('overlapping_host_commands',lambda:check_timing(rows,1))
 rows=copy.deepcopy(read(R/'cohort1/result1.json')['timing']);rows.pop();reject('missing_timing_cell',lambda:check_timing(rows,1))
 z=inputs[1,0].copy();z[0]=0;reject('input_guard',lambda:check_input(z,1,0))
 gate=all(s['ratio']>=1.10 for s in summaries if s['first']==2044 and s['mask']==15) and all(s['ratio']>=1.0 for s in summaries if s['first']==508 and s['mask']==15)
 result=dict(passed=True,scope='Isolated operator evidence only; no native/model/serving admission',cases=440,parent_word_pairs=parent,probability_words_checked=probwords,independent_fp64_head_samples=cpusamples,all_probability_parent_words_exact=True,fresh_process_results_exact=True,readonly_before_after_exact=True,negative_controls_rejected=negative,timing=summaries,declared_component_gate_passed=gate,files_sha256={str(p):sha(p) for p in [R/'audit1.py',R/'preflight1.json',R/'postflight1.json',R/'cohort1/result1.json',R/'cohort2/result1.json']},native_integration_qualified=False,serving_speed_claim=False,no_comparator_execution=True,goal_complete=False)
 path=R/'audit1.json';need(not path.exists(),'audit overwrite');path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
