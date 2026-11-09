"""Verify the uninstrumented preparation produces the same exact half bits."""
from pathlib import Path
import hashlib,json,itertools,sys
import numpy as np
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent;T=ROOT.parent/'mistral-fused-mlp-trained1'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def need(v,m):
 if not v:raise ValueError(m)
shapes={'gate':(4096,14336,14336,False),'down':(14336,4096,4096,True),'gate-prefix-proxy':(4096,4096,14336,False)}
layers=[0,15,31]
weight_hashes={};weight_raw_files={};weight_writers={}
def prepared_reference(layer,name,layout):
    key=(layer,name,layout)
    if key in weight_hashes:return weight_hashes[key]
    k,n,full,down=shapes[name];folder=T/'assets1'/('layer-'+str(layer));prefix='down' if down else 'gate'
    w=np.memmap(folder/(prefix+'-weight.u32'),dtype='<u4',mode='r',shape=(full,k//8));s=np.memmap(folder/(prefix+'-scales.f16'),dtype='<f2',mode='r',shape=(full,k//64));b=np.memmap(folder/(prefix+'-biases.f16'),dtype='<f2',mode='r',shape=(full,k//64))
    ks=np.arange(k);h=hashlib.sha256();guard=np.full(32,0x35a5,dtype='<u2').tobytes();h.update(guard)
    for start in range(0,n,64):
        codes=((w[start:start+64,ks//8]>>((ks%8)*4))&15).astype(np.float64)
        values=(codes*s[start:start+64,ks//64].astype(np.float64)+b[start:start+64,ks//64].astype(np.float64)).astype(np.float32).astype('<f2').view('<u2')
        if layout:
            padded=np.zeros((k//128,64,136),dtype='<u2');padded[:,:,:128]=values.reshape(64,k//128,128).transpose(1,0,2);h.update(padded.tobytes())
        else:h.update(values.tobytes())
    h.update(guard);weight_hashes[key]=h.hexdigest();return weight_hashes[key]
def preparation_writer_sha(elements):
    if elements in weight_writers:return weight_writers[elements]
    h=hashlib.sha256();guard=np.full(16,0x1937abef,dtype='<u4').tobytes();h.update(guard);chunk=np.zeros(1<<20,dtype='<u4').tobytes();remaining=elements
    while remaining:
        count=min(remaining,1<<20);h.update(chunk[:count*4]);remaining-=count
    h.update(guard);weight_writers[elements]=h.hexdigest();return weight_writers[elements]
def verify_weights(d,check_raw=False):
    items=d['prepared_weights'];wanted=set(itertools.product(layers,shapes,[0,1]));keys={(e['layer'],e['shape'],e['layout']) for e in items}
    need(len(items)==len(keys)==18 and keys==wanted,'missing/duplicate prepared weights')
    half_words=writer_words=0
    for e in items:
        layer,name,layout=e['layer'],e['shape'],e['layout'];k,n,full,down=shapes[name];elements=n*k if layout==0 else n*k//128*136
        need(e['inner']==k and e['columns']==n and e['elements']==elements and e['payload_bytes']==elements*2 and e['allocation_bytes']==elements*2+128,'wrong requested cache size/layout')
        need(not e['counted'] and e['guards_exact'],'preparation ownership/guards not checked')
        need(np.isfinite(e['gpu_ms']) and e['gpu_ms']>0 and np.isclose(e['gpu_ms'],(e['gpu_end_seconds']-e['gpu_start_seconds'])*1000,rtol=1e-12,atol=1e-9) and e['allocation_encode_wait_wall_ms']>=e['gpu_ms'],'invalid preparation clock')
        expected=prepared_reference(layer,name,layout)
        need(e['half_sha256']==expected,'prepared half values/padding/guards differ from independent Q4 decode')
        need(e['writers_sha256']==preparation_writer_sha(elements),'prepared weight missing/duplicate writer')
        need(('raw_half' in e)==(layer==0),'wrong retained prepared-array coverage')
        if check_raw and 'raw_half' in e:
            p=ROOT/'normal-prepare1'/e['raw_half'];need(p.stat().st_size==elements*2+128 and sha(p)==expected,'raw prepared half differs');weight_raw_files[e['raw_half']]=expected
        half_words+=elements+64;writer_words+=elements+32
    return dict(prepared_half_words=half_words,preparation_writer_words=writer_words,prepared_arrays=18,raw_prepared_arrays=len(weight_raw_files),all_prepared_bits_independently_reconstructed=True,physical_allocation_peak_proven=False)

need(read(ROOT/'normal-prepare1-status.json')['returncode']==0,'normal preparation failed')
for name,h in read(ROOT/'correctness1-preflight.json')['files_sha256'].items():need(sha(Path(name))==h,'changed original input '+name)
derived=read(ROOT/'normal-prepare-derivation1.json');text=(ROOT/'screen1.mm').read_text()
need(sha(ROOT/'screen1.mm')==derived['original_host_sha256'],'changed qualified host')
for change in derived['changes']:
 need(text.count(change['old'])==1,'ambiguous derivation');text=text.replace(change['old'],change['new'])
need(text==(ROOT/'normal_prepare1.mm').read_text(),'normal host contains an unrecorded change')
need(sha(ROOT/'combined1.metal')==read(ROOT/'correctness-audit1.json')['combined_sha256'],'GPU shader changed')
d=read(ROOT/'normal-prepare1/result1.json');need(d['completed'] and d['preparation_only'] and not d['timing'] and not d['exact_native_captured_controls'] and d['cases_count']==0 and d['samples_count']==0,'incorrect normal preparation scope')
reference=read(ROOT/'correctness1/result1.json');need(d['pipeline_resources']==reference['pipeline_resources'],'different pipeline resources')
result=verify_weights(d,True)
prior={(e['layer'],e['shape'],e['layout']):e['half_sha256'] for e in reference['prepared_weights']}
for e in d['prepared_weights']:
 need(e['half_sha256']==prior[(e['layer'],e['shape'],e['layout'])],'observer altered half bits')
 need(e['allocated_size_bytes']>=e['allocation_bytes'],'invalid allocated/requested cache bytes')
result.update(passed=True,scope=__doc__,normal_and_counted_half_bits_exact=True,qualified_shader_sha256=sha(ROOT/'combined1.metal'),result_sha256=sha(ROOT/'normal-prepare1/result1.json'),normal_host_sha256=sha(ROOT/'normal_prepare1.mm'),normal_binary_sha256=sha(ROOT/'normal_prepare1'),raw_prepared_arrays=weight_raw_files,native_runtime_changed=False,serving_speed_claim=False,goal_complete=False)
dest=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'normal-preparation-audit1.json';need(not dest.exists(),'Never overwrite audit');dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='raw_prepared_arrays'},indent=2))
