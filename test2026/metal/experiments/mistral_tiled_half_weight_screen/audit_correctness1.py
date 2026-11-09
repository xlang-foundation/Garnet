"""Independent cache half bits, full-parent results, writer maps and FP64 checks."""
from pathlib import Path
import copy,hashlib,json,itertools,sys
import numpy as np
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent;T=ROOT.parent/'mistral-fused-mlp-trained1'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def need(v,message):
    if not v:raise ValueError(message)
need(read(ROOT/'correctness1-status.json')['returncode']==0 and read(ROOT/'correctness1-postflight.json')['passed'],'incomplete run')
for name,h in read(ROOT/'correctness1-preflight.json')['files_sha256'].items():need(sha(Path(name))==h,'changed input '+name)
shapes={'gate':(4096,14336,14336,False),'down':(14336,4096,4096,True),'gate-prefix-proxy':(4096,4096,14336,False)}
row_counts=[127,128,129,255,256,257,383,384,511,512];layers=[0,15,31];tiles=[128,256]
variants=[(-1,False)]+list(itertools.product(range(4),[False,True]))
expected_keys=set(itertools.product(layers,shapes,row_counts,tiles,variants))
cache={};writer_cache={};gold={};assets={};inputs={};raw_files={}
def source(layer,name):
    key=(layer,name)
    if key not in gold:
        k,n,full,down=shapes[name]
        gold[key]=np.memmap(T/'captures1'/('layer-'+str(layer))/('down.f32' if down else 'gate.f32'),dtype='<u4',mode='r',shape=(512,full))
    return gold[key]
def expected(layer,name,rows,counted):
    k,n,full,down=shapes[name];key=(layer,name,rows)
    ids=(511-np.arange(rows,dtype=np.int64)*257)%512
    if key not in cache:
        out=np.full(512*n+32,0x1937abef,dtype='<u4');out[16:-16].reshape(512,n)[ids]=source(layer,name)[ids,:n]
        cache[key]=hashlib.sha256(out.tobytes()).hexdigest()
    wk=(n,rows,counted)
    if wk not in writer_cache:
        counts=np.full(512*n+32,0x1937abef,dtype='<u4');counts[16:-16]=0
        if counted:counts[16:-16].reshape(512,n)[ids]=1
        writer_cache[wk]=hashlib.sha256(counts.tobytes()).hexdigest()
    return cache[key],writer_cache[wk]
def fp64(p):
    layer,name,row,col=p['layer'],p['shape'],p['row'],p['column'];k,n,full,down=shapes[name];key=(layer,name)
    if key not in assets:
        folder=T/'assets1'/('layer-'+str(layer));prefix='down' if down else 'gate'
        assets[key]=(np.memmap(folder/(prefix+'-weight.u32'),dtype='<u4',mode='r',shape=(full,k//8)),np.memmap(folder/(prefix+'-scales.f16'),dtype='<f2',mode='r',shape=(full,k//64)),np.memmap(folder/(prefix+'-biases.f16'),dtype='<f2',mode='r',shape=(full,k//64)))
        inputs[key]=np.memmap(T/'captures1'/('layer-'+str(layer))/('silu.f32' if down else 'norm-input.f32'),dtype='<f4',mode='r',shape=(512,k))
    w,s,b=assets[key];ks=np.arange(k);codes=((w[col,ks//8]>>((ks%8)*4))&15).astype(np.float64)
    dequant=(codes*s[col,ks//64].astype(np.float64)+b[col,ks//64].astype(np.float64)).astype(np.float32).astype(np.float16).astype(np.float64)
    x=inputs[key][row].astype(np.float16).astype(np.float64);terms=x*dequant;value=float(np.sum(terms,dtype=np.float64));sumabs=float(np.sum(np.abs(terms),dtype=np.float64))
    actual=float(source(layer,name)[row,col:col+1].view('<f4')[0]);fraction=abs(actual-value)/(2e-6+4e-6*sumabs)
    need(np.isfinite(actual) and actual==p['actual'] and np.isclose(value,p['sum'],rtol=2e-12,atol=2e-12) and np.isclose(sumabs,p['sum_abs'],rtol=2e-12,atol=2e-12),'independent FP64 recomputation differs')
    need(fraction<=1 and np.isclose(fraction,p['fraction'],rtol=1e-7,atol=1e-12),'fixed FP64 gate failed');return fraction
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
    h=hashlib.sha256();guard=np.full(16,0x1937abef,dtype='<u4').tobytes();h.update(guard);chunk=np.ones(1<<20,dtype='<u4').tobytes();remaining=elements
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
        need(e['counted'] and e['guards_exact'],'preparation ownership/guards not checked')
        need(np.isfinite(e['gpu_ms']) and e['gpu_ms']>0 and np.isclose(e['gpu_ms'],(e['gpu_end_seconds']-e['gpu_start_seconds'])*1000,rtol=1e-12,atol=1e-9) and e['allocation_encode_wait_wall_ms']>=e['gpu_ms'],'invalid preparation clock')
        expected=prepared_reference(layer,name,layout)
        need(e['half_sha256']==expected,'prepared half values/padding/guards differ from independent Q4 decode')
        need(e['writers_sha256']==preparation_writer_sha(elements),'prepared weight missing/duplicate writer')
        need(('raw_half' in e)==(layer==0),'wrong retained prepared-array coverage')
        if check_raw and 'raw_half' in e:
            p=ROOT/'correctness1'/e['raw_half'];need(p.stat().st_size==elements*2+128 and sha(p)==expected,'raw prepared half differs');weight_raw_files[e['raw_half']]=expected
        half_words+=elements+64;writer_words+=elements+32
    return dict(prepared_half_words=half_words,preparation_writer_words=writer_words,prepared_arrays=18,raw_prepared_arrays=len(weight_raw_files),all_prepared_bits_independently_reconstructed=True,physical_allocation_peak_proven=False)

def verify(d,check_raw=False):
    need(d['completed'] and not d['timing'] and d['cases_count']==1620 and d['samples_count']==0,'wrong/incomplete mode')
    keys={(e['layer'],e['shape'],e['rows'],e['tile'],(e['mode'],e['counted'])) for e in d['cases']}
    need(len(d['cases'])==len(keys)==1620 and keys==expected_keys,'missing or duplicate factor coverage')
    total=0
    for e in d['cases']:
        layer,name,rows,counted=e['layer'],e['shape'],e['rows'],e['counted'];k,n,full,down=shapes[name]
        out,counts=expected(layer,name,rows,counted)
        need(e['output_sha256']==e['expected_output_sha256']==out,'full trained output differs')
        need(e['writers_sha256']==e['expected_writers_sha256']==counts,'full writer/guard map differs')
        need(e['complete_parent_words']==512*n+32 and e['guards_exact'],'false coverage/guards');total+=512*n+32
        for field,digest in [('raw_output',out),('raw_writers',counts)]:
            if field in e and check_raw:
                path=ROOT/'correctness1'/e[field];need(path.stat().st_size==(512*n+32)*4 and sha(path)==digest,'retained raw array differs')
                raw_files[e[field]]=digest
    need(total==d['complete_parent_word_comparisons']==d['writer_word_comparisons'],'false total comparisons')
    verify_weights(d,check_raw)
    resources=d['pipeline_resources'];need(len(resources)==54,'missing pipeline resources')
    need(all(e['shared_bytes']==17408 and e['maximum_threads']>=256 and e['simd_width']==32 for e in resources),'different resource/ABI profile')
    need(len(d['fp64'])==243,'missing FP64 samples')
    sample_keys={(p['layer'],p['shape'],p['row'],p['column']) for p in d['fp64']}
    wanted={(l,name,r,c) for l in layers for name,(_,n,_,_) in shapes.items() for r in [0,256,511] for c in [0,1,15,16,31,32,n//2,n-2,n-1]}
    need(sample_keys==wanted,'wrong FP64 coverage')
    maximum=max(fp64(p) for p in d['fp64']);return dict(complete_parent_words=total,writer_words=total,maximum_fp64_fraction=maximum)
d=read(ROOT/'correctness1/result1.json');result=verify(d,True);rejected=[]
mutations={
    'wrong_prepared_half':lambda x:x['prepared_weights'][0].update(half_sha256='0'*64),
    'wrong_preparation_writer':lambda x:x['prepared_weights'][0].update(writers_sha256='0'*64),
    'wrong_cache_size':lambda x:x['prepared_weights'][0].update(payload_bytes=0),
    'missing_prepared_weight':lambda x:x['prepared_weights'].pop(),
    'wrong_output':lambda x:x['cases'][0].update(output_sha256='0'*64),
    'wrong_expected_and_actual_output':lambda x:x['cases'][0].update(output_sha256='0'*64,expected_output_sha256='0'*64),
    'wrong_writers':lambda x:x['cases'][1].update(writers_sha256='0'*64),
    'missing_case':lambda x:x['cases'].pop(),
    'duplicate_case':lambda x:x['cases'].__setitem__(1,copy.deepcopy(x['cases'][0])),
    'wrong_tile':lambda x:x['cases'][0].update(tile=64),
    'wrong_guard_flag':lambda x:x['cases'][0].update(guards_exact=False),
    'wrong_total':lambda x:x.update(complete_parent_word_comparisons=0),
    'wrong_FP64':lambda x:x['fp64'][0].update(actual=1e10),
    'wrong_shared_size':lambda x:x['pipeline_resources'][0].update(shared_bytes=0),
}
for name,mutate in mutations.items():
    bad=copy.deepcopy(d);mutate(bad)
    try:verify(bad)
    except (ValueError,KeyError,TypeError,IndexError):rejected.append(name)
    else:raise ValueError('Corruption admitted: '+name)
result.update(prepared_weights=verify_weights(d,True),raw_prepared_arrays=weight_raw_files)
result.update(passed=True,scope=__doc__,cases=1620,layers=layers,shapes=list(shapes),independent_fp64_samples=243,
    all_complete_parent_and_writer_hashes_independently_reconstructed=True,retained_raw_arrays=raw_files,negative_controls_rejected=rejected,
    result_sha256=sha(ROOT/'correctness1/result1.json'),preflight_sha256=sha(ROOT/'correctness1-preflight.json'),binary_sha256=sha(ROOT/'screen1'),combined_sha256=sha(ROOT/'combined1.metal'),
    proxy_is_actual_QO=False,native_runtime_changed=False,serving_speed_claim=False,quality_qualified=False,goal_complete=False)
dest=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'correctness-audit1.json';need(not dest.exists(),'Never overwrite audit');dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='retained_raw_arrays'},indent=2))
