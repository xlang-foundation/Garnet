"""Independent saved-data packed-descriptor qualification; no inference."""
import sys
from pathlib import Path
import copy,hashlib,json,math
import numpy as np
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
b=load(root/'bindings1.json');h=load(root/'host-binding1.json');r=load(root/'correctness1/result1.json')
assert load(root/'correctness1-status.json')['returncode']==0
assert sha(root/'host1.mm')==h['source_sha256'] and sha(root/'host1')==h['binary_sha256']
assert sha(root/'combined1.metal')==b['shader_sha256']==h['shader_sha256']
for name,value in b['assets_sha256'].items():assert sha(Path(name))==value,name
shapes=[(14336,4096),(4096,14336),(4096,4096)]
cells=[(n,False,False) for n in [128,129,255,256,257,383,384,511,512]]+[(512,False,True)]+[(n,True,False) for n in [128,256,512]]
expected=[(fi,count,dense,reverse,v) for fi in range(3) for count,dense,reverse in cells for v in range(3)]
def verify_metadata(report):
    assert report['completed'] and report['mode']=='correctness' and report['order']=='forward'
    assert report['math_mode']=='Safe' and report['relaxed_descriptor_only'] and not report['timing_admitted']
    assert not report['native_integration'] and not report['quality_qualified'] and not report['serving_speed_claim']
    assert report['FP64_values']==2340 and 0<=report['maximum_FP64_fraction']<=1
    assert [(g['kernel'],g['values']) for g in report['guards']]==[
        ('packed_affine_layout_guard',[0]*32),('relaxed_layout_guard',[0]*32),('fault_layout_guard',[1]*32),('descriptor_coordinates',[1]*1280+[0])]
    assert len(report['cases'])==len(expected)==117
    for c,(fi,count,dense,reverse,v) in zip(report['cases'],expected):
        assert (c['fixture'],c['count'],c['dense'],c['reverse'],c['variant'])==(fi,count,dense,reverse,v)
        n,k=shapes[fi];tile=256 if count>=256 and (n==14336 or (k==14336 and count>=384)) else 128
        assert c['tile']==tile and math.isfinite(c['gpu_ms_including_gather']) and c['gpu_ms_including_gather']>0
        label=f'case{fi}-n{count}'+('-dense' if dense else '-indexed')+('-reverse' if reverse else '')+f'-v{v}.f32'
        assert c['output_file']==(label if v>0 else '')
verify_metadata(r)
def exact(a,b):assert a.shape==b.shape and np.array_equal(a.view('<u4'),b.view('<u4')),'Complete output/guard bits differ'
native_gold=[];inputs=[];weights=[];scales=[];biases=[]
for folder,(n,k) in zip(['capture-up2','capture-down1','capture-qo1'],shapes):
    p=Path(b['captures'])/folder
    native_gold.append(np.memmap(p/'reference-parent.f32',dtype='<f4',mode='r').reshape(514,n))
    inputs.append(np.memmap(p/'input-parent.f32',dtype='<f4',mode='r').reshape(514,k))
    weights.append(np.memmap(p/'weight.u32',dtype='<u4',mode='r').reshape(n,k//8))
    scales.append(np.memmap(p/'scales.f16',dtype='<f2',mode='r').reshape(n,k//64))
    biases.append(np.memmap(p/'biases.f16',dtype='<f2',mode='r').reshape(n,k//64))
fp64=0;fraction=0.;arithmetic_words=0;writer_words=0;files={}
for c in r['cases']:
    fi,count,dense,reverse,v=[c[k] for k in ['fixture','count','dense','reverse','variant']]
    if not v:continue
    n,k=shapes[fi];ids=np.arange(count) if dense else (np.arange(count)*127+5)%512
    if reverse:ids=ids[::-1]
    active=np.zeros(512,dtype=bool);active[ids]=True
    path=root/'correctness1'/c['output_file'];actual=np.memmap(path,dtype='<f4',mode='r').reshape(514,n)
    reference=native_gold[fi].copy()
    if v==2:
        reference[1:-1].view('<u4')[:]=active[:,None].astype('<u4');writer_words+=actual.size
    else:
        reference[1:-1][~active]=0;arithmetic_words+=actual.size
    exact(actual,reference);files[path.name]=sha(path)
    if v==1:
        j=np.arange(k)
        for pos in sorted({0,count//2,count-1}):
            row=int(ids[pos]);x=inputs[fi][row+1].astype(np.float16).astype(np.float64)
            for col in sorted({0,1,15,16,31,32,63,64,n//2,n-1}):
                codes=(weights[fi][col,j//8]>>((j%8)*4))&15
                decoded=(codes.astype(np.float64)*scales[fi][col,j//64].astype(np.float64)+biases[fi][col,j//64].astype(np.float64)).astype(np.float32).astype(np.float16).astype(np.float64)
                products=x*decoded;total=float(products.sum());absolute=float(np.abs(products).sum())
                for data in [actual,native_gold[fi]]:
                    value=abs(float(data[row+1,col])-total)/(2e-6+4e-6*absolute);assert math.isfinite(value) and value<=1
                    fraction=max(fraction,value);fp64+=1
assert fp64==2340
negative=[]
mutations={
    'missing_cell':lambda x:x['cases'].pop(),
    'wrong_geometry':lambda x:x['cases'][0].update(fixture=2),
    'wrong_count':lambda x:x['cases'][0].update(count=127),
    'wrong_tile':lambda x:x['cases'][0].update(tile=256),
    'wrong_variant':lambda x:x['cases'][0].update(variant=1),
    'bad_layout':lambda x:x['guards'][1]['values'].__setitem__(0,1),
    'missing_coordinate':lambda x:x['guards'][3]['values'].__setitem__(0,0),
    'invalid_coordinate':lambda x:x['guards'][3]['values'].__setitem__(-1,1),
    'nonfinite_time':lambda x:x['cases'][0].update(gpu_ms_including_gather=float('nan')),
    'premature_timing_admission':lambda x:x.update(timing_admitted=True),
}
for name,mutate in mutations.items():
    bad=copy.deepcopy(r);mutate(bad)
    try:verify_metadata(bad)
    except (AssertionError,KeyError,TypeError,ValueError):negative.append(name)
    else:raise AssertionError('Invalid metadata admitted: '+name)
sample=np.array([1.,2.,3.],dtype='<f4')
for name,index in [('output_bit',1),('guard_bit',0),('writer_count',2)]:
    bad=sample.copy();bad.view('<u4')[index]^=1
    try:exact(bad,sample)
    except AssertionError:negative.append(name)
    else:raise AssertionError('Invalid saved payload admitted')
result=dict(passed=True,scope=__doc__,cells=39,arithmetic_parent_words=arithmetic_words,writer_parent_words=writer_words,FP64_values=fp64,maximum_original_FP64_fraction=fraction,
    full_candidate_bits_equal_frozen_native=True,one_writer_per_selected_output=True,cooperative_coordinates_exact=1280,
    negative_controls_rejected=negative,source_binding_sha256=sha(root/'bindings1.json'),host_binding_sha256=sha(root/'host-binding1.json'),result_sha256=sha(root/'correctness1/result1.json'),payload_sha256=files,
    native_integration=False,quality_qualified=False,serving_speed_claim=False,goal_complete=False)
path=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else root/'correctness-audit1.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='payload_sha256'},indent=2))
