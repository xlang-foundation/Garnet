"""Independently compare every retained full-MLP output with native captures."""
from pathlib import Path
import copy,hashlib,json,sys
import numpy as np
ROOT=Path(sys.argv[1]).resolve()
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def require(v,m):
    if not v:raise ValueError(m)
def check(d):
    require(d['completed'] and d['candidate_exact'] and d['trained_activations'] and d['original_main_options']=='nil/default, matching native compiled profile','incomplete/wrong native policy replay')
    require([x['layer'] for x in d['layers']]==[0,15,31],'missing/wrong layer')
    for layer in d['layers']:
        require(layer['rows']==512 and layer['native_control_mismatches']==[0]*4,'native control differs')
        require(layer['prior_explicit31_silu_word_mismatches']==layer['prior_explicit31_half_word_mismatches']==0,'prior surrogate differs numerically on captured data')
        require([v['variant'] for v in layer['variants']]==[0,1,2],'wrong fused variant coverage')
        for v in layer['variants']:
            require(v['guards_exact'] and v['writers_exact'] and v['half_mismatches']==v['down_mismatches']==0,'fused output/ownership mismatch')
            if v['variant']==0:require(v['gate_mismatches']==v['up_mismatches']==0,'fused shadow mismatch')
        require(len(layer['fp64'])==81,'missing independent-reference sample coverage')
        for p in layer['fp64']:require(np.isfinite(p['actual']) and abs(p['actual']-p['sum'])<=2e-6+4e-6*p['sum_abs'],'fixed native FP64 gate failed')
def main():
    require(read(ROOT/'replay1-status.json')['returncode']==0 and read(ROOT/'replay1-postflight.json')['passed'],'failed process/postflight')
    for path,digest in read(ROOT/'replay1-command.json')['files_sha256'].items():require(sha(Path(path))==digest,'changed replay input '+path)
    d=read(ROOT/'replay-output1/result1.json');check(d);files={};sample_count=0;maximum=0.;counts=dict(native_fp32=0,shadow_fp32=0,half=0,down=0,writer=0)
    def array(p,count,dtype):
        require(p.stat().st_size==count*np.dtype(dtype).itemsize,'truncated replay array')
        files[str(p.relative_to(ROOT))]=sha(p);return np.memmap(p,dtype=dtype,shape=(count,),mode='r')
    def f32(p,words):
        a=array(p,words+32,'<u4');require(np.all(a[:16]==0x1937abef) and np.all(a[-16:]==0x1937abef),'FP32 guard changed');return a[16:-16]
    for layer in d['layers']:
        index=layer['layer'];prefix='layer-'+str(index);capture=ROOT/'captures1'/prefix;folder=ROOT/'replay-output1';asset=ROOT/'assets1'/prefix
        expected={name:array(capture/(name+'.f32'),512*width,'<f4') for name,width in [('norm-input',4096),('gate',14336),('up',14336),('silu',14336),('down',4096)]}
        half_gold=expected['silu'].astype('<f2').view('<u2')
        for name in ['gate','up','silu','down']:
            words=len(expected[name]);a=f32(folder/(prefix+'-control-'+name+'.u32'),words)
            require(np.array_equal(a,expected[name].view('<u4')),'original replay differs from native full '+name);counts['native_fp32']+=words
        prior=f32(folder/(prefix+'-explicit31-silu.u32'),512*14336)
        require(np.array_equal(prior,expected['silu'].view('<u4')),'prior31 numerical control differs on actual data')
        def half(path):
            a=array(path,512*14336+64,'<u2');require(np.all(a[:32]==0x35a5) and np.all(a[-32:]==0x35a5),'half output guards changed');require(np.array_equal(a[32:-32],half_gold),'down-input half bits differ from actual native SiLU');return a
        half(folder/(prefix+'-control-half.u16'))
        for v in layer['variants']:
            label=prefix+'-fused-'+str(v['variant']);half(folder/(label+'-half.u16'));counts['half']+=512*14336
            down=f32(folder/(label+'-down.u32'),512*4096);require(np.array_equal(down,expected['down'].view('<u4')),'final down output differs');counts['down']+=512*4096
            if v['variant']==0:
                for name in ['gate','up']:
                    a=f32(folder/(prefix+'-fused-shadow-'+name+'.u32'),512*14336);require(np.array_equal(a,expected[name].view('<u4')),'full trained fused shadow differs');counts['shadow_fp32']+=512*14336
            if v['variant']==2:
                a=array(folder/(label+'-writers.u32'),512*14336+32,'<u4');require(np.all(a[:16]==0x1937abef) and np.all(a[-16:]==0x1937abef) and np.all(a[16:-16]==1),'missing/duplicate trained half writers');counts['writer']+=512*14336
        weights=[]
        for name,n,k in [('gate',14336,4096),('up',14336,4096),('down',4096,14336)]:
            w=np.memmap(asset/(name+'-weight.u32'),dtype='<u4',shape=(n,k//8),mode='r');s=np.memmap(asset/(name+'-scales.f16'),dtype='<f2',shape=(n,k//64),mode='r');b=np.memmap(asset/(name+'-biases.f16'),dtype='<f2',shape=(n,k//64),mode='r');weights.append((w,s,b))
        for p in layer['fp64']:
            stream,row,col=p['stream'],p['row'],p['column'];name=['gate','up','down'][stream];k=14336 if stream==2 else 4096;w,s,b=weights[stream];ks=np.arange(k);codes=((w[col,ks//8]>>((ks%8)*4))&15).astype(np.float64)
            dequant=(codes*s[col,ks//64].astype(np.float64)+b[col,ks//64].astype(np.float64)).astype(np.float32).astype(np.float16).astype(np.float64)
            x=expected['silu' if stream==2 else 'norm-input'].reshape(512,k)[row].astype(np.float16).astype(np.float64);terms=x*dequant;value=float(np.sum(terms,dtype=np.float64));sumabs=float(np.sum(np.abs(terms),dtype=np.float64));actual=float(expected[name].reshape(512,-1)[row,col])
            require(actual==p['actual'] and np.isclose(value,p['sum'],rtol=2e-12,atol=2e-12) and np.isclose(sumabs,p['sum_abs'],rtol=2e-12,atol=2e-12),'independent FP64 reference differs')
            fraction=abs(actual-value)/(2e-6+4e-6*sumabs);require(fraction<=1,'trained reference bound failed');maximum=max(maximum,fraction);sample_count+=1
    require(counts['native_fp32']==d['original_native_fp32_words'] and counts['half']==d['fused_half_word_comparisons'] and counts['down']==d['fused_down_word_comparisons'],'false full array counts')
    rejected=[]
    mutations={
      'native_control_mismatch':lambda x:x['layers'][0]['native_control_mismatches'].__setitem__(3,1),
      'wrong_nonlinear_policy':lambda x:x.update(original_main_options='forced MSL3.1'),
      'half_input_mismatch':lambda x:x['layers'][0]['variants'][1].update(half_mismatches=1),
      'down_output_mismatch':lambda x:x['layers'][0]['variants'][1].update(down_mismatches=1),
      'gate_shadow_mismatch':lambda x:x['layers'][0]['variants'][0].update(gate_mismatches=1),
      'writer_mismatch':lambda x:x['layers'][0]['variants'][2].update(writers_exact=False),
      'guard_mismatch':lambda x:x['layers'][0]['variants'][1].update(guards_exact=False),
      'missing_layer':lambda x:x['layers'].pop(),
      'false_fp64':lambda x:x['layers'][0]['fp64'][0].update(actual=1e10),
    }
    for label,mutate in mutations.items():
        bad=copy.deepcopy(d);mutate(bad)
        try:check(bad)
        except (ValueError,KeyError,IndexError,TypeError):rejected.append(label)
        else:raise ValueError('Corruption admitted '+label)
    result=dict(passed=True,scope=__doc__,layers=[0,15,31],actual_rows=512,word_comparisons=counts,independent_fp64_samples=sample_count,maximum_fp64_fraction=maximum,negative_controls_rejected=rejected,actual_native_default_options_reproduced=True,prior_explicit31_numerically_identical_on_observed_data=True,prior_explicit31_performance_equivalence_established=False,trained_activations_and_weights=True,all_complete_mlp_down_outputs_exact_native=True,source_replay_binding_sha256=sha(ROOT/'replay-bindings1.json'),raw_result_sha256=sha(ROOT/'replay-output1/result1.json'),capture_audit_sha256=sha(ROOT/'capture-audit1.json'),files_sha256=files,native_runtime_fusion_integrated=False,serving_speed_claim=False,quality_qualified=False,goal_complete=False)
    out=Path(sys.argv[2]) if len(sys.argv)>2 else ROOT/'replay-audit1.json';require(not out.exists(),'Never overwrite audit');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
