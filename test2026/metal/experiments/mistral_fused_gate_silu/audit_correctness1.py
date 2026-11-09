"""Independently inspect retained complete arrays, ownership maps and FP64 samples."""
from pathlib import Path
import hashlib,json,sys
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
def mmap(p,dtype,count):
    require(p.stat().st_size==np.dtype(dtype).itemsize*count,'wrong retained array size '+str(p))
    return np.memmap(p,dtype=dtype,mode='r',shape=(count,))
def validate_record(d):
    require(d['completed'] and len(d['cases'])==36 and d['fp64_values']==1080,'unfinished screen')
    require([(x['label'],x['rows']) for x in d['cases']]==[(label,rows) for label in ['synthetic','layer-0','layer-15','layer-31'] for rows in [127,128,129,255,256,257,383,511,512]],'changed shape/corpus coverage')
    for c in d['cases']:
        require(c['tile']==(256 if c['rows']>=256 else 128) and len(c['variants'])==6,'wrong tile/variant coverage')
        require([(v['math'],v['variant']) for v in c['variants']]==[(m,v) for m in ['Safe','Fast'] for v in range(3)],'wrong policy coverage')
        for v in c['variants']:
            require(v['guards_exact'] and v['writers_exact'] and v['gate_mismatches']==v['up_mismatches']==0,'invalid shadow or ownership')
            if v['math']=='Fast':require(v['half_mismatches']==0,'Fast candidate does not preserve half bits')
        require(len(c['fp64'])==30,'missing FP64 coverage')
        for p in c['fp64']:
            require(np.isfinite(p['actual']) and abs(p['actual']-p['sum'])<=2e-6+4e-6*p['sum_abs'],'FP64 bound failed')
        if c['rows']==512:require(c['truncated_grid_missing']==256*14336,'fault grid not detected')

def main():
    require(read(ROOT/'correctness1-status.json')['returncode']==0 and read(ROOT/'correctness1-postflight.json')['passed'],'failed process/postflight')
    for path,digest in read(ROOT/'correctness1-command.json')['files_sha256'].items():require(sha(Path(path))==digest,'changed code/input '+path)
    binding=read(ROOT/'bindings1.json')
    for name,digest in binding['assets_sha256'].items():require(sha(ROOT/name)==digest,'changed frozen asset '+name)
    d=read(ROOT/'correctness1/result1.json');validate_record(d)
    N,K,OS=14336,4096,14344;matrix_words=514*OS+32;half_words=512*N+64;writer_words=512*N+32
    total_half=total_f32=total_writers=0;maximum=0.;sample_count=0
    input=np.fromfile(ROOT/'assets1/input-parent.f32',dtype='<f4').reshape(514,K)
    assets={}
    for label in ['synthetic','layer-0','layer-15','layer-31']:
        assets[label]=[]
        for stream in ['gate','up']:
            folder=ROOT/'assets1'/label
            assets[label].append((np.memmap(folder/(stream+'-weight.u32'),dtype='<u4',mode='r',shape=(N,K//8)),np.memmap(folder/(stream+'-scales.f16'),dtype='<f2',mode='r',shape=(N,K//64)),np.memmap(folder/(stream+'-biases.f16'),dtype='<f2',mode='r',shape=(N,K//64))))
    for c in d['cases']:
        prefix=c['label']+'-m'+str(c['rows']);folder=ROOT/'correctness1';rows=c['rows']
        gold_half=mmap(folder/(prefix+'-control-half.u16'),'<u2',half_words)
        require(np.all(gold_half[:32]==0x35a5) and np.all(gold_half[32+rows*N:]==0x35a5),'control half guard changed')
        gold=[mmap(folder/(prefix+'-control-'+stream+'.u32'),'<u4',matrix_words) for stream in ['gate','up']]
        for stream in range(2):
            require(np.all(gold[stream][:16]==0x1937abef) and np.all(gold[stream][-16:]==0x1937abef),'control FP32 guard changed')
            physical=gold[stream][16:-16].reshape(514,OS)
            require(np.all(physical[0]==0) and np.all(physical[rows+1:]==0) and np.all(physical[:,-8:]==0),'inactive FP32 output changed')
        for v in c['variants']:
            label=prefix+'-'+v['math'].lower()+'-'+['shadow','normal','writer'][v['variant']]
            half=mmap(folder/(label+'-half.u16'),'<u2',half_words);difference=int(np.count_nonzero(half!=gold_half))
            require(difference==v['half_mismatches'],'false half parity report')
            require(np.all(half[:32]==0x35a5) and np.all(half[32+rows*N:]==0x35a5),'half output ownership violation')
            if v['math']=='Fast':require(difference==0,'nonidentical admitted output')
            total_half+=half_words
            if v['variant']==0:
                for stream,name in enumerate(['gate','up']):
                    actual=mmap(folder/(label+'-'+name+'.u32'),'<u4',matrix_words)
                    require(np.array_equal(actual,gold[stream]),'changed complete gate/up FP32 shadow')
                    total_f32+=matrix_words
            if v['variant']==2:
                counts=mmap(folder/(label+'-counts.u32'),'<u4',writer_words)
                require(np.all(counts[:16]==0x1937abef) and np.all(counts[-16:]==0x1937abef),'writer guards changed')
                require(np.all(counts[16:16+rows*N]==1) and np.all(counts[16+rows*N:-16]==0),'missing/duplicate writer')
                total_writers+=writer_words
        if rows==512:
            fault=mmap(folder/(prefix+'-fault-counts.u32'),'<u4',writer_words)
            require(np.all(fault[:16]==0x1937abef) and np.all(fault[-16:]==0x1937abef),'fault guards changed')
            require(np.all(fault[16:16+256*N]==1) and np.all(fault[16+256*N:-16]==0),'fault missing region differs')
        for p in c['fp64']:
            stream,row,col=p['stream'],p['row'],p['column'];physical_row=rows-row
            w,s,b=assets[c['label']][stream];ks=np.arange(K)
            codes=((w[col,ks//8]>>((ks%8)*4))&15).astype(np.float64)
            # Half coefficients times integer codes and their sum are exactly
            # representable in FP32; double evaluation then half rounding is
            # an independent oracle for the native explicit float fma/cast.
            weight=(codes*s[col,ks//64].astype(np.float64)+b[col,ks//64].astype(np.float64)).astype(np.float32).astype(np.float16).astype(np.float64)
            x=input[physical_row].astype(np.float16).astype(np.float64);terms=x*weight
            value=float(np.sum(terms,dtype=np.float64));sumabs=float(np.sum(np.abs(terms),dtype=np.float64))
            actual=float(gold[stream][16+physical_row*OS+col:17+physical_row*OS+col].view('<f4')[0])
            require(actual==p['actual'] and np.isclose(value,p['sum'],rtol=2e-12,atol=2e-12) and np.isclose(sumabs,p['sum_abs'],rtol=2e-12,atol=2e-12),'false FP64 sample')
            fraction=abs(actual-value)/(2e-6+4e-6*sumabs);require(fraction<=1,'independent FP64 bound failed');maximum=max(maximum,fraction);sample_count+=1
    import copy
    rejected=[]
    def reject(label,mutate):
        bad=copy.deepcopy(d);mutate(bad)
        try:validate_record(bad)
        except (ValueError,KeyError,IndexError,TypeError):rejected.append(label)
        else:raise ValueError('Corrupt evidence admitted '+label)
    reject('false_half_parity',lambda x:x['cases'][0]['variants'][4].update(half_mismatches=1))
    reject('false_shadow_parity',lambda x:x['cases'][0]['variants'][3].update(gate_mismatches=1))
    reject('writer_failure',lambda x:x['cases'][0]['variants'][4].update(writers_exact=False))
    reject('guard_failure',lambda x:x['cases'][0]['variants'][4].update(guards_exact=False))
    reject('missing_case',lambda x:x['cases'].pop())
    reject('wrong_tile',lambda x:x['cases'][0].update(tile=256))
    reject('wrong_policy',lambda x:x['cases'][0]['variants'][4].update(math='Safe'))
    reject('false_fp64',lambda x:x['cases'][0]['fp64'][0].update(actual=1e10))
    reject('undetected_grid_fault',lambda x:x['cases'][8].update(truncated_grid_missing=0))
    result=dict(passed=True,scope=__doc__,candidate_admitted_to_operator_timing='Fast only',rejected_candidate='Safe:14121 half-bit differences across shadow/normal/writer observations; original gate/up FP32 shadows exact.',cases=36,full_half_word_comparisons=total_half,full_fp32_shadow_word_comparisons=total_f32,full_retained_writer_words=total_writers,independent_fp64_samples=sample_count,maximum_fp64_fraction=maximum,negative_controls_rejected=rejected,main_control='Original MSL3.1 ordinary silu_mul',projection_control='Original MSL4 Safe packed affine',fused_candidate='MSL4 Fast, unchanged strict MMA descriptors; ordinary nonlinear policy preserved in tested data.',bindings_sha256=sha(ROOT/'bindings1.json'),raw_result_sha256=sha(ROOT/'correctness1/result1.json'),trained_weight_layers=[0,15,31],trained_activation_inputs=False,full_native_graph_qualified=False,serving_speed_claim=False,quality_qualified=False,goal_complete=False)
    dest=Path(sys.argv[2]) if len(sys.argv)>2 else ROOT/'correctness-audit1.json';require(not dest.exists(),'Never overwrite audit');dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
