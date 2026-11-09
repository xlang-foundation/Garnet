"""Audit full FP32 decode-product/shadow parents and independent trained dot samples."""
from pathlib import Path
import copy,hashlib,itertools,json,math,sys
import numpy as np
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'correctness-audit1.json'
T=ROOT.parent/'mistral-fused-mlp-trained1';files={};rawcache={};assetcache={};dotcache={}
def need(v,m):
    if not v:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def raw(p):
    if p not in rawcache:
        b=p.read_bytes();files[str(p)]=hashlib.sha256(b).hexdigest();rawcache[p]=b
    return rawcache[p]
def read(p):return json.loads(raw(p))
def words(p):return np.frombuffer(raw(p),dtype='<u4')
def data(layer,name,dtype):
    key=layer,name,dtype
    if key not in assetcache:
        folder='captures1' if name=='norm-input.f32' else 'assets1'
        assetcache[key]=np.frombuffer(raw(T/folder/('layer-'+str(layer))/name),dtype=dtype)
    return assetcache[key]
def source(layer,n,padded):
    k=4096;stride=k+3 if padded else k;parent=np.full(32+4*stride,0x1937abef,dtype='<u4')
    x=data(layer,'norm-input.f32','<f4').reshape(512,k)
    for row,original in enumerate([0,127,256,511]):parent[16+row*stride:16+row*stride+k]=x[original].view('<u4')
    hashes=[hashlib.sha256(parent.tobytes()).hexdigest()]
    for role in ['gate','up']:
        for kind,dtype,width in [('weight.u32','<u4',k//8),('scales.f16','<u2',k//64),('biases.f16','<u2',k//64)]:
            v=data(layer,role+'-'+kind,dtype)[:n*width]
            b=np.full(16,0x1937abef,dtype='<u4').tobytes()+v.tobytes()+np.full(16,0x1937abef,dtype='<u4').tobytes()
            hashes.append(hashlib.sha256(b).hexdigest())
    return hashes
def oracle(layer,row,col,role):
    key=layer,row,col,role
    if key not in dotcache:
        name='up' if role else 'gate';k=4096
        packed=data(layer,name+'-weight.u32','<u4').reshape(14336,k//8)[col]
        q=((packed[:,None]>>(np.arange(8,dtype=np.uint32)*4))&15).reshape(k).astype(np.float64)
        scale=data(layer,name+'-scales.f16','<f2').reshape(14336,k//64)[col].astype(np.float64).repeat(64)
        bias=data(layer,name+'-biases.f16','<f2').reshape(14336,k//64)[col].astype(np.float64).repeat(64)
        # Half coefficient * integer nibble is exact in binary64; rounding
        # the sum once reproduces the dequantizing FP32 FMA.
        weight=(q*scale+bias).astype(np.float32).astype(np.float64)
        x=data(layer,'norm-input.f32','<f4').reshape(512,k)[[0,127,256,511][row]].astype(np.float64)
        product=x*weight;dotcache[key]=float(product.sum()),float(np.abs(product).sum())
    return dotcache[key]
def check(report,layout,raw_read=words):
    padded=layout=='padded';folder=ROOT/(layout+'1')
    need(report['completed'] and report['mode']=='correctness' and report['padded'] is padded and report['candidate_selected']==-1,'wrong/unfinished correctness scope')
    need(report['pipelines_prepared_before_collection'] and report['pipeline_count']==len(report['pipelines'])==280,'missing prepared profile family')
    need(all(p['simd_width']==32 and p['max_threads']>=128 and p['threadgroup_bytes']==0 for p in report['pipelines']),'invalid pipeline resources')
    need(report['case_combinations']==300 and len(report['cases'])==1800 and len(report['fp64'])==3780,'incomplete shape/mask/oracle coverage')
    expected={(layer,n,physical,mask,indexed,method,shadow) for layer in [0,15,31] for n in [14336,1,7,17,33] for physical,mask,indexed in [(i,(1<<i)-1,False) for i in range(1,5)]+[(4,i,True) for i in range(16)] for method in range(3) for shadow in [False,True]}
    actual=set();mismatch=[0]*3;exact=0;compared=0;case_map={};product_by_case={};control_checked=set()
    for cell in report['cases']:
        key=tuple(cell[k] for k in ['layer','columns','physical_rows','mask','indexed','method','shadow']);need(key not in actual,'duplicate case');actual.add(key)
        n=cell['columns'];physical=cell['physical_rows'];stride=n+(5 if padded else 0);count=32+4*stride
        need(cell['parent_words']==count,'wrong bounded output parent')
        controls=[raw_read(folder/name) for name in cell['raw_controls']];got=raw_read(folder/cell['raw_product'])
        need(len(got)==count and all(len(c)==count for c in controls),'truncated parent')
        need(hashlib.sha256(got.tobytes()).hexdigest()==cell['product_sha256'] and hashlib.sha256(controls[0].tobytes()).hexdigest()==cell['control_product_sha256'],'raw/digest disagreement')
        active=cell['mask'];expected_guard=np.ones(count,dtype=bool);inactive=np.zeros(count,dtype=bool)
        for row in range(physical):
            expected_guard[16+row*stride:16+row*stride+n]=False
            if not(active&(1<<row)):inactive[16+row*stride:16+row*stride+n]=True
        need(np.all(controls[0][expected_guard]==0x1937abef) and np.all(got[expected_guard]==0x1937abef),'product prefix/suffix/padding overwritten')
        need(np.all(controls[0][inactive]==0) and np.all(got[inactive]==0),'inactive product not cleared')
        need(np.all(np.isfinite(controls[0][~expected_guard].view('<f4'))),'nonfinite control product')
        identity=cell['case'];identity_key=key[:5]
        if identity in case_map:need(case_map[identity]==identity_key,'case ID changed identity')
        case_map[identity]=identity_key
        if identity not in control_checked:
            control_checked.add(identity)
            for c in controls[1:]:
                guard=np.ones(count,dtype=bool)
                if cell['indexed'] and active!=(1<<physical)-1:guard[16:16+(physical-1)*stride+n]=False
                for row in range(physical):
                    if active&(1<<row):guard[16+row*stride:16+row*stride+n]=False
                need(np.all(c[guard]==0x1937abef),'control projection parent guard changed')
        diff=int(np.count_nonzero(got!=controls[0]));shadow_diff=0
        if cell['shadow']:
            shadows=[raw_read(folder/name) for name in cell['raw_shadows']];need(all(len(v)==count for v in shadows),'truncated shadow')
            shadow_diff=int(np.count_nonzero((shadows[0]!=controls[1])|(shadows[1]!=controls[2])))
        need((diff,shadow_diff)==(cell['product_mismatches'],cell['shadow_mismatches']),'false complete-parent equality/mismatch claim')
        mismatch[cell['method']]+=diff+shadow_diff
        compared+=count*(3 if cell['shadow'] else 1)
        if not diff and not shadow_diff:exact+=count*(3 if cell['shadow'] else 1)
        product_by_case[identity,cell['method'],cell['shadow']]=got
    need(actual==expected and set(case_map)==set(range(300)),'missing/noncanonical case family')
    need(mismatch==report['mismatches_by_method'] and exact==report['exact_parent_words'],'wrong complete-parent totals')
    maximum=0;fpkeys=[]
    for sample in report['fp64']:
        identity=sample['case'];layer,n,physical,mask,indexed=case_map[identity];row,col,role=sample['row'],sample['column'],sample['role']
        need((sample['layer'],sample['columns'])==(layer,n) and mask&(1<<row) and row<physical and role in [0,1] and col in [0,n//2,n-1],'unbound FP64 sample')
        fpkeys.append((identity,row,col,role))
        total,absolute=oracle(layer,row,col,role)
        need(math.isclose(total,sample['sum'],rel_tol=2e-12,abs_tol=1e-10) and math.isclose(absolute,sample['sum_abs'],rel_tol=2e-12,abs_tol=1e-10),'independent FP64 dot differs')
        cell=next(c for c in report['cases'] if c['case']==identity)
        values=raw_read(folder/cell['raw_controls'][role+1]).view('<f4');actual_value=float(values[16+row*(n+(5 if padded else 0))+col])
        need(actual_value==sample['actual'] and math.isfinite(actual_value),'FP64 sample not bound to control word')
        fraction=abs(actual_value-total)/(2e-6+4e-6*absolute);need(fraction<=1,'original FP64 allowance exceeded');maximum=max(maximum,fraction)
    expected_fp=[]
    for identity,(layer,n,physical,mask,indexed) in case_map.items():
        expected_fp.extend((identity,row,col,role) for row in range(physical) if mask&(1<<row) for col in [0,n//2,n-1] for role in [0,1])
    need(sorted(fpkeys)==sorted(expected_fp),'independent sample coverage missing')
    need(len(report['readonly_checks'])==15,'missing read-only parent checks')
    need({(e['layer'],e['columns']) for e in report['readonly_checks']}==set(itertools.product([0,15,31],[14336,1,7,17,33])),'read-only coverage changed')
    for entry in report['readonly_checks']:need(entry['unchanged'] is True and entry['sha256']==source(entry['layer'],entry['columns'],padded),'input/weight guarded parent identity differs')
    need(report['after_close_allocated_bytes']<=report['baseline_allocated_bytes']+4194304,'GPU owners retained')
    return dict(cells=1800,case_combinations=300,compared_parent_words=compared,exact_parent_words=exact,mismatches_by_method=mismatch,fp64_checks=3780,maximum_original_fp64_fraction=maximum,eligible_methods=[i for i,n in enumerate(mismatch) if n==0],after_close_allocated_bytes=report['after_close_allocated_bytes'])
def main():
    need(not DEST.exists(),'never overwrite audit');pre=read(ROOT/'correctness1-preflight.json');post=read(ROOT/'correctness1-postflight.json')
    need(pre['passed'] and post['passed'] and post['all_owned_processes_terminal'],'unfinished probes')
    for name,h in pre['files_sha256'].items():need(sha(Path(name))==h,'changed bound input '+name)
    statuses=[read(ROOT/(layout+'1-status.json')) for layout in ['contiguous','padded']]
    need(all(s['status']=='terminal' and s['returncode']==0 for s in statuses) and statuses[1]['started_ns']>=statuses[0]['finished_ns'],'probes were not serial/completed')
    data={layout:read(ROOT/(layout+'1')/'result1.json') for layout in ['contiguous','padded']}
    result={layout:check(report,layout) for layout,report in data.items()}
    eligible=set.intersection(*(set(r['eligible_methods']) for r in result.values()));need(eligible=={2},'candidate gate changed; review before timing')
    rejected=[]
    mutations={
        'false_product_equality':lambda r:r['cases'][0].update(product_mismatches=1),
        'false_shadow_equality':lambda r:next(c for c in r['cases'] if c['shadow']).update(shadow_mismatches=1),
        'changed_mask':lambda r:r['cases'][0].update(mask=0),
        'missing_case':lambda r:r['cases'].pop(),
        'unprepared_pipeline':lambda r:r.update(pipelines_prepared_before_collection=False),
        'wrong_pipeline_count':lambda r:r.update(pipeline_count=1),
        'wrong_fp64_value':lambda r:r['fp64'][0].update(actual=1e9),
        'wrong_fp64_sum':lambda r:r['fp64'][0].update(sum=1e9),
        'changed_readonly_parent':lambda r:r['readonly_checks'][0]['sha256'].__setitem__(0,'0'*64),
        'false_math_eligibility':lambda r:r.update(mismatches_by_method=[0,0,0]),
        'owner_leak':lambda r:r.update(after_close_allocated_bytes=r['baseline_allocated_bytes']+4194305),
    }
    for name,mutate in mutations.items():
        bad=copy.deepcopy(data['contiguous']);mutate(bad)
        try:check(bad,'contiguous')
        except (ValueError,KeyError,IndexError,TypeError):rejected.append(name)
        else:raise ValueError('corruption admitted '+name)
    # Raw buffer corruption must reject independently of report claims.
    target=ROOT/'contiguous1'/data['contiguous']['cases'][0]['raw_product']
    def corrupt(path):
        v=words(path)
        if path==target:v=v.copy();v[0]^=1
        return v
    try:check(data['contiguous'],'contiguous',corrupt)
    except ValueError:rejected.append('raw_parent_guard_corruption')
    else:raise ValueError('corrupt parent admitted')
    output=dict(passed=True,scope=__doc__,layouts=result,eligible_methods=sorted(eligible),candidate_names=['strict-dot/exp','strict-dot/fast::exp','nil-default/exp'],negative_controls_rejected=rejected,files_sha256=files,no_native_runtime_integration=True,serving_speed_claim=False,goal_complete=False)
    DEST.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({k:v for k,v in output.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
