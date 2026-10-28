"""Fault tests for the actual factory against CPU SDK stubs, never GPU results."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess
PREFIX=b'GARNET_Q128_FACTORY '
MODES=['success','cached','major','minor','version','found0','found_negative','found9','skip_failed','all_failed','opaque','first_bad_then_good']+[f'get_{what}_{i}' for what in ('value','bytes') for i in range(7)]+[f'{what}_{m}' for what in ('check_state','workspace') for m in (16,32,128)]+[f'{what}_{i}' for what in ('fail','throw') for i in range(1,28)]
SUCCEED={'success','cached','skip_failed','opaque'}
def check(report,stderr,mode,enabled,unsupported=False):
    assert report['mode']==mode and report['unexpected_matmuls']==0
    accepted=mode in SUCCEED and not unsupported
    assert report['created']==report['info_okay']==accepted and report['cached']==(mode=='cached' and accepted)
    assert report['thrown']==(mode.startswith('throw_') and not unsupported)
    if accepted:assert report['algorithm']==67 and report['workspace']==0
    if unsupported:assert report['calls']==[]
    if not enabled:assert stderr==b'';return
    rows=[json.loads(line[len(PREFIX):]) for line in stderr.splitlines() if line.startswith(PREFIX)]
    assert rows and len(rows)==len(stderr.splitlines())
    groups={}
    for r in rows:
        assert set(r)=={'id','seq','event','name','a','b','c','opaque'};groups.setdefault(r['id'],[]).append(r)
    assert len(groups)==(2 if mode=='cached' and not unsupported else 1)
    for group in groups.values():
        assert [r['seq'] for r in group]==list(range(1,len(group)+1)) and group[0]['event']=='begin' and group[-1]['event']=='end'
    if accepted:
        main=groups[1];configs=[r for r in main if r['event']=='configuration'];assert [(r['name'],r['a'],r['b'],r['c']) for r in configs]==[(n,v,4,v) for n,v in zip(('ID','TILE_ID','STAGES_ID','SPLITK_NUM','REDUCTION_SCHEME','CTA_SWIZZLING','CUSTOM_OPTION'),(67,13,17,1,0,1,18))]
        assert [(r['name'],r['a'],r['b']) for r in main if r['event']=='check']==[('M16',0,0),('M32',0,0),('M128',0,0)]
        assert main[-1]['name']=='created' and len([r for r in main if r['event']=='api'])==27
        assert report['calls'].count('cublasLtMatmulAlgoCheck:16')==report['calls'].count('cublasLtMatmulAlgoCheck:32')==report['calls'].count('cublasLtMatmulAlgoCheck:128')==1
        assert report['calls'][7:14]==['cublasLtMatrixLayoutCreate:28:5120:12288:5120','cublasLtMatrixLayoutCreate:28:5120:128:5120','cublasLtMatrixLayoutCreate:14:12288:128:12288','cublasLtMatrixLayoutCreate:28:5120:16:5120','cublasLtMatrixLayoutCreate:14:12288:16:12288','cublasLtMatrixLayoutCreate:28:5120:32:5120','cublasLtMatrixLayoutCreate:14:12288:32:12288']
        if mode=='cached':assert groups[2][-1]['name']=='cached' and len([r for r in groups[2] if r['event']=='api'])==3
    else:assert groups[1][-1]['name']==('exception' if report['thrown'] else 'empty')
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir();rows=[];baseline={}
    cases=[('contract.exe',m,v) for m in MODES for v in (None,'1')]+[('contract.exe','success',v) for v in ('','0','true','01',' 1','1 ','-1','2')]+[('unsupported.exe','success',v) for v in (None,'1')]
    for i,(exe,mode,policy) in enumerate(cases):
        env=dict(os.environ);env.pop('GARNET_Q128_FACTORY_TRACE',None)
        if policy is not None:env['GARNET_Q128_FACTORY_TRACE']=policy
        child=subprocess.run([str(a.directory/exe),mode],capture_output=True,env=env);stem=str(i);(a.output/(stem+'-stdout.log')).write_bytes(child.stdout);(a.output/(stem+'-stderr.log')).write_bytes(child.stderr)
        assert child.returncode==0,(i,mode,child.returncode);q=json.loads(child.stdout);check(q,child.stderr,mode,policy=='1',exe=='unsupported.exe')
        key=(exe,mode)
        if key in baseline:assert baseline[key]==q,('Observation changed behavior',key)
        else:baseline[key]=q
        rows.append(dict(exe=exe,mode=mode,policy=policy,stdout_sha256=hashlib.sha256(child.stdout).hexdigest(),stderr_sha256=hashlib.sha256(child.stderr).hexdigest(),report=q))
    result=dict(status='Q128_actual_factory_CPU_fault_contract_replayed',cases=len(rows),modes=len(MODES),executables={n:hashlib.sha256((a.directory/n).read_bytes()).hexdigest() for n in ('contract.exe','unsupported.exe')},rows=rows,GPU_model_or_matmul_executed=False)
    (a.output/'report.json').write_bytes(json.dumps(result,indent=2).encode());print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
