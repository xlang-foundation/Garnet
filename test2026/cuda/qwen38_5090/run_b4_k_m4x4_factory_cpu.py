"""Actual K four-M4 factory and pointer-stride contracts under CPU SDK stubs, never GPU results."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess
PREFIX=b'GARNET_KM4X4_FACTORY '
RUN_MODES=['run_valid','run_batch','run_rows','run_columns','run_zero_scale','run_nan_scale','run_weight','run_input','run_output','run_workspace','run_workspace_bytes','run_wrong_device','run_device_error','run_device_throw','run_negative_scale','run_infinite_scale','run_null_context']+[f'run_matmul_{what}_{i}' for what in ('fail','throw') for i in range(4)]
MODES=['success','cached','cross_geometry','cross_device','major','minor','version','found0','found_negative','found9','skip_failed','all_failed','opaque','first_bad_then_good']+[f'get_{what}_{i}' for what in ('value','bytes') for i in range(7)]+[f'{what}_{m}' for what in ('check_state','workspace') for m in (1,2)]+[f'{what}_{i}' for what in ('fail','throw') for i in range(1,25)]+RUN_MODES
SUCCEED={'success','cached','cross_geometry','cross_device','skip_failed','opaque',*RUN_MODES}

def check(q,stderr,mode,geometry,enabled,unsupported):
    accepted=mode in SUCCEED and not unsupported and geometry == 1024
    assert q['mode']==mode and q['created']==q['info_okay']==accepted and q['cached']==(mode=='cached' and accepted)
    assert q['thrown']==(mode.startswith('throw_') and not unsupported and geometry == 1024)
    attempts=(4 if mode=='run_valid' else int(mode[-1])+1 if mode.startswith(('run_matmul_fail_','run_matmul_throw_')) else 0) if accepted else 0
    assert q['unexpected_matmuls']==attempts and q['matmul_contract_okay']
    assert q['input_offsets']==[i*20480 for i in range(attempts)] and q['output_offsets']==[i*8192 for i in range(attempts)]
    expected_runtime=(-999 if not mode.startswith('run_') else 1 if mode=='run_null_context' else 0 if mode=='run_valid' else 14 if mode=='run_device_throw' or mode.startswith(('run_matmul_fail_','run_matmul_throw_')) else 7)
    assert q['runtime_status']==expected_runtime
    if accepted:assert q['algorithm']==35 and q['workspace']==0
    if unsupported or geometry != 1024:assert q['calls']==[]
    if not enabled:assert stderr==b'';return
    rows=[json.loads(line[len(PREFIX):]) for line in stderr.splitlines() if line.startswith(PREFIX)];assert rows and len(rows)==len(stderr.splitlines());groups={}
    for row in rows:
        assert set(row)=={'id','seq','event','name','a','b','c','opaque'};groups.setdefault(row['id'],[]).append(row)
    assert len(groups)==(2 if mode in ('cached','cross_geometry','cross_device') and accepted else 1)
    for group in groups.values():assert [r['seq'] for r in group]==list(range(1,len(group)+1)) and group[0]['event']=='begin' and group[-1]['event']=='end'
    if accepted:
        for i,group in groups.items():
            if i==2 and mode=='cached':assert group[-1]['name']=='cached' and len([r for r in group if r['event']=='api'])==3;continue
            if i==2 and mode=='cross_geometry':assert group[-1]['name']=='empty' and not any(r['event']=='api' for r in group);continue
            current_geometry=geometry
            expected_configuration=(35,18,16,7,2,1,2) if current_geometry==12288 else (35,13,17,3,2,0,1)
            configs=[r for r in group if r['event']=='configuration'];assert [(r['name'],r['a'],r['b'],r['c']) for r in configs]==[(n,v,4,v) for n,v in zip(('ID','TILE_ID','STAGES_ID','SPLITK_NUM','REDUCTION_SCHEME','CTA_SWIZZLING','CUSTOM_OPTION'),expected_configuration)]
            assert [(r['name'],r['a'],r['b']) for r in group if r['event']=='check']==[('M4',0,0),('M4_actual',0,0)] and group[-1]['name']=='created' and len([r for r in group if r['event']=='api'])==24
        expected=[f'cublasLtMatrixLayoutCreate:28:5120:{geometry}:5120','cublasLtMatrixLayoutCreate:28:5120:4:5120',f'cublasLtMatrixLayoutCreate:14:{geometry}:4:{geometry}','cublasLtMatrixLayoutCreate:28:5120:4:5120',f'cublasLtMatrixLayoutCreate:14:{geometry}:4:{geometry}'];assert q['calls'][7:12]==expected
        assert q['calls'].count('cublasLtMatmulAlgoCheck:1')==q['calls'].count('cublasLtMatmulAlgoCheck:2')==(2 if mode=='cross_device' else 1)
    else:assert groups[1][-1]['name']==('exception' if q['thrown'] else 'empty')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();assert not a.output.exists();a.output.mkdir();rows=[];baseline={}
    cases=[('contract.exe',g,m,v) for g in (1024,) for m in MODES for v in (None,'1')]+[('contract.exe',g,'success',v) for g in (1024,) for v in ('','0','true','01',' 1','1 ','-1','2')]+[('unsupported.exe',g,'success',v) for g in (1024,) for v in (None,'1')]+[('contract.exe',g,'success',v) for g in (0,-1,1,5120,6144,12288) for v in (None,'1')]
    for i,(exe,g,mode,policy) in enumerate(cases):
        env=dict(os.environ);env.pop('GARNET_KM4X4_FACTORY_TRACE',None)
        if policy is not None:env['GARNET_KM4X4_FACTORY_TRACE']=policy
        child=subprocess.run([str(a.directory/exe),mode,str(g)],capture_output=True,env=env);(a.output/(str(i)+'-stdout.log')).write_bytes(child.stdout);(a.output/(str(i)+'-stderr.log')).write_bytes(child.stderr)
        assert child.returncode==0,(i,g,mode,child.returncode,child.stdout,child.stderr);q=json.loads(child.stdout);check(q,child.stderr,mode,g,policy=='1',exe=='unsupported.exe');key=(exe,g,mode)
        if key in baseline:assert baseline[key]==q,('Tracing changed behavior',key)
        else:baseline[key]=q
        rows.append(dict(exe=exe,geometry=g,mode=mode,policy=policy,stdout_sha256=hashlib.sha256(child.stdout).hexdigest(),stderr_sha256=hashlib.sha256(child.stderr).hexdigest(),report=q))
    result=dict(status='B4T4_K_M4X4_actual_factory_CPU_fault_contract_replayed',cases=len(rows),modes=len(MODES),executables={n:hashlib.sha256((a.directory/n).read_bytes()).hexdigest() for n in ('contract.exe','unsupported.exe')},rows=rows,GPU_model_or_SDK_matmul_executed=False)
    (a.output/'report.json').write_bytes(json.dumps(result,indent=2).encode());print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
