"""Exclusive standalone T128 attention correctness; no model or performance run."""
from pathlib import Path
import argparse,hashlib,json,os,sys,zipfile

JOB='prefill128-attention-component-attempt1'
REFERENCE='vllm-observed-fixed4k-ordinary-attempt1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def closed_reference(root,digest):
    archive=root/'results'/(REFERENCE+'-evidence.zip');assert sha(archive)==digest
    w=json.loads((root/'results'/REFERENCE/'workflow.json').read_bytes())
    assert type(w['exit_code']) is int and w['exit_code']==0 and w['cleanup_finished'] is True
    assert w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after']
    assert type(w['server_exit_code']) is int and w['server_exit_code']==0
    with zipfile.ZipFile(archive) as z:assert json.loads(z.read('results/'+REFERENCE+'/workflow.json'))==w
    return {'job':REFERENCE,'archive_sha256':digest,'exit_code':w['exit_code'],'cleanup_finished':True}

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--closed-reference-archive-sha256',required=True);args=a.parse_args()
    root=Path('D:/CantorAI/5090-study-20261023');m=Path(__file__).resolve().parent;assert m==root/'Garnet/test2026/cuda/qwen38_5090'
    sys.path.insert(0,str(m));from run_logged import run
    from run_bf16_low_batch_native_build import idle
    parent=closed_reference(root,args.closed_reference_archive_sha256)
    preparation=m/'evidence/prefill128-attention-component-preparation.json';p=json.loads(preparation.read_bytes())
    assert p['status']=='prefill128_attention_component_prepared_gpu_unrun'
    assert parent['archive_sha256']==p['reference_parent']['archive_sha256']
    profiled=p['profiled_parent'];proof=root/'results'/(profiled['job']+'-verified.json');q=json.loads(proof.read_bytes())
    assert sha(proof)==profiled['proof_sha256'] and q['status']=='native_cuda_stdio_profiled_replayed'
    assert sha(root/'results'/(profiled['job']+'-evidence.zip'))==q['archive_sha256']==profiled['archive_sha256']
    for n,h in p['files'].items():assert sha(root/n)==h
    for n,h in p['runtime_dependencies'].items():assert sha(root/'build/bin'/n)==h
    for n,h in p['environment'].items():assert sha(Path(n))==h
    out=root/'results'/JOB;archive=root/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists();idle();out.mkdir();os.chdir(root/'Garnet')
    sdk=root/'tools/cuda13/nvidia/cu13'
    sdk_names=('bin/nvcc.exe','bin/ptxas.exe','nvvm/bin/cicc.exe','include/crt/host_config.h','include/cuda_runtime.h','include/cuda_runtime_api.h','include/cuda_bf16.h','include/cuda_bf16.hpp')
    sdk_before={str(sdk/n):sha(sdk/n) for n in sdk_names}
    binary=root/'tools/bf16-kv-native/prefill128-attention-probe.exe'
    assert not binary.exists() and binary.parent.is_dir()
    common=[root/n for n in p['files']]+[preparation,root/'results'/REFERENCE/'workflow.json'];code=1
    import run_logged as ledger
    ledger.source_identity=lambda:{'scope':'explicit owned standalone component inputs; unrelated untracked files excluded','owned_source_sha256':p['files']}
    w={'status':'running','source_sha256':p['files'],'runtime_dependencies':p['runtime_dependencies'],'environment':p['environment'],
       'sdk_before':sdk_before,'preparation_sha256':sha(preparation),'closed_reference':parent,'cleanup_finished':False,
       'runtime_loaded':False,'performance_admitted':False,'scope':p['scope']}
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    save()
    try:
        code=run([sys.executable,'-m','unittest',*p['test_modules']],
                 out/'cpu-ledger','correctness','Actual11CPU profile/full FP64 oracle/closed-parent and complete-case controls before component build/GPU',common)
        if code:raise RuntimeError('T128 attention CPU controls failed')
        code=run(['cmd.exe','/d','/c',str(m/'build_bf16_prefill128_attention_component.bat')],out/'build-ledger','build',
                 'Actual pinned remote compiler version, explicitSM120 build/link into fixed private component folder; no Garnet DLL/SDK mutation',common)
        if code:raise RuntimeError('Remote T128 attention build failed')
        w['executable_sha256']=sha(binary);save();common.append(binary)
        native=out/'native';native.mkdir()
        code=run([sys.executable,str(m/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--',str(binary),str(native/'witnesses.f32')],
                 out/'gpu-ledger','correctness','Actual36cases/full outputs, plain-tiled probability/output bit checks,9graph launches,256malformed calls, complete canonical readonly/guards; full FP64 replay required',common)
        if code:raise RuntimeError('Actual T128 attention GPU collection failed')
        w['status']='prefill128_attention_component_collected_needs_replay'
    except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
    finally:
        w.update(exit_code=code,source_after_sha256={n:sha(root/n) for n in p['files']},
            runtime_dependencies_after={n:sha(root/'build/bin'/n) for n in p['runtime_dependencies']},
            environment_after={n:sha(Path(n)) for n in p['environment']},sdk_after={n:sha(Path(n)) for n in sdk_before})
        save();idle();w['cleanup_finished']=True;save()
        paths={f.relative_to(root).as_posix():f for f in [*out.rglob('*'),*common] if f.is_file()}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for n,f in paths.items():z.write(f,n)
            z.writestr('evidence-manifest.json',json.dumps({'files':{n:sha(f) for n,f in paths.items()},'scope':p['scope']},indent=2))
        print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(paths)}),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
