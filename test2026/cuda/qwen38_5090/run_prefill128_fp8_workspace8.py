"""Preserved-source untimed real FP8 M16/M32/M128 control, no application DLL loading."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='prefill128-fp8-workspace8-attempt1';FAILED='bf16-prefill128-model-admission-attempt2';OLD='actual-layer0-fp8-control-attempt1'
NAMES=('actual_prefill128_fp8_workspace8.cu','actual_prefill128_fp8_projection.cu','actual_layer0_fp8_control.cu','build_prefill128_fp8_workspace8.bat','run_prefill128_fp8_workspace8.py','verify_prefill128_fp8_workspace8.py','prefill128_fp8_projection.py','test_prefill128_fp8_projection.py','test_prefill128_fp8_workspace8.py','evidence/prefill128-fp8-projection-derivation.json','evidence/prefill128-fp8-workspace8-derivation.json')
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()

PRIMITIVES=('src/backends/tensorrt/modelopt_cublaslt.cpp','src/backends/tensorrt/modelopt_cublaslt.h','src/cuda/modelopt_fp8.cu','src/cuda/modelopt_fp8.cuh')

def primitive_source_hashes(repo):
    return {'Garnet/'+name:sha(repo/name) for name in PRIMITIVES}

def prepare():
    from run_bf16_low_batch_native_build import idle
    idle();results=ROOT/'results';w=json.loads((results/FAILED/'workflow.json').read_bytes());q=json.loads((results/(FAILED+'-verified.json')).read_bytes());analysis=json.loads((results/(FAILED+'-analysis.json')).read_bytes())
    assert w['exit_code']==1 and w['cleanup_finished'] and w['baseline_source_restored'] and w['baseline_library_restored']
    assert w['source_before_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after'] and w['environment_before']==w['environment_after']
    assert q['status']=='bf16_prefill128_full4k_model_failure_preserved' and analysis['status']=='bf16_prefill128_first_full4k_failure_analyzed' and q['archive_sha256']==analysis['archive_sha256']==sha(results/(FAILED+'-evidence.zip'))
    old=json.loads((results/(OLD+'-verified.json')).read_bytes());assert old['status']=='real_fp8_projection_control_verified' and old['archive_sha256']==sha(results/(OLD+'-evidence.zip'))
    fixture=results/'actual-layer0-fp8-fixture';fraw=(fixture/'manifest.json').read_bytes();f=json.loads(fraw)
    with zipfile.ZipFile(results/(OLD+'-evidence.zip')) as z:assert z.read('results/actual-layer0-fp8-fixture/manifest.json')==fraw
    for name,row in f['files'].items():assert (fixture/name).stat().st_size==row['bytes'] and sha(fixture/name)==row['sha256']
    assert sha(ROOT/'models/Qwen3.8-27B-NVFP4/config.json')==f['config_sha256'] and sha(ROOT/'models/Qwen3.8-27B-NVFP4/model.safetensors.index.json')==f['index_sha256']
    methods={'Garnet/test2026/cuda/qwen38_5090/'+name:sha(M/name) for name in NAMES};files={**w['source_after_sha256'],**methods,**primitive_source_hashes(ROOT/'Garnet')}
    for name,h in files.items():assert sha(ROOT/name)==h
    for name,h in w['runtime_dependencies_after'].items():assert sha(ROOT/'build/bin'/name)==h
    for name,h in w['environment_after'].items():assert sha(Path(name))==h
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6'
    record=json.loads((M/'evidence/prefill128-fp8-projection-derivation.json').read_bytes());source=(M/record['source']).read_bytes();assert hashlib.sha256(source).hexdigest()==record['source_sha256'];text=source.decode()
    for step in record['replacements']:
        assert text.count(step['old'])==step['count'];text=text.replace(step['old'],step['new'])
    assert text.encode()==(M/record['target']).read_bytes() and hashlib.sha256(text.encode()).hexdigest()==record['target_sha256']
    environment=dict(w['environment_after']);cuda=ROOT/'tools/cuda13/nvidia/cu13'
    for name in ('include/cublasLt.h','include/cublas_v2.h','include/cuda_runtime.h','lib/x64/cublasLt.lib','bin/x86_64/cublasLt64_13.dll'):
        path=cuda/name;environment[str(path)]=sha(path)
    assert environment[str(cuda/'bin/x86_64/cublasLt64_13.dll')]==w['runtime_dependencies_after']['cublasLt64_13.dll']
    previous=json.loads((results/'prefill128-fp8-projection-attempt1-verified.json').read_bytes())
    assert previous['status']=='prefill128_fp8_projection_failure_preserved' and previous['workspace_failure']['required_workspace_bytes']==5246879
    assert sha(results/'prefill128-fp8-projection-attempt1-evidence.zip')==previous['archive_sha256']
    parents={name:{'proof_sha256':sha(results/(name+'-verified.json')),'archive_sha256':data['archive_sha256'],'status':data['status']} for name,data in ((FAILED,q),(OLD,old))}
    parents['prefill128-fp8-projection-attempt1']={'proof_sha256':sha(results/'prefill128-fp8-projection-attempt1-verified.json'),'archive_sha256':previous['archive_sha256'],'status':previous['status']}
    p={'status':'prefill128_fp8_projection_prepared_gpu_unrun','files':files,'methods':methods,'runtime':w['runtime_dependencies_after'],'environment':environment,'fixture_manifest_sha256':hashlib.sha256(fraw).hexdigest(),'fixture_files':f['files'],'parents':parents,'failure_analysis_sha256':sha(results/(FAILED+'-analysis.json')),'vcvars':'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat','gpu_uuid':uuid,'cpu_tests':6,'scope':'Untimed standalone real-layer0 FP8 matrix control only. Preserve failed4MiB attempt. Allocate8MiB privately; only sharedM128 call receives8MiB, all M16/M32 and defaultM128 calls retain4MiB. Declare every native CUDA/header source before compilation. Reproduce archivedM16/M32 first; actualM16 input repeated8times givesM128, defaultM128 differences observed without admission, then actual AlgoCheck-qualified M16algorithm applied to32/128 and every BF16 output word must match repeatedM16. Independent CPU input cast/FP8 quantization/full-word replay on both machines. Frozen source/runtime/SDK/fixture, no GarnetDLL load/change, model, quality, HTTP, serving-memory or performance admission; cause of full-model mismatch remains unproven.'}
    path=results/'prefill128-fp8-workspace8-preparation.json';assert not path.exists();path.write_bytes(json.dumps(p,indent=2).encode());print(json.dumps({'status':p['status'],'source_pins':len(files),'fixture_manifest_sha256':p['fixture_manifest_sha256'],'sha256':sha(path)}))

def execute():
    from run_bf16_low_batch_native_build import idle
    import run_logged as ledger
    from prefill128_fp8_projection import replay_projection
    prep=ROOT/'results/prefill128-fp8-workspace8-preparation.json';p=json.loads(prep.read_bytes());fixture=ROOT/'results/actual-layer0-fp8-fixture'
    for name,h in p['files'].items():assert sha(ROOT/name)==h
    for name,h in p['runtime'].items():assert sha(ROOT/'build/bin'/name)==h
    for name,h in p['environment'].items():assert sha(Path(name))==h
    assert sha(fixture/'manifest.json')==p['fixture_manifest_sha256']
    for name,row in p['fixture_files'].items():assert sha(fixture/name)==row['sha256'] and (fixture/name).stat().st_size==row['bytes']
    for name,row in p['parents'].items():assert sha(ROOT/'results'/(name+'-verified.json'))==row['proof_sha256'] and sha(ROOT/'results'/(name+'-evidence.zip'))==row['archive_sha256']
    assert sha(ROOT/'results'/(FAILED+'-analysis.json'))==p['failure_analysis_sha256']
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists();idle();out.mkdir();(out/'build').mkdir();os.chdir(ROOT/'Garnet');code=1
    common=[prep,fixture/'manifest.json']+[ROOT/name for name in p['methods']]+[ROOT/'results'/(name+'-verified.json') for name in p['parents']]+[ROOT/'results'/(FAILED+'-analysis.json')]
    common += [ROOT/'Garnet'/name for name in ('src/backends/tensorrt/modelopt_cublaslt.cpp','src/backends/tensorrt/modelopt_cublaslt.h','src/cuda/modelopt_fp8.cu','src/cuda/modelopt_fp8.cuh')]
    ledger.source_identity=lambda:{'scope':'Explicit owned standalone FP8 control source inventory; unrelated untracked files excluded','owned_source_sha256':{name:sha(ROOT/name) for name in p['files']}}
    w={'status':'running','source_before':p['files'],'runtime_before':p['runtime'],'environment_before':p['environment'],'preparation_sha256':sha(prep),'parents':p['parents'],'fixture_manifest_sha256':p['fixture_manifest_sha256'],'cleanup_finished':False,'gpu_model_executed':False,'application_library_loaded':False,'performance_admitted':False,'scope':p['scope']}
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    save()
    try:
        code=ledger.run([sys.executable,'-m','unittest','test2026.cuda.qwen38_5090.test_prefill128_fp8_projection','test2026.cuda.qwen38_5090.test_prefill128_fp8_workspace8'],out/'cpu-ledger','correctness','Six synthetic input/word/provenance/workspace and complete-native-source closure faults; no model',common)
        if code:raise RuntimeError('CPU replay controls failed')
        code=ledger.run(['cmd.exe','/d','/c',str(M/'build_prefill128_fp8_workspace8.bat'),p['vcvars'],str(out/'build')],out/'build-ledger','build','Private explicit sm120 standalone CUDA control; no production build or DLL mutation',common)
        if code:raise RuntimeError('Standalone build failed')
        exe=out/'build/control.exe';w['executable_sha256']=sha(exe);save();os.environ['PATH']=str(ROOT/'tools/cuda13/nvidia/cu13/bin/x86_64')+os.pathsep+os.environ['PATH']
        for mode in ('control','shared'):
            idle();native=out/mode;native.mkdir();command=[str(exe),str(fixture),str(native)]
            if mode=='shared':command.append('--same-algorithm')
            code=ledger.run(command,out/(mode+'-ledger'),'correctness','Untimed real FP8 '+mode+' M16/M32/M128;2repeats; default128 observation, checked shared128 requires full exact output',common+[exe]);w[mode+'_exit_code']=code;save()
            if code:raise RuntimeError('Standalone '+mode+' failed; preserve all outputs')
            w[mode+'_replay']=replay_projection(lambda name:(fixture/name).read_bytes(),lambda name:(native/name).read_bytes(),mode);save()
        w['status']='prefill128_fp8_projection_collected_needs_replay'
    except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
    finally:
        idle();w.update(exit_code=code,source_after={name:sha(ROOT/name) for name in p['files']},runtime_after={name:sha(ROOT/'build/bin'/name) for name in p['runtime']},environment_after={name:sha(Path(name)) for name in p['environment']},fixture_after={name:sha(fixture/name) for name in p['fixture_files']},cleanup_finished=True);save()
        paths=[path for path in out.rglob('*') if path.is_file()]+common+[fixture/name for name in p['fixture_files'] if name!='weight-n5120-k6144.e4m3'];names={path.relative_to(ROOT).as_posix():path for path in paths}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for name,path in names.items():z.write(path,name)
            z.writestr('evidence-manifest.json',json.dumps({'files':{name:sha(path) for name,path in names.items()},'linked_parent_archives':{name:row['archive_sha256'] for name,row in p['parents'].items()},'external_weight':p['fixture_files']['weight-n5120-k6144.e4m3'],'scope':p['scope']},indent=2))
        print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(names)}),flush=True)
    return code

if __name__=='__main__':
    assert Path(__file__).resolve().parent==M.resolve();sys.path.insert(0,str(M));parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    else:raise SystemExit(execute())
