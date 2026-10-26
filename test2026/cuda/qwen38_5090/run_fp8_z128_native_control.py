"""Exclusive actual typed Z helper and separate CPU plugin contract qualification."""
from pathlib import Path
import argparse,hashlib,json,os,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='fp8-z128-native-control-attempt1';PARENT='gdn-z-native-control-attempt1'
NAMES=('actual_fp8_z128_native_control.cu','fp8_z128_observed_helper.cpp','build_fp8_z128_native_control.bat','build_fp8_z128_plugin_cpu.bat','fp8_z128_plugin_cpu_stubs.cpp','probe_fp8_z128_plugin_contract.cpp','fp8_z128_native_control.py','test_fp8_z128_native_control.py','test_fp8_z128_native_source.py','run_fp8_z128_native_control.py','verify_fp8_z128_native_control.py','evidence/fp8-z128-native-source-derivation.json','evidence/fp8-z128-native-control-derivation.json','evidence/fp8-z128-native-control-origin.json','evidence/gdn-z-native-control-attempt1-verified.json')
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()

def prepare():
    from run_bf16_low_batch_native_build import idle
    from run_gdn_z_projection import live_fixture
    idle();results=ROOT/'results';w=json.loads((results/PARENT/'workflow.json').read_bytes());q=json.loads((results/(PARENT+'-verified.json')).read_bytes());previous=json.loads((results/'gdn-z-native-control-preparation.json').read_bytes())
    assert w['exit_code']==0 and w['cleanup_finished'] and w['before']==w['after'] and q['status']=='real_gdn_z_default_and_m32_shared_algorithm_fully_replayed' and q['archive_sha256']==sha(results/(PARENT+'-evidence.zip'))
    assert previous['files']==w['after']['source'] and previous['runtime']==w['after']['runtime'] and previous['environment']==w['after']['environment']
    origin=json.loads((M/'evidence/fp8-z128-native-control-origin.json').read_bytes());copies=origin['candidate_copies'];assert len(copies)==10
    methods={'Garnet/test2026/cuda/qwen38_5090/'+name:sha(M/name) for name in NAMES+tuple(copies)}
    for name,row in copies.items():assert methods['Garnet/test2026/cuda/qwen38_5090/'+name]==row['sha256'] and row['production_source'].startswith('src/backends/tensorrt/')
    files={**previous['files'],**methods}
    for name,h in files.items():assert sha(ROOT/name)==h,('Source drift',name)
    runtime=previous['runtime'];environment=dict(previous['environment']);trt=ROOT/'tools/TensorRT'
    for name,h in runtime.items():assert sha(ROOT/'build/bin'/name)==h
    for name,h in environment.items():assert sha(Path(name))==h
    for name in ('include/NvInfer.h','include/NvInferRuntime.h','include/NvInferRuntimeCommon.h','include/NvInferRuntimeBase.h','include/NvInferPlugin.h','include/NvInferVersion.h','lib/nvinfer_10.lib','bin/nvinfer_10.dll'):
        path=trt/name;environment[str(path)]=sha(path)
    fixture=results/'actual-gdn-z-128-fixture';f=json.loads((fixture/'manifest.json').read_bytes());assert sha(fixture/'manifest.json')==previous['fixture_manifest_sha256'];live_fixture(fixture,f)
    opaque={row['actual_opaque'] for profile in q['profiles'] for row in profile['algorithm_replay']['calls'] if row['m']==32 or profile['algorithm_replay']['mode']=='shared'};assert len(opaque)==1
    p={'status':'fp8_z128_native_control_prepared_gpu_unrun','files':files,'methods':methods,'runtime':runtime,'environment':environment,'fixture_manifest_sha256':previous['fixture_manifest_sha256'],'fixture_files':previous['fixture_files'],'checkpoint_config_sha256':previous['checkpoint_config_sha256'],'checkpoint_index_sha256':previous['checkpoint_index_sha256'],'gpu_uuid':previous['gpu_uuid'],'vcvars':previous['vcvars'],'parent':{'job':PARENT,'proof_sha256':sha(results/(PARENT+'-verified.json')),'archive_sha256':q['archive_sha256']},'expected_m32_opaque':opaque.pop(),'cpu_tests':13,'candidate_copies':copies,'scope':'Exclusive untimed actual typed Z128 helper in separate observedTU plus distinct actualplugin CPUwire contract withunusedGPUstubs. Exactnew fiveproduction-source copies andold fivecomparisoncopies only; no production source/DLL/build/SDK mutation. CanonicalM32 opaque67pinned/actual128AlgoCheck/4MiBnative allocation mustexactlymatchreal fourM32outputs atbothlayers/tworepeats, bothdefaultprofilesreproducedfirst. CompleteF32/BF16/FP8 byte/word replay,15realinvalidargs perlayer withoutmatmul. No actualapplicationDLL/model/quality/HTTP/servingmemory/timing/2xadmission.'}
    target=results/'fp8-z128-native-control-preparation.json';assert not target.exists();target.write_bytes(json.dumps(p,indent=2).encode());print(json.dumps({'status':p['status'],'source_pins':len(files),'candidate_copies':len(copies),'preparation_sha256':sha(target)}))

def execute():
    from run_bf16_low_batch_native_build import idle
    from run_gdn_z_projection import live_fixture
    from test2026.cuda.qwen38_5090.fp8_z128_native_control import replay_algorithms,replay_arrays
    import run_logged as ledger
    results=ROOT/'results';prep=results/'fp8-z128-native-control-preparation.json';p=json.loads(prep.read_bytes());fixture=results/'actual-gdn-z-128-fixture';f=json.loads((fixture/'manifest.json').read_bytes())
    def snapshot():return {'source':{name:sha(ROOT/name) for name in p['files']},'runtime':{name:sha(ROOT/'build/bin'/name) for name in p['runtime']},'environment':{name:sha(Path(name)) for name in p['environment']},'fixture':{name:sha(fixture/name) for name in p['fixture_files']},'checkpoints':{'config':sha(ROOT/'models/Qwen3.8-27B-NVFP4/config.json'),'index':sha(ROOT/'models/Qwen3.8-27B-NVFP4/model.safetensors.index.json')}}
    expected={'source':p['files'],'runtime':p['runtime'],'environment':p['environment'],'fixture':{name:row['sha256'] for name,row in p['fixture_files'].items()},'checkpoints':{'config':p['checkpoint_config_sha256'],'index':p['checkpoint_index_sha256']}}
    assert snapshot()==expected and sha(fixture/'manifest.json')==p['fixture_manifest_sha256'];live_fixture(fixture,f)
    assert sha(results/(PARENT+'-evidence.zip'))==p['parent']['archive_sha256'] and sha(results/(PARENT+'-verified.json'))==p['parent']['proof_sha256']
    out=results/JOB;archive=results/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists();idle();out.mkdir();(out/'build').mkdir();os.chdir(ROOT/'Garnet');code=1
    common=[prep,fixture/'manifest.json',results/(PARENT+'-verified.json')]+[ROOT/name for name in p['methods']]
    common += [ROOT/'Garnet'/name for name in ('src/backends/tensorrt/modelopt_cublaslt.cpp','src/backends/tensorrt/modelopt_cublaslt.h','src/backends/tensorrt/modelopt_fp8_plugin.cpp','src/backends/tensorrt/modelopt_fp8_plugin.h','src/cuda/modelopt_fp8.cu','src/cuda/modelopt_fp8.cuh')]
    ledger.source_identity=lambda:{'scope':'Explicit owned typed Z128 native control inputs; unrelated untracked excluded','owned_source_sha256':{name:sha(ROOT/name) for name in p['files']}}
    w={'status':'running','before':expected,'preparation_sha256':sha(prep),'parent':p['parent'],'fixture_manifest_sha256':p['fixture_manifest_sha256'],'application_library_loaded':False,'gpu_model_executed':False,'model_or_performance_qualified':False,'cleanup_finished':False,'scope':p['scope']}
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    save()
    try:
        modules=['test2026.cuda.qwen38_5090.'+name for name in ('test_fp8_z128_native_source','test_fp8_z128_native_control','test_gdn_z_projection','test_gdn_z_algorithm_log')]
        code=ledger.run([sys.executable,'-m','unittest',*modules,'-v'],out/'cpu-ledger','correctness','13 source/real128/completeword/context/algorithm/workspace CPUfault controls',common)
        if code:raise RuntimeError('CPU controls failed')
        code=ledger.run(['cmd.exe','/d','/c',str(M/'build_fp8_z128_plugin_cpu.bat'),p['vcvars'],str(out/'wire-cpu')],out/'wire-cpu-ledger','correctness','Actual separate Z plugin creator/wire/constructor/descriptor/offtarget contracts; unusedGPUstubs',common)
        if code:raise RuntimeError('Actual CPU plugin wire control failed')
        wire=json.loads((out/'wire-cpu/report.json').read_bytes());assert wire['status']=='fp8_z128_plugin_cpu_contract_passed' and wire['unexpected_gpu_calls']==0 and wire['workspace_bytes']==4849664 and wire['wire_policy']==0x5a313238
        w['wire_contract']=wire;save()
        code=ledger.run(['cmd.exe','/d','/c',str(M/'build_fp8_z128_native_control.bat'),p['vcvars'],str(out/'build')],out/'build-ledger','build','Private sm120 actual typedZ helper plus separate observedTU; no productionDLL/SDK mutations',common)
        if code:raise RuntimeError('Actual typed helper build failed')
        exe=out/'build/control.exe';w['executable_sha256']=sha(exe);save();os.environ['PATH']=str(ROOT/'tools/cuda13/nvidia/cu13/bin/x86_64')+os.pathsep+os.environ['PATH']
        for mode in ('control','native'):
            for layer in (0,1):
                idle();label=mode+'-layer'+str(layer);native=out/label;native.mkdir();command=[str(exe),str(fixture/('layer'+str(layer))),str(native)]
                if mode=='native':command.append('--native-z128')
                code=ledger.run(command,out/(label+'-ledger'),'correctness','Untimed typedZ '+mode+' layer'+str(layer)+' real128rows/2repeats, fulloutputs/15invalidargs fornative',common+[exe]);w[label+'_exit_code']=code;save()
                if code:raise RuntimeError('Actual '+label+' failed; preserve rawoutputs')
                w[label+'_algorithms']=replay_algorithms((out/(label+'-ledger/commands.log')).read_bytes(),mode,p['expected_m32_opaque']);w[label+'_replay']=replay_arrays(lambda name:(fixture/('layer'+str(layer))/name).read_bytes(),lambda name:(native/name).read_bytes(),mode);save()
        w['status']='fp8_z128_actual_typed_helper_collected_needs_replay'
    except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
    finally:
        idle();w.update(exit_code=code,after=snapshot(),cleanup_finished=True);save();assert w['before']==w['after']
        selected=list(out.rglob('*'))+common+[fixture/name for name in p['fixture_files'] if not name.endswith('.e4m3')];names={path.relative_to(ROOT).as_posix():path for path in selected if path.is_file()}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for name,path in names.items():z.write(path,name)
            z.writestr('evidence-manifest.json',json.dumps({'files':{name:sha(path) for name,path in names.items()},'linked_parent_archive':p['parent'],'external_weight_files':{name:row for name,row in p['fixture_files'].items() if name.endswith('.e4m3')},'scope':p['scope']},indent=2))
        print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(names)}),flush=True)
    return code

if __name__=='__main__':
    assert Path(__file__).resolve().parent==M.resolve();sys.path[:0]=[str(M),str(ROOT/'Garnet')];parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    else:raise SystemExit(execute())
