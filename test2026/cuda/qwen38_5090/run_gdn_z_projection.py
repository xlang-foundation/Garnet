"""Exclusive untimed real128 Z control; no production source/DLL/SDK mutations."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='gdn-z-native-control-attempt1';PARENT='gdn-chunk-stage-witness-attempt2';EXTRACT='gdn-z-fixture-extraction-attempt1'
NAMES=('actual_gdn_z_projection.cu','build_gdn_z_projection.bat','run_gdn_z_projection.py','verify_gdn_z_projection.py','gdn_z_projection.py','prepare_gdn_z_projection_fixture.py','test_gdn_z_projection.py','verify_gdn_z_fixture.py','gdn_z_algorithm_log.py','test_gdn_z_algorithm_log.py')
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()

def live_fixture(fixture,f):
    from test2026.cuda.qwen38_5090.gdn_z_projection import checkpoint_tensor
    checkpoint=ROOT/'models/Qwen3.8-27B-NVFP4';assert sha(checkpoint/'config.json')==f['config_sha256'] and sha(checkpoint/'model.safetensors.index.json')==f['index_sha256']
    index=json.loads((checkpoint/'model.safetensors.index.json').read_bytes())
    for name,row in f['files'].items():assert (fixture/name).stat().st_size==row['bytes'] and sha(fixture/name)==row['sha256']
    for row in f['layers']:
        for key,info in row['tensors'].items():
            raw,metadata=checkpoint_tensor(checkpoint,index,key,info['dtype'],info['shape'])
            assert {k:info[k] for k in metadata}==metadata
            assert hashlib.sha256(raw).hexdigest()==info['sha256']
            assert raw==(fixture/('layer'+str(row['layer']))/info['file']).read_bytes()

def prepare():
    from run_bf16_low_batch_native_build import idle
    idle();results=ROOT/'results';parent=json.loads((results/PARENT/'workflow.json').read_bytes());q=json.loads((results/(PARENT+'-verified.json')).read_bytes())
    assert parent['exit_code']==0 and parent['cleanup_finished'] and parent['baseline_source_restored'] and parent['baseline_library_restored']
    assert q['status']=='gdn_chunk_stage_witness_arrays_replayed' and q['archive_sha256']==sha(results/(PARENT+'-evidence.zip'))
    verifier=M/'verify_gdn_z_fixture.py';subprocess.run([sys.executable,str(verifier),'--proofs',str(results)],check=True)
    extracted=json.loads((results/(EXTRACT+'-verified.json')).read_bytes());assert extracted['status']=='gdn_z_fixture_stage_vectors_and_extraction_provenance_replayed' and extracted['archive_sha256']==sha(results/(EXTRACT+'-evidence.zip'))
    fixture=results/'actual-gdn-z-128-fixture';fraw=(fixture/'manifest.json').read_bytes();f=json.loads(fraw);assert hashlib.sha256(fraw).hexdigest()==extracted['fixture_manifest_sha256'];live_fixture(fixture,f)
    files={**parent['source_after_sha256'],**{'Garnet/test2026/cuda/qwen38_5090/'+name:sha(M/name) for name in NAMES}}
    for name,h in files.items():assert sha(ROOT/name)==h,('Source drift',name)
    runtime=parent['runtime_dependencies_after'];environment=dict(parent['environment_after']);cuda=ROOT/'tools/cuda13/nvidia/cu13'
    for name,h in runtime.items():assert sha(ROOT/'build/bin'/name)==h
    for name,h in environment.items():assert sha(Path(name))==h
    for name in ('include/cublasLt.h','include/cublas_v2.h','include/cuda_runtime.h','lib/x64/cublasLt.lib','bin/x86_64/cublasLt64_13.dll'):
        path=cuda/name;environment[str(path)]=sha(path)
    assert environment[str(cuda/'bin/x86_64/cublasLt64_13.dll')]==runtime['cublasLt64_13.dll']
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6'
    p={'status':'gdn_z_native_control_prepared_gpu_unrun','files':files,'methods':{'Garnet/test2026/cuda/qwen38_5090/'+name:sha(M/name) for name in NAMES},'runtime':runtime,'environment':environment,'fixture_manifest_sha256':hashlib.sha256(fraw).hexdigest(),'fixture_files':f['files'],'parents':{job:{'archive_sha256':sha(results/(job+'-evidence.zip')),'proof_sha256':sha(results/(job+'-verified.json'))} for job in (PARENT,EXTRACT)},'checkpoint_config_sha256':f['config_sha256'],'checkpoint_index_sha256':f['index_sha256'],'gpu_uuid':uuid,'vcvars':'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat','scope':'Exclusive untimed actual GDN in_proj_z N6144K5120 on real first128inputs atlayers0/1. CPU-recheck actual checkpoint rawweight/scale bytes andheaders againstfixture, sixsyntheticfaultcontrols. Reproduce bothdefault fourM32/M128 profiles at4MiB first, bothlayers/tworepeats, then observeactualM32algorithm and AlgoCheck actualM128 withexplicit8MiBcap. Shared intervention must exactly matchallfourM32outputs andCPU inputcast/FP8quantization. Defaulthelper andCUDA sources unchanged, noGarnetDLL/model/805quality/HTTP/servingVRAM/timing/2xadmission. No assumption M32algorithm or8MiB willqualify.'}
    path=results/'gdn-z-native-control-preparation.json';assert not path.exists();path.write_bytes(json.dumps(p,indent=2).encode());print(json.dumps({'status':p['status'],'source_pins':len(files),'preparation_sha256':sha(path)}))

def execute():
    from run_bf16_low_batch_native_build import idle
    import run_logged as ledger
    from test2026.cuda.qwen38_5090.gdn_z_projection import replay
    from test2026.cuda.qwen38_5090.gdn_z_algorithm_log import replay_log
    results=ROOT/'results';prep=results/'gdn-z-native-control-preparation.json';p=json.loads(prep.read_bytes());fixture=results/'actual-gdn-z-128-fixture';f=json.loads((fixture/'manifest.json').read_bytes())
    def snapshot():return {'source':{name:sha(ROOT/name) for name in p['files']},'runtime':{name:sha(ROOT/'build/bin'/name) for name in p['runtime']},'environment':{name:sha(Path(name)) for name in p['environment']},'fixture':{name:sha(fixture/name) for name in p['fixture_files']},'checkpoints':{'config':sha(ROOT/'models/Qwen3.8-27B-NVFP4/config.json'),'index':sha(ROOT/'models/Qwen3.8-27B-NVFP4/model.safetensors.index.json')}}
    expected={'source':p['files'],'runtime':p['runtime'],'environment':p['environment'],'fixture':{name:row['sha256'] for name,row in p['fixture_files'].items()},'checkpoints':{'config':p['checkpoint_config_sha256'],'index':p['checkpoint_index_sha256']}}
    assert snapshot()==expected and sha(fixture/'manifest.json')==p['fixture_manifest_sha256'];live_fixture(fixture,f)
    for job,row in p['parents'].items():assert sha(results/(job+'-evidence.zip'))==row['archive_sha256'] and sha(results/(job+'-verified.json'))==row['proof_sha256']
    out=results/JOB;archive=results/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists();idle();out.mkdir();(out/'build').mkdir();os.chdir(ROOT/'Garnet');code=1
    common=[prep,fixture/'manifest.json']+[ROOT/name for name in p['methods']]+[results/(job+'-verified.json') for job in p['parents']]
    common += [ROOT/'Garnet'/name for name in ('src/backends/tensorrt/modelopt_cublaslt.cpp','src/backends/tensorrt/modelopt_cublaslt.h','src/cuda/modelopt_fp8.cu','src/cuda/modelopt_fp8.cuh')]
    ledger.source_identity=lambda:{'scope':'Explicit owned standalone Z control source; unrelated untracked excluded','owned_source_sha256':{name:sha(ROOT/name) for name in p['files']}}
    w={'status':'running','before':expected,'preparation_sha256':sha(prep),'parents':p['parents'],'fixture_manifest_sha256':p['fixture_manifest_sha256'],'native_projection_control_executed':False,'gpu_model_executed':False,'application_library_loaded':False,'model_or_performance_qualified':False,'cleanup_finished':False,'scope':p['scope']}
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    save()
    try:
        code=ledger.run([sys.executable,'-m','unittest','test2026.cuda.qwen38_5090.test_gdn_z_projection','test2026.cuda.qwen38_5090.test_gdn_z_algorithm_log','-v'],out/'cpu-ledger','correctness','Eight synthetic Z fullword/input/shape/scale/lineage/algorithm/workspace controls',common)
        if code:raise RuntimeError('CPU replay controls failed')
        code=ledger.run(['cmd.exe','/d','/c',str(M/'build_gdn_z_projection.bat'),p['vcvars'],str(out/'build')],out/'build-ledger','build','Private explicit sm120 actualZ control; no applicationDLL/productionbuild/SDK mutation',common)
        if code:raise RuntimeError('Standalone build failed')
        exe=out/'build/control.exe';w['executable_sha256']=sha(exe);save();os.environ['PATH']=str(ROOT/'tools/cuda13/nvidia/cu13/bin/x86_64')+os.pathsep+os.environ['PATH']
        for mode in ('control','shared'):
            for layer in (0,1):
                idle();native=out/(mode+'-layer'+str(layer));native.mkdir();command=[str(exe),str(fixture/('layer'+str(layer))),str(native)]
                if mode=='shared':command.append('--m32-shared')
                w['native_projection_control_executed']=True;save()
                code=ledger.run(command,out/(mode+'-layer'+str(layer)+'-ledger'),'correctness','Untimed realZ '+mode+' layer'+str(layer)+' actual128rows/fourM32;2repeats; completeoutputs',common+[exe]);w[mode+'_layer'+str(layer)+'_exit_code']=code;save()
                if code:raise RuntimeError('Actual '+mode+' layer'+str(layer)+' failed; preserve outputs')
                w[mode+'_layer'+str(layer)+'_algorithm_replay']=replay_log((out/(mode+'-layer'+str(layer)+'-ledger/commands.log')).read_bytes(),mode);save()
                w[mode+'_layer'+str(layer)+'_replay']=replay(lambda name:(fixture/('layer'+str(layer))/name).read_bytes(),lambda name:(native/name).read_bytes(),mode);save()
        w['status']='real_gdn_z_native_control_collected_needs_replay'
    except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
    finally:
        idle();w.update(exit_code=code,after=snapshot(),cleanup_finished=True);save();assert w['after']==w['before']
        selected=list(out.rglob('*'))+common+[fixture/name for name in p['fixture_files'] if not name.endswith('.e4m3')]
        names={path.relative_to(ROOT).as_posix():path for path in selected if path.is_file()}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for name,path in names.items():z.write(path,name)
            z.writestr('evidence-manifest.json',json.dumps({'files':{name:sha(path) for name,path in names.items()},'linked_parent_archives':{job:row['archive_sha256'] for job,row in p['parents'].items()},'external_weight_files':{name:row for name,row in p['fixture_files'].items() if name.endswith('.e4m3')},'scope':p['scope']},indent=2))
        print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'files':len(names)}),flush=True)
    return code

if __name__=='__main__':
    assert Path(__file__).resolve().parent==M.resolve();sys.path[:0]=[str(M),str(ROOT/'Garnet')];parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    else:raise SystemExit(execute())
