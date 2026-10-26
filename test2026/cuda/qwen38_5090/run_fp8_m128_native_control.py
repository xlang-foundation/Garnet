"""Isolated actual-helper and CPU wire qualification; no application DLL or model."""
from pathlib import Path
import argparse,hashlib,json,os,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='fp8-m128-native-control-attempt1';PARENT='prefill128-fp8-workspace8-attempt1'
NAMES=('actual_fp8_m128_native_control.cu','fp8_m128_observed_helper.cpp','build_fp8_m128_native_control.bat','build_fp8_m128_plugin_cpu.bat','fp8_m128_plugin_cpu_stubs.cpp','probe_fp8_m128_plugin_contract.cpp','fp8_m128_native_control.py','test_fp8_m128_native_control.py','run_fp8_m128_native_control.py','verify_fp8_m128_native_control.py','evidence/fp8-m128-native-control-derivation.json','evidence/fp8-m128-native-control-origin.json')
CANDIDATES=tuple('native_fp8_m128_candidate/'+name for name in ('modelopt_fp8_m128_canonical.h','modelopt_fp8_m128_canonical.cpp','modelopt_fp8_m128_plugin.h','modelopt_fp8_m128_plugin.cpp','modelopt_fp8_m128_profile.h'))
PRIMITIVES=('src/backends/tensorrt/modelopt_cublaslt.cpp','src/backends/tensorrt/modelopt_cublaslt.h','src/backends/tensorrt/modelopt_fp8_plugin.cpp','src/backends/tensorrt/modelopt_fp8_plugin.h','src/cuda/modelopt_fp8.cu','src/cuda/modelopt_fp8.cuh')
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()

def prepare():
    from run_bf16_low_batch_native_build import idle
    idle();previous=json.loads((ROOT/'results/prefill128-fp8-workspace8-preparation.json').read_bytes());w=json.loads((ROOT/'results'/PARENT/'workflow.json').read_bytes());q=json.loads((ROOT/'results'/(PARENT+'-verified.json')).read_bytes())
    assert w['exit_code']==0 and w['cleanup_finished'] and q['status']=='prefill128_fp8_real_projection_replayed' and q['archive_sha256']==sha(ROOT/'results'/(PARENT+'-evidence.zip'))
    assert w['source_before']==w['source_after']==previous['files'] and w['runtime_before']==w['runtime_after']==previous['runtime'] and w['environment_before']==w['environment_after']==previous['environment']
    methods={'Garnet/test2026/cuda/qwen38_5090/'+name:sha(M/name) for name in NAMES+CANDIDATES}
    files={**previous['files'],**methods,**{'Garnet/'+name:sha(ROOT/'Garnet'/name) for name in PRIMITIVES}}
    for name,h in files.items():assert sha(ROOT/name)==h
    for name,h in previous['runtime'].items():assert sha(ROOT/'build/bin'/name)==h
    for name,h in previous['environment'].items():assert sha(Path(name))==h
    origin=json.loads((M/'evidence/fp8-m128-native-control-origin.json').read_bytes())
    for name,row in origin['candidate_copies'].items():assert methods['Garnet/test2026/cuda/qwen38_5090/'+name]==row['sha256'] and row['production_source'].startswith('src/backends/tensorrt/')
    assert set(origin['candidate_copies'])==set(CANDIDATES)
    fixture=ROOT/'results/actual-layer0-fp8-fixture';assert sha(fixture/'manifest.json')==previous['fixture_manifest_sha256']
    for name,row in previous['fixture_files'].items():assert sha(fixture/name)==row['sha256'] and (fixture/name).stat().st_size==row['bytes']
    p={'status':'fp8_m128_native_control_prepared_unrun','files':files,'methods':methods,'runtime':previous['runtime'],'environment':previous['environment'],'fixture_manifest_sha256':previous['fixture_manifest_sha256'],'fixture_files':previous['fixture_files'],'vcvars':previous['vcvars'],'parent':{'job':PARENT,'proof_sha256':sha(ROOT/'results'/(PARENT+'-verified.json')),'archive_sha256':q['archive_sha256']},'cpu_tests':8,'scope':'Standalone exact production helper copies plus distinct plugin wire controls. Real retained FP8 GDN matrix; compare all BF16 output words/casts/quantization with qualified8MiB control. Strict actual algorithm35/opaque/5246879B requirement/8MiB allocation and actual loaded DLL identity. Preserve full application source/runtime/SDK/fixture. No Garnet DLL load/change, full-model cause, quality, serving memory or performance qualification.'}
    path=ROOT/'results/fp8-m128-native-control-preparation.json';assert not path.exists();path.write_bytes(json.dumps(p,indent=2).encode());print(json.dumps({'status':p['status'],'source_pins':len(files),'sha256':sha(path)}))

def execute():
    from run_bf16_low_batch_native_build import idle
    from test2026.cuda.qwen38_5090.fp8_m128_native_control import wire_contract,native_log,replay
    import run_logged as ledger
    prep=ROOT/'results/fp8-m128-native-control-preparation.json';p=json.loads(prep.read_bytes());fixture=ROOT/'results/actual-layer0-fp8-fixture'
    def validate():
        for name,h in p['files'].items():assert sha(ROOT/name)==h
        for name,h in p['runtime'].items():assert sha(ROOT/'build/bin'/name)==h
        for name,h in p['environment'].items():assert sha(Path(name))==h
        assert sha(fixture/'manifest.json')==p['fixture_manifest_sha256']
        for name,row in p['fixture_files'].items():assert sha(fixture/name)==row['sha256'] and (fixture/name).stat().st_size==row['bytes']
        assert sha(ROOT/'results'/(PARENT+'-verified.json'))==p['parent']['proof_sha256'] and sha(ROOT/'results'/(PARENT+'-evidence.zip'))==p['parent']['archive_sha256']
    validate();idle();out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists();out.mkdir();(out/'build').mkdir();os.chdir(ROOT/'Garnet');code=1
    common=[prep,fixture/'manifest.json',ROOT/'results'/(PARENT+'-verified.json')]+[ROOT/name for name in p['methods']]+[ROOT/'Garnet'/name for name in PRIMITIVES]
    ledger.source_identity=lambda:{'scope':'Explicit owned actual FP8 M128 helper control inventory; unrelated untracked files excluded','owned_source_sha256':{name:sha(ROOT/name) for name in p['files']}}
    w={'status':'running','preparation_sha256':sha(prep),'source_before':p['files'],'runtime_before':p['runtime'],'environment_before':p['environment'],'parent':p['parent'],'cleanup_finished':False,'gpu_model_executed':False,'application_library_loaded':False,'performance_admitted':False,'scope':p['scope']}
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    save()
    try:
        code=ledger.run([sys.executable,'-m','unittest',*('test2026.cuda.qwen38_5090.'+name for name in ('test_prefill128_fp8_projection','test_prefill128_fp8_workspace8','test_fp8_m128_native_control'))],out/'cpu-ledger','correctness','8 CPU input/full-word/source and actual-wire/algorithm/workspace parser fault controls; no native runtime/model',common)
        if code:raise RuntimeError('CPU controls')
        code=ledger.run(['cmd.exe','/d','/c',str(M/'build_fp8_m128_plugin_cpu.bat'),p['vcvars'],str(out/'wire-cpu')],out/'wire-cpu-ledger','correctness','Actual distinct M128 creator/wire/descriptor/default and cross-wire compatibility; GPU stubs must remain untouched',common)
        if code:raise RuntimeError('Actual plugin wire control')
        w['wire_contract']=wire_contract(json.loads((out/'wire-cpu/report.json').read_bytes()));save()
        code=ledger.run(['cmd.exe','/d','/c',str(M/'build_fp8_m128_native_control.bat'),p['vcvars'],str(out/'build')],out/'build-ledger','build','Explicit sm120 standalone observer plus separate actual native helper translation unit; no production build or application DLL',common)
        if code:raise RuntimeError('Actual helper standalone build')
        exe=out/'build/control.exe';w['executable_sha256']=sha(exe);save();os.environ['PATH']=str(ROOT/'tools/cuda13/nvidia/cu13/bin/x86_64')+os.pathsep+os.environ['PATH']
        with zipfile.ZipFile(ROOT/'results'/(PARENT+'-evidence.zip')) as prior:
            for mode in ('control','native'):
                idle();native=out/mode;native.mkdir();command=[str(exe),str(fixture),str(native)]
                if mode=='native':command.append('--native-m128')
                code=ledger.run(command,out/(mode+'-ledger'),'correctness','Untimed '+mode+' actual FP8 helper M16/M32/M128 with 2 repeats; complete exact outputs required for native',common+[exe]);w[mode+'_exit_code']=code;save()
                if code:raise RuntimeError('Actual '+mode+' GPU control failed')
                w[mode+'_algorithms']=native_log((out/(mode+'-ledger/commands.log')).read_text(),mode)
                prefix='results/'+PARENT+'/'+('shared' if mode=='native' else 'control')+'/'
                w[mode+'_replay']=replay(lambda name:(fixture/name).read_bytes(),lambda name:(native/name).read_bytes(),mode,lambda name:prior.read(prefix+name));save()
        validate();w['status']='fp8_m128_actual_helper_collected_needs_replay'
    except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
    finally:
        idle();w.update(exit_code=code,source_after={name:sha(ROOT/name) for name in p['files']},runtime_after={name:sha(ROOT/'build/bin'/name) for name in p['runtime']},environment_after={name:sha(Path(name)) for name in p['environment']},fixture_after={name:sha(fixture/name) for name in p['fixture_files']},cleanup_finished=True);save()
        paths=[path for path in out.rglob('*') if path.is_file()]+common+[fixture/name for name in p['fixture_files'] if name!='weight-n5120-k6144.e4m3'];names={path.relative_to(ROOT).as_posix():path for path in paths}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for name,path in names.items():z.write(path,name)
            z.writestr('evidence-manifest.json',json.dumps({'files':{name:sha(path) for name,path in names.items()},'linked_parent_archive':p['parent'],'external_weight':p['fixture_files']['weight-n5120-k6144.e4m3'],'scope':p['scope']},indent=2))
        print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(names)}),flush=True)
    return code

if __name__=='__main__':
    assert Path(__file__).resolve().parent==M.resolve();sys.path[:0]=[str(M),str(ROOT/'Garnet')];parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    else:raise SystemExit(execute())
