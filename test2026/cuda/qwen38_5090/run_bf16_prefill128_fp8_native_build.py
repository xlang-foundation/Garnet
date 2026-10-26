"""Build combined T128 plus default-off M128 FP8 candidate; restore baseline."""
from pathlib import Path
import hashlib,json,os,shutil,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090'
JOB='bf16-prefill128-fp8-native-build-attempt1'
digest=lambda raw:hashlib.sha256(raw).hexdigest()
sha=lambda path:digest(path.read_bytes())

def candidate_bytes(bundle,expected):
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(expected)
        assert set(z.namelist())==set(expected)
        result={name:z.read(name) for name in expected}
        for name,raw in result.items():
            assert name.startswith('src/') and '..' not in Path(name).parts and digest(raw)==expected[name]
    return result

def snapshot_targets(repo,candidate,base):
    assert set(candidate)==set(base)
    old={}
    for name,raw in candidate.items():
        target=repo/name;assert target.resolve().is_relative_to(repo.resolve()) and target.parent.is_dir()
        if base[name] is None:assert not target.exists()
        else:assert sha(target)==base[name];old[name]=target.read_bytes()
    return old

def restore_targets(repo,candidate,old):
    # Validate every installed byte first; retain any unexpected outside edit.
    for name,raw in candidate.items():assert (repo/name).read_bytes()==raw,('Candidate changed during build',name)
    for name in candidate:
        path=repo/name
        if name in old:path.write_bytes(old[name])
        else:path.unlink()
    assert all((repo/name).read_bytes()==raw for name,raw in old.items())
    assert all(not(repo/name).exists() for name in candidate if name not in old)

def validate_cuda_objects(before,after):
    assert len(before)==len(after)==34 and set(before)==set(after)
    changed=sorted(name for name in before if after[name]!=before[name])
    allowed={'qwen38_bf16_kv_low_batch_overlay.cu.obj','qwen38_bf16_prefill128_attention.cu.obj','qwen38_bf16_prefill128_commit.cu.obj'}
    assert all(Path(name).name in allowed for name in changed)
    return changed

def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path[:0]=[str(M),str(ROOT/'Garnet')]
    import run_logged as ledger
    from run_bf16_low_batch_native_build import idle
    prep=M/'evidence/bf16-prefill128-fp8-native-build-preparation.json';p=json.loads(prep.read_bytes())
    assert p['status']=='bf16_prefill128_fp8_native_build_prepared_unrun' and len(p['candidate_files'])==22
    for name,h in {**p['base_files'],**p['methods']}.items():assert sha(ROOT/name)==h,('Pre-build source drift',name)
    for name,h in p['runtime'].items():assert sha(ROOT/'build/bin'/name)==h
    for name,h in p['environment'].items():assert sha(Path(name))==h
    for job,row in p['parents'].items():
        proof=ROOT/'results'/(job+'-verified.json');q=json.loads(proof.read_bytes())
        assert sha(proof)==row['proof_sha256'] and q['status']==row['status'] and sha(ROOT/'results'/(job+'-evidence.zip'))==q['archive_sha256']==row['archive_sha256']
        w=json.loads((ROOT/'results'/job/'workflow.json').read_bytes());assert w['exit_code']==row['exit_code'] and w['cleanup_finished']
    cache=ROOT/'build/CMakeCache.txt';assert sha(cache)==p['cache_before_sha256']
    objects={path.relative_to(ROOT).as_posix():sha(path) for path in (ROOT/'build/garnet/CMakeFiles/garnet.dir').rglob('*.cu.obj')}
    assert objects==p['cuda_objects_before']
    assert all(flag in cache.read_bytes() for flag in (b'GARNET_STAGE_VENDOR_DLLS:BOOL=OFF',b'GARNET_QWEN35_WHOLE_PROGRAM_CUDA:BOOL=OFF'))
    bundle=ROOT/p['bundle'];assert sha(bundle)==p['bundle_sha256'];candidate=candidate_bytes(bundle,p['candidate_files'])
    old=snapshot_targets(ROOT/'Garnet',candidate,p['candidate_base'])
    library=ROOT/'build/bin/garnet.dll';baseline=ROOT/'build/bin/garnet-native-tokenizer-nfc-attempt1.dll'
    saved=ROOT/'build/bin/garnet-native-bf16-prefill128-fp8-m128-attempt1.dll';assert sha(library)==sha(baseline)==p['runtime']['garnet.dll'] and not saved.exists()
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists()
    idle();assert shutil.disk_usage(ROOT).free>=12<<30;out.mkdir();os.chdir(ROOT/'Garnet');code=1;installed={}
    with zipfile.ZipFile(out/'original-source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name,raw in old.items():z.writestr(name,raw)
    (out/'CMakeCache-before.txt').write_bytes(cache.read_bytes())
    w={'status':'running','preparation_sha256':sha(prep),'source_before':p['base_files'],'methods':p['methods'],'candidate_files':p['candidate_files'],'candidate_base':p['candidate_base'],
       'runtime_before':p['runtime'],'environment_before':p['environment'],'parents':p['parents'],'cuda_objects_before':objects,'cleanup_finished':False,'gpu_model_executed':False,'performance_admitted':False,'scope':p['scope']}
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    common=[prep,bundle,ROOT/p['proposal']]+[ROOT/name for name in p['methods']]+[ROOT/'results'/(job+'-verified.json') for job in p['parents']]
    inventory=dict(p['base_files']);inventory.update(p['methods'])
    ledger.source_identity=lambda:{'scope':'Explicit owned native build inventory; unrelated untracked files excluded','owned_source_sha256':{name:sha(ROOT/name) for name in inventory}}
    save()
    try:
        code=ledger.run([sys.executable,'-m','unittest',*p['test_ids']],out/'cpu-ledger','correctness','Retained CPU regressions plus transactional source snapshot/restore and bundle controls before native install',common)
        if code:raise RuntimeError('Pre-build CPU controls')
        for name,raw in candidate.items():
            installed[name]=raw;(ROOT/'Garnet'/name).write_bytes(raw)
        inventory.update({'Garnet/'+name:h for name,h in p['candidate_files'].items()})
        w['installed_source']={name:sha(ROOT/name) for name in inventory};save()
        code=ledger.run(['cmd.exe','/d','/c',str(M/'build_integrated_batch_top1_native.bat'),p['vcvars']],out/'build-ledger','build','Reconfigure existing Release/MT/sm120 tree for complete T128 route plus separately qualified M128 FP8 helper/plugin and graph counters',common)
        if code:raise RuntimeError('Full native candidate build')
        w['candidate_library_sha256']=sha(library);assert w['candidate_library_sha256']!=p['runtime']['garnet.dll'];save()
        after={path.relative_to(ROOT).as_posix():sha(path) for path in (ROOT/'build/garnet/CMakeFiles/garnet.dir').rglob('*.cu.obj')}
        added=sorted(set(after)-set(objects));changed=sorted(name for name in objects if after.get(name)!=objects[name])
        w.update(cuda_objects_built=after,added_cuda_objects=added,changed_cuda_objects=changed);save()
        assert not added and changed==validate_cuda_objects(objects,after)
        code=ledger.run(['cmd.exe','/d','/c',str(M/'build_bf16_prefill128_plugin_cpu.bat'),p['vcvars'],str(out/'wire-cpu')],out/'wire-cpu-ledger','correctness','Actual CPU-only plugin creator/wire/descriptor compatibility across new16profiles and old B1/B4/B8 dialects; no enqueue/model',common+[library])
        if code:raise RuntimeError('Actual T128 plugin wire contract')
        wire=json.loads((out/'wire-cpu/report.json').read_bytes())
        assert wire['status']=='prefill128_plugin_cpu_contract_passed' and wire['profiles']==16 and wire['bad_wires']==832 and wire['bad_descriptors']==496 and wire['bad_formats']==336 and wire['cross_dialect_rejections']==128 and wire['legacy_roundtrips']==96 and wire['unexpected_operator_calls']==0
        code=ledger.run(['cmd.exe','/d','/c',str(M/'build_fp8_m128_plugin_cpu.bat'),p['vcvars'],str(out/'fp8-wire-cpu')],out/'fp8-wire-cpu-ledger','correctness','Separate M128 FP8 wire/descriptor/default and legacy compatibility; no GPU stubs may run',common+[library])
        if code:raise RuntimeError('Actual M128 FP8 plugin wire contract')
        from test2026.cuda.qwen38_5090.fp8_m128_native_control import wire_contract as fp8_wire_contract
        w['fp8_wire_contract']=fp8_wire_contract(json.loads((out/'fp8-wire-cpu/report.json').read_bytes()));save()
        for name,argv in (
            ('selector-cpu',[sys.executable,str(M/'probe_native_tiled_scores_selector_cpu.py'),'--root',str(ROOT),'--output',str(out/'selector-cpu')]),
            ('embedded-selector-cpu',[sys.executable,str(M/'run_tiled_scores_embedded_selector.py'),'--root',str(ROOT),'--output',str(out/'embedded-selector-cpu')]),
            ('tokenizer-cpu',[sys.executable,str(M/'probe_native_tokenizer_frozen_cpu.py'),'--study',str(ROOT),'--oracle',str(ROOT/'results/tokenizer-nfc-cpu-oracles.zip'),'--output',str(out/'tokenizer-cpu')])):
            code=ledger.run(argv,out/(name+'-ledger'),'correctness','Actual candidate native '+name+' before any model',common+[library])
            if code:raise RuntimeError('Candidate '+name)
        assert all(sha(ROOT/name)==h for name,h in w['installed_source'].items())
        saved.write_bytes(library.read_bytes());assert sha(saved)==w['candidate_library_sha256'];w.update(status='bf16_prefill128_fp8_native_built_cpu_qualified_model_unqualified',candidate_saved=saved.relative_to(ROOT).as_posix())
    except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
    finally:
        if code and library.exists() and sha(library)!=p['runtime']['garnet.dll']:(out/'failed-garnet.dll').write_bytes(library.read_bytes())
        library.write_bytes(baseline.read_bytes())
        try:
            if installed:restore_targets(ROOT/'Garnet',installed,old)
            w['baseline_source_restored']=True
        except BaseException as error:w.update(baseline_source_restored=False,restoration_error=repr(error));code=1
        w.update(exit_code=code,source_after={name:sha(ROOT/name) for name in p['base_files']},methods_after={name:sha(ROOT/name) for name in p['methods']},runtime_after={name:sha(ROOT/'build/bin'/name) for name in p['runtime']},
                 environment_after={name:sha(Path(name)) for name in p['environment']},cache_after_sha256=sha(cache),baseline_library_restored=sha(library)==p['runtime']['garnet.dll'])
        save();idle();w['cleanup_finished']=True;save()
        paths=[path for path in out.rglob('*') if path.is_file()]+common
        if saved.exists():paths.append(saved)
        names={path.relative_to(ROOT).as_posix():path for path in paths}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for name,path in names.items():z.write(path,name)
            z.writestr('evidence-manifest.json',json.dumps({'files':{name:sha(path) for name,path in names.items()},'scope':p['scope']},indent=2))
        print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(names)}),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
