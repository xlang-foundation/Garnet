"""Install hash-reviewed opt-in sources, preserve old DLL/source, compile only."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,zipfile
root=Path('D:/CantorAI/5090-study-20261023');repo=root/'Garnet';results=root/'results';methods=repo/'test2026/cuda/qwen38_5090'
sys.path.insert(0,str(methods));from run_logged import run
sha=lambda b:hashlib.sha256(b).hexdigest()
prep_path=methods/'evidence/bf16-model-integration-build-preparation.json';prep=json.loads(prep_path.read_bytes())
for n,h in prep['methods'].items():assert sha((methods/n).read_bytes())==h
proof_path=results/'bf16-plugin-cpu-contract-attempt1-verified.json';proof=json.loads(proof_path.read_bytes())
assert proof['status']=='bf16_plugin_cpu_contract_replayed' and proof['contract']['profiles']==64
assert sha((results/'bf16-plugin-cpu-contract-attempt1-evidence.zip').read_bytes())==proof['archive_sha256']
assert sha((methods/'verify_bf16_plugin_cpu_contract.py').read_bytes())==proof['replay_verifier_sha256']
library=root/'build/bin/garnet.dll';retained=root/'build/bin/garnet-native-kv-delta-attempt1.dll'
assert sha(library.read_bytes())==sha(retained.read_bytes())==prep['original_library_sha256']
bundle=results/'bf16-model-integration-source.zip';assert sha(bundle.read_bytes())==prep['source_zip_sha256']
new={};old={}
with zipfile.ZipFile(bundle) as z:
    assert z.testzip() is None and set(z.namelist())==set(prep['candidate_source_sha256'])
    for n,h in prep['candidate_source_sha256'].items():
        p=repo/n;assert p.resolve().is_relative_to(repo.resolve())
        new[n]=z.read(n);assert sha(new[n])==h
        if p.exists():
            raw=p.read_bytes();base=prep['base_source'][n]
            assert sha(raw) in base['allowed_raw'] or sha(raw.replace(b'\r\n',b'\n')) in base['allowed_normalized'],('Source drift',n)
            old[n]=raw
        else:assert prep['base_source'][n]['may_be_absent'],('Required source missing',n)
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
name='bf16-model-integration-build-attempt1';out=results/name;assert not out.exists();out.mkdir()
manifest={'status':'installing_unbuilt_candidate','preparation_sha256':sha(prep_path.read_bytes()),'old_library_sha256':prep['original_library_sha256'],
          'source_zip_sha256':prep['source_zip_sha256'],'candidate_source_sha256':prep['candidate_source_sha256'],
          'original_source_sha256':{n:sha(v) for n,v in old.items()},'scope':prep['scope']}
with zipfile.ZipFile(out/'original-source.zip','x',zipfile.ZIP_DEFLATED) as z:
    for n,v in old.items():z.writestr(n,v)
before_dependencies={p.name:sha(p.read_bytes()) for p in (root/'build/bin').iterdir() if p.name in ('xlang3.exe','xlang3_runtime.dll','nvinfer_10.dll','cudart64_13.dll','cublasLt64_13.dll')}
manifest['runtime_dependencies_before_sha256']=before_dependencies
def save():(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
save();code=1;os.chdir(repo)
try:
    for n,v in new.items():(repo/n).parent.mkdir(parents=True,exist_ok=True);(repo/n).write_bytes(v)
    command=[sys.executable,'-m','unittest']+prep['cpu_tests']
    code=run(command,out/'cpu-ledger','correctness','BF16 admission/construction and unchanged default contracts',[prep_path,proof_path]+[repo/n for n in new])
    if code:raise RuntimeError('CPU model admission failed')
    code=run(['cmd.exe','/d','/c',str(methods/'build_bf16_model_integration.bat')],out/'build-ledger','build','Native opt-in BF16 helper and model lowering',[library,prep_path,proof_path,Path(__file__)]+[repo/n for n in new])
    if code:raise RuntimeError('Native BF16 integration build failed')
    assert {n:sha((repo/n).read_bytes()) for n in new}==prep['candidate_source_sha256']
    assert sha(library.read_bytes())!=prep['original_library_sha256']
    after={n:sha((root/'build/bin'/n).read_bytes()) for n in before_dependencies};assert after==before_dependencies
    saved=root/'build/bin/garnet-native-bf16-model-attempt1.dll';assert not saved.exists();saved.write_bytes(library.read_bytes())
    manifest.update(status='bf16_model_native_build_finished_unqualified',exit_code=0,library_sha256=sha(library.read_bytes()),retained_candidate_library=str(saved),runtime_dependencies_after_sha256=after)
except BaseException as error:
    manifest.update(status='bf16_model_native_build_failed_restored',exit_code=code,error=repr(error))
    if library.exists() and sha(library.read_bytes())!=prep['original_library_sha256']:(out/'failed-garnet.dll').write_bytes(library.read_bytes())
    library.write_bytes(retained.read_bytes());assert sha(library.read_bytes())==prep['original_library_sha256']
    skipped=[]
    for n,v in new.items():
        p=repo/n
        if p.exists() and sha(p.read_bytes())!=sha(v):skipped.append(n);continue
        if n in old:p.write_bytes(old[n])
        elif p.exists():assert p.resolve().is_relative_to(repo.resolve());p.unlink()
    manifest['restore_skipped_source_drift']=skipped;raise
finally:save()
print(json.dumps(manifest,indent=2));raise SystemExit(code)
