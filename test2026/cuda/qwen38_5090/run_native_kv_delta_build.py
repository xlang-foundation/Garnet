"""Install a reviewed native KV-delta handoff after operator closure, then build.

Preserves original source/DLL and restores both if compilation fails. This is
build qualification only; neither the new final-write helper nor model is admitted.
"""
from pathlib import Path
import hashlib,json,subprocess,sys,zipfile
root=Path('D:/CantorAI/5090-study-20261023');results=root/'results';repo=root/'Garnet';methods=repo/'test2026/cuda/qwen38_5090'
sys.path.insert(0,str(methods))
from run_logged import run
sha=lambda b:hashlib.sha256(b).hexdigest()
parent='canonical-kv-overlay-plugin-attempt1';proof_path=results/(parent+'-verified.json');proof=json.loads(proof_path.read_bytes())
assert proof['status']=='canonical_kv_overlay_operator_replayed' and proof['raw_context_arrays']==96 and len(proof['cases'])==24 and len(proof['engines'])==4
archive=results/(parent+'-evidence.zip');assert sha(archive.read_bytes())==proof['archive_sha256']
previous=json.loads((results/parent/'manifest.json').read_bytes());assert previous['status']=='operator_finished_needs_independent_replay' and previous['exit_code']==0
for stage in ('build','correctness'):
    events=[json.loads(l) for l in (results/parent/(stage+'-ledger/commands.jsonl')).read_text().splitlines()]
    assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
plan_path=methods/'evidence/native-kv-delta-source-handoff.json';plan=json.loads(plan_path.read_bytes())
staged=results/'native-kv-delta-source.zip';assert sha(staged.read_bytes())==plan['source_zip_sha256']
library=root/'build/bin/garnet.dll';old_library=previous['library_sha256'];retained=root/'build/bin/garnet-native-host-embedding-attempt1.dll'
assert sha(library.read_bytes())==sha(retained.read_bytes())==old_library==plan['original_library_sha256']
out=results/'native-kv-delta-build-attempt1';assert not out.exists()
old={};new={}
with zipfile.ZipFile(staged) as z:
    assert z.testzip() is None and set(z.namelist())==set(plan['candidate_source_sha256'])
    for n,h in plan['candidate_source_sha256'].items():
        new[n]=z.read(n);assert sha(new[n])==h
        p=repo/n
        if n in plan['base_normalized_sha256']:
            old[n]=p.read_bytes();assert sha(old[n].replace(b'\r\n',b'\n'))==plan['base_normalized_sha256'][n],('Source drift',n)
        else:assert not p.exists(),('New source path already exists',n)
with zipfile.ZipFile(archive) as z:
    for n,recipe in plan['qualified_source_derivation'].items():
        source=z.read('Garnet/test2026/cuda/qwen38_5090/'+recipe['source']).decode()
        for before,after in recipe['changes']:
            assert source.count(before)==1;source=source.replace(before,after)
        assert new[n]==source.encode()
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
launcher=root/'build-layer0-stages.bat';assert launcher.is_file()
out.mkdir()
for n,b in old.items():
    p=out/'original'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
receipt={'status':'installing_unbuilt_candidate','old_library_sha256':old_library,'retained_original_library':str(retained),'operator_archive_sha256':proof['archive_sha256'],
         'handoff_sha256':sha(plan_path.read_bytes()),'original_source_sha256':{n:sha(b) for n,b in old.items()},'candidate_source_sha256':plan['candidate_source_sha256']}
(out/'source-receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
code=None;failure=None
try:
    for n,b in new.items():
        (repo/n).write_bytes(b);assert sha((repo/n).read_bytes())==plan['candidate_source_sha256'][n]
    code=run(['cmd.exe','/d','/c',str(launcher)],out/'build-ledger','build','default-off-canonical-KV-and-final-write-native-integration',
        [launcher,library,plan_path,proof_path,Path(__file__)]+[repo/n for n in sorted(new)])
    if code:raise RuntimeError('Native KV delta build failed with exit '+str(code))
    for n,h in plan['candidate_source_sha256'].items():assert sha((repo/n).read_bytes())==h
    assert sha(library.read_bytes())!=old_library
    saved=root/'build/bin/garnet-native-kv-delta-attempt1.dll';assert not saved.exists();saved.write_bytes(library.read_bytes())
    receipt.update(status='native_kv_delta_build_finished_unqualified',exit_code=0,library_sha256=sha(library.read_bytes()),retained_candidate_library=str(saved))
except BaseException as error:
    failure=repr(error);receipt.update(status='native_kv_delta_build_failed_restored',exit_code=code,error=failure)
    if library.exists() and sha(library.read_bytes())!=old_library:
        p=out/'failed-garnet.dll';assert not p.exists();p.write_bytes(library.read_bytes());receipt['retained_failed_output_library']=str(p)
    library.write_bytes(retained.read_bytes());assert sha(library.read_bytes())==old_library
    for n,b in old.items():(repo/n).write_bytes(b)
    # New paths were verified absent before installation. Remove only those exact
    # candidate files, after retaining their bytes in the immutable staged ZIP.
    for n in set(new)-set(old):
        p=repo/n;assert p.resolve().is_relative_to(repo.resolve())
        if p.exists():assert sha(p.read_bytes())==plan['candidate_source_sha256'][n];p.unlink()
    receipt['restored_original_library_sha256']=sha(library.read_bytes())
finally:
    receipt['scope']='Native build only. Final-write helper, same-DLL default regression, compact whole-model state/IDs/hidden, streaming and HTTP gates still required. No speed/VRAM/2x admission.'
    (out/'manifest.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
print(json.dumps(receipt,indent=2))
if failure:raise SystemExit(code or 1)
