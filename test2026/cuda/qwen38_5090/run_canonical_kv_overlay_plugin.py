"""Qualify a standalone canonical KV TensorRT operator; no Garnet DLL rebuild."""
from pathlib import Path
import hashlib,json,re,shutil,subprocess,sys,zipfile
root=Path('D:/CantorAI/5090-study-20261023');methods=root/'Garnet/test2026/cuda/qwen38_5090'
sys.path.insert(0,str(methods))
from run_logged import run
sha=lambda b:hashlib.sha256(b).hexdigest()
parent='attention-kv-delta-strided-attempt1';proof_path=root/'results'/(parent+'-verified.json')
proof=json.loads(proof_path.read_bytes())
assert proof['status']=='attention_kv_delta_strided_replayed' and len(proof['cases'])==12 and proof['raw_context_arrays']==144 and proof['raw_prefix_count_vectors']==36 and proof['retained_parent_context_arrays_compared']==108
archive=root/'results'/(parent+'-evidence.zip');assert sha(archive.read_bytes())==proof['archive_sha256']
previous=json.loads((root/'results'/parent/'manifest.json').read_bytes())
assert previous['status']=='strided_component_finished_needs_independent_replay' and previous['exit_code']==0
events=[json.loads(l) for l in (root/'results'/parent/'correctness-ledger/commands.jsonl').read_text().splitlines()]
assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
library=root/'build/bin/garnet.dll';assert sha(library.read_bytes())==previous['library_sha256']
prep_path=methods/'evidence/canonical-kv-overlay-plugin-preparation.json';prep=json.loads(prep_path.read_bytes())
for n,h in prep['files'].items():assert sha((methods/n).read_bytes())==h
for n,h in prep['unchanged_source_sha256'].items():assert sha((root/'Garnet'/n).read_bytes())==h
for n,h in prep['method_sha256'].items():
    mapped={'build_canonical_kv_overlay_plugin.bat':'build-canonical-kv-overlay-plugin.bat','test_canonical_kv_overlay_plugin.bat':'test-canonical-kv-overlay-plugin.bat'}
    path=root/mapped[n] if n in mapped else methods/n
    assert sha(path.read_bytes())==h
fixture=(methods/prep['fixture_source_file']).read_bytes();assert sha(fixture)==prep['fixture_source_sha256']
text=fixture.decode();assert (methods/'canonical_kv_overlay_fixture.cuh').read_bytes()==text[:text.index(prep['fixture_prefix_end'])].encode()
start=text.index(prep['host_fixture_start']);end=text.index(prep['host_fixture_end'],start)
assert (methods/'canonical_kv_overlay_plugin_test.cu').read_text(encoding='utf8').count(text[start:end])==1
references=[root/'results'/parent/('context-case'+str(i)+'-baseline.f32') for i in range(12)]
with zipfile.ZipFile(archive) as z:
    for p in references:assert p.read_bytes()==z.read(p.relative_to(root).as_posix())
dependencies=[root/'tools/TensorRT'/n for n in ('bin/nvinfer_10.dll','lib/nvinfer_10.lib','include/NvInfer.h','include/NvInferRuntime.h')]
dependency_hashes={p.relative_to(root).as_posix():sha(p.read_bytes()) for p in dependencies}
out=root/'results/canonical-kv-overlay-plugin-attempt1';exe=root/'build/bin/garnet_canonical_kv_overlay_plugin_test.exe'
assert not out.exists() and not exe.exists()
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
assert shutil.disk_usage(root).free>=1024**3
out.mkdir();(out/'native').mkdir()
build=root/'build-canonical-kv-overlay-plugin.bat';test=root/'test-canonical-kv-overlay-plugin.bat'
files=[Path(__file__),build,test,prep_path,library,proof_path,archive]+[methods/n for n in prep['files']]+[root/'Garnet'/n for n in prep['unchanged_source_sha256']]+references
report={'status':'running','library_sha256':sha(library.read_bytes()),'parent_archive_sha256':proof['archive_sha256'],
        'source_sha256':{p.relative_to(root).as_posix():sha(p.read_bytes()) for p in files if p!=archive},
        'dependency_sha256':dependency_hashes,'timing_enabled':False,
        'scope':'Standalone context-only canonical-state TensorRT operator, four engines and24cases after strided qualification. Source/serialization/shape/format/fullspan alias/clone/deserialize/eager/graph/read-only gates. Current Garnet DLL/runtime/CMake unchanged; no model or performance admission.'}
path=out/'manifest.json';code=1
def save():path.write_text(json.dumps(report,indent=2),encoding='utf8')
save()
try:
    code=run(['cmd.exe','/d','/c',str(build)],out/'build-ledger','build','standalone-canonical-kv-context-operator',files+dependencies)
    report['build_exit_code']=code
    if code:raise RuntimeError('Canonical operator build failed')
    report['executable_sha256']=sha(exe.read_bytes())
    code=run([sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--','cmd.exe','/d','/c',str(test),str(out/'native/plugin')],out/'correctness-ledger','correctness','canonical-kv-operator-full-context-and-lifecycle-gates',files+dependencies+[exe])
    report['correctness_exit_code']=code
    if code:raise RuntimeError('Canonical operator qualification failed')
    raw=(out/'correctness-ledger/commands.log').read_text(encoding='utf8')
    rows=[json.loads(l) for l in raw.splitlines() if l.startswith('{"case":')]
    assert [(r['case'],r['strategy']) for r in rows]==[(i,s) for i in range(12) for s in (0,1)]
    assert all(r['status']=='operator_correctness_pass' and r['engine_legacy_bit_exact'] and r['graph_replays']==3 and r['clone_exact'] and r['restored_exact'] and r['readonly_canonical_preserved'] and r['negative_enqueues_preserved_state'] for r in rows)
    loaded=Path(re.search(r'TRT_LOADED_LIBRARY path=([^\r\n]+)',raw).group(1))
    assert sha(loaded.read_bytes())==dependency_hashes['tools/TensorRT/bin/nvinfer_10.dll']
    report.update(loaded_tensorrt_library=str(loaded),loaded_tensorrt_sha256=sha(loaded.read_bytes()),
        metadata_checks=int(re.search(r'CANONICAL_PLUGIN_METADATA_PASS checks=(\d+)',raw).group(1)),
        total_checks=int(re.search(r'CANONICAL_PLUGIN_QUALIFICATION_PASS checks=(\d+) cases=24 engines=4',raw).group(1)))
    report.update(status='operator_finished_needs_independent_replay',rows=rows);code=0
except BaseException as failure:
    report.update(status='failed',error=repr(failure))
    if code==0:code=1
    raise
finally:
    report['exit_code']=code;report['library_unchanged']=sha(library.read_bytes())==report['library_sha256']
    report['source_after_sha256']={p.relative_to(root).as_posix():sha(p.read_bytes()) for p in files if p not in (archive,proof_path,library)}
    report['dependency_after_sha256']={p.relative_to(root).as_posix():sha(p.read_bytes()) for p in dependencies}
    save()
raise SystemExit(code)
