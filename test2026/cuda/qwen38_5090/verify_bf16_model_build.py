"""Independently replay build/source lineage; no GPU/model qualification."""
from pathlib import Path
import hashlib,io,json,sys,zipfile
out=Path(sys.argv[1]);name='bf16-model-integration-build-attempt1';archive=out/(name+'-evidence.zip')
sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(archive) as z:
    doc=lambda n:json.loads(z.read(n));m=doc('evidence-manifest.json')
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
    for n,h in m['files'].items():assert sha(z.read(n))==h
    for n,h in m['linked_archives'].items():assert sha((out/n).read_bytes())==h
    base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/'
    b=doc(base+'manifest.json');prep_raw=z.read(methods+'evidence/bf16-model-integration-build-preparation.json');p=json.loads(prep_raw)
    assert b['status']=='bf16_model_native_build_finished_unqualified' and b['exit_code']==0
    assert sha(prep_raw)==b['preparation_sha256']
    assert b['old_library_sha256']==p['original_library_sha256'] and b['library_sha256']==m['library_sha256']!=b['old_library_sha256']
    assert b['runtime_dependencies_before_sha256']==b['runtime_dependencies_after_sha256']
    assert b['candidate_source_sha256']==p['candidate_source_sha256']
    source=z.read('results/bf16-model-integration-source.zip');assert sha(source)==p['source_zip_sha256']==b['source_zip_sha256']
    with zipfile.ZipFile(io.BytesIO(source)) as s:
        assert s.testzip() is None and set(s.namelist())==set(b['candidate_source_sha256'])
        for n,h in b['candidate_source_sha256'].items():assert sha(s.read(n))==sha(z.read('Garnet/'+n))==h
    with zipfile.ZipFile(io.BytesIO(z.read(base+'original-source.zip'))) as s:
        assert s.testzip() is None and set(s.namelist())==set(b['original_source_sha256'])
        for n,h in b['original_source_sha256'].items():
            raw=s.read(n);assert sha(raw)==h
            allowed=p['base_source'][n]
            assert h in allowed['allowed_raw'] or sha(raw.replace(b'\r\n',b'\n')) in allowed['allowed_normalized']
    for n,h in p['methods'].items():assert sha(z.read(methods+n))==h
    for ledger in ('cpu-ledger','build-ledger'):
        events=[json.loads(l) for l in z.read(base+ledger+'/commands.jsonl').splitlines()]
        assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and events[1]['exit_code']==0
        assert events[0]['id']==events[1]['id']
        raw=z.read(base+ledger+'/commands.log');e=events[1]
        assert 0<=e['raw_start_offset']<e['raw_end_offset']<=len(raw)
        log=raw[e['raw_start_offset']:e['raw_end_offset']]
        if ledger=='cpu-ledger':assert b'Ran 33 tests' in log and b'\nOK' in log
        else:assert b'Linking CXX shared library' in log and b'garnet.dll' in log
    parent=doc('results/bf16-plugin-cpu-contract-attempt1-verified.json')
    assert parent['status']=='bf16_plugin_cpu_contract_replayed' and parent['contract']['profiles']==64
    assert parent['archive_sha256']==m['linked_archives']['bf16-plugin-cpu-contract-attempt1-evidence.zip']
    cache=z.read('build/CMakeCache.txt');ninja=z.read('build/build.ninja').decode().replace('\r\n','\n').replace('\\','/')
    assert b'GARNET_QWEN35_WHOLE_PROGRAM_CUDA:BOOL=OFF' in cache
    assert b'GARNET_STAGE_VENDOR_DLLS:BOOL=OFF' in cache
    helper=[block for block in ninja.split('\n\n') if block.startswith('build ') and '/cuda_bf16_kv_delta_commit.cpp.obj:' in block]
    assert len(helper)==1 and ' -MT ' in helper[0] and '/MD ' not in helper[0] and '-MD ' not in helper[0]
    archived=sha(z.read(methods+'verify_bf16_model_build.py'))
r={'status':'bf16_model_native_build_replayed','archive_sha256':sha(archive.read_bytes()),'files_verified':len(m['files']),
   'library_sha256':b['library_sha256'],'original_library_sha256':b['old_library_sha256'],'cpu_tests':33,'helper_crt':'/MT',
   'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,
   'model_qualified':False,'performance_admitted':False,'scope':b['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(r,indent=2),encoding='utf8');print(json.dumps(r,indent=2))
