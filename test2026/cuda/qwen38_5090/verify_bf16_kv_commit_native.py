"""Independent archive replay of standalone BF16 native commit operator controls."""
from pathlib import Path
import hashlib,json,sys,zipfile
out=Path(sys.argv[1]);name='bf16-kv-commit-native-attempt1';archive=out/(name+'-evidence.zip')
sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(archive) as z:
    doc=lambda n:json.loads(z.read(n));m=doc('evidence-manifest.json')
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
    for n,h in m['files'].items():assert sha(z.read(n))==h
    base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/'
    w=doc(base+'workflow.json');prepraw=z.read(methods+'evidence/bf16-kv-commit-native-preparation.json');prep=json.loads(prepraw)
    assert sha(prepraw)==w['preparation_sha256'] and w['library_unchanged'] and w['cleanup_finished']
    assert w['source_sha256']==w['source_after_sha256']==prep['files']
    for n,h in prep['files'].items():assert sha(z.read(n))==h
    proof=doc('results/register128-padded-http-pilot-attempt1-verified.json')
    assert proof['status']=='register128_padded_http_pilot_replayed' and proof['archive_sha256']==w['parent_http_archive_sha256']==m['parent_http_archive_sha256']
    assert sha((out/'register128-padded-http-pilot-attempt1-evidence.zip').read_bytes())==proof['archive_sha256']
    assert proof['library_sha256']==w['library_sha256']==m['observed_unused_library_sha256']
    for ledger in ('cpu-layout','cpu-fixture','native'):
        events=[json.loads(l) for l in z.read(base+ledger+'/commands.jsonl').splitlines()]
        assert events[0]['event']=='started' and events[-1]['event']=='finished' and events[-1]['exit_code']==0
    assert b'Ran 6 tests' in z.read(base+'cpu-layout/commands.log') and b'\nOK' in z.read(base+'cpu-layout/commands.log').replace(b'\r\n',b'\n')
    assert sha(z.read(base+'regenerated-layout.h'))==prep['fixture_header_sha256']
    compile_receipt=doc(methods+'evidence/bf16-kv-native-local-compile.json')
    for n,h in compile_receipt['source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
    assert sha(z.read('tools/bf16-kv-native/probe_bf16_kv_commit_native.exe'))==compile_receipt['local_artifact_sha256']['probe_bf16_kv_commit_native.exe']
    raw=z.read(base+'native/commands.log');records=[]
    for line in raw.splitlines():
        try:r=json.loads(line)
        except (ValueError,UnicodeDecodeError):continue
        if isinstance(r,dict):records.append(r)
    identity=[r for r in records if r.get('event')=='gpu_identity'];profiles=[r for r in records if r.get('event')=='profile_passed']
    assert len(identity)==1 and identity[0]['uuid']==prep['gpu_uuid'] and identity[0]['sm']==120 and '5090' in identity[0]['name']
    assert len(profiles)==4 and [(r['batch'],r['chunk']) for r in profiles]==[(4,4),(4,32),(8,4),(8,32)]
    for r in profiles:
        assert r['full_state_byte_cases']==(11 if r['chunk']==4 else 6) and r['rejected_calls']==(16 if r['batch']==4 else 15) and r['readonly_inputs_exact'] is True
    finals=[r for r in records if r.get('status')=='bf16_kv_commit_native_operator_passed'];assert len(finals)==1 and finals[0]['performance_admitted'] is False and finals[0]['model_qualified'] is False
    resource=doc(base+'resources/resource-monitor.json');samples=[json.loads(l) for l in z.read(base+'resources/resource-vram.jsonl').splitlines()]
    assert resource['gpu']['uuid']==prep['gpu_uuid'] and resource['child_exit_code']==0
    peak=resource['sampled_memory']['sampled_peak_adapter_used_bytes'];assert resource['sampled_memory']['samples']==len(samples) and peak==max(r['used_bytes'] for r in samples)
    assert w['exit_code']==0 and w['status']=='bf16_commit_operator_finished_needs_independent_replay'
    verifier=sha(z.read(methods+'verify_bf16_kv_commit_native.py'))
result={'status':'bf16_kv_commit_native_operator_replayed','archive_sha256':sha(archive.read_bytes()),'files_verified':len(m['files']),
        'profiles':profiles,'full_state_byte_cases':34,'rejected_calls':62,'sampled_correctness_adapter_peak_bytes':peak,
        'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':verifier,
        'model_qualified':False,'performance_admitted':False,'scope':prep['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
