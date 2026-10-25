"""Replay raw attention witnesses with independent dense NumPy FP64 arithmetic."""
from pathlib import Path
import hashlib,json,platform,sys,zipfile
import numpy as np
from bf16_kv_overlay_oracle import ATOL,RTOL,case_plan,compare_witness,sample_triples
out=Path(sys.argv[1]);name='bf16-kv-overlay-native-attempt1';archive=out/(name+'-evidence.zip')
sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(archive) as z:
    doc=lambda n:json.loads(z.read(n));m=doc('evidence-manifest.json')
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
    for n,h in m['files'].items():assert sha(z.read(n))==h
    base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/'
    w=doc(base+'workflow.json');prepraw=z.read(methods+'evidence/bf16-kv-overlay-native-preparation.json');prep=json.loads(prepraw)
    assert sha(prepraw)==w['preparation_sha256'] and w['library_unchanged'] and w['cleanup_finished']
    assert w['source_sha256']==w['source_after_sha256']==prep['files']
    for n,h in prep['files'].items():assert sha(z.read(n))==h
    proof=doc('results/bf16-kv-commit-native-attempt1-verified.json')
    assert proof['status']=='bf16_kv_commit_native_operator_replayed' and proof['full_state_byte_cases']==34 and proof['rejected_calls']==62
    assert proof['archive_sha256']==w['parent_commit_archive_sha256']==m['parent_commit_archive_sha256']
    assert sha((out/'bf16-kv-commit-native-attempt1-evidence.zip').read_bytes())==proof['archive_sha256']
    assert w['library_sha256']==m['observed_unused_library_sha256']==prep['observed_unused_library_sha256']
    oracle_source=Path(__file__).with_name('bf16_kv_overlay_oracle.py');assert sha(oracle_source.read_bytes())==sha(z.read(methods+oracle_source.name))
    compile_receipt=doc(methods+'evidence/bf16-kv-overlay-local-compile.json')
    for n,h in compile_receipt['source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
    assert sha(z.read('tools/bf16-kv-native/probe_bf16_kv_overlay_native.exe'))==compile_receipt['executable_sha256']
    assert sha(z.read(methods+'evidence/bf16-kv-overlay-probe-compile.log'))==compile_receipt['compile_log_sha256']
    assert sha(z.read(base+'regenerated-layout.h'))==prep['fixture_header_sha256']
    numerical=[];passed=w['exit_code']==0
    if passed:
        for ledger in ('cpu-oracle','cpu-fixture','native'):
            events=[json.loads(l) for l in z.read(base+ledger+'/commands.jsonl').splitlines()]
            assert events[0]['event']=='started' and events[-1]['event']=='finished' and events[-1]['exit_code']==0
        cpu=z.read(base+'cpu-oracle/commands.log').replace(b'\r\n',b'\n');assert b'Ran 8 tests' in cpu and b'\nOK' in cpu
        records=[]
        for line in z.read(base+'native/commands.log').splitlines():
            try:r=json.loads(line)
            except (ValueError,UnicodeDecodeError):continue
            if isinstance(r,dict):records.append(r)
        identity=[r for r in records if r.get('event')=='gpu_identity'];cases=[r for r in records if r.get('event')=='case_collected']
        assert len(identity)==1 and identity[0]['uuid']==prep['gpu_uuid'] and identity[0]['sm']==120 and '5090' in identity[0]['name']
        plan=case_plan();assert len(cases)==len(plan)==70
        raw=z.read(base+'witnesses.bin');cursor=0;negative=0;graphs=0;prefix=0
        for case,expected in zip(cases,plan):
            for key,value in expected.items():assert case[key]==value
            assert case['samples']==[list(v) for v in sample_triples(expected)] and case['witness_offset']==cursor
            size=len(case['samples'])*256*4;assert case['witness_bytes']==size
            assert case['readonly_exact'] is True and case['canaries_exact'] is True
            assert case['graph_replay_exact']==(case['id'] in (1,15,29,43))
            assert case['prefix_cache_exact']==(case['end']>0)
            assert case['rejected_calls']==(50 if case['id'] in (0,14,28,42) else 0)
            numerical.append(dict(expected,**compare_witness(expected,raw[cursor:cursor+size])))
            cursor+=size;negative+=case['rejected_calls'];graphs+=case['graph_replay_exact'];prefix+=case['prefix_cache_exact']
        assert cursor==len(raw) and negative==200 and graphs==4 and prefix==62
        finals=[r for r in records if r.get('status')=='bf16_overlay_collected_needs_fp64_replay']
        assert len(finals)==1 and finals[0]['cases']==70 and finals[0]['witness_bytes']==len(raw) and finals[0]['model_qualified'] is False and finals[0]['performance_admitted'] is False
        assert w['status']=='bf16_overlay_collected_needs_independent_fp64_replay'
    resource=doc(base+'resources/resource-monitor.json');samples=[json.loads(l) for l in z.read(base+'resources/resource-vram.jsonl').splitlines()]
    assert resource['gpu']['uuid']==prep['gpu_uuid'] and resource['child_exit_code']==w['exit_code']
    peak=resource['sampled_memory']['sampled_peak_adapter_used_bytes'];assert resource['sampled_memory']['samples']==len(samples) and peak==max(r['used_bytes'] for r in samples)
    verifier=sha(z.read(methods+'verify_bf16_kv_overlay_native.py'))
result={'status':'bf16_kv_overlay_native_fp64_replayed' if passed else 'bf16_kv_overlay_native_failure_preserved',
 'archive_sha256':sha(archive.read_bytes()),'files_verified':len(m['files']),'cases':numerical,
 'fp64_witness_scalars':sum(c['scalars'] for c in numerical),'maximum_absolute_error':max((c['max_absolute_error'] for c in numerical),default=None),
 'maximum_tolerance_fraction':max((c['max_tolerance_fraction'] for c in numerical),default=None),'absolute_tolerance':ATOL,'relative_tolerance':RTOL,
 'rejected_calls':negative if passed else None,'graph_replays':graphs if passed else None,'prefix_cache_exact_cases':prefix if passed else None,
 'sampled_correctness_adapter_peak_bytes':peak,'numpy_version':np.__version__,'python_version':platform.python_version(),
 'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':verifier,
 'oracle_sha256':sha(oracle_source.read_bytes()),'model_qualified':False,'performance_admitted':False,'scope':prep['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
