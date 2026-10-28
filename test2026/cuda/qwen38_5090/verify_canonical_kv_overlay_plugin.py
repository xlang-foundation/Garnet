"""Replay standalone TRT output/source lineage; full-state checks are witnesses."""
from pathlib import Path
from array import array
import hashlib,json,math,re,sys,zipfile
out=Path(sys.argv[1]);name='canonical-kv-overlay-plugin-attempt1';archive=out/(name+'-evidence.zip');sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(archive) as z:
    doc=lambda n:json.loads(z.read(n));m=doc('evidence-manifest.json')
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
    for n,h in m['files'].items():assert sha(z.read(n))==h
    for n,h in m['linked_archives'].items():assert sha((out/n).read_bytes())==h
    base='results/'+name+'/';report=doc(base+'manifest.json');passed=report['exit_code']==0
    assert not report['timing_enabled'] and report['library_unchanged'] and report['library_sha256']==m['library_sha256']
    for n,h in report['source_sha256'].items():
        if n!='build/bin/garnet.dll':assert sha(z.read(n))==h
    for n,h in report['source_after_sha256'].items():assert sha(z.read(n))==h==report['source_sha256'][n]
    assert report['dependency_sha256']==report['dependency_after_sha256']==m['dependency_sha256']
    methods='Garnet/test2026/cuda/qwen38_5090/';prep=doc(methods+'evidence/canonical-kv-overlay-plugin-preparation.json')
    assert prep['engines']==4 and prep['profiles']==12 and prep['operator_cases']==24 and prep['strategies']==[0,1] and prep['layers']==[3,63]
    for n,h in prep['files'].items():assert sha(z.read(methods+n))==h
    for n,h in prep['unchanged_source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
    for n,h in prep['method_sha256'].items():assert sha(z.read(n.replace('_','-') if n.endswith('.bat') else methods+n))==h
    fixture=z.read(methods+prep['fixture_source_file']);assert sha(fixture)==prep['fixture_source_sha256']
    text=fixture.decode();assert z.read(methods+'canonical_kv_overlay_fixture.cuh')==text[:text.index(prep['fixture_prefix_end'])].encode()
    start=text.index(prep['host_fixture_start']);end=text.index(prep['host_fixture_end'],start)
    assert z.read(methods+'canonical_kv_overlay_plugin_test.cu').decode().count(text[start:end])==1
    parent='attention-kv-delta-strided-attempt1';proof=doc('results/'+parent+'-verified.json')
    assert proof['status']=='attention_kv_delta_strided_replayed' and proof['archive_sha256']==m['linked_archives'][parent+'-evidence.zip']
    assert proof['raw_context_arrays']==144 and proof['retained_parent_context_arrays_compared']==108 and len(proof['cases'])==12
    arrays=0;checks=[];engines=[];peak=None
    with zipfile.ZipFile(out/(parent+'-evidence.zip')) as previous:
        old=doc('results/'+parent+'-verified.json');oldmanifest=json.loads(previous.read('evidence-manifest.json'))
        assert oldmanifest['library_sha256']==m['library_sha256']
        for n in prep['unchanged_source_sha256']:
            if 'Garnet/'+n in oldmanifest['files']:assert z.read('Garnet/'+n)==previous.read('Garnet/'+n)
        if passed:
            assert report['status']=='operator_finished_needs_independent_replay'
            assert sha(z.read('build/bin/garnet_canonical_kv_overlay_plugin_test.exe'))==report['executable_sha256']
            log=z.read(base+'correctness-ledger/commands.log').decode('utf8')
            rows=[json.loads(l) for l in log.splitlines() if l.startswith('{"case":')]
            assert rows==report['rows'] and [(r['case'],r['strategy']) for r in rows]==[(i,s) for i in range(12) for s in (0,1)]
            metadata=int(re.search(r'CANONICAL_PLUGIN_METADATA_PASS checks=(\d+)',log).group(1));total=int(re.search(r'CANONICAL_PLUGIN_QUALIFICATION_PASS checks=(\d+) cases=24 engines=4',log).group(1))
            assert metadata==report['metadata_checks'] and total==report['total_checks'] and total>metadata>0
            loaded=re.search(r'TRT_LOADED_LIBRARY path=([^\r\n]+)',log).group(1)
            assert loaded==report['loaded_tensorrt_library'] and report['loaded_tensorrt_sha256']==m['dependency_sha256']['tools/TensorRT/bin/nvinfer_10.dll']
            controls=re.findall(r'B4_ATTENTION_CONTROLS case=(\d+) pattern=(\d+) slots=([^\n]+) active=([^\n]+)',log)
            oldlog=previous.read('results/'+parent+'/correctness-ledger/commands.log').decode('utf8')
            assert controls==re.findall(r'B4_ATTENTION_CONTROLS case=(\d+) pattern=(\d+) slots=([^\n]+) active=([^\n]+)',oldlog) and len(controls)==12
            for layer in (3,63):
                for strategy in (0,1):
                    stem=base+'native/plugin-layer'+str(layer)+'-strategy'+str(strategy)
                    engine=z.read(stem+'.engine');inspector=doc(stem+'-inspector.json')
                    assert engine and inspector and 'readonly_canonical_kv_overlay' in json.dumps(inspector)
                    engines.append({'layer':layer,'strategy':strategy,'engine_sha256':sha(engine)})
            for row in rows:
                i,s=row['case'],row['strategy'];assert row['canonical_layer']==(3 if i%2==0 else 63)
                assert row['status']=='operator_correctness_pass' and row['graph_replays']==3
                assert all(row[k] for k in ('engine_legacy_bit_exact','clone_exact','restored_exact','readonly_canonical_preserved','negative_enqueues_preserved_state'))
                reference=z.read('results/'+parent+'/context-case'+str(i)+'-baseline.f32')
                assert reference==previous.read('results/'+parent+'/context-case'+str(i)+'-baseline.f32') and len(reference)==4*4*24*256*4
                values=array('f');values.frombytes(reference)
                if sys.byteorder!='little':values.byteswap()
                assert all(math.isfinite(v) for v in values)
                for stage in ('eager','graph','clone','restored'):
                    assert z.read(base+'native/plugin-case'+str(i)+'-strategy'+str(s)+'-'+stage+'.f32')==reference;arrays+=1
                checks.append({'case':i,'strategy':s,'canonical_layer':row['canonical_layer'],'four_raw_contexts_legacy_exact':True,'readonly_and_rejected_mutation_in_process_witness':True})
            monitor=doc(base+'resources/resource-monitor.json');samples=[json.loads(l) for l in z.read(base+'resources/resource-vram.jsonl').decode().splitlines()]
            assert monitor['child_exit_code']==0 and monitor['sampled_memory']['samples']==len(samples)
            peak=max(s['used_bytes'] for s in samples);assert monitor['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
            previous_monitor=json.loads(previous.read('results/'+parent+'/resources/resource-monitor.json'))
            assert monitor['gpu']==previous_monitor['gpu'] and '5090' in monitor['gpu']['name']
    for stage in ('build','correctness'):
        path=base+stage+'-ledger/commands.jsonl'
        if path not in m['files']:assert not passed;continue
        events=[json.loads(l) for l in z.read(path).decode().splitlines()];assert events[-1]['event']=='finished'
        if passed:assert events[-1]['exit_code']==0
    archived_verifier=sha(z.read(methods+'verify_canonical_kv_overlay_plugin.py'))
result={'status':'canonical_kv_overlay_operator_replayed' if passed else 'canonical_kv_overlay_operator_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(m['files']),
        'cases':checks,'engines':engines,'raw_context_arrays':arrays,'sampled_peak_adapter_used_bytes':peak,
        'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived_verifier,
        'model_integration_admitted':False,'performance_admitted':False,
        'scope':'Standalone context-only TensorRT canonical KV operator, source/fixture/parent lineage, engine artifacts and raw eager/graph/clone/restored contexts. Metadata, actual loaded vendor binary identity, negative enqueue and readonly full-state/canaries are in-process witnesses; no raw canonical arrays or independent engine execution on replay. Dependency binaries fingerprinted, not redistributed. No model/HTTP/2x admission.'}
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
