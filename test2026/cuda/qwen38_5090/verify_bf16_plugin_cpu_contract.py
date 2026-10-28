"""Replay CPU wrapper assertions and source/SDK lineage; never a model gate."""
from pathlib import Path
import hashlib,json,sys,zipfile
out=Path(sys.argv[1]);name='bf16-plugin-cpu-contract-attempt1';archive=out/(name+'-evidence.zip');sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(archive) as z:
    doc=lambda n:json.loads(z.read(n));m=doc('evidence-manifest.json')
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
    for n,h in m['files'].items():assert sha(z.read(n))==h
    base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/'
    w=doc(base+'workflow.json');raw=z.read(methods+'evidence/bf16-plugin-cpu-contract-preparation.json');prep=json.loads(raw)
    assert sha(raw)==w['preparation_sha256'] and w['library_unchanged']
    assert w['source_sha256']==w['source_after_sha256']==prep['files'] and w['sdk_sha256']==w['sdk_after_sha256']==prep['sdk_sha256']
    for n,h in prep['files'].items():assert sha(z.read(n))==h
    parent=doc('results/bf16-kv-overlay-native-attempt1-verified.json')
    assert parent['status']=='bf16_kv_overlay_native_fp64_replayed' and len(parent['cases'])==70 and parent['fp64_witness_scalars']==1172480 and parent['maximum_tolerance_fraction']<=1
    assert parent['archive_sha256']==w['parent_overlay_archive_sha256']==sha((out/'bf16-kv-overlay-native-attempt1-evidence.zip').read_bytes())
    events=[json.loads(l) for l in z.read(base+'cpu/commands.jsonl').splitlines()]
    assert events[0]['event']=='started' and events[-1]['event']=='finished' and events[-1]['exit_code']==w['exit_code']
    receipt=doc(methods+'evidence/bf16-plugin-cpu-local-build.json')
    observation=doc(methods+'evidence/bf16-plugin-sdk-runtime-observation.json')
    assert observation['sha256']==prep['sdk_sha256']['tools/TensorRT/bin/nvinfer_10.dll']
    assert observation['producer_sha256']==sha(z.read(methods+'observe_bf16_plugin_sdk.py'))
    assert receipt['header_sha256']==sha(z.read('tools/bf16-kv-native/bf16_kv_fixture_layout.h'))
    for n,h in receipt['build_logs_sha256'].items():assert sha(z.read(methods+'evidence/'+(n+'.txt' if n.endswith('.bat') else n)))==h
    for n,h in receipt['source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
    assert sha(z.read('tools/bf16-kv-native/probe_bf16_kv_plugin_contract.exe'))==receipt['executable_sha256']
    records=[]
    for line in z.read(base+'cpu/commands.log').splitlines():
        try:r=json.loads(line)
        except (ValueError,UnicodeDecodeError):continue
        if isinstance(r,dict) and r.get('status')=='bf16_plugin_cpu_contract_passed':records.append(r)
    passed=w['exit_code']==0
    if passed:
        assert w['status']=='bf16_plugin_cpu_contract_finished_needs_independent_replay' and len(records)==1
        r=records[0]
        assert r['loaded_tensorrt_dll'].casefold()=='D:/CantorAI/5090-study-20261023/tools/TensorRT/bin/nvinfer_10.dll'.casefold()
        for key,value in {'profiles':64,'roundtrips':64,'clones':64,'bad_wires':3328,'bad_descriptors':1984,'bad_formats':1344,'runtime_version':101601,'header_version':101601,'enqueue_tested':False,'model_qualified':False,'performance_admitted':False}.items():assert r[key]==value
    archived=sha(z.read(methods+'verify_bf16_plugin_cpu_contract.py'))
result={'status':'bf16_plugin_cpu_contract_replayed' if passed else 'bf16_plugin_cpu_contract_failure_preserved','archive_sha256':sha(archive.read_bytes()),
        'files_verified':len(m['files']),'contract':records[0] if passed else None,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),
        'archived_replay_verifier_sha256':archived,'enqueue_tested':False,'model_qualified':False,'performance_admitted':False,'scope':prep['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
