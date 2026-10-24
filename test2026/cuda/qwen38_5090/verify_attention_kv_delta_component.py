"""Replay KV-overlay contexts/counts and source lineage; raw full caches absent."""
from pathlib import Path
from array import array
import hashlib,json,math,re,sys,zipfile
out=Path(sys.argv[1]);name='attention-kv-delta-correctness-attempt1';archive=out/(name+'-evidence.zip')
sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(archive) as z:
    doc=lambda n:json.loads(z.read(n));m=doc('evidence-manifest.json')
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
    for n,h in m['files'].items():assert sha(z.read(n))==h
    for n,h in m['linked_archives'].items():assert sha((out/n).read_bytes())==h
    base='results/'+name+'/';report=doc(base+'manifest.json');passed=report['exit_code']==0
    assert report['library_unchanged'] and report['library_sha256']==m['library_sha256'] and not report['timing_enabled']
    for n,h in report['source_sha256'].items():
        if n!='build/bin/garnet.dll':assert sha(z.read(n))==h
    for n,h in report['source_after_sha256'].items():assert sha(z.read(n))==h==report['source_sha256'][n]
    methods='Garnet/test2026/cuda/qwen38_5090/'
    prep=doc(methods+'evidence/attention-kv-delta-component-preparation.json')
    for n,h in prep['files'].items():assert sha(z.read(methods+n))==h
    for n,h in prep['method_sha256'].items():
        path='build-attention-kv-delta-component.bat' if n.endswith('.bat') else methods+n
        assert sha(z.read(path))==h
    native=z.read('Garnet/src/cuda/qwen35_attention_core.cu')
    assert sha(native)==prep['native_base_sha256']
    native=native.decode().replace('\r\n','\n')
    core=z.read(methods+'attention_kv_delta_core.cu').decode()
    helpers=native[native.index('__device__ float warpSum'):native.index('__global__ void writeKeys')]
    assert core.count(helpers)==1
    start=native.index('template<bool tiled> __global__ void compact(')
    kernel=native[start:native.index('// Split-key research path:',start)]
    for change in prep['kernel_derivation']:
        assert kernel.count(change['old'])==1;kernel=kernel.replace(change['old'],change['new'])
    assert core.count(kernel)==1
    driver=z.read(methods+'attention_b4_t4_test.cu')
    assert sha(driver)==prep['driver_base_sha256']
    driver=driver.decode().replace('\r\n','\n')
    for change in prep['driver_derivation']:
        assert driver.count(change['old'])==1;driver=driver.replace(change['old'],change['new'])
    start=driver.index('  for(int mode=0;mode<4;++mode)for(int i=0;i<5;')
    end=driver.index('\n }\n Check(cudaEventDestroy',start)
    driver=driver[:start]+driver[end:]
    assert z.read(methods+'attention_kv_delta_test.cu')==driver.encode()
    parent=doc('results/attention-b4-t4-component-attempt1-verified.json')
    assert parent['status']=='attention_b4_t4_component_replayed' and len(parent['checks'])==16
    assert parent['archive_sha256']==report['parent_archive_sha256']==m['linked_archives']['attention-b4-t4-component-attempt1-evidence.zip']
    checks=[];arrays=0;count_vectors=0
    def floats(raw):
        assert len(raw)%4==0;v=array('f');v.frombytes(raw)
        if sys.byteorder!='little':v.byteswap()
        assert all(math.isfinite(x) for x in v);return v
    if passed:
        assert report['status']=='correctness_finished_needs_independent_replay'
        assert sha(z.read('build/bin/garnet_attention_kv_delta_test.exe'))==report['executable_sha256']
        raw=z.read(base+'correctness-ledger/commands.log').decode('utf8')
        rows=[json.loads(l) for l in raw.splitlines() if l.startswith('{"case":')]
        controls=re.findall(r'B4_ATTENTION_CONTROLS case=(\d+) pattern=(\d+) slots=([^\n]+) active=([^\n]+)',raw)
        assert rows==report['rows'] and len(rows)==len(controls)==8
        for i,(row,control) in enumerate(zip(rows,controls)):
            assert row['case']==int(control[0])==i and int(control[1])==([0]*5+[1,2,3])[i]
            slots=[int(v) for v in control[2].strip().strip(',').split(',')]
            mask=[int(v) for v in control[3].strip().strip(',').split(',')]
            assert len(slots)==len(mask)==16 and all(0<=s<4096 for s in slots) and all(a in (0,1) for a in mask)
            counts=[sum(mask[b*4:b*4+4]) for b in range(4)]
            assert counts==([[4]*4]*5+[[2,2,2,0],[0]*4,[4,1,2,0]])[i]
            for b in range(4):
                active_slots=[slots[b*4+t] for t in range(4) if mask[b*4+t]]
                assert len(active_slots)==len(set(active_slots))
            prefix=base+'context-case'+str(i)
            cpu=floats(z.read(prefix+'-cpu.f32'));assert len(cpu)==4*4*24*256
            legacy=z.read(prefix+'-baseline.f32');legacy_values=floats(legacy)
            for variant in ('baseline','compact','tiled','split','overlay'):
                raw_context=z.read(prefix+'-'+variant+'.f32');v=floats(raw_context);arrays+=1
                assert len(v)==len(cpu) and all(abs(a-b)<=2e-5+2e-4*abs(b) for a,b in zip(v,cpu))
                if variant in ('compact','tiled','overlay'):assert raw_context==legacy
                if variant=='split':assert all(abs(a-b)<=2e-5+2e-4*abs(b) for a,b in zip(v,legacy_values))
                for bt,active in enumerate(mask):
                    if not active:assert all(x==0 for x in v[bt*24*256:(bt+1)*24*256])
            arrays+=1
            for trial in range(3):
                accepted=floats(z.read(prefix+'-commit'+str(trial)+'-counts.f32'));count_vectors+=1
                expected=[0]*4 if trial==0 else [min(n,b if trial==1 else [4,0,3,1][b]) for b,n in enumerate(counts)]
                assert list(accepted)==expected
            assert row['status']=='correctness_pass' and row['variants_checked']==5 and row['overlay_legacy_bit_exact'] and row['readonly_overlay_preserved_kv'] and row['zero_partial_prefix_trials']==3 and row['full_kv_bit_exact'] and row['graph_replays']==3
            checks.append({'case':i,'context_legacy_bit_exact':True,'active_counts_by_row':counts,'accepted_prefix_vectors_replayed':3,'retained_cpu_tolerance':True})
    for stage in ('build','correctness'):
        path=base+stage+'-ledger/commands.jsonl'
        if path not in m['files']:assert not passed;continue
        events=[json.loads(l) for l in z.read(path).decode().splitlines()]
        assert events[-1]['event']=='finished'
        if passed:assert events[-1]['exit_code']==0
    if passed:
        monitor=doc(base+'resources/resource-monitor.json');samples=[json.loads(l) for l in z.read(base+'resources/resource-vram.jsonl').decode().splitlines()]
        assert monitor['child_exit_code']==0 and monitor['sampled_memory']['samples']==len(samples)
        assert monitor['sampled_memory']['sampled_peak_adapter_used_bytes']==max(s['used_bytes'] for s in samples)
result={'status':'attention_kv_delta_component_replayed' if passed else 'attention_kv_delta_component_failure_preserved',
        'archive_sha256':sha(archive.read_bytes()),'files_verified':len(m['files']),'cases':checks,'raw_context_arrays':arrays,'raw_prefix_count_vectors':count_vectors,
        'performance_admitted':False,'model_integration_admitted':False,
        'scope':'Standalone synthetic readonly KV overlay plus explicit accepted-prefix commit; no existing native/plugin changes. Retained CPU-context and legacy/overlay byte comparisons independently replayed; full cache/prefix/guards/input checks remain in-process witnesses, raw full caches absent. No model correctness, timing, DRAM-traffic or speed claim.'}
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
