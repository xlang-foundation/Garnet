"""Replay component timing after context/source/ownership checks; no model claim."""
from pathlib import Path
from array import array
import hashlib,json,math,re,statistics,sys,zipfile
out=Path(sys.argv[1]);name='attention-kv-delta-timing-attempt1';archive=out/(name+'-evidence.zip')
sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(archive) as z:
    doc=lambda n:json.loads(z.read(n));m=doc('evidence-manifest.json')
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
    for n,h in m['files'].items():assert sha(z.read(n))==h
    for n,h in m['linked_archives'].items():assert sha((out/n).read_bytes())==h
    base='results/'+name+'/';report=doc(base+'manifest.json');passed=report['exit_code']==0
    assert report['library_unchanged'] and report['library_sha256']==m['library_sha256'] and report['timing_enabled']
    for n,h in report['source_sha256'].items():
        if n!='build/bin/garnet.dll':assert sha(z.read(n))==h
    for n,h in report['source_after_sha256'].items():assert sha(z.read(n))==h==report['source_sha256'][n]
    methods='Garnet/test2026/cuda/qwen38_5090/'
    prep=doc(methods+'evidence/attention-kv-delta-timing-preparation.json')
    for n,h in prep['native_source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
    for n,h in prep['unchanged_source_sha256'].items():assert sha(z.read(methods+n))==h
    for n,h in prep['method_sha256'].items():
        path='build-attention-kv-delta-timing.bat' if n.endswith('.bat') else methods+n
        assert sha(z.read(path))==h
    driver=z.read(methods+prep['base_driver']);assert sha(driver)==prep['base_driver_sha256']
    driver=driver.decode()
    for old,new in prep['driver_derivation']:
        assert driver.count(old)==1;driver=driver.replace(old,new)
    marker='\n }\n Check(cudaEventDestroy';assert driver.count(marker)==1
    driver=driver.replace(marker,'\n'+prep['timing_block']+marker)
    assert z.read(methods+prep['driver'])==driver.encode() and sha(driver.encode())==prep['driver_sha256']
    parent_name='attention-kv-delta-correctness-attempt1'
    parent=doc('results/'+parent_name+'-verified.json')
    assert parent['status']=='attention_kv_delta_component_replayed' and len(parent['cases'])==8 and parent['raw_context_arrays']==48 and parent['raw_prefix_count_vectors']==24
    assert parent['archive_sha256']==report['parent_archive_sha256']==m['linked_archives'][parent_name+'-evidence.zip']
    checks=[];ratios=[];arrays=0;vectors=0
    def floats(raw):
        assert len(raw)%4==0;v=array('f');v.frombytes(raw)
        if sys.byteorder!='little':v.byteswap()
        assert all(math.isfinite(x) for x in v);return v
    with zipfile.ZipFile(out/(parent_name+'-evidence.zip')) as previous:
        for n in list(prep['unchanged_source_sha256'])+[prep['base_driver']]:assert z.read(methods+n)==previous.read(methods+n)
        for n in prep['native_source_sha256']:assert z.read('Garnet/'+n)==previous.read('Garnet/'+n)
        if passed:
            assert report['status']=='component_timing_finished_needs_independent_replay'
            assert sha(z.read('build/bin/garnet_attention_kv_delta_timing_test.exe'))==report['executable_sha256']
            for stage in ('qualification','timing'):
                log=z.read(base+stage+'-ledger/commands.log').decode('utf8')
                rows=[json.loads(l) for l in log.splitlines() if l.startswith('{"case":')]
                controls=re.findall(r'B4_ATTENTION_CONTROLS case=(\d+) pattern=(\d+) slots=([^\n]+) active=([^\n]+)',log)
                assert len(rows)==len(controls)==8
                if stage=='timing':assert rows==report['rows']
                for i,(row,control) in enumerate(zip(rows,controls)):
                    assert row['case']==int(control[0])==i and int(control[1])==([0]*5+[1,2,3])[i]
                    slots=[int(v) for v in control[2].strip().strip(',').split(',')];mask=[int(v) for v in control[3].strip().strip(',').split(',')]
                    assert len(slots)==len(mask)==16 and all(0<=s<4096 for s in slots) and all(a in (0,1) for a in mask)
                    counts=[sum(mask[b*4:b*4+4]) for b in range(4)]
                    assert counts==([[4]*4]*5+[[2,2,2,0],[0]*4,[4,1,2,0]])[i]
                    for b in range(4):
                        active_slots=[slots[b*4+t] for t in range(4) if mask[b*4+t]]
                        assert len(active_slots)==len(set(active_slots))
                    prefix=base+stage+'-context-case'+str(i);old='results/'+parent_name+'/context-case'+str(i)
                    cpu=floats(z.read(prefix+'-cpu.f32'));assert len(cpu)==4*4*24*256
                    legacy=z.read(prefix+'-baseline.f32');legacy_values=floats(legacy)
                    for variant in ('cpu','baseline','compact','tiled','split','overlay'):
                        raw=z.read(prefix+'-'+variant+'.f32');v=floats(raw);arrays+=1
                        assert raw==previous.read(old+'-'+variant+'.f32')
                        assert len(v)==len(cpu) and all(abs(a-b)<=2e-5+2e-4*abs(b) for a,b in zip(v,cpu))
                        if variant in ('compact','tiled','overlay'):assert raw==legacy
                        if variant=='split':assert all(abs(a-b)<=2e-5+2e-4*abs(b) for a,b in zip(v,legacy_values))
                        for bt,active in enumerate(mask):
                            if not active:assert all(x==0 for x in v[bt*24*256:(bt+1)*24*256])
                    for trial in range(3):
                        raw=z.read(prefix+'-commit'+str(trial)+'-counts.f32');accepted=floats(raw);vectors+=1
                        expected=[0]*4 if trial==0 else [min(n,b if trial==1 else [4,0,3,1][b]) for b,n in enumerate(counts)]
                        assert list(accepted)==expected and raw==previous.read(old+'-commit'+str(trial)+'-counts.f32')
                    if stage=='qualification':
                        assert row['status']=='correctness_pass' and row['variants_checked']==5 and row['overlay_legacy_bit_exact'] and row['readonly_overlay_preserved_kv'] and row['zero_partial_prefix_trials']==3 and row['full_kv_bit_exact'] and row['graph_replays']==3
                    else:
                        assert row['batch']==4 and row['tokens']==4 and row['heads']==24 and row['kv']==4 and row['capacity']==4096 and row['dim']==256
                        assert row['prefix_parameter']==[32,128,512,3968,4096,128,128,128][i] and row['pattern']==int(control[1])
                        assert row['order']=='ABCDEEDCBAx3' and row['iterations_per_sample']==20 and row['warmups_per_variant']==5
                        assert row['overlay_legacy_bit_exact'] and row['timed_context_exact'] and row['timed_full_kv_exact']
                        for key in ('baseline_ms','compact_ms','tiled_ms','split_ms','overlay_commit_ms'):
                            assert len(row[key])==6 and all(math.isfinite(v) and v>0 for v in row[key])
                        medians={key:statistics.median(row[key]) for key in ('baseline_ms','compact_ms','tiled_ms','split_ms','overlay_commit_ms')}
                        ratios.append({'case':i,'prefix_parameter':row['prefix_parameter'],'pattern':row['pattern'],'median_ms':medians,'baseline_over_overlay_commit':medians['baseline_ms']/medians['overlay_commit_ms']})
                    checks.append({'stage':stage,'case':i,'retained_parent_contexts_exact':True,'overlay_legacy_bit_exact':True,'active_counts_by_row':counts,'accepted_count_vectors_replayed':3})
    for stage in ('build','qualification','timing'):
        path=base+stage+'-ledger/commands.jsonl'
        if path not in m['files']:assert not passed;continue
        events=[json.loads(l) for l in z.read(path).decode().splitlines()];assert events[-1]['event']=='finished'
        if passed:assert events[-1]['exit_code']==0
        if stage!='build':
            monitor=doc(base+stage+'-resources/resource-monitor.json');samples=[json.loads(l) for l in z.read(base+stage+'-resources/resource-vram.jsonl').decode().splitlines()]
            assert monitor['child_exit_code']==events[-1]['exit_code'] and monitor['sampled_memory']['samples']==len(samples)
            assert monitor['sampled_memory']['sampled_peak_adapter_used_bytes']==max(s['used_bytes'] for s in samples)
    archived_verifier_sha=sha(z.read(methods+'verify_attention_kv_delta_timing.py'))
result={'status':'attention_kv_delta_timing_replayed' if passed else 'attention_kv_delta_timing_failure_preserved',
        'archive_sha256':sha(archive.read_bytes()),'files_verified':len(m['files']),'cases':checks,'raw_context_arrays':arrays,'raw_prefix_count_vectors':vectors,
        'component_event_ratios':ratios,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived_verifier_sha,
        'performance_admitted':False,'model_integration_admitted':False,
        'scope':'Frozen-slot synthetic B4/T4 component; legacy full-cache output versus readonly overlay and in-place full accepted-prefix commit, changed ownership/output contract. Same initialized canonical values, internal CUDA events, per-sample readbacks/checks outside events. Retained context arrays/counts replayed; full-cache checks remain in-process witnesses. No model/HTTP/DRAM-traffic/2x claim.'}
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
