"""Replay cached/hybrid component outputs and event arithmetic; no model claim."""
from pathlib import Path
from array import array
import hashlib,json,math,re,statistics,sys,zipfile
out=Path(sys.argv[1]);name='attention-kv-delta-cached-attempt1';archive=out/(name+'-evidence.zip')
sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(archive) as z:
    doc=lambda n:json.loads(z.read(n));m=doc('evidence-manifest.json')
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
    for n,h in m['files'].items():assert sha(z.read(n))==h
    for n,h in m['linked_archives'].items():assert sha((out/n).read_bytes())==h
    base='results/'+name+'/';report=doc(base+'manifest.json');passed=report['exit_code']==0
    assert report['library_unchanged'] and report['library_sha256']==m['library_sha256'] and report['hybrid_threshold']==512
    for n,h in report['source_sha256'].items():
        if n!='build/bin/garnet.dll':assert sha(z.read(n))==h
    for n,h in report['source_after_sha256'].items():assert sha(z.read(n))==h==report['source_sha256'][n]
    methods='Garnet/test2026/cuda/qwen38_5090/'
    prep=doc(methods+'evidence/attention-kv-delta-cached-preparation.json')
    assert prep['hybrid_visible_prefix_threshold']==512 and prep['profile_count']==12 and prep['original_profiles_retained']==8
    for n,h in prep['files'].items():assert sha(z.read(methods+n))==h
    for group in ('base_source_sha256','unchanged_source_sha256'):
        for n,h in prep[group].items():assert sha(z.read('Garnet/'+n))==h
    for n,h in prep['method_sha256'].items():
        path='build-attention-kv-delta-cached.bat' if n.endswith('.bat') else methods+n
        assert sha(z.read(path))==h
    native=z.read('Garnet/src/cuda/qwen35_attention_core.cu').decode().replace('\r\n','\n')
    core=z.read(methods+'attention_kv_delta_cached_core.cu').decode()
    helpers=native[native.index('__device__ float warpSum'):native.index('__global__ void writeKeys')]
    assert core.count(helpers)==1
    old_core=z.read(methods+'attention_kv_delta_core.cu').decode()
    start=old_core.index('__global__ void overlay(');compact=old_core[start:old_core.index('__global__ void commitPrefix',start)]
    for old,new in prep['compact_derivation']:
        assert compact.count(old)==1;compact=compact.replace(old,new)
    assert core.count(compact)==1
    start=native.index('__global__ void scores(');legacy=native[start:native.index('// Research candidate: one block per query',start)]
    for old,new in prep['legacy_derivation']:
        assert legacy.count(old)==1;legacy=legacy.replace(old,new)
    assert core.count(legacy)==1
    driver=z.read(methods+'attention_kv_delta_timing_test.cu').decode()
    for old,new in prep['driver_derivation']:
        assert driver.count(old)==1;driver=driver.replace(old,new)
    assert driver.count('mode==4?inputs[')==prep['canonical_cache_comparison_replacement_count']==6
    driver=driver.replace('mode==4?inputs[','mode>=4?inputs[')
    assert z.read(methods+'attention_kv_delta_cached_test.cu')==driver.encode()
    parent_name='attention-kv-delta-timing-attempt1';parent=doc('results/'+parent_name+'-verified.json')
    assert parent['status']=='attention_kv_delta_timing_replayed' and len(parent['cases'])==16 and parent['raw_context_arrays']==96 and parent['raw_prefix_count_vectors']==48
    assert parent['archive_sha256']==report['parent_archive_sha256']==m['linked_archives'][parent_name+'-evidence.zip']
    checks=[];ratios=[];arrays=0;vectors=0;retained=0
    prefixes=[32,128,512,3968,4096,128,128,128,511,513,1024,2048]
    patterns=[0]*5+[1,2,3]+[0]*4
    metric_keys=('baseline_ms','compact_ms','tiled_ms','split_ms','overlay_commit_ms','cached_compact_commit_ms','cached_legacy_commit_ms','hybrid_commit_ms')
    variants=('cpu','baseline','compact','tiled','split','overlay','cached-compact','cached-legacy','hybrid')
    def floats(raw):
        assert len(raw)%4==0;v=array('f');v.frombytes(raw)
        if sys.byteorder!='little':v.byteswap()
        assert all(math.isfinite(x) for x in v);return v
    with zipfile.ZipFile(out/(parent_name+'-evidence.zip')) as previous:
        for group in ('base_source_sha256','unchanged_source_sha256'):
            for n in prep[group]:assert z.read('Garnet/'+n)==previous.read('Garnet/'+n)
        if passed:
            assert report['status']=='cached_component_finished_needs_independent_replay'
            assert sha(z.read('build/bin/garnet_attention_kv_delta_cached_test.exe'))==report['executable_sha256']
            for stage in ('qualification','timing'):
                log=z.read(base+stage+'-ledger/commands.log').decode('utf8')
                rows=[json.loads(l) for l in log.splitlines() if l.startswith('{"case":')]
                controls=re.findall(r'B4_ATTENTION_CONTROLS case=(\d+) pattern=(\d+) slots=([^\n]+) active=([^\n]+)',log)
                assert len(rows)==len(controls)==12
                if stage=='timing':assert rows==report['rows']
                for i,(row,control) in enumerate(zip(rows,controls)):
                    assert row['case']==int(control[0])==i and int(control[1])==patterns[i]
                    slots=[int(v) for v in control[2].strip().strip(',').split(',')];mask=[int(v) for v in control[3].strip().strip(',').split(',')]
                    assert len(slots)==len(mask)==16 and all(0<=s<4096 for s in slots) and all(a in (0,1) for a in mask)
                    counts=[sum(mask[b*4:b*4+4]) for b in range(4)]
                    assert counts==([[4]*4]*5+[[2,2,2,0],[0]*4,[4,1,2,0]]+[[4]*4]*4)[i]
                    for b in range(4):
                        active_slots=[slots[b*4+t] for t in range(4) if mask[b*4+t]]
                        assert len(active_slots)==len(set(active_slots))
                    short=sum(a and slot+1<=512 for a,slot in zip(mask,slots));long=sum(mask)-short
                    prefix=base+stage+'-context-case'+str(i);old='results/'+parent_name+'/'+stage+'-context-case'+str(i)
                    cpu=floats(z.read(prefix+'-cpu.f32'));assert len(cpu)==4*4*24*256
                    legacy=z.read(prefix+'-baseline.f32');legacy_values=floats(legacy)
                    for variant in variants:
                        raw=z.read(prefix+'-'+variant+'.f32');v=floats(raw);arrays+=1
                        if i<8 and variant in variants[:6]:assert raw==previous.read(old+'-'+variant+'.f32');retained+=1
                        assert len(v)==len(cpu) and all(abs(a-b)<=2e-5+2e-4*abs(b) for a,b in zip(v,cpu))
                        if variant not in ('cpu','baseline','split'):assert raw==legacy
                        if variant=='split':assert all(abs(a-b)<=2e-5+2e-4*abs(b) for a,b in zip(v,legacy_values))
                        for bt,active in enumerate(mask):
                            if not active:assert all(x==0 for x in v[bt*24*256:(bt+1)*24*256])
                    for trial in range(3):
                        raw=z.read(prefix+'-commit'+str(trial)+'-counts.f32');accepted=floats(raw);vectors+=1
                        expected=[0]*4 if trial==0 else [min(n,b if trial==1 else [4,0,3,1][b]) for b,n in enumerate(counts)]
                        assert list(accepted)==expected
                        if i<8:assert raw==previous.read(old+'-commit'+str(trial)+'-counts.f32')
                    if stage=='qualification':
                        assert row['status']=='correctness_pass' and row['variants_checked']==8 and row['cached_readonly_legacy_exact'] and row['overlay_legacy_bit_exact'] and row['readonly_overlay_preserved_kv'] and row['zero_partial_prefix_trials']==3 and row['full_kv_bit_exact'] and row['graph_replays']==3
                    else:
                        assert row['batch']==4 and row['tokens']==4 and row['heads']==24 and row['kv']==4 and row['capacity']==4096 and row['dim']==256
                        assert row['prefix_parameter']==prefixes[i] and row['pattern']==patterns[i]
                        assert row['order']=='ABCDEFGHHGFEDCBAx3' and row['iterations_per_sample']==20 and row['warmups_per_variant']==5
                        assert row['overlay_legacy_bit_exact'] and row['timed_context_exact'] and row['timed_full_kv_exact']
                        for key in metric_keys:assert len(row[key])==6 and all(math.isfinite(v) and v>0 for v in row[key])
                        medians={key:statistics.median(row[key]) for key in metric_keys}
                        ratios.append({'case':i,'prefix_parameter':prefixes[i],'pattern':patterns[i],'median_ms':medians,
                                       'baseline_over_candidate_medians':{k:medians['baseline_ms']/medians[k] for k in metric_keys[4:]}})
                    checks.append({'stage':stage,'case':i,'overlay_contexts_legacy_exact':True,'active_counts_by_row':counts,
                                   'hybrid_eligible_short_tokens':short,'hybrid_eligible_long_tokens':long,'accepted_count_vectors_replayed':3})
    for stage in ('build','qualification','timing'):
        path=base+stage+'-ledger/commands.jsonl'
        if path not in m['files']:assert not passed;continue
        events=[json.loads(l) for l in z.read(path).decode().splitlines()];assert events[-1]['event']=='finished'
        if passed:assert events[-1]['exit_code']==0
        if stage!='build':
            monitor=doc(base+stage+'-resources/resource-monitor.json');samples=[json.loads(l) for l in z.read(base+stage+'-resources/resource-vram.jsonl').decode().splitlines()]
            assert monitor['child_exit_code']==events[-1]['exit_code'] and monitor['sampled_memory']['samples']==len(samples)
            assert monitor['sampled_memory']['sampled_peak_adapter_used_bytes']==max(s['used_bytes'] for s in samples)
    archived_verifier_sha=sha(z.read(methods+'verify_attention_kv_delta_cached.py'))
result={'status':'attention_kv_delta_cached_replayed' if passed else 'attention_kv_delta_cached_failure_preserved',
        'archive_sha256':sha(archive.read_bytes()),'files_verified':len(m['files']),'cases':checks,'raw_context_arrays':arrays,
        'raw_prefix_count_vectors':vectors,'retained_parent_context_arrays_compared':retained,'component_event_ratios':ratios,
        'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived_verifier_sha,
        'performance_admitted':False,'model_integration_admitted':False,
        'scope':'Synthetic B4/T4 cached KV-overlay component; twelve profiles/eight variants. Context/count/source/parent/ledger/resource replay; full-cache/input/guard checks remain in-process witnesses. Legacy full-cache output versus readonly overlay plus accepted-prefix in-place commit is a changed ownership/output contract. No model/HTTP/DRAM/2x result.'}
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
