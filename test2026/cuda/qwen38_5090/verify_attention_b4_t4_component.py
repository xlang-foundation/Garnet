"""Replay retained component output comparisons; never admit model performance."""
from pathlib import Path
from array import array
import hashlib,json,math,re,statistics,sys,zipfile

out=Path(sys.argv[1]);name='attention-b4-t4-component-attempt1'
archive=out/(name+'-evidence.zip');sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(archive) as z:
    doc=lambda n:json.loads(z.read(n));m=doc('evidence-manifest.json')
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
    for n,h in m['files'].items():assert sha(z.read(n))==h
    for n,h in m['linked_archives'].items():assert sha((out/n).read_bytes())==h
    base='results/'+name+'/'
    report=doc(base+'manifest.json');passed=report['exit_code']==0
    assert report['library_sha256']==m['library_sha256'] and report['library_unchanged']
    for n,h in report['source_sha256'].items():
        if n!='build/bin/garnet.dll':assert sha(z.read(n))==h
    for n,h in report['source_after_sha256'].items():assert sha(z.read(n))==h==report['source_sha256'][n]
    prep=doc('Garnet/test2026/cuda/qwen38_5090/evidence/b4-t4-attention-component-preparation.json')
    original=z.read('Garnet/'+prep['base_file'])
    assert sha(original)==prep['base_sha256']
    for n,h in prep['native_source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
    for n,h in prep['method_sha256'].items():
        path='build-attention-b4-t4-component.bat' if n.endswith('.bat') else 'Garnet/test2026/cuda/qwen38_5090/'+n
        assert sha(z.read(path))==h
    parent=doc('results/b4-joined-layer-profile-attempt1-verified.json')
    assert parent['status']=='b4_joined_layer_profile_replayed' and parent['correctness_passed']
    assert parent['archive_sha256']==report['parent_archive_sha256']==m['linked_archives']['b4-joined-layer-profile-attempt1-evidence.zip']
    derived=original.decode().replace('\r\n','\n')
    for r in prep['derivation']:
        assert derived.count(r['old'])==1;derived=derived.replace(r['old'],r['new'])
    assert z.read('Garnet/'+prep['candidate_file'])==derived.encode()
    assert sha(derived.encode())==prep['candidate_sha256']
    checks=[];speed_ratios=[]
    def floats(raw):
        assert len(raw)%4==0;v=array('f');v.frombytes(raw)
        if sys.byteorder!='little':v.byteswap()
        assert all(math.isfinite(x) for x in v);return v
    if passed:
        assert report['status']=='component_finished_needs_independent_replay'
        assert sha(z.read('build/bin/garnet_attention_b4_t4_test.exe'))==report['executable_sha256']
        assert len(report['rows'])==8
        for stage in ('qualification','timing'):
            log=z.read(base+stage+'-ledger/commands.log').decode('utf8')
            controls=re.findall(r'B4_ATTENTION_CONTROLS case=(\d+) pattern=(\d+) slots=([^\n]+) active=([^\n]+)',log)
            rows=[json.loads(l) for l in log.splitlines() if l.startswith('{"case":')]
            assert len(controls)==len(rows)==8
            if stage=='timing':assert rows==report['rows']
            for i,(control,row) in enumerate(zip(controls,rows)):
                ci,pattern,slots,mask=control
                assert int(ci)==row['case']==i and int(pattern)==([0]*5+[1,2,3])[i]
                slots=[int(v) for v in slots.strip().strip(',').split(',')]
                mask=[int(v) for v in mask.strip().strip(',').split(',')]
                assert len(slots)==len(mask)==16 and all(0<=s<4096 for s in slots)
                assert all(v in (0,1) for v in mask)
                counts=[sum(mask[b*4:b*4+4]) for b in range(4)]
                assert counts==([[4]*4]*5+[[2,2,2,0],[0]*4,[4,1,2,0]])[i]
                for b in range(4):
                    used=[slots[b*4+t] for t in range(4) if mask[b*4+t]]
                    assert len(used)==len(set(used))
                prefix=base+stage+'-context-case'+str(i)
                cpu=floats(z.read(prefix+'-cpu.f32'));assert len(cpu)==4*4*24*256
                legacy=z.read(prefix+'-baseline.f32');legacy_values=floats(legacy)
                for variant in ('baseline','compact','tiled','split'):
                    raw=z.read(prefix+'-'+variant+'.f32');actual=floats(raw)
                    assert len(actual)==len(cpu)
                    assert all(abs(a-b)<=2e-5+2e-4*abs(b) for a,b in zip(actual,cpu))
                    if variant in ('compact','tiled'):assert raw==legacy
                    if variant=='split':assert all(abs(a-b)<=2e-5+2e-4*abs(b) for a,b in zip(actual,legacy_values))
                    for bt,active in enumerate(mask):
                        if not active:assert all(v==0 for v in actual[bt*24*256:(bt+1)*24*256])
                if stage=='qualification':
                    assert row['status']=='correctness_pass' and row['full_kv_bit_exact'] and row['variants_checked']==4
                else:
                    assert row['batch']==4 and row['tokens']==4 and row['capacity']==4096
                    assert row['order']=='ABCDDCBAx3' and row['full_kv_exact'] and row['row_tiled_bit_exact'] and row['split_within_tolerance']
                    for key in ('baseline_ms','compact_ms','tiled_ms','split_ms'):
                        assert len(row[key])==6 and all(math.isfinite(v) and v>0 for v in row[key])
                    speed_ratios.append({'case':i,'baseline_median_ms':statistics.median(row['baseline_ms']),
                        'baseline_over_variant_medians':{key:statistics.median(row['baseline_ms'])/statistics.median(row[key]) for key in ('compact_ms','tiled_ms','split_ms')}})
                checks.append({'stage':stage,'case':i,'retained_cpu_context_tolerance':True,'compact_tiled_legacy_bit_exact':True,'inactive_context_zero':True,'active_counts_by_row':counts})
    for stage in ('build','qualification','timing'):
        ledger=base+stage+'-ledger/commands.jsonl'
        if ledger not in m['files']:
            assert not passed;continue
        events=[json.loads(l) for l in z.read(ledger).decode().splitlines()]
        assert events[-1]['event']=='finished'
        if passed:assert events[-1]['exit_code']==0
        if stage!='build':
            monitor=doc(base+stage+'-resources/resource-monitor.json')
            samples=[json.loads(l) for l in z.read(base+stage+'-resources/resource-vram.jsonl').decode().splitlines()]
            assert monitor['child_exit_code']==events[-1]['exit_code']
            assert monitor['sampled_memory']['samples']==len(samples)
            assert monitor['sampled_memory']['sampled_peak_adapter_used_bytes']==max(s['used_bytes'] for s in samples)
result={'status':'attention_b4_t4_component_replayed' if passed else 'attention_b4_t4_component_failure_preserved',
        'archive_sha256':sha(archive.read_bytes()),'files_verified':len(m['files']),'checks':checks,'component_event_ratios':speed_ratios,
        'performance_admitted':False,'scope':'Synthetic standalone attention with KV copies, two processes, unchanged existing kernels. Retained C++ double-CPU-context arrays replayed; full KV/input/guards remain in-process witnesses, not independently retained raw caches. No full-model correctness, actual4K prompt, production or2x result.'}
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
