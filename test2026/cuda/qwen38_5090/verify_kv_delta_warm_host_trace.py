"""Independent CPU replay of strict model witnesses and nested warm host traces."""
from pathlib import Path
import hashlib,json,sys,zipfile
out=Path(sys.argv[1]);repo=Path(sys.argv[2]) if len(sys.argv)>2 else Path('D:/CantorAI/work/qwen38-5090')
sys.path.insert(0,str(repo/'test2026/cuda/qwen38_5090'))
from warm_host_trace import WarmHostTrace
sha=lambda b:hashlib.sha256(b).hexdigest()
name='kv-delta-warm-host-trace-attempt1';archive=out/(name+'-evidence.zip');base='results/'+name+'/'
methods='Garnet/test2026/cuda/qwen38_5090/'
with zipfile.ZipFile(archive) as z:
    doc=lambda n:json.loads(z.read(n));evidence=doc('evidence-manifest.json')
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(evidence['files'])+1
    for n,h in evidence['files'].items():assert sha(z.read(n))==h
    for n,h in evidence['linked_archives'].items():assert sha((out/n).read_bytes())==h
    workflow=doc(base+'workflow.json');assert workflow['cleanup_finished'] and workflow['library_unchanged']
    assert workflow['library_sha256']==evidence['library_sha256']
    for key,after,prefix in (('source_sha256','source_after_sha256','Garnet/'),('helper_sha256','helper_after_sha256',methods),('method_sha256','method_after_sha256',methods)):
        assert workflow[key]==workflow[after]
        for n,h in workflow[key].items():assert sha(z.read(prefix+n))==h
    prepraw=z.read(methods+'evidence/kv-delta-warm-host-trace-preparation.json');prep=json.loads(prepraw)
    assert sha(prepraw)==workflow['preparation_sha256'] and prep['case_count']==8
    for n,h in prep['files'].items():assert sha(z.read(methods+n))==h
    assert sha((repo/'test2026/cuda/qwen38_5090/warm_host_trace.py').read_bytes())==sha(z.read(methods+'warm_host_trace.py'))
    parent='kv-delta-qpc-http-pilot-attempt1';proof=doc('results/'+parent+'-verified.json')
    assert proof['status']=='compact_kv_delta_qpc_http_pilot_replayed' and proof['measured_sse_worker_pairs']==64
    assert proof['archive_sha256']==workflow['parent_http_archive_sha256']==evidence['linked_archives'][parent+'-evidence.zip']
    assert proof['library_sha256']==evidence['library_sha256']
    closed=doc('results/'+parent+'/workflow.json');assert closed['cleanup_finished'] and closed['exit_code']==0
    assert closed['source_sha256']==workflow['source_sha256']
    full='kv-delta-full-b4-t4-attempt1';fullproof=doc('results/'+full+'-verified.json')
    assert fullproof['status']=='compact_kv_delta_full_model_replayed' and fullproof['raw_last_hidden_all_exact']
    assert fullproof['archive_sha256']==workflow['full_model_archive_sha256']==prep['full_model_archive_sha256']==evidence['linked_archives'][full+'-evidence.zip']
    with zipfile.ZipFile(out/(full+'-evidence.zip')) as previous:
        text=previous.read(methods+'probe_kv_delta_full_b4_t4.py').decode().replace('\r\n','\n')
        for before,after,count in prep['probe_derivation']:assert text.count(before)==count;text=text.replace(before,after)
        assert text.encode()==z.read(methods+'probe_kv_delta_warm_full_b4_t4.py') and sha(text.encode())==prep['probe_sha256']
        ledger=[json.loads(l) for l in z.read(base+'ledger/commands.jsonl').splitlines()]
        assert ledger[-1]['event']=='finished' and ledger[-1]['exit_code']==workflow['exit_code']
        cpu=doc(base+'cpu-tests.json');assert cpu['exit_code']==0 and 'Ran 4 tests' in cpu['stderr']
        if workflow['exit_code']!=0:
            result={'status':'compact_warm_host_trace_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(evidence['files']),'performance_admitted':False}
        else:
            assert workflow['status']=='warm_host_trace_finished_needs_independent_replay'
            current=doc(base+'native/correctness/report.json')
            original=json.loads(previous.read('results/'+full+'/native/report.json'))
            assert current['status']==original['status']=='backend_smoke_completed' and len(current['cases'])==8
            for field in ('batch','capacity','chunk','mtp_depth','draft_chunk','deferred_state_commit','draft_builder_workspace_bytes','kv_delta_state','shared_output_storage','joined_mlp_projections','host_embedding_lookup','garnet_dll_sha256','retained_state_report_sha256','retained_unjoined_b4_report_sha256','oracles','negative_commit_gate','kv_delta_contract'):
                assert current[field]==original[field],field
            golden={(c['group'],c['repeat']):c for c in original['cases']}
            seen=set();hidden=0
            for case in current['cases']:
                group,repeat=case['group'],case['repeat'];assert (group,repeat) not in seen;seen.add((group,repeat))
                old=golden[(group,repeat%2)]
                for field in ('keys','prompt_lengths','limits','generated_ids','expected_ids','ordinary_ids','ordinary_state','mtp_state'):
                    assert case[field]==old[field],(group,repeat,field)
                assert case['default_off_ids_and_state_exact'] and case['duplicate_pair_isolation_exact'] and case['full_target_state_exact'] and case['exact_b1_ids_match']
                for mode in ('ordinary','mtp'):
                    stem=f'group-{group}-repeat-{repeat}-{mode}-last-hidden.f32'
                    oldstem=f'group-{group}-repeat-{repeat%2}-{mode}-last-hidden.f32'
                    raw=z.read(base+'native/correctness/'+stem)
                    assert raw==previous.read('results/'+full+'/native/'+oldstem)
                    assert len(raw)==4*5120*4 and sha(raw)==case[mode+'_state']['last_hidden_sha256'];hidden+=1
            assert seen=={(g,r) for g in (0,1) for r in range(4)} and hidden==16
            trace=doc(base+'native/trace-report.json')
            assert trace['status']=='warm_host_trace_correctness_finished_needs_replay' and trace['library_sha256']==evidence['library_sha256']
            assert len(trace['captures'])==8 and trace['parent_http_archive_sha256']==proof['archive_sha256']
            tags=set()
            for c in trace['captures']:
                assert c['group'] in (0,1) and c['repeat'] in range(4)
                if c['repeat']<2:
                    assert c['phase']=='warmup' and c['graphs_before'] is None and c['graphs_after'] is None
                else:
                    assert c['phase']=='capture';tags.add(c['tag'])
                    assert c['tag']==f"group-{c['group']}-repeat-{c['repeat']}"
                    for graph,before in c['graphs_before'].items():
                        after=c['graphs_after'][graph]
                        assert before['graph_capture_failures']==after['graph_capture_failures']==0
                        assert before['graph_captures']>=1 and before['graph_captures']==after['graph_captures']
                        assert after['graph_replays']>=before['graph_replays']
            records=[json.loads(l) for l in z.read(base+'native/stages.jsonl').splitlines()]
            assert {r['tag'] for r in records}==tags=={f'group-{g}-repeat-{r}' for g in (0,1) for r in (2,3)}
            assert all(r['error'] is None and r['host_wall_seconds']==r['ended']-r['started'] for r in records)
            helper=WarmHostTrace();helper.records=records;assert helper.summaries()==trace['summaries']
            assert all(sum(r['stage']=='generate' and r['parent'] is None and r['tag']==tag for r in records)==1 for tag in tags)
            resource=doc(base+'resources/resource-monitor.json');samples=[json.loads(l) for l in z.read(base+'resources/resource-vram.jsonl').splitlines()]
            assert resource['child_exit_code']==0 and resource['sampled_memory']['samples']==len(samples)
            assert resource['sampled_memory']['sampled_peak_adapter_used_bytes']==max(v['used_bytes'] for v in samples)
            result={'status':'compact_warm_host_trace_replayed','archive_sha256':sha(archive.read_bytes()),'files_verified':len(evidence['files']),
                'library_sha256':evidence['library_sha256'],'cases':8,'raw_last_hidden_arrays_compared':hidden,'raw_hidden_all_exact':True,
                'captured_generate_calls':4,'captured_real_rows':16,'host_intervals':len(records),'summaries':trace['summaries'],'performance_admitted':False,
                'scope':prep['scope']+' Full canonical state remains a hash witness; raw full state absent. Parent spans include children. No CUDA kernel time or speed claim.'}
    result['replay_verifier_sha256']=sha(Path(__file__).read_bytes());result['archived_replay_verifier_sha256']=sha(z.read(methods+'verify_kv_delta_warm_host_trace.py'))
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
