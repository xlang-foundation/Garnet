"""Replay full-model witnesses, then select decode windows by raw CUDA markers."""
from pathlib import Path
import hashlib,json,sqlite3,sys,zipfile
from cuda_trace_windows import select_window

out=Path(sys.argv[1]);name='kv-delta-decode-cuda-trace-attempt1'
archive=out/(name+'-evidence.zip');base='results/'+name+'/'
methods='Garnet/test2026/cuda/qwen38_5090/'
sha=lambda b:hashlib.sha256(b).hexdigest()

def aggregate(rows):
    groups={}
    for row in rows:
        item=groups.setdefault(row['name'],{'name':row['name'],'count':0,'sum_duration_ns':0,'max_duration_ns':0})
        dt=row['end']-row['start'];assert dt>=0
        item['count']+=1;item['sum_duration_ns']+=dt;item['max_duration_ns']=max(item['max_duration_ns'],dt)
    return sorted(groups.values(),key=lambda r:r['sum_duration_ns'],reverse=True)

def union_ns(rows):
    total=0;left=right=None
    for row in sorted(rows,key=lambda v:v['start']):
        a,b=row['start'],row['end'];assert b>=a
        if right is None:left,right=a,b
        elif a<=right:right=max(right,b)
        else:total+=right-left;left,right=a,b
    return total+(right-left if right is not None else 0)

with zipfile.ZipFile(archive) as z:
    doc=lambda n:json.loads(z.read(n));e=doc('evidence-manifest.json')
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(e['files'])+1
    for n,h in e['files'].items():assert sha(z.read(n))==h
    for n,h in e['linked_archives'].items():assert sha((out/n).read_bytes())==h
    w=doc(base+'workflow.json');assert w['cleanup_finished'] and w['library_unchanged'] and w['library_sha256']==e['library_sha256']
    for key,after,prefix in (('source_sha256','source_after_sha256','Garnet/'),('helper_sha256','helper_after_sha256',methods),('method_sha256','method_after_sha256',methods)):
        assert w[key]==w[after]
        for n,h in w[key].items():assert sha(z.read(prefix+n))==h
    prepraw=z.read(methods+'evidence/kv-delta-decode-cuda-trace-preparation.json');prep=json.loads(prepraw)
    assert sha(prepraw)==w['preparation_sha256'] and prep['case_count']==8 and prep['capture_calls']==[2,3]
    for n,h in prep['files'].items():assert sha(z.read(methods+n))==h
    assert sha(Path(__file__).with_name('cuda_trace_windows.py').read_bytes())==sha(z.read(methods+'cuda_trace_windows.py'))
    assert z.read('test-kv-delta-decode-cuda-trace.bat')==z.read(methods+'test_kv_delta_decode_cuda_trace.bat')
    smoke='nsight-cuda-smoke-2026-attempt1-evidence.zip'
    assert e['linked_archives'][smoke]==w['nsight_smoke_archive_sha256']=='cc37ade5bd19ba07aa35e93172b88ddcfd2d20e3b38f5ff02cccb1da4a615b9f'
    setup_path='results/nsight-2026-cab-extraction/setup.json'
    with zipfile.ZipFile(out/smoke) as old:assert z.read(setup_path)==old.read(setup_path)
    setup=doc(setup_path)
    assert setup['nsys_sha256']==w['nsys_sha256']==w['nsys_after_sha256']=='e597347ef6cb45456612c0c9593401bb6d1b85497bb18534968f309955831291'
    assert '2026.5.1.161-265138896106v0' in w['nsys_version']
    parent='padded-mtp-http-pilot-attempt1';proof=doc('results/'+parent+'-verified.json')
    assert proof['status']=='padded_mtp_http_pilot_replayed' and proof['measured_sse_worker_pairs']==64
    assert proof['archive_sha256']==w['parent_http_archive_sha256']==e['linked_archives'][parent+'-evidence.zip']
    closed=doc('results/'+parent+'/workflow.json');assert closed['cleanup_finished'] and closed['exit_code']==0 and closed['source_sha256']==w['source_sha256']
    assert closed['library_sha256']==proof['library_sha256']==e['library_sha256']
    config='models/Qwen3.8-27B-NVFP4/config.json'
    assert sha(z.read(config))==w['model_config_sha256']==w['model_config_after_sha256']==closed['model_config_sha256']==closed['model_config_after_sha256']
    full='kv-delta-full-b4-t4-attempt1';fullproof=doc('results/'+full+'-verified.json')
    assert fullproof['status']=='compact_kv_delta_full_model_replayed' and fullproof['raw_last_hidden_all_exact']
    assert fullproof['archive_sha256']==w['full_model_archive_sha256']==prep['full_model_archive_sha256']==e['linked_archives'][full+'-evidence.zip']
    ledger=[json.loads(l) for l in z.read(base+'ledger/commands.jsonl').splitlines()]
    assert ledger[-1]['event']=='finished' and ledger[-1]['exit_code']==w['exit_code']
    cpu=doc(base+'cpu-tests.json');assert cpu['exit_code']==0 and 'Ran 10 tests' in cpu['stderr']
    result={'status':'compact_decode_cuda_trace_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(e['files']),'performance_admitted':False}
    if w['exit_code']==0:
        assert w['status']=='decode_cuda_trace_finished_needs_independent_replay' and w['profile_exit_code']==w['stats_exit_code']==0
        with zipfile.ZipFile(out/(full+'-evidence.zip')) as previous:
            text=previous.read(methods+'probe_kv_delta_full_b4_t4.py').decode().replace('\r\n','\n')
            for before,after,count in prep['probe_derivation']:assert text.count(before)==count;text=text.replace(before,after)
            assert text.encode()==z.read(methods+'probe_kv_delta_warm_full_b4_t4.py') and sha(text.encode())==prep['warm_probe_sha256']
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
        assert trace['status']=='decode_cuda_trace_correctness_finished_needs_replay' and trace['generate_calls']==8
        assert [c['call'] for c in trace['captures']]==[2,3]
        expected_markers={2:(8973*4,8977*4),3:(8981*4,8989*4)}
        for capture in trace['captures']:
            assert capture['group']==0 and capture['repeat']==capture['call']
            assert (capture['start_copy_bytes'],capture['end_copy_bytes'])==expected_markers[capture['call']]
            for graph,before in capture['graphs_before'].items():
                after=capture['graphs_after'][graph]
                assert before['graph_capture_failures']==after['graph_capture_failures']==0
                assert before['graph_captures']>=1 and before['graph_captures']==after['graph_captures']
                assert after['graph_replays']>=before['graph_replays']
            case=next(c for c in current['cases'] if c['group']==0 and c['repeat']==capture['repeat'])
            assert capture['speculative_metrics']==case['speculative_metrics']
            before=capture['metrics_before_generate'];after=capture['speculative_metrics']
            assert before['mtp_prefill_calls']>0 and before['mtp_head_calls']==1
            assert before['iterations']==[]
            for key in ('target_verification_calls','target_prefix_replays','target_prefix_recomputes','mtp_draft_calls','mtp_correction_calls'):
                assert before[key]==0
            assert before['mtp_prefill_calls']==after['mtp_prefill_calls']
        # Extract only into this output's fixed scratch directory; open read-only.
        scratch=out/'replay-scratch';scratch.mkdir(exist_ok=True)
        sql=scratch/(name+'.sqlite');sql.write_bytes(z.read(base+'startup-trace.sqlite'))
        connection=sqlite3.connect('file:'+sql.as_posix()+'?mode=ro',uri=True);connection.row_factory=sqlite3.Row
        try:
            kernels=[dict(r) for r in connection.execute('select k.*,s.value as name from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id order by k.start')]
            apis=[dict(r) for r in connection.execute('select r.*,s.value as name from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id')]
            copies=[dict(r) for r in connection.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY')]
            gpu=[dict(r) for r in connection.execute('select name,uuid,computeMajor,computeMinor from TARGET_INFO_GPU where computeMajor>0')]
            assert len(gpu)==1 and gpu[0]['computeMajor']==12 and '5090' in gpu[0]['name']
            windows=[]
            for capture in trace['captures']:
                left,right,selected=select_window(kernels,copies,capture['start_copy_bytes'],capture['end_copy_bytes'])
                assert len(selected)>5
                selected_apis=[r for r in apis if r['start']>=left and r['end']<=right];assert selected_apis
                node=[k for k in selected if k['graphNodeId']];outside=[k for k in selected if not k['graphNodeId']]
                windows.append({'call':capture['call'],'group':0,'repeat':capture['repeat'],
                    'marker_interval_ns':[left,right],'kernel_records':len(selected),'graph_node_kernel_records':len(node),
                    'kernel_sum_duration_ns':sum(k['end']-k['start'] for k in selected),
                    'kernel_interval_union_ns':union_ns(selected),'kernels_by_name':aggregate(selected),
                    'graph_node_kernels_by_name':aggregate(node),'non_graph_node_kernels_by_name':aggregate(outside),
                    'apis_by_name':aggregate(selected_apis),'speculative_metrics':capture['speculative_metrics'],
                    'metrics_before_generate':capture['metrics_before_generate']})
            assert windows[0]['marker_interval_ns'][1]<windows[1]['marker_interval_ns'][0]
        finally:connection.close()
        resource=doc(base+'resources/resource-monitor.json');samples=[json.loads(l) for l in z.read(base+'resources/resource-vram.jsonl').splitlines()]
        with zipfile.ZipFile(out/(parent+'-evidence.zip')) as parent_zip:
            assert resource['gpu']==json.loads(parent_zip.read('results/'+parent+'/resources/resource-monitor.json'))['gpu']
        assert resource['child_exit_code']==0 and resource['sampled_memory']['samples']==len(samples)
        peak=max(v['used_bytes'] for v in samples);assert resource['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
        result.update(status='compact_decode_cuda_trace_replayed',library_sha256=e['library_sha256'],cases=8,
            raw_last_hidden_arrays_compared=hidden,raw_hidden_all_exact=True,captured_generate_calls=2,captured_real_rows=8,
            windows=windows,nsys_version=w['nsys_version'],gpu=gpu,sampled_peak_adapter_used_bytes=peak,
            scope=prep['scope']+' Full canonical state remains a producer hash witness. Graph-node membership is an observation, not operator attribution. Kernel sums may overlap; CPU/API/GPU durations are not additive. Profiler overhead applies; no HTTP speed or2x claim.')
    result['replay_verifier_sha256']=sha(Path(__file__).read_bytes());result['archived_replay_verifier_sha256']=sha(z.read(methods+'verify_kv_delta_decode_cuda_trace.py'))
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='windows'},indent=2))
if 'windows' in result:
    for w in result['windows']:
        print(json.dumps({'call':w['call'],'kernel_interval_union_ns':w['kernel_interval_union_ns'],'top_kernels':w['kernels_by_name'][:12]},indent=2))
