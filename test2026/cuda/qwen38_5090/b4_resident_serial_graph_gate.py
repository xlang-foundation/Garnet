"""Actual snapshot policy/branch evidence; requested flags alone are insufficient."""
import json,re
NAMES={'target4','target32','decoder_norm','verifier_norm','prefix_commit','draft4','draft16','wide_norm'}
COUNTERS=('contexts','graph_captures','graph_replays','graph_capture_failures','serial_graph_reuses','pending_graph_reuses','serial_graph_warmup_waits')

def admit(raw,expected_ids):
    assert expected_ids and all(isinstance(n,str) and re.fullmatch('[0-9a-f]{32}',n) for n in expected_ids)
    seen=set();rows=[];previous={}
    for line in raw.splitlines():
        try:event=json.loads(json.loads(line)['line'])
        except (ValueError,TypeError,KeyError):continue
        if not isinstance(event,dict) or event.get('event')!='study_graph_snapshot':continue
        ids=event['request_ids'];assert type(ids) is list and 1<=len(ids)<=4 and len(set(ids))==len(ids)
        assert event['physical_batch']==4 and event['real_rows']==len(ids) and not seen.intersection(ids) and set(ids)<=set(expected_ids)
        assert event['CUDA_kernel_elapsed_claimed'] is event['clean_throughput_result_claimed'] is False
        graphs=event['graphs'];assert set(graphs)==NAMES
        selected={}
        for name,g in graphs.items():
            assert g['cuda_graph_requested'] is g['serial_graph_reuse_requested'] is True
            assert all(type(g[n]) is int and g[n]>=0 for n in COUNTERS)
            counters={n:g[n] for n in COUNTERS};assert counters['graph_capture_failures']==0
            assert counters['graph_replays']>=counters['serial_graph_reuses']>=counters['pending_graph_reuses']
            if name in previous:assert all(counters[n]>=previous[name][n] for n in COUNTERS),('Counter decreased',name)
            selected[name]=counters
        assert event['io']['embedding_mode']=='host_mapped_bf16' and event['io']['head_precision']=='modelopt_nvfp4_w4a4'
        rows.append(dict(request_ids=ids,graphs=selected));previous=selected;seen.update(ids)
    assert seen==set(expected_ids) and rows
    # These are the primary repeatedly invoked target/draft programs. Other
    # graph counters remain recorded without inventing a speed attribution.
    for name in ('target4','target32','draft4','draft16'):
        assert previous[name]['graph_captures']>0 and previous[name]['graph_replays']>0 and previous[name]['serial_graph_reuses']>0,('Reuse branch not exercised',name)
    return dict(status='resident_serial_requested_policy_and_actual_graph_reuse_replayed',snapshot_groups=len(rows),real_control_requests=len(seen),last_counters=previous,rows=rows,
                math_or_output_ID_equivalence_proved_by_this_gate=False,CUDA_kernel_time_or_clean_throughput_qualified=False,
                scope='Counter/flag evidence only. Independent actual output-ID and transport gates must also pass. Resident IO bytes require their independent native legacy comparison. Joint policy is one configuration; no individual causal speed attribution.')
