"""Audit both frozen warmed stage traces, native controls, memory ownership."""
import collections
import datetime
import hashlib
import json
import sys
from pathlib import Path

root=Path(sys.argv[1]).resolve();output=root/'qwen35-nvtx-stage-v1-summary.json'
assert not output.exists();sources={}


def read(path):
    raw=path.read_bytes();sources[path.name]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))


epoch_time=lambda s:datetime.datetime.fromisoformat(s).timestamp()
identity_path=root/'qwen35-nvtx-stage-v1-before-run-identity.json'
identity=read(identity_path);epoch=sources[identity_path.name]
p=root/'qwen35-nvtx-stage-v1-processes.jsonl';blob=p.read_bytes();sources[p.name]=hashlib.sha256(blob).hexdigest()
obs=[json.loads(line) for line in blob.decode('utf-8-sig').splitlines()]
assert obs[0]['event']=='observer_started' and obs[-1]['event']=='stopped'
errors=[r for r in obs if r['event'] in ('observer_failure','process_metadata_unavailable')]
processes=[r for r in obs if r['event']=='process_observed']
cases=[];previous_end=None
for size,batch in [('2b',1),('4b',8)]:
    stem=f'profile-qwen35-nvtx-{size}-b{batch}-v1'
    terminal=read(root/(stem+'-exit.json'));assert terminal['exit_code']==0
    assert terminal['execution_identity_sha256'].lower()==epoch
    assert terminal['garnet_dll_sha256'].lower()==identity['garnet_binary_sha256'].lower()
    assert terminal['gpu_uuid']=='GPU-472bb21a-0ca8-9d60-47a0-c527bb3d2d81'
    rows=read(root/(stem+'-requests.json'))
    prior=read(root/f'matched512-garnet-{size}-b{batch}-study-shared-mtp-v1-a1.json')
    assert len(rows)==len(prior)==28
    for index,(row,ref) in enumerate(zip(rows,prior)):
        for field in ('profile','native_build_policy','input_sha256','prompt_token_ids','generated_token_ids',
                      'sampled_tokens_including_stop','finish_reasons','stop_token_ids','cache_policy',
                      'speculative_metrics','visual_tokens_per_row'):
            assert row[field]==ref[field],(size,index,field)
        assert row['owned_memory']['owned_model_bytes']==ref['owned_memory']['owned_model_bytes'],(size,index,'owned bytes')
        assert row['owned_memory']['speculative']['mtp_graph']['contexts']==ref['owned_memory']['speculative']['mtp_graph']['contexts']==1
        assert row['warmup']==(index<8) and row['vision_compile_ms']==0
        probe=row['nvtx_stage_diagnostic']
        assert probe['request_index']==index and probe['capture']==(index==8)
        assert not probe['native_layer_profiler_enabled'] and not probe['primary_latency_accepted']
        if index!=8:assert not probe['events'] and probe['native_thread_id'] is None
    assert read(root/(stem+'-preservation.json'))['same_all28_input_output_stop_profile_speculative_counters']
    data=read(root/(stem+'-stages.json'))
    assert data['schema_marker_protocol_passed'] and data['predeclared_attribution_gate_passed']
    assert data['kernel_count_attributed_fraction']>=.99 and data['kernel_time_attributed_fraction']>=.99
    counts=data['marker_label_counts'];metrics=rows[8]['speculative_metrics']
    for label in ('request','generation','spec_reset','speculative_prefill','target_prefill'):assert counts[label]==1
    assert counts['input_row']==batch
    assert counts.get('mtp_prompt_cache',0)+counts.get('mtp_initial_draft',0)==metrics['mtp_prefill_calls']
    assert counts.get('mtp_draft',0)==metrics['mtp_draft_calls']
    assert counts.get('mtp_correction',0)==metrics['mtp_correction_calls']
    assert counts['target_verify']==counts['target_commit']==metrics['target_verification_calls']
    assert data['range_count']==sum(e['operation']=='push' for e in rows[8]['nvtx_stage_diagnostic']['events'])
    lo,hi=map(epoch_time,(terminal['started_utc'],terminal['ended_utc']))
    assert previous_end is None or previous_end<=lo;previous_end=hi
    assert epoch_time(obs[0]['utc'])<=lo<hi<=epoch_time(obs[-1]['utc'])
    own=next(r for r in processes if r['pid']==rows[0]['process_id'] and lo<=epoch_time(r['observed_utc'])<=hi)
    allowed={own['pid']};parent=own['parent_pid']
    while parent not in allowed:
        ancestor=next((r for r in processes if r['pid']==parent),None)
        if ancestor is None or not ancestor['name'].lower().removesuffix('.exe').startswith('nsys'):break
        allowed.add(parent);parent=ancestor['parent_pid']
    foreign=[]
    for snap in obs:
        if snap['event']!='snapshot' or max(lo,epoch_time(snap['begin_utc']))>min(hi,epoch_time(snap['end_utc'])):continue
        other=[p for p in snap['processes'] if p['pid'] not in allowed]
        if other:foreign.append({'begin_utc':snap['begin_utc'],'end_utc':snap['end_utc'],'processes':other})
    phase_errors=[r for r in errors if 'observed_utc' not in r or lo<=epoch_time(r['observed_utc'])<=hi]
    cases.append({'model':size,'batch':batch,'process_id':own['pid'],'stage_trace':data,
        'observed_foreign_process_snapshots':foreign,'observer_metadata_errors':phase_errors,
        'process_observation_qualified':not foreign and not phase_errors})
result={'all56_native_work_owner_and_stage_controls_passed':True,'cases':cases,'execution_identity_sha256':epoch,
    'garnet_dll_sha256':identity['garnet_binary_sha256'],'sources_sha256':sources,
    'maximum_snapshot_gap_seconds':obs[-1]['maximum_snapshot_gap_seconds'],
    'process_observation_qualified':all(c['process_observation_qualified'] for c in cases),
    'primary_latency_accepted':False,'memory_accepted':False,'production_promoted':False,
    'scope':'One warmed request/model, original MTP policy, full28 signatures and owned bytes/context1 preserved per model. '
            'Per-thread NVTX nesting, actual CUDA API correlation and>=99percent kernel count/time attribution. '
            'Instrumentation/single-request observations are not primary latency/memory, independent quality, causal speedup or2x proof.'}
output.write_text(json.dumps(result,indent=2)+'\n')
print('QWEN35_NVTX_STAGE_FULL_AUDIT_PASS',len(cases))
