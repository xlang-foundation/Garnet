"""Audit actual-model early-KV fault isolation/reset/immutable-prefix recovery.
Usage: python3 audit_fault_model1.py ABSOLUTE_STUDY
No GPU work or inference is started. Diagnostic GPU spans are safety witnesses.
"""
from pathlib import Path
import copy
import hashlib
import json
import math
import re
import sys

r=Path(sys.argv[1]).resolve();load=lambda n:json.loads((r/n).read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
b=load('bindings2.json')
assert load('model-status2.json')['status']=='terminal' and load('model-status2.json')['returncode']==0
assert load('build-test-status1.json')['completed']
assert load('compile-context-guards1-status.json')['returncode']==load('context-guards1-status.json')['returncode']==0
assert 'context abort/reuse PASS' in (r/'context-guards1.stdout').read_text()
for mode in ['on','off']:
    assert load('ctest-'+mode+'1-status.json')['returncode']==0
    assert '100% tests passed out of 50' in (r/('ctest-'+mode+'1.stdout')).read_text()
assert sha(Path(b['worker']))==b['worker_sha256']
assert sha(Path(b['library'])/'libgarnet.dylib')==b['library_sha256']
assert sha(Path(b['fixture']))==b['fixture_sha256'] and sha(r/'source-audit1.json')==b['source_audit_sha256']
for name,h in load('source-audit1.json')['candidate_owned_files_sha256'].items():assert sha(Path(b['source'])/name)==h,name
config=json.loads((Path(b['model'])/'config.json').read_text())
assert [config[k] for k in ['num_hidden_layers','hidden_size','num_attention_heads','num_key_value_heads']]==[32,4096,32,8]
assert config.get('head_dim',config['hidden_size']//config['num_attention_heads'])==128
# Independent IEEE-F32 +zero byte references, not hashes copied from native output.
zero=hashlib.sha256();block=bytes(4*1024*1024)
for _ in range(16):zero.update(block)
zero_kv=zero.hexdigest();zero_hidden=hashlib.sha256(bytes(4*4096*4)).hexdigest()
prior=r.parent/'mistral-kv-commit-hoist1'
prior_manifest=json.loads((prior/'provenance1.json').read_text())
assert sha(prior/'model-on1.stdout')==prior_manifest['files_sha256']['model-on1.stdout']
prior_events=[json.loads(line) for line in (prior/'model-on1.stdout').read_text().splitlines() if line.startswith('{')]
prior_seed=next(e for e in prior_events if e.get('event')=='diagnostic_full_state' and e['invocation']==1)
expected_seed=dict(kv=prior_seed['layers'],hidden=prior_seed['hidden'])
labels={'tracked_generation','leased_generation','predict_rows','snapshot','final_hidden','hidden','row_reset','row_release','prefix_capture','prefix_restore','arm_while_failed'}
retention=['entries','accounted_host_bytes','private_host_bytes','shared_resource_bytes','shared_resource_count','capture_policy']

def check(data):
    assert data['completed'] and 'kv_hoist_test_faults' in data['backend'] and 'chunk_kv_commit_hoist' in data['backend']
    def state(s):
        assert set(s)=={'kv','hidden'} and len(s['kv'])==32 and all(len(pair)==2 for pair in s['kv'])
        assert all(re.fullmatch('[0-9a-f]{64}',h) for pair in s['kv'] for h in pair) and re.fullmatch('[0-9a-f]{64}',s['hidden'])
    def pred(p):assert len(p)==4 and all(type(v)is int and 0<=v<32768 for v in p)
    def invalid(e):assert e['rejected'] is True and isinstance(e['error'],str) and e['error']
    base=data['baseline'];state(base['seed_state']);state(base['healthy_state']);pred(base['seed_predictions']);pred(base['healthy_predictions'])
    assert base['seed_state']==expected_seed,'Seed differs from prior qualified hoist engine'
    assert {e['label'] for e in base['invalid_arm_calls']}=={'negative_fault','excess_fault'}
    for e in base['invalid_arm_calls']:invalid(e)
    initial=base['initial_cache'];assert initial['entries']==4 and initial['accounted_host_bytes']<=384*1024*1024
    def reset(s):
        state(s['zero_state']);assert s['zero_state']['kv']==[[zero_kv,zero_kv] for _ in range(32)] and s['zero_state']['hidden']==zero_hidden,'Reset did not zero complete state'
        assert {e['label'] for e in s['stale_rejections']}=={'old_epoch_tracked','old_epoch_leased'}
        for e in s['stale_rejections']:invalid(e)
        invalid(s['unready_rejection']);state(s['restored_state']);assert s['restored_state']==base['seed_state']
        assert s['restored_predictions']==base['seed_predictions']
    assert [c['mode'] for c in data['cases']]==[1,2]
    previous=base['initial_epochs'];modes=[]
    for c in data['cases']:
        mode=c['mode'];reset(c['reset_before']);reset(c['recovery']);invalid(c['failure'])
        assert 'Injected KV hoist fault' in c['failure']['error']
        w=c['witness'];assert w['mode']==mode and w['armed']==0 and w['encoded_commits']==2 and w['native_abort_called']
        assert w['pending_outputs']==w['live_submissions']==0 and w['live_command'] is False
        assert math.isfinite(w['observed_gpu_ms'])
        if mode==1:assert w['submitted_commands']==w['drained_commands']==w['completed_commands']==0 and w['observed_gpu_ms']==0
        else:assert w['submitted_commands']>=1 and w['submitted_commands']==w['drained_commands']==w['completed_commands'] and w['observed_gpu_ms']>0
        assert {e['label'] for e in c['blocked']}==labels and len(c['blocked'])==11
        for e in c['blocked']:invalid(e)
        assert c['invalid_metadata']==dict(tracked_count=-1,prefill_count=-1)
        assert c['cache_before']==c['cache_after'],'Failed state changed immutable-prefix metadata'
        for key in retention:assert c['cache_before'][key]==initial[key]
        state(c['recovered_state']);assert c['recovered_state']==base['healthy_state'] and c['recovered_predictions']==base['healthy_predictions']
        assert len(c['epochs_at_failure'])==len(c['new_epochs'])==4
        assert all(c['epochs_at_failure'][i]>previous[i] and c['new_epochs'][i]>c['epochs_at_failure'][i] for i in range(4));previous=c['new_epochs']
        modes.append(dict(mode=mode,submitted=w['submitted_commands'],drained=w['drained_commands'],completed=w['completed_commands'],blocked_calls=11,invalid_metadata_sentinels=2,exact_full_state_prefix_and_continuation=True))
    disarmed=data['disarmed'];reset(disarmed['reset']);state(disarmed['state'])
    assert disarmed['state']==base['healthy_state'] and disarmed['predictions']==base['healthy_predictions']
    assert all(disarmed['epochs'][i]>previous[i] for i in range(4));invalid(data['closed_rejection'])
    for key in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']:assert data['cleanup'][key]==0
    assert data['cleanup']['metal_allocated_bytes']<=4194304
    assert data['memory']['metal_allocated_bytes']<data['memory']['metal_recommended_working_set_bytes']-256*1024*1024
    return modes

raw=load('model-results2.json');modes=check(raw)
mutations=[('wrong_submission_drain',lambda d:d['cases'][1]['witness'].__setitem__('drained_commands',0)),
 ('exposed_token_count',lambda d:d['cases'][0]['invalid_metadata'].__setitem__('tracked_count',128)),
 ('reset_stale_state',lambda d:d['cases'][0]['recovery']['zero_state']['kv'][0].__setitem__(0,'0'*64)),
 ('wrong_restored_state',lambda d:d['cases'][1]['recovery']['restored_state'].__setitem__('hidden','0'*64)),
 ('wrong_recovered_prediction',lambda d:d['cases'][1]['recovered_predictions'].__setitem__(0,-1)),
 ('missing_poison_guard',lambda d:d['cases'][0]['blocked'].pop()),
 ('retained_native_owner',lambda d:d['cleanup'].__setitem__('garnet_live_matrix_owners',1))]
rejected=[]
for name,mutate in mutations:
    changed=copy.deepcopy(raw);mutate(changed)
    try:check(changed)
    except (AssertionError,KeyError,ValueError):rejected.append(name)
    else:raise AssertionError('Invalid fault evidence accepted: '+name)
result=dict(passed=True,scope=__doc__,modes=modes,native_tests=dict(faults_on=50,faults_off=50),
    prior_qualified_seed_digest_matches=65,independent_complete_zero_state_digest_checks=325,
    exact_restored_or_continued_digest_comparisons=520,
    old_epoch_and_uninitialized_prediction_rejections=15,reset_disarms_unconsumed_fault=True,standalone_context_arming_guards_passed=True,
    cache_entries_preserved=raw['baseline']['initial_cache']['entries'],postclose_metal_bytes=raw['cleanup']['metal_allocated_bytes'],
    corruption_controls_rejected=rejected,
    limitation='Two explicit software fault sites at first eligible layer, with/without prior GPU submission. Not every hardware error, cancellation, cache pressure, model, or serving failure protocol; general nonfinite arithmetic and coding/serving performance remain unqualified.',
    two_product_attention_implemented=False,serving_speed_claim=False,quality_gate_passed=False,goal_complete=False)
(r/'fault-model-audit1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
