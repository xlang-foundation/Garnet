"""Audit full-output admission before comparing any projection boundaries."""
import datetime
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
root=Path(sys.argv[1]).resolve()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
stem=root/'qwen35-mtp-stage-v2'
output=Path(str(stem)+'-summary.json');assert not output.exists()
reference=read(root/'qwen35-mtp-stage-v1-reference.json')
assert reference['all_original_complete_outputs_bit_exact']
cases=read(root/'mtp-component-fixtures.json')
def timestamp(s):return datetime.datetime.fromisoformat(s.replace('Z','+00:00'))
observer=[json.loads(line) for line in Path(str(stem)+'-processes.jsonl').read_text(encoding='utf-8-sig').splitlines()]
assert not any(x['event']=='observer_failure' for x in observer)
assert observer[-1]['event']=='stopped'
epoch=digest(Path(str(stem)+'-before-build-identity.json'))
binary=digest(Path(str(stem)+'.exe'))
records=[];native_failures=[];observed=0
for geometry,indices in [('2b',range(5)),('4b',range(5,7))]:
    native=read(Path(str(stem)+'-native-'+geometry+'.json'))
    native_exit=read(Path(str(stem)+'-native-'+geometry+'-exit.json'))
    assert native_exit['exit_code']==0 and native_exit['execution_identity_sha256'].lower()==epoch
    assert native['diagnostic_complete'] and native['full_capture_replay_outputs_exact'] and native['inactive_cache_and_input_bytes_preserved']
    assert [x['case'] for x in native['cases']]==list(indices)
    for item in native['cases']:
        assert len(item['runs'])==6
        for run in item['runs']:
            for tensor in run['tensors']:
                path=Path(tensor['path']);assert digest(path)==tensor['sha256'] and path.stat().st_size==tensor['words']*4
                expected=np.fromfile(cases[item['case']]['steps'][run['step']]['expected'][('hidden','key','value').index(tensor['name'])]['path'],dtype='<f4')
                actual=np.fromfile(path,dtype='<f4');assert np.isfinite(actual).all()
                bad=np.abs(actual.astype(np.float64)-expected.astype(np.float64))>2e-5+2e-4*np.abs(expected.astype(np.float64))
                assert int(bad.sum())==tensor['strict_bad_count']
                if tensor['strict_bad_count']:
                    native_failures.append({'case':item['case'],'repeat':run['repeat'],'step':run['step'],'tensor':tensor['name'],'count':tensor['strict_bad_count'],'max_abs':tensor['max_abs']})
    for index in indices:
        ref=reference['cases'][index];payload=Path(ref['file']).read_bytes();assert hashlib.sha256(payload).hexdigest()==ref['sha256']
        for mode in (0,1):
            tag=Path(str(stem)+f'-c{index}-o{mode}')
            result=read(Path(str(tag)+'.json'));exit=read(Path(str(tag)+'-exit.json'))
            assert exit['exit_code']==0 and exit['binary_sha256'].lower()==binary and exit['execution_identity_sha256'].lower()==epoch
            assert result['diagnostic_complete'] and result['admitted_exact_native']
            assert result['case']==index and result['observe']==bool(mode) and len(result['runs'])==2
            start,end=timestamp(exit['started_utc']),timestamp(exit['ended_utc'])
            snapshots=[x for x in observer if x['event']=='snapshot' and timestamp(x['begin_utc'])<end and timestamp(x['end_utc'])>start]
            # Short synthetic cases may miss a sample; retain that limitation.
            foreign=[{'snapshot':s['begin_utc'],'process':p} for s in snapshots for p in s['processes'] if p['pid']!=result['pid']]
            errors=[x for x in observer if x['event']=='process_metadata_unavailable' and start<=timestamp(x['observed_utc'])<=end]
            record={'case':index,'observe':bool(mode),'all_complete_outputs_native_exact':True,'stage_comparisons':[],
                    'sampled_observations':len(snapshots),'foreign_process_snapshots':foreign,'observer_metadata_errors':errors}
            observed+=sum(p['pid']==result['pid'] for s in snapshots for p in s['processes'])
            for run in result['runs']:
                assert run['complete_native_output_exact']
                files={x['name']:x for x in run['tensors']}
                raw=Path(files['packed']['file']).read_bytes()
                b,t,h=cases[index]['batch'],cases[index]['chunk'],cases[index]['config']['hidden_size']
                width=cases[index]['config']['num_key_value_heads']*cases[index]['capacity']*cases[index]['config']['head_dim']
                packed=np.frombuffer(raw,dtype='<f4').reshape(b,t*h+2*width)
                for name,sl in [('hidden',slice(0,t*h)),('key',slice(t*h,t*h+width)),('value',slice(t*h+width,None))]:
                    expected=Path(str(stem)+f'-native-{geometry}-c{index}-r0-s{run["step"]}-{name}.f32').read_bytes()
                    assert packed[:,sl].copy().tobytes()==expected
                if not mode:continue
                stages={s['weight']:s for s in result['observations']}
                for stage in ref['steps'][run['step']]['stages']:
                    actual_stage=stages[stage['weight']]
                    comparisons=[]
                    for kind,spec,actual_spec in zip(('precast_f32','input_bf16','output_bf16'),stage['parts'],actual_stage['parts']):
                        expected=payload[spec['offset']:spec['offset']+spec['bytes']]
                        assert hashlib.sha256(expected).hexdigest()==spec['sha256']
                        actual=Path(files[actual_spec['name']]['file']).read_bytes()
                        assert len(actual)==len(expected) and actual_spec['bf16']==spec['bf16']
                        dtype='<u2' if spec['bf16'] else '<u4'
                        a,e=np.frombuffer(actual,dtype=dtype),np.frombuffer(expected,dtype=dtype)
                        af=(a.astype(np.uint32)<<16).view(np.float32) if spec['bf16'] else a.view(np.float32)
                        ef=(e.astype(np.uint32)<<16).view(np.float32) if spec['bf16'] else e.view(np.float32)
                        assert np.isfinite(af).all() and np.isfinite(ef).all()
                        changed=np.flatnonzero(a!=e)
                        comparisons.append({'part':kind,'words':len(a),'different_words':len(changed),
                            'max_abs':float(np.abs(af.astype(np.float64)-ef.astype(np.float64)).max()),
                            'first_changed':[{'index':int(i),'actual':float(af[i]),'reference':float(ef[i])} for i in changed[:5]]})
                    record['stage_comparisons'].append({'step':run['step'],'weight':stage['weight'],'comparisons':comparisons})
            records.append(record)
assert len(records)==14
result={'all14_plain_and_observed_components_native_exact':True,'all_native_capture_replay_and_preservation_controls':True,
        'current_native_strict_numerical_gate_passed':not native_failures,'current_native_strict_failures':native_failures,
        'records':records,'sampled_native_component_observations':observed,
        'sampled_component_process_qualification':all(not x['foreign_process_snapshots'] and not x['observer_metadata_errors'] for x in records),
        'maximum_snapshot_gap_seconds':observer[-1]['maximum_snapshot_gap_seconds'],
        'execution_identity_sha256':epoch,'binary_sha256':binary,
        'serving_modified':False,'quality_accepted':False,'performance_accepted':False,
        'scope':'Complete-output-admitted synthetic stage boundary differences only; first differing boundary does not prove bug/root cause. Strict tolerance failures remain failures; no relaxed gate or learned-model speed/memory/quality acceptance.'}
with output.open('x') as stream:json.dump(result,stream,indent=2)
print('MTP_STAGE_ATTRIBUTION_AUDITED',len(records),'STRICT_NUMERICAL_GATE',not native_failures)
