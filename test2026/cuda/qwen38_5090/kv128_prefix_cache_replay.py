"""Strict retained T32 ordinary oracle and complete layer3 cache-prefix replay."""
import hashlib,json,math,struct,zipfile
from .kv128_prefix_cache_witness import check_graphs,snapshots,compare,same_state
from .bf16_packed_state_gate import _validate
from .fa3_prefix_cache_witness import PREFIXES,read_cache,check_graphs as legacy_check_graphs
REFERENCE='mtp-prefill16-full4k-admission-attempt1';OLD_PREFIX='fa3-prefix-cache-witness-attempt1'
SHA_REF='92eef5433c946f4f2f9dc831902cb80f48b060a92bc6d91879f07137367b1ea5'
SHA_PREFIX='0158289d1b6706905dc10fc101f3064eed332a76ebb591c4c6ccf85c87031592'
sha=lambda b:hashlib.sha256(b).hexdigest()
def diagnose_state(actual,reference,read,oldread):
    _validate(actual['logical']);_validate(reference['logical'])
    different=[dict(index=i,layer=a['layer'],role=a['role'],kind=a['kind'],actual_sha256=a['row_sha256'],reference_sha256=b['row_sha256']) for i,(a,b) in enumerate(zip(actual['logical']['components'],reference['logical']['components'])) if a['row_sha256']!=b['row_sha256']]
    vectors={}
    for name in ('hidden','normalized'):
        a=actual['vectors'][name];b=reference['vectors'][name];x=read(a['file']);y=oldread(b['file']);assert len(x)==len(y)==20480 and sha(x)==a['sha256'] and sha(y)==b['sha256']
        av=struct.unpack('<5120f',x);bv=struct.unpack('<5120f',y);assert all(math.isfinite(v) for v in av+bv)
        changed=[i for i in range(5120) if x[i*4:(i+1)*4]!=y[i*4:(i+1)*4]]
        vectors[name]=dict(byte_exact=x==y,different_F32_words=len(changed),first_different_word=changed[0] if changed else None,max_absolute_difference=max(abs(a-b) for a,b in zip(av,bv)))
    return dict(physical_state_exact=actual['logical']['physical_row_sha256']==reference['logical']['physical_row_sha256'],logical_state_exact=actual['logical']['logical_row_sha256']==reference['logical']['logical_row_sha256'],different_components=different,first_different_component=different[0] if different else None,vectors=vectors)
def replay(reports,reads,proofs,require_pass):
    path=proofs/(REFERENCE+'-evidence.zip');assert sha(path.read_bytes())==SHA_REF
    observations={};cache_result=None
    with zipfile.ZipFile(path) as old:
        assert old.testzip() is None;base='results/'+REFERENCE+'/native/';reference=json.loads(old.read(base+'report.json'));control=reference['cases'][0];oldread=lambda n:old.read(base+n)
        for phase,tokens in (('baseline',32),('native',128)):
            if phase not in reports:continue
            r=reports[phase];read=reads[phase];assert r['tokens']==tokens and r['prompt_ids']==control['prompt_ids'] and len(r['prompt_ids'])==3968
            assert not r['performance_admitted'] and not r['cause_established'] and r['reference_job']==REFERENCE
            if 'graphs' in r:check_graphs(r['graphs'],tokens)
            states={}
            for name in ('empty_state','ordinary_prefill','ordinary_final'):
                if name in r:states[name]=diagnose_state(r[name],reference['empty_state'] if name=='empty_state' else control[name],read,oldread)
            ids=r.get('ordinary_ids');complete=r['status']=='fa3_prefix_caches_collected_retained_ordinary_control_exact'
            if complete or require_pass:
                assert complete and r['retained_empty_prefill_final_vectors_and_states_exact'] is r['retained_128_output_ids_exact'] is True
                same_state(r['empty_state'],reference['empty_state'],read,oldread)
                for name in ('ordinary_prefill','ordinary_final'):same_state(r[name],control[name],read,oldread)
                assert len(ids)==128 and ids==control['ordinary_ids'];snapshots(r,read)
            observations[phase]=dict(status=r['status'],error=r.get('error'),complete_ordinary_T32_oracle_passed=complete,states=states,ordinary_128_ids_equal_T32=ids==control['ordinary_ids'] if ids is not None else None,first_different_output=next((i for i,(a,b) in enumerate(zip(ids,control['ordinary_ids'])) if a!=b),None) if ids else None,captured_prefixes=[row['prefix'] for row in r['snapshots']])
    if 'baseline' in reports and observations['baseline']['complete_ordinary_T32_oracle_passed']:
        path=proofs/(OLD_PREFIX+'-evidence.zip');assert sha(path.read_bytes())==SHA_PREFIX
        with zipfile.ZipFile(path) as old:
            assert old.testzip() is None;base='results/'+OLD_PREFIX+'/baseline/';prior=json.loads(old.read(base+'report.json'));a=snapshots(reports['baseline'],reads['baseline'])
            legacy_check_graphs(prior['graphs'],32);assert [r['prefix'] for r in prior['snapshots']]==list(PREFIXES)
            b={(r['prefix'],name):read_cache(r['caches'][name],lambda n:old.read(base+n)) for r in prior['snapshots'] for name in ('key','value')}
            assert a==b;observations['baseline']['all12_retained_T32_prefix_cache_arrays_byte_exact']=True
    if set(reports)=={'baseline','native'} and all(len(reports[k]['snapshots'])==6 for k in reports):
        cache_result=compare(reports['baseline'],reports['native'],reads['baseline'],reads['native'])
    if require_pass:
        assert set(observations)=={'baseline','native'} and all(row['complete_ordinary_T32_oracle_passed'] for row in observations.values()) and cache_result['all_prefix_KEY_and_VALUE_byte_exact']
    return dict(status='KV128_full_chain_prefix_and_strict_ordinary_oracle_replayed' if require_pass else 'KV128_prefix_failure_diagnostics_replayed',ordinary_observations=observations,cache_comparison=cache_result,full_six_case_MTP_quality_or_performance_qualified=False,two_times_target_achieved=False)
