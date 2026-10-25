"""Replay per-real-row state witnesses and raw hidden arrays for private padding."""
from pathlib import Path
import hashlib,json,math,re,struct,sys,zipfile
out=Path(sys.argv[1]);name='register128-padded-state-attempt1';archive=out/(name+'-evidence.zip')
sha=lambda b:hashlib.sha256(b).hexdigest();base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/'
strip=lambda state:{k:v for k,v in state.items() if k!='last_hidden_row_sha256'}
row_bytes=5120*4
def check_state(state,raw):
    assert len(raw)==4*row_bytes and all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',raw))
    assert state['state_words_per_row']==173932544 and len(state['state_row_sha256'])==len(state['slots'])==4
    assert all(isinstance(h,str) and re.fullmatch('[0-9a-f]{64}',h) for h in state['state_row_sha256'])
    assert all(type(n) is int and 0<=n<=4096 for n in state['slots'])
    assert sha(raw)==state['last_hidden_sha256']
    assert [sha(raw[i*row_bytes:(i+1)*row_bytes]) for i in range(4)]==state['last_hidden_row_sha256']
    assert [sha(raw[i*2*row_bytes:(i+1)*2*row_bytes]) for i in range(2)]==state['last_hidden_pair_sha256']

with zipfile.ZipFile(archive) as z:
    doc=lambda n:json.loads(z.read(n));evidence=doc('evidence-manifest.json')
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(evidence['files'])+1
    for n,h in evidence['files'].items():assert sha(z.read(n))==h
    for n,h in evidence['linked_archives'].items():assert sha((out/n).read_bytes())==h
    w=doc(base+'workflow.json');assert w['cleanup_finished'] and w['library_unchanged'] and w['library_sha256']==evidence['library_sha256']
    for key,after,prefix in (('source_sha256','source_after_sha256','Garnet/'),('helper_sha256','helper_after_sha256',methods),('method_sha256','method_after_sha256',methods)):
        assert w[key]==w[after]
        for n,h in w[key].items():assert sha(z.read(prefix+n))==h
    prepraw=z.read(methods+'evidence/register128-padded-state-preparation.json');prep=json.loads(prepraw)
    assert sha(prepraw)==w['preparation_sha256'] and prep['cases']==12 and prep['physical_batch']==4 and prep['repeats']==2
    for n,h in prep['files'].items():assert sha(z.read(methods+n))==h
    warm='kv-delta-decode-cuda-trace-attempt1';warmproof=doc('results/'+warm+'-verified.json')
    assert warmproof['status']=='compact_decode_cuda_trace_replayed' and warmproof['cases']==8 and warmproof['raw_hidden_all_exact']
    assert warmproof['archive_sha256']==w['decode_trace_archive_sha256']==prep['required_decode_trace_archive_sha256']==evidence['linked_archives'][warm+'-evidence.zip']
    assert warmproof['library_sha256']==evidence['library_sha256']
    config='models/Qwen3.8-27B-NVFP4/config.json';assert sha(z.read(config))==w['model_config_sha256']==w['model_config_after_sha256']
    warmw=doc('results/'+warm+'/workflow.json');assert warmw['exit_code']==0 and warmw['cleanup_finished']
    assert w['source_sha256']==warmw['source_sha256'] and w['helper_sha256']==warmw['helper_sha256']
    full='kv-delta-full-b4-t4-attempt1';fullproof=doc('results/'+full+'-verified.json')
    assert fullproof['status']=='compact_kv_delta_full_model_replayed' and fullproof['raw_last_hidden_all_exact']
    assert fullproof['archive_sha256']==w['full_model_archive_sha256']==prep['parent_full_archive_sha256']==prep['full_model_archive_sha256']==evidence['linked_archives'][full+'-evidence.zip']
    ledger=[json.loads(l) for l in z.read(base+'ledger/commands.jsonl').splitlines()]
    assert ledger[-1]['event']=='finished' and ledger[-1]['exit_code']==w['exit_code']
    cpu=doc(base+'cpu-tests.json');assert cpu['exit_code']==0 and 'Ran 5 tests' in cpu['stderr']
    with zipfile.ZipFile(out/(full+'-evidence.zip')) as previous:
        raw=previous.read(methods+'probe_kv_delta_full_b4_t4.py');assert sha(raw)==prep['parent_probe_sha256']
        text=raw.decode().replace('\r\n','\n')
        for before,after,count in prep['probe_derivation']:assert text.count(before)==count;text=text.replace(before,after)
        assert text.encode()==z.read(methods+'probe_register128_padded_state.py') and sha(text.encode())==prep['probe_sha256']
        launcher=previous.read(methods+'test_kv_delta_full_b4_t4.bat').decode().replace('\r\n','\n')
        assert launcher.count('probe_kv_delta_full_b4_t4.py')==1
        launcher=launcher.replace('probe_kv_delta_full_b4_t4.py','probe_register128_padded_state.py').encode()
        assert launcher==z.read(methods+'test_register128_padded_state.bat')==z.read('test-register128-padded-state.bat')
        if w['exit_code']!=0:
            result={'status':'compact_register128_padded_state_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(evidence['files']),'real_state_all_exact':False,'performance_admitted':False}
        else:
            assert w['status']=='register128_padded_state_finished_needs_independent_replay'
            report=doc(base+'native/report.json');gold=json.loads(previous.read('results/'+full+'/native/report.json'))
            assert report['status']=='padded_mtp_state_passed_needs_replay' and len(report['cases'])==12 and report['private_padding']
            assert report['register_delta_128'] is True
            for c in report['cases']:assert c['decoder_memory']['packed_profiles']['register_delta_128'] is True
            for field in ('batch','capacity','chunk','mtp_depth','draft_chunk','deferred_state_commit','draft_builder_workspace_bytes','kv_delta_state','shared_output_storage','joined_mlp_projections','host_embedding_lookup','garnet_dll_sha256','retained_state_report_sha256','retained_unjoined_b4_report_sha256','oracles','kv_delta_contract'):
                assert report[field]==gold[field],field
            negative=report['negative_commit_gate'];old_negative=gold['negative_commit_gate']
            assert negative['status']=='passed' and negative['metadata_rejections']==136 and negative['valid_zero_prefix_modes']==2
            assert negative['rejected_calls']==old_negative['rejected_calls'] and len(negative['rejected_calls'])==11
            for mode in ('before','after'):
                raw=z.read(base+'native/kv-negative-'+mode+'-last-hidden.f32');check_state(negative[mode],raw)
                assert raw==previous.read('results/'+full+'/native/kv-negative-'+mode+'-last-hidden.f32')==bytes(4*row_bytes)
                assert strip(negative[mode])==old_negative[mode]
            assert negative['before']==negative['after']
            teachers={};golden_cases={(c['group'],c['repeat']):c for c in gold['cases']}
            for case in gold['cases']:
                for i,key in enumerate(case['keys']):
                    value={'ids':case['expected_ids'][i],'length':case['prompt_lengths'][i],'limit':case['limits'][i]}
                    assert key not in teachers or teachers[key]==value;teachers[key]=value
            seen=set();cases=[];raw_hidden=reset_hidden=full_controls=0
            for case in report['cases']:
                g,r=case['group'],case['repeat'];assert (g,r) not in seen;seen.add((g,r));assert r in (0,1) and g in range(6)
                keys=prep['keys_by_group'][g];real=len(keys);assert real==prep['real_rows_by_group'][g]==case['real_rows']
                assert case['keys']==keys and case['padded_rows']==4-real
                lengths=[teachers[k]['length'] for k in keys];assert case['prompt_lengths']==lengths+[lengths[0]]*(4-real)
                assert case['limits']==[teachers[k]['limit'] for k in keys] and len(set(case['limits']))==1
                expected=[teachers[k]['ids'] for k in keys]
                assert case['expected_ids']==case['ordinary_ids']==case['generated_ids']==expected and case['exact_b1_ids_match']
                reset_raw=z.read(base+f'native/group-{g}-repeat-{r}-reset-last-hidden.f32');check_state(case['reset_state'],reset_raw)
                assert reset_raw==bytes(4*row_bytes) and case['reset_state']==negative['before'];reset_hidden+=1
                arrays={}
                for mode in ('ordinary','mtp'):
                    arrays[mode]=z.read(base+f'native/group-{g}-repeat-{r}-{mode}-last-hidden.f32')
                    check_state(case[mode+'_state'],arrays[mode]);raw_hidden+=1
                a,b=case['ordinary_state'],case['mtp_state']
                assert case['real_rows_state_exact'] and all(a[key][:real]==b[key][:real] for key in ('state_row_sha256','last_hidden_row_sha256','slots'))
                assert arrays['ordinary'][:real*row_bytes]==arrays['mtp'][:real*row_bytes]
                assert case['all_physical_rows_state_equal']==(a==b)
                duplicates=[(i,j) for i in range(real) for j in range(i) if keys[i]==keys[j]]
                assert case['duplicate_real_rows_exact'] and all(state[key][i]==state[key][j] for state in (a,b) for key in ('state_row_sha256','last_hidden_row_sha256','slots') for i,j in duplicates)
                if real==4:
                    control=golden_cases[(0,r)];assert case['same_DLL_default_off_full_control_exact'] is True
                    for mode in ('ordinary','mtp'):
                        assert strip(case[mode+'_state'])==control[mode+'_state']
                        assert arrays[mode]==previous.read('results/'+full+f'/native/group-0-repeat-{r}-{mode}-last-hidden.f32')
                    full_controls+=1
                else:assert case['same_DLL_default_off_full_control_exact'] is None
                metrics=case['speculative_metrics'];assert metrics['target_forward_calls']==metrics['target_verification_calls']+metrics['target_prefix_recomputes']
                assert metrics['target_verification_calls']==len(metrics['iterations']) and 0<=metrics['accepted_draft_tokens']<=metrics['proposed_draft_tokens']
                cases.append({'group':g,'repeat':r,'real_rows':real,'padded_rows':4-real,'real_rows_state_exact':True,'raw_real_hidden_exact':True,'full_physical_state_equal':a==b})
            assert seen=={(g,r) for g in range(6) for r in (0,1)} and raw_hidden==24 and reset_hidden==12 and full_controls==2
            resource=doc(base+'resources/resource-monitor.json');samples=[json.loads(l) for l in z.read(base+'resources/resource-vram.jsonl').splitlines()]
            assert resource['child_exit_code']==0 and resource['sampled_memory']['samples']==len(samples)
            peak=max(v['used_bytes'] for v in samples);assert resource['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
            with zipfile.ZipFile(out/(warm+'-evidence.zip')) as parent:
                assert resource['gpu']==json.loads(parent.read('results/'+warm+'/resources/resource-monitor.json'))['gpu']
            with zipfile.ZipFile(out/'padded-mtp-state-attempt1-evidence.zip') as prior:
                assert sha((out/'padded-mtp-state-attempt1-evidence.zip').read_bytes())==prep['retained_padding_archive_sha256']==w['retained_padding_archive_sha256']==evidence['linked_archives']['padded-mtp-state-attempt1-evidence.zip']
                oldbase='results/padded-mtp-state-attempt1/native/'
                original=json.loads(prior.read(oldbase+'report.json'));assert original['status']=='padded_mtp_state_passed_needs_replay' and len(original['cases'])==12
                controls={(c['group'],c['repeat']):c for c in original['cases']}
                for c in report['cases']:
                    control=controls[(c['group'],c['repeat'])]
                    for key in ('keys','real_rows','padded_rows','prompt_lengths','limits','generated_ids','expected_ids','ordinary_ids','ordinary_state','mtp_state','reset_state','speculative_metrics'):
                        assert c[key]==control[key],(c['group'],c['repeat'],key)
                raw_names=[n for n in prior.namelist() if n.startswith(oldbase) and n.endswith('.f32')]
                assert len(raw_names)==38
                for n in raw_names:assert prior.read(n)==z.read(base+'native/'+n[len(oldbase):])
            result={'status':'compact_register128_padded_state_replayed','archive_sha256':sha(archive.read_bytes()),'files_verified':len(evidence['files']),
                'library_sha256':evidence['library_sha256'],'cases':cases,'real_state_all_exact':True,'raw_ordinary_mtp_hidden_arrays':24,'reset_hidden_arrays':12,
                'full_four_default_off_controls':2,'sampled_peak_adapter_used_bytes':peak,'performance_admitted':False,
                'register_delta_128':True,'raw_hidden_register_off_controls':38,'candidate_worker_sha256':prep['files']['native_kv_delta_register128_padded_mtp_stream_worker.py'],'padding_policy_sha256':prep['files']['padded_mtp_policy.py'],
                'scope':prep['scope']+' Canonical state remains a producer hash witness. All physical hidden arrays retained; dummy differences are explicitly permitted outside real rows. Streaming callbacks/HTTP/timings remain unqualified.'}
    result['replay_verifier_sha256']=sha(Path(__file__).read_bytes());result['archived_replay_verifier_sha256']=sha(z.read(methods+'verify_register128_padded_state.py'))
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
