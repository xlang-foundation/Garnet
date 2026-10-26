"""Independently replay actual first full4K ordinary failure; no equivalence or speed admission."""
from pathlib import Path
import argparse,hashlib,json,math,struct,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.verify_fp8_z128_full4k_model import checked,JOB,PARENT,M as METHODS
from test2026.cuda.qwen38_5090.fp8_z128_full4k_model import validate_selection
from test2026.cuda.qwen38_5090.bf16_packed_state_gate import same_packed_bf16_row
from test2026.cuda.qwen38_5090.mtp_prefill16_full4k import replay_report
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def vector(entry,read):
    assert set(entry)=={'file','bytes','sha256'} and Path(entry['file']).name==entry['file'] and entry['file'].endswith('.f32')
    raw=read(entry['file']);assert type(entry['bytes']) is int and len(raw)==entry['bytes']==20480 and sha(raw)==entry['sha256']
    values=struct.unpack('<5120f',raw);assert all(math.isfinite(value) for value in values)
    return raw,values

def compare(actual,reference,read,read_reference):
    a=actual['logical'];b=reference['logical'];assert same_packed_bf16_row(a,0,a,0) and same_packed_bf16_row(b,0,b,0)
    assert len(a['slots'])==len(b['slots'])==1 and a['slots']==b['slots']
    different=[]
    for x,y in zip(a['components'],b['components']):
        if x['row_sha256']!=y['row_sha256']:different.append({key:x[key] for key in ('layer','kind','role','logical_words')})
    assert set(actual['vectors'])==set(reference['vectors'])=={'hidden','normalized'}
    stats={}
    for name in ('hidden','normalized'):
        raw,x=vector(actual['vectors'][name],read);old,y=vector(reference['vectors'][name],read_reference)
        delta=sum((float(v)-float(w))**2 for v,w in zip(x,y));norm=sum(float(w)**2 for w in y)
        stats[name]={'raw_bytes_exact':raw==old,'equal_scalars':sum(v==w for v,w in zip(x,y)),'scalars':5120,'max_absolute_difference':max(abs(float(v)-float(w)) for v,w in zip(x,y)),'relative_l2_difference':math.sqrt(delta/norm) if norm else (0.0 if delta==0 else None)}
    return {'all_complete_hashes_and_raw_vectors_exact':same_packed_bf16_row(a,0,b,0) and all(value['raw_bytes_exact'] for value in stats.values()),'different_components':different,'first_different_component':different[0] if different else None,'vectors':stats,'complete_component_count':128}

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);args=a.parse_args()
    archive=args.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive);old,old_manifest=checked(args.proofs/(PARENT+'-evidence.zip'))
    try:
        base='results/'+JOB+'/';p=json.loads(z.read(METHODS+'evidence/fp8-z128-full4k-model-preparation.json'));w=json.loads(z.read(base+'workflow.json'));r=json.loads(z.read(base+'native/report.json'))
        fail_raw=(args.proofs/(JOB+'-verified.json')).read_bytes();failure=json.loads(fail_raw)
        assert failure['status']=='fp8_z128_full4k_model_failure_preserved' and failure['archive_sha256']==sha(archive.read_bytes())
        assert w['exit_code']==1 and w['cleanup_finished'] and w['baseline_source_restored'] and w['baseline_library_restored']
        assert w['source_before_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment']
        assert r['status']=='failed' and r['library_sha256']==p['candidate_library_sha256'] and r['target128_enabled'] is True
        assert (r['target_prefill_chunk'],r['target_decode_chunk'],r['draft_prefill_chunk'])==(128,4,16) and r['target_program_chunks']==[4,128]
        assert failure['default_off_selection']=={'m128':{'4':0,'128':0},'z128':{'4':0,'128':0}} and failure['enabled_native_selection']==r['fp8_selection']==validate_selection(r)
        assert r['cases']==[] and len(r['ordinary_observations'])==1 and r['quality_qualified'] is r['performance_admitted'] is False
        ref=json.loads(old.read('results/'+PARENT+'/native/report.json'));cases=json.loads(old.read(METHODS+'evidence/native-host-stage-cases.json'))['cases']
        assert sha((args.proofs/(PARENT+'-evidence.zip')).read_bytes())==p['baseline_archive_sha256']
        read=lambda name:z.read(base+'native/'+name);read_reference=lambda name:old.read('results/'+PARENT+'/native/'+name)
        baseline=replay_report(ref,read_reference,cases);assert baseline['captured_states']==32
        actual=r['ordinary_observations'][0];expected=ref['cases'][0];control=cases[0]
        for key in ('sequence','case_index','key','request_sha256','image_sha256','prompt_ids'):assert actual[key]==expected[key]
        assert actual['case_index']==actual['sequence']==0 and control['case']=='text' and actual['prompt_ids']==control['prompt_ids'] and len(actual['prompt_ids'])==3968
        x=actual['ordinary_ids'];y=expected['ordinary_ids'];assert len(x)==len(y)==128 and all(type(token) is int and 0<=token<248320 for token in x+y)
        phases={'empty':compare(r['empty_state'],ref['empty_state'],read,read_reference)}
        for name in ('reset_state','ordinary_prefill','ordinary_final'):phases[name]=compare(actual[name],expected[name],read,read_reference)
        assert phases['empty']['all_complete_hashes_and_raw_vectors_exact'] and phases['reset_state']['all_complete_hashes_and_raw_vectors_exact']
        assert actual['ordinary_prefill']['logical']['slots']==[3968] and actual['ordinary_final']['logical']['slots']==[4095] and x!=y
        resource=json.loads(z.read(base+'native-resources/resource-monitor.json'));assert resource['child_exit_code']==1 and resource['gpu']['uuid']==p['gpu_uuid']
        cpu=z.read(base+'cpu-ledger/commands.log');assert b'Ran 70 tests' in cpu and b'\nOK' in cpu
        for name in ('bf16_packed_state_gate.py','mtp_prefill16_full4k.py','verify_fp8_z128_full4k_model.py','prefill128_fp8_model.py','fp8_z128_full4k_model.py'):
            assert sha(Path(__file__).with_name(name).read_bytes())==p['files'][METHODS+name]
        result={'status':'fp8_z128_first_full4k_failure_analyzed','default_off_selection':failure['default_off_selection'],'enabled_native_selection':failure['enabled_native_selection'],'archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'baseline_archive_sha256':p['baseline_archive_sha256'],'original_failure_proof_sha256':sha(fail_raw),'actual_prompt_tokens':3968,'actual_output_tokens':128,'first_different_output_zero_based':next(i for i,(v,w) in enumerate(zip(x,y)) if v!=w),'equal_output_ids':sum(v==w for v,w in zip(x,y)),'actual_ids':x,'reference_ids':y,'phases':phases,'complete_raw_vector_scalars_compared':40960,'mtp_generate_executed':False,'image_cases_executed':False,'remaining_cases_unrun':5,'source_runtime_environment_restored':True,'external_analyzer_sha256':sha(Path(__file__).read_bytes()),'cause_established':False,'harmlessness_or_quality_equivalence_proven':False,'performance_admitted':False,'two_times_target_achieved':False,'scope':'Actual first single-row text3968+128ordinary T128-target versus retainedT32. Validate complete128component geometry/slots, all40960finite raw hidden/normalized scalars and actual IDs. This is a failure boundary observation, not numerical/semantic harmlessness, cause or timing admission; raw caches are not serialized.'}
    finally:z.close();old.close()
    path=args.proofs/(JOB+'-analysis.json');raw=json.dumps(result,indent=2).encode();assert not path.exists() or path.read_bytes()==raw
    if not path.exists():path.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'first_different_output_zero_based':result['first_different_output_zero_based'],'first_prefill_state_difference':phases['ordinary_prefill']['first_different_component'],'prefill_hidden_relative_l2':phases['ordinary_prefill']['vectors']['hidden']['relative_l2_difference']}));return 0

if __name__=='__main__':raise SystemExit(main())
