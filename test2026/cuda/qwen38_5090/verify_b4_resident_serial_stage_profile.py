"""Independent raw fixed4k output/dispatch and physical-group host-span replay."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.b4_full4k_inputs import load_corpus
from test2026.cuda.qwen38_5090.b4_padding_lifecycle_inputs import select_lifecycle
from test2026.cuda.qwen38_5090.b4_http_event_gate import events,match_native,dispatches
from test2026.cuda.qwen38_5090.b4_stage_profile_gate import admit
from test2026.cuda.qwen38_5090.b4_rowwise_http_admission import LIBRARY,sha_file
JOB='b4-resident-serial-stage-profile-attempt1';P='Garnet/test2026/cuda/qwen38_5090/';B='results/'+JOB+'/';sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--proofs',type=Path,required=True);a=ap.parse_args();archive=a.proofs/(JOB+'-evidence.zip')
    with zipfile.ZipFile(archive) as z:
        mf=json.loads(z.read('evidence-manifest.json'))['files'];assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(mf)+1
        for n,h in mf.items():assert sha(z.read(n))==h
        doc=lambda n:json.loads(z.read(n));raw=z.read(P+'evidence/b4-resident-serial-stage-profile-preparation.json');p=json.loads(raw);w=doc(B+'workflow.json')
        assert sha(raw)==w['preparation_sha256'] and w['cleanup_finished'] and w['all_restored'] and w['baseline_source_restored'] and w['baseline_library_restored'] and not w['performance_admitted']
        for first,last,key in [('source_before','source_after','files'),('runtime_before','runtime_after','runtime'),('environment_before','environment_after','environment'),('checkpoint_before','checkpoint_after','checkpoints'),('cuda_objects_before','cuda_objects_after','cuda_objects'),('model_files_before','model_files_after','model_files')]:assert w[first]==w[last]==p[key]
        assert w['cache_before_sha256']==w['cache_after_sha256']==p['cache_before_sha256'] and len(p['cuda_objects'])==34 and len(p['model_files'])==18
        for n in p['archive_files']:assert sha(z.read(n))==p['files'][n]
        for n in p['reader_dependencies']:assert sha(Path(__file__).with_name(n).read_bytes())==p['files'][P+n]
        assert sha(z.read(p['model_file_manifest']))==p['model_file_manifest_sha256'] and doc(p['model_file_manifest'])['files']==p['model_files']
        assert sha(z.read(p['bundle']))==p['bundle_sha256'] and len(p['model_source_files'])==89 and all(p['source_files'][n]==h for n,h in p['model_source_files'].items())
        with zipfile.ZipFile(io.BytesIO(z.read(p['bundle']))) as source:assert len(source.namelist())==len(set(source.namelist()))==91 and {n:sha(source.read(n)) for n in source.namelist()}==p['source_files']
        with zipfile.ZipFile(io.BytesIO(z.read(B+'original-source.zip'))) as old:assert {n:sha(old.read(n)) for n in old.namelist()}=={n:h for n,h in p['candidate_base'].items() if h is not None}
        for job,row in p['parents'].items():
            raw_parent=z.read('results/'+job+'-verified.json');q=json.loads(raw_parent)
            assert sha(raw_parent)==row['proof_sha256'] and raw_parent==(a.proofs/(job+'-verified.json')).read_bytes() and q['status']==row['status'] and sha_file(a.proofs/(job+'-evidence.zip'))==q['archive_sha256']==row['archive_sha256']
        stages={}
        for stage in ('cpu','embedded-cpu','candidate'):
            name=B+stage+'-ledger/commands.jsonl'
            if name not in mf:continue
            rows=[json.loads(s) for s in z.read(name).splitlines()];assert len(rows)==2 and rows[0]['event']=='started' and rows[1]['event']=='finished'
            expected=p['files'] if stage=='cpu' else p['installed_files'];assert rows[0]['source']['owned_source_sha256']==rows[1]['source_after']['owned_source_sha256']==expected;stages[stage]=rows[1]['exit_code']
        for stage,count in (('cpu',6),('embedded-cpu',18)):
            if stages.get(stage)==0:assert ('Ran '+str(count)+' tests').encode() in z.read(B+stage+'-ledger/commands.log') and b'\nOK' in z.read(B+stage+'-ledger/commands.log')
        summary=None
        if w['exit_code']==0:
            assert stages=={'cpu':0,'embedded-cpu':0,'candidate':0} and w['status']=='B4_resident_serial_stage_profile_collected_needs_independent_replay' and w['candidate_library_sha256']==LIBRARY==p['library_sha256'] and w['installed_source']==p['installed_files'] and doc(B+'resources/resource-monitor.json')['child_exit_code']==0
            quality=doc('results/'+p['quality_job']+'-verified.json');benchmark=doc('results/'+p['benchmark_job']+'-verified.json')
            assert quality['exit_code']==0 and quality['B4_HTTP_quality_case_gate_passed'] and quality['arrays']['resident_IO_controls_byte_exact'] and quality['arrays']['actual_graph_policy']['real_control_requests']==31
            assert benchmark['exit_code']==0 and benchmark['B4_HTTP_benchmark_case_gate_passed'] and benchmark['benchmark_arrays']['all372_prompt_and_output_IDs_exact_retained_B1']
            r=doc(B+'native/report.json');assert r['status']=='B4_resident_serial_host_stage_controls_collected_needs_independent_replay' and r['real_request_count']==62 and r['CUDA_kernel_elapsed_claimed'] is r['clean_throughput_result_claimed'] is r['two_times_target_achieved'] is False
            ready=r['ready'];assert (ready['batch'],ready['capacity'],ready['prefill_chunk'],ready['chunk'])==(4,4096,32,4) and ready['generation_mode']=='verified_mtp' and ready['library_sha256']==LIBRARY and ready['stage_profile'] is True and ready['profile_clock_domain']=='native_worker'
            assert ready['CUDA_kernel_elapsed_claimed'] is ready['clean_throughput_result_claimed'] is False and ready['backend_policy']==dict(resident_controls=True,serial_target_graph_reuse=True,serial_draft_graph_reuse=True,accepted_head_reuse=False)
            ad=ready['source_admission'];assert ad['model_parents']==p['model_parents'] and ad['source_bundle_sha256']==p['bundle_sha256'] and ad['full_model_file_manifest_sha256']==p['model_file_manifest_sha256']
            fixed=z.read(P+'evidence/fixed-length-4096-corpus.zip');assert sha(fixed)==p['input_corpus_sha256'];selection=load_corpus(fixed);keys=list(selection['inputs']);groups=[g['real_keys'] for g in select_lifecycle(selection['inputs'])['groups']]+[keys,keys]
            with zipfile.ZipFile(a.proofs/'b4-full4k-exact-attempt1-evidence.zip') as old:gold_report=json.loads(old.read('results/b4-full4k-exact-attempt1/reference/report.json'))
            assert gold_report['inputs']==selection['inputs'];gold={row['keys'][0]:row['generated_ids'][0] for row in gold_report['rows']}
            native_raw=z.read(B+'native/worker/worker-stdout.jsonl');native=events(native_raw,ready);used=set();bodies={};profiled=set()
            def match(key,body,prompt,ids,finish):
                assert key not in used and key in native and finish=='length';match_native(native[key],body,prompt,ids,finish);used.add(key);bodies[key]=body
                if body.get('study_profile') is True:profiled.add(key)
            assert set(r['controls'])=={'unprofiled','profiled'}
            for phase in ('unprofiled','profiled'):
                saved=r['controls'][phase];assert len(saved['boundary'])==7 and len(saved['streams'])==2
                for index,group in enumerate(groups):
                    current=saved['boundary'][index];assert current['index']==index and current['requested_concurrency']==len(group) and len(current['results'])==len(group)
                    for row,key in enumerate(group):
                        label=phase+'-boundary-'+str(index)+'-'+str(row);body=doc(B+'native/'+label+'-request.json');expected={**selection['inputs'][key]['body'],'stream':False}
                        if phase=='profiled':expected['study_profile']=True
                        assert body==expected;response=doc(B+'native/'+label+'-response.json');c=response['choices'][0]
                        assert c['prompt_token_ids']==selection['inputs'][key]['prompt_ids'] and c['response_token_ids']==gold[key] and response['usage']==dict(prompt_tokens=3968,completion_tokens=128,total_tokens=4096)
                        assert current['results'][row]==dict(source_key=key,id=response['id'],body=body,prompt=c['prompt_token_ids'],ids=c['response_token_ids'],finish=c['finish_reason']);match(response['id'],body,c['prompt_token_ids'],c['response_token_ids'],c['finish_reason'])
                for index,key in enumerate((keys[0],keys[4])):
                    label=phase+'-stream-'+str(index);body=doc(B+'native/'+label+'-request.json');expected={**selection['inputs'][key]['body'],'stream':True}
                    if phase=='profiled':expected['study_profile']=True
                    assert body==expected;wire=[json.loads(s)['data'] for s in z.read(B+'native/'+label+'-events.jsonl').splitlines()];assert wire[-1]=='[DONE]';values=[json.loads(s) for s in wire[:-1]];final=[v for v in values if 'response_token_ids' in v];assert len(final)==1;final=final[0]
                    assert [v['token_id'] for v in values if 'token_id' in v]==gold[key] and final['response_token_ids']==gold[key] and final['prompt_token_ids']==selection['inputs'][key]['prompt_ids'] and final['usage']==dict(prompt_tokens=3968,completion_tokens=128,total_tokens=4096)
                    assert saved['streams'][index]==dict(id=final['id'],source_key=key,label=label) and doc(B+'native/'+label+'-result.json')['status']=='completed';match(final['id'],body,final['prompt_token_ids'],final['response_token_ids'],final['choices'][0]['finish_reason'])
            assert len(used)==len(native)==62 and len(profiled)==31
            host=admit(native_raw,profiled);assert host==r['host_stages'];dispatch=dispatches(z.read(B+'native/worker/batch-dispatch.jsonl'),bodies,native);assert dispatch==r['dispatch'] and 4 in dispatch['real_rows']
            summary=dict(actual_native_HTTP_requests=62,profiled_real_request_IDs=31,unprofiled_real_request_IDs=31,public_output_IDs=7936,all62_prompt_and_output_IDs_exact_retained_B1=True,all_real_native_callbacks_and_dispatches_matched=True,host_stages=host,dispatch=dispatch,CUDA_kernel_elapsed_or_clean_serving_speed_qualified=False)
        else:assert w['status']=='failed' and isinstance(w.get('error'),str)
        result=dict(status='actual_B4_resident_serial_host_stage_controls_replayed' if w['exit_code']==0 else 'actual_B4_resident_serial_host_stage_failure_preserved',exit_code=w['exit_code'],error=w.get('error'),archive_sha256=sha_file(archive),archive_bytes=archive.stat().st_size,files_verified=len(mf),stages=stages,arrays=summary,source_runtime_SDK_config_index_CMake_all34_and_full_model_files_restored=True,B4_host_stage_case_gate_passed=w['exit_code']==0,CUDA_kernel_elapsed_or_clean_serving_speed_qualified=False,two_times_target_achieved=False,reader_sha256=sha(Path(__file__).read_bytes()),scope=p['scope'])
    target=a.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw;target.write_bytes(raw);print(json.dumps(dict(status=result['status'],proof_sha256=sha(raw),exit_code=w['exit_code'])))
if __name__=='__main__':main()
