"""Independent source91 three-trial raw HTTP timing, IDs, dispatch and restoration replay."""
from pathlib import Path
import argparse,gzip,hashlib,io,json,sys,tempfile,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.b4_http_event_gate import events,match_native,dispatches
from test2026.cuda.qwen38_5090.o128_http_benchmark_contract import replay_trial,REFERENCE,REFERENCE_SHA
from test2026.cuda.qwen38_5090.b4_rowwise_http_admission import LIBRARY,sha_file
JOB='b4-resident-serial-http-benchmark-attempt1';P='Garnet/test2026/cuda/qwen38_5090/';B='results/'+JOB+'/';sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--proofs',type=Path,required=True);a=ap.parse_args();archive=a.proofs/(JOB+'-evidence.zip')
    with zipfile.ZipFile(archive) as z:
        mf=json.loads(z.read('evidence-manifest.json'))['files'];assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(mf)+1
        for n,h in mf.items():assert sha(z.read(n))==h
        doc=lambda n:json.loads(z.read(n));raw=z.read(P+'evidence/b4-resident-serial-http-benchmark-preparation.json');p=json.loads(raw);w=doc(B+'workflow.json')
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
            proof=z.read('results/'+job+'-verified.json');q=json.loads(proof)
            assert sha(proof)==row['proof_sha256'] and proof==(a.proofs/(job+'-verified.json')).read_bytes() and q['status']==row['status'] and sha_file(a.proofs/(job+'-evidence.zip'))==q['archive_sha256']==row['archive_sha256']
        stages={}
        name=B+'candidate-ledger/commands.jsonl'
        if name in mf:
            records=[json.loads(n) for n in z.read(name).splitlines()];assert len(records)==2 and records[0]['event']=='started' and records[1]['event']=='finished'
            assert records[0]['source']['owned_source_sha256']==records[1]['source_after']['owned_source_sha256']==p['installed_files'];stages['candidate']=records[1]['exit_code']
        summary=None
        if w['exit_code']==0:
            assert stages=={'candidate':0} and w['status']=='B4_resident_serial_HTTP_benchmark_collected_needs_independent_replay'
            assert w['installed_source']==p['installed_files'] and w['candidate_library_sha256']==LIBRARY==p['library_sha256'] and doc(B+'resources/resource-monitor.json')['child_exit_code']==0
            r=doc(B+'native/report.json');assert r['status']=='B4_resident_serial_three_fixed4k_HTTP_trials_collected_needs_independent_replay' and len(r['trials'])==3 and r['physical_batch']==4 and r['grouping_delay_ms']==10 and r['grouping_clock']=='perf_counter' and not r['two_times_target_achieved']
            quality=p['quality_job'];parent=doc('results/'+quality+'-verified.json');assert parent['exit_code']==0 and parent['B4_HTTP_quality_case_gate_passed'] and parent['arrays']['all805_prompt_and_output_IDs_exact_T32'] and parent['arrays']['actual_native_HTTP_requests']==836 and r['quality_proof_sha256']==p['parents'][quality]['proof_sha256']
            assert parent['arrays']['resident_IO_controls_byte_exact'] and parent['arrays']['actual_graph_policy']['status']=='resident_serial_requested_policy_and_actual_graph_reuse_replayed'
            ready=r['ready'];assert (ready['batch'],ready['capacity'],ready['prefill_chunk'],ready['chunk'])==(4,4096,32,4) and ready['generation_mode']=='verified_mtp' and ready['library_sha256']==LIBRARY
            assert ready['backend_policy']==dict(resident_controls=True,serial_target_graph_reuse=True,serial_draft_graph_reuse=True,accepted_head_reuse=False)
            ad=ready['source_admission'];assert ad['model_parents']==p['model_parents'] and ad['source_bundle_sha256']==p['bundle_sha256'] and ad['full_model_file_manifest_sha256']==p['model_file_manifest_sha256'] and ready['frame_max_bytes']==16*1024*1024
            outputs=events(z.read(B+'native/worker/worker-stdout.jsonl'),ready);used=set();bodies={};trials=[]
            reference=a.proofs/(REFERENCE+'-evidence.zip');assert sha_file(reference)==REFERENCE_SHA
            corpus_raw=z.read(P+'evidence/fixed-length-4096-corpus.zip');assert sha(corpus_raw)==p['input_corpus_sha256']
            with zipfile.ZipFile(io.BytesIO(corpus_raw)) as corpus,zipfile.ZipFile(reference) as old:
                corpus_manifest=json.loads(corpus.read('manifest.json'));oldbase='results/'+REFERENCE+'/mtp/benchmark4k/';oldmanifests=[n for n in old.namelist() if n.startswith(oldbase) and n.endswith('-manifest.json')];assert len(oldmanifests)==1;oldmanifest=json.loads(old.read(oldmanifests[0]))
                for index in range(3):
                    prefix=B+'native/trial-'+str(index)+'/';manifests=[n for n in mf if n.startswith(prefix) and n.endswith('-manifest.json')];assert len(manifests)==1
                    manifest=doc(manifests[0]);read=lambda n:z.read(prefix+n);fresh=replay_trial(manifest,read,corpus,oldmanifest,lambda n:old.read(oldbase+n));saved=r['trials'][index]
                    assert saved['index']==index and saved['manifest_sha256']==sha(z.read(manifests[0])) and saved['local_replay']==fresh
                    metadata=doc(B+'native/trial-'+str(index)+'-metadata.json');assert metadata==manifest['engine_metadata'] and metadata['ready']==ready and metadata['trial_index']==index and metadata['repeat_count']==3 and metadata['physical_batch']==4 and metadata['grouping_delay_ms']==10 and metadata['quality_proof_sha256']==r['quality_proof_sha256']
                    for row in corpus_manifest['rows']:
                        key=manifest['run_id']+'-'+row['key'];body=json.loads(read(key+'-request.json'));assert 'study_transport_padding' not in body and 'study_graph_snapshot' not in body
                        wire=[json.loads(json.loads(n)['data']) for n in read(key+'-events.jsonl').splitlines() if json.loads(n)['data']!='[DONE]'];final=[n for n in wire if 'response_token_ids' in n];assert len(final)==1;final=final[0];native_key=final['id'];assert native_key not in used and native_key in outputs
                        match_native(outputs[native_key],body,final['prompt_token_ids'],final['response_token_ids'],final['choices'][0]['finish_reason']);used.add(native_key);bodies[native_key]=body
                    trials.append(fresh)
            assert len(used)==len(outputs)==372
            dispatch=dispatches(z.read(B+'native/worker/batch-dispatch.jsonl'),bodies,outputs);assert dispatch==r['dispatch'] and 4 in dispatch['real_rows'] and dispatch['max_joined_frame_bytes']<=16*1024*1024
            summary=dict(trials=trials,actual_native_HTTP_requests=372,measured_requests=192,warmup_requests=180,all372_prompt_and_output_IDs_exact_retained_B1=True,all_real_native_callbacks_and_dispatches_matched=True,public_output_IDs=47616,measured_output_tokens=24576,private_dummy_tokens_counted_in_throughput=False,dispatch=dispatch)
        else:assert w['status']=='failed' and isinstance(w.get('error'),str)
        result=dict(status='actual_B4_resident_serial_three_fixed4k_HTTP_trials_replayed' if w['exit_code']==0 else 'actual_B4_resident_serial_HTTP_benchmark_failure_preserved',exit_code=w['exit_code'],error=w.get('error'),archive_sha256=sha_file(archive),archive_bytes=archive.stat().st_size,files_verified=len(mf),stages=stages,benchmark_arrays=summary,source_runtime_SDK_config_index_CMake_all34_and_full_model_files_restored=True,B4_HTTP_benchmark_case_gate_passed=w['exit_code']==0,strongest_tuned_references_qualified=False,cross_engine_output_equivalence_qualified=False,two_times_target_achieved=False,reader_sha256=sha(Path(__file__).read_bytes()),scope=p['scope'])
    target=a.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw;target.write_bytes(raw);print(json.dumps(dict(status=result['status'],proof_sha256=sha(raw),exit_code=w['exit_code'])))
if __name__=='__main__':main()
