"""Independent source91 HTTP callback, dispatch, adapted805 and restoration replay."""
from pathlib import Path
import argparse,gzip,hashlib,io,json,sys,tempfile,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.b4_full4k_inputs import load_corpus as load_fixed
from test2026.cuda.qwen38_5090.b4_padding_lifecycle_inputs import select_lifecycle
from test2026.cuda.qwen38_5090.b4_http_event_gate import events,match_native,dispatches
from test2026.cuda.qwen38_5090.mmmu_http_quality import load_corpus
from test2026.cuda.qwen38_5090.verify_mmmu_http_quality import replay
from test2026.cuda.qwen38_5090.probe_o128_http_quality import REFERENCE,REFERENCE_SHA
from test2026.cuda.qwen38_5090.b4_rowwise_http_admission import LIBRARY,sha_file
JOB='b4-rowwise-http-quality-attempt2';P='Garnet/test2026/cuda/qwen38_5090/';B='results/'+JOB+'/';sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--proofs',type=Path,required=True);ap.add_argument('--corpus',type=Path,required=True);ap.add_argument('--scratch',type=Path,required=True);a=ap.parse_args();archive=a.proofs/(JOB+'-evidence.zip')
    with zipfile.ZipFile(archive) as z:
        mf=json.loads(z.read('evidence-manifest.json'))['files'];assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(mf)+1
        for n,h in mf.items():assert sha(z.read(n))==h
        doc=lambda n:json.loads(z.read(n));raw=z.read(P+'evidence/b4-rowwise-http-quality-preparation.json');p=json.loads(raw);w=doc(B+'workflow.json')
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
        for stage in ('cpu','embedded-cpu','candidate'):
            name=B+stage+'-ledger/commands.jsonl'
            if name not in mf:continue
            rows=[json.loads(s) for s in z.read(name).splitlines()];assert len(rows)==2 and rows[0]['event']=='started' and rows[1]['event']=='finished'
            expected=p['files'] if stage=='cpu' else p['installed_files'];assert rows[0]['source']['owned_source_sha256']==rows[1]['source_after']['owned_source_sha256']==expected
            stages[stage]=rows[1]['exit_code']
        for stage in ('cpu','embedded-cpu'):
            if stages.get(stage)==0:assert b'Ran 17 tests' in z.read(B+stage+'-ledger/commands.log') and b'\nOK' in z.read(B+stage+'-ledger/commands.log')
        summary=None
        if w['exit_code']==0:
            assert stages=={'cpu':0,'embedded-cpu':0,'candidate':0} and w['status']=='B4_rowwise_HTTP_quality_collected_needs_independent_replay'
            assert w['installed_source']==p['installed_files'] and w['candidate_library_sha256']==LIBRARY==p['library_sha256'] and doc(B+'resources/resource-monitor.json')['child_exit_code']==0
            r=doc(B+'native/report.json');assert r['status']=='B4_rowwise_HTTP_controls_and_adapted805_collected_needs_independent_replay' and r['token_mismatches']==[] and not r['HTTP_quality_or_performance_qualified']
            ready=r['ready'];assert (ready['batch'],ready['capacity'],ready['prefill_chunk'],ready['chunk'])==(4,4096,32,4) and ready['generation_mode']=='verified_mtp' and ready['library_sha256']==LIBRARY
            ad=ready['source_admission'];assert ad['model_parents']==p['model_parents'] and ad['source_bundle_sha256']==p['bundle_sha256'] and ad['full_model_file_manifest_sha256']==p['model_file_manifest_sha256'] and ready['frame_max_bytes']==16*1024*1024
            outputs=events(z.read(B+'native/worker/worker-stdout.jsonl'),ready);used=set();bodies={}
            def match(key,body,prompt,ids,finish):
                assert key not in used and key in outputs;match_native(outputs[key],body,prompt,ids,finish);used.add(key);bodies[key]=body
            fixed=z.read(P+'evidence/fixed-length-4096-corpus.zip');assert sha(fixed)==p['input_corpus_sha256'];selection=load_fixed(fixed);keys=list(selection['inputs']);groups=[g['real_keys'] for g in select_lifecycle(selection['inputs'])['groups']]+[keys,keys]
            with zipfile.ZipFile(a.proofs/'b4-full4k-exact-attempt1-evidence.zip') as parent:oracle=json.loads(parent.read('results/b4-full4k-exact-attempt1/reference/report.json'))
            assert oracle['inputs']==selection['inputs'];gold={r['keys'][0]:r['generated_ids'][0] for r in oracle['rows']}
            assert len(r['boundary_controls'])==7
            for index,group in enumerate(groups):
                saved=r['boundary_controls'][index];assert saved['index']==index and saved['requested_concurrency']==len(group) and saved['large_transport_fixture_only'] is (index==6) and len(saved['results'])==len(group)
                for row,key in enumerate(group):
                    label='boundary-'+str(index)+'-'+str(row);body=doc(B+'native/'+label+'-request.json');expected={**selection['inputs'][key]['body'],'stream':False}
                    if index==6:expected['study_transport_padding']='x'*600000
                    assert body==expected;response=doc(B+'native/'+label+'-response.json');c=response['choices'][0]
                    assert c['prompt_token_ids']==selection['inputs'][key]['prompt_ids'] and c['response_token_ids']==gold[key] and response['usage']==dict(prompt_tokens=3968,completion_tokens=128,total_tokens=4096)
                    match(response['id'],body,c['prompt_token_ids'],c['response_token_ids'],c['finish_reason'])
            assert len(r['stream_controls'])==2
            for index,key in enumerate((keys[0],keys[4])):
                label='stream-'+str(index);body=doc(B+'native/'+label+'-request.json');assert body=={**selection['inputs'][key]['body'],'stream':True}
                wire=[json.loads(n)['data'] for n in z.read(B+'native/'+label+'-events.jsonl').splitlines()];assert wire[-1]=='[DONE]'
                values=[json.loads(n) for n in wire[:-1]];final=[n for n in values if 'response_token_ids' in n];assert len(final)==1;final=final[0]
                assert [n['token_id'] for n in values if 'token_id' in n]==gold[key] and final['response_token_ids']==gold[key] and final['prompt_token_ids']==selection['inputs'][key]['prompt_ids']
                assert doc(B+'native/'+label+'-result.json')['status']=='completed';match(final['id'],body,final['prompt_token_ids'],final['response_token_ids'],final['choices'][0]['finish_reason'])
            assert r['negative_controls']==[dict(ignore_eos_nonboolean_rejected_HTTP400=True,worker_remained_healthy=True)] and doc(B+'native/invalid-ignore-eos-request.json')['ignore_eos']==1
            assert json.loads(z.read(B+'native/invalid-ignore-eos-response.bin'))['error']
            reference=a.proofs/(REFERENCE+'-evidence.zip');assert sha_file(reference)==REFERENCE_SHA;_,corpus_rows,_=load_corpus(a.corpus)
            with tempfile.TemporaryDirectory(prefix='b4-http-quality-replay-',dir=a.scratch) as temporary,zipfile.ZipFile(reference) as old:
                folder=Path(temporary);qualities={}
                for key,container,prefix in [('candidate',z,B+'native/mmmu-quality/'),('reference',old,'results/'+REFERENCE+'/ordinary/mmmu-quality/')]:
                    dst=folder/key;dst.mkdir()
                    for n in container.namelist():
                        if n.startswith(prefix):
                            target=dst/n.removeprefix(prefix);assert target.resolve().is_relative_to(dst.resolve());target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(container.read(n))
                    qualities[key]=replay(a.corpus,dst);assert qualities[key]['status']=='adapted_mmmu_http_quality_responses_replayed'
                assert qualities['candidate']==r['quality'] and qualities['reference']==r['reference_quality']
                for row in corpus_rows:
                    path='mmmu-quality/cases/'+row['id']+'/response.bin';response=doc(B+'native/'+path);c=response['choices'][0];control=json.loads(old.read('results/'+REFERENCE+'/ordinary/'+path))['choices'][0]
                    assert (c['prompt_token_ids'],c['response_token_ids'])==(control['prompt_token_ids'],control['response_token_ids'])
                    body=json.loads(gzip.decompress(z.read(B+'native/mmmu-quality/cases/'+row['id']+'/wire-request.json.gz')));match(response['id'],body,c['prompt_token_ids'],c['response_token_ids'],c['finish_reason'])
            assert len(used)==len(outputs)==836
            d=dispatches(z.read(B+'native/worker/batch-dispatch.jsonl'),bodies,outputs);assert d['joined_frames_over_default_2MiB']>=1 and d['max_joined_frame_bytes']<=16*1024*1024 and 4 in d['real_rows']
            summary=dict(actual_native_HTTP_requests=836,adapted_requests=805,all805_prompt_and_output_IDs_exact_T32=True,full4k_nonstream_controls=29,text_image_SSE_controls=2,all_real_requests_matched_no_public_dummy_IDs=True,dispatch=d,quality=qualities['candidate'],performance_qualified=False)
        else:assert w['status']=='failed' and isinstance(w.get('error'),str)
        result=dict(status='actual_B4_HTTP_805_IDs_dispatch_and_full4k_controls_replayed' if w['exit_code']==0 else 'actual_B4_HTTP_quality_failure_preserved',exit_code=w['exit_code'],error=w.get('error'),archive_sha256=sha_file(archive),archive_bytes=archive.stat().st_size,files_verified=len(mf),stages=stages,arrays=summary,source_runtime_SDK_config_index_CMake_all34_and_full_model_files_restored=True,B4_HTTP_quality_case_gate_passed=w['exit_code']==0,official900_MMMU_qualified=False,production_or_performance_qualified=False,two_times_target_achieved=False,reader_sha256=sha(Path(__file__).read_bytes()),scope=p['scope'])
    target=a.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw;target.write_bytes(raw);print(json.dumps(dict(status=result['status'],proof_sha256=sha(raw),exit_code=w['exit_code'])))
if __name__=='__main__':main()
