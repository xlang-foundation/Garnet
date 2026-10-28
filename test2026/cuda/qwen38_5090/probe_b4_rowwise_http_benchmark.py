"""Prospective clean B4 HTTP trials; requires an independently passed quality proof."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,threading,time,zipfile
from native_batch_http_bridge import BatchWorker,BatchHandler,ThreadingHTTPServer
sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();root=a.root;out=a.output;assert not out.exists();out.mkdir();sys.path.insert(0,str(root/'Garnet'));m=root/'Garnet/test2026/cuda/qwen38_5090'
    from test2026.cuda.qwen38_5090.b4_rowwise_http_admission import LIBRARY,sha_file
    from test2026.cuda.qwen38_5090.o128_http_benchmark_contract import replay_trial,REFERENCE,REFERENCE_SHA
    from test2026.cuda.qwen38_5090.b4_http_event_gate import events,match_native,dispatches
    prep=json.loads((m/'evidence/b4-rowwise-http-benchmark-preparation.json').read_bytes());quality=prep['quality_job'];proof=root/'results'/(quality+'-verified.json');q=json.loads(proof.read_bytes())
    assert sha_file(proof)==prep['parents'][quality]['proof_sha256'] and q['exit_code']==0 and q['B4_HTTP_quality_case_gate_passed'] and q['arrays']['all805_prompt_and_output_IDs_exact_T32'] and q['arrays']['actual_native_HTTP_requests']==836
    assert sha_file(root/'results'/(quality+'-evidence.zip'))==q['archive_sha256']==prep['parents'][quality]['archive_sha256'] and q['source_runtime_SDK_config_index_CMake_all34_and_full_model_files_restored']
    cache=json.loads((root/'results/study-ssd-engine-cache/receipt.json').read_bytes())['cache_root'];worker=server=thread=None
    report=dict(status='starting',library_sha256=LIBRARY,quality_proof_sha256=sha_file(proof),reference_archive_sha256=REFERENCE_SHA,trials=[],physical_batch=4,private_padding=True,queued_concurrency=True,grouping_delay_ms=10,grouping_clock='perf_counter',strongest_tuned_references_qualified=False,two_times_target_achieved=False)
    def save():(out/'report.json').write_bytes(json.dumps(report,indent=2).encode())
    save()
    try:
        worker=BatchWorker([sys.executable,str(m/'launch_b4_rowwise_http_worker.py'),str(root),cache],out/'worker',4,10)
        deadline=time.monotonic()+1800
        while not worker.healthy():
            assert worker.process.poll() is None and worker.failure is None,worker.failure;assert time.monotonic()<deadline,'Worker startup deadline exceeded';time.sleep(.25)
        ready=worker.ready_info;assert ready['library_sha256']==LIBRARY and (ready['batch'],ready['capacity'],ready['prefill_chunk'],ready['chunk'])==(4,4096,32,4) and ready['generation_mode']=='verified_mtp'
        assert ready['source_admission']['source_bundle_sha256']==prep['bundle_sha256'] and ready['source_admission']['full_model_file_manifest_sha256']==prep['model_file_manifest_sha256'];report['ready']=ready;save()
        server=ThreadingHTTPServer(('127.0.0.1',18082),BatchHandler);server.worker=worker;server.model='qwen38-27b-nvfp4';thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        reference=root/'results'/(REFERENCE+'-evidence.zip');assert sha_file(reference)==REFERENCE_SHA
        with zipfile.ZipFile(reference) as old,zipfile.ZipFile(m/'evidence/fixed-length-4096-corpus.zip') as corpus:
            oldbase='results/'+REFERENCE+'/mtp/benchmark4k/';manifests=[n for n in old.namelist() if n.startswith(oldbase) and n.endswith('-manifest.json')];assert len(manifests)==1;oldmanifest=json.loads(old.read(manifests[0]));corpus_manifest=json.loads(corpus.read('manifest.json'));bodies={};responses={}
            for trial in range(3):
                report['status']='benchmark_trial_'+str(trial);save();folder=out/('trial-'+str(trial));metadata=out/('trial-'+str(trial)+'-metadata.json')
                metadata.write_bytes(json.dumps(dict(ready=ready,generation_mode='verified_mtp',library_sha256=LIBRARY,quality_proof_sha256=sha_file(proof),trial_index=trial,repeat_count=3,physical_batch=4,private_padding=True,grouping_delay_ms=10,queued_concurrency=True,performance_admitted=False,scope='Unprofiled actual frozen3968+128 text/image HTTP C1/2/4/8.64 measured+60 warmups/trial. Real output tokens only; private dummy tokens excluded. Three repeats on one worker; queue and grouping time included. NativeWindows vs saved referenceWSL. Ordinary references provisional; stronger tuned/MTP baselines and cross-engine output equivalence not established.')).encode())
                subprocess.run([sys.executable,str(m/'benchmark_fixed_length_native.py'),'--url','http://127.0.0.1:18082','--corpus-archive',str(m/'evidence/fixed-length-4096-corpus.zip'),'--output',str(folder),'--metadata',str(metadata)],check=True)
                files=list(folder.glob('*-manifest.json'));assert len(files)==1;manifest=json.loads(files[0].read_bytes());result=replay_trial(manifest,lambda n:(folder/n).read_bytes(),corpus,oldmanifest,lambda n:old.read(oldbase+n))
                for row in corpus_manifest['rows']:
                    key=manifest['run_id']+'-'+row['key'];body=json.loads((folder/(key+'-request.json')).read_bytes());wire=[json.loads(json.loads(n)['data']) for n in (folder/(key+'-events.jsonl')).read_bytes().splitlines() if json.loads(n)['data']!='[DONE]'];final=[n for n in wire if 'response_token_ids' in n];assert len(final)==1;final=final[0];assert final['id'] not in bodies
                    bodies[final['id']]=body;responses[final['id']]=final
                report['trials'].append(dict(index=trial,manifest_sha256=sha_file(files[0]),local_replay=result));save()
            native=events((out/'worker/worker-stdout.jsonl').read_bytes(),ready);assert len(native)==len(bodies)==len(responses)==372
            for key,body in bodies.items():
                final=responses[key];match_native(native[key],body,final['prompt_token_ids'],final['response_token_ids'],final['choices'][0]['finish_reason'])
            report['dispatch']=dispatches((out/'worker/batch-dispatch.jsonl').read_bytes(),bodies,native);assert 4 in report['dispatch']['real_rows'];save()
        report['status']='B4_three_fixed4k_HTTP_trials_collected_needs_independent_replay';save()
    except BaseException as error:report.update(status='failed',error=repr(error));save();raise
    finally:
        if server is not None:server.shutdown();server.server_close()
        if thread is not None:thread.join()
        if worker is not None:worker.close()
    return 0
if __name__=='__main__':raise SystemExit(main())
