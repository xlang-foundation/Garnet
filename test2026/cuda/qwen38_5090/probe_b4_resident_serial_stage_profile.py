"""Separate fixed4k host-call diagnosis; output parity before timing interpretation."""
from pathlib import Path
import argparse,concurrent.futures,copy,hashlib,json,sys,threading,time,zipfile
from urllib.request import Request,urlopen
from native_batch_http_bridge import BatchWorker,BatchHandler,ThreadingHTTPServer
sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();root=a.root;out=a.output
    assert root.resolve()==Path('D:/CantorAI/5090-study-20261023').resolve() and not out.exists(),'Remote-only diagnostic';out.mkdir();m=root/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(root/'Garnet'))
    from test2026.cuda.qwen38_5090.b4_full4k_inputs import load_corpus
    from test2026.cuda.qwen38_5090.b4_padding_lifecycle_inputs import select_lifecycle
    from test2026.cuda.qwen38_5090.benchmark_stream import request as stream_request
    from test2026.cuda.qwen38_5090.b4_http_event_gate import events,match_native,dispatches
    from test2026.cuda.qwen38_5090.b4_stage_profile_gate import admit
    from test2026.cuda.qwen38_5090.b4_rowwise_http_admission import sha_file
    p=json.loads((m/'evidence/b4-resident-serial-stage-profile-preparation.json').read_bytes())
    quality=p['quality_job'];qp=root/'results'/(quality+'-verified.json');q=json.loads(qp.read_bytes());assert sha_file(qp)==p['parents'][quality]['proof_sha256'] and q['exit_code']==0 and q['B4_HTTP_quality_case_gate_passed'] and q['arrays']['resident_IO_controls_byte_exact'] and q['arrays']['actual_graph_policy']['real_control_requests']==31
    benchmark=p['benchmark_job'];bp=root/'results'/(benchmark+'-verified.json');b=json.loads(bp.read_bytes());assert sha_file(bp)==p['parents'][benchmark]['proof_sha256'] and b['exit_code']==0 and b['B4_HTTP_benchmark_case_gate_passed'] and b['benchmark_arrays']['all372_prompt_and_output_IDs_exact_retained_B1']
    fixed=(m/'evidence/fixed-length-4096-corpus.zip').read_bytes();assert sha(fixed)==p['input_corpus_sha256'];selection=load_corpus(fixed);keys=list(selection['inputs']);groups=[g['real_keys'] for g in select_lifecycle(selection['inputs'])['groups']]+[keys,keys]
    oracle=root/'results/b4-full4k-exact-attempt1-evidence.zip';assert sha_file(oracle)==p['parents']['b4-full4k-exact-attempt1']['archive_sha256']
    with zipfile.ZipFile(oracle) as z:gold_report=json.loads(z.read('results/b4-full4k-exact-attempt1/reference/report.json'))
    assert gold_report['inputs']==selection['inputs'];gold={r['keys'][0]:r['generated_ids'][0] for r in gold_report['rows']}
    cache=json.loads((root/'results/study-ssd-engine-cache/receipt.json').read_bytes())['cache_root'];worker=server=thread=None
    report=dict(status='starting',controls={},expected_real_requests=62,CUDA_kernel_elapsed_claimed=False,clean_throughput_result_claimed=False,two_times_target_achieved=False,
                scope='Same31fixed4k controls twice: unprofiled and profiled in one instrumented worker. Exact prompt/output IDs, real native callbacks/dispatch and raw group-span replay required. Instrumentation changes host scheduling and includes logging; never clean throughput or isolated GPU/kernel cost. No large transport fixtures or805repeat.')
    def save():(out/'report.json').write_bytes(json.dumps(report,indent=2).encode())
    save();url='http://127.0.0.1:18082';bodies={};responses={};profiled_ids=set()
    try:
        worker=BatchWorker([sys.executable,str(m/'launch_b4_resident_serial_stage_profile_worker.py'),str(root),cache],out/'worker',4,10);deadline=time.monotonic()+1800
        while not worker.healthy():
            assert worker.process.poll() is None and worker.failure is None,worker.failure;assert time.monotonic()<deadline,'Diagnostic worker startup exceeded';time.sleep(.25)
        ready=worker.ready_info;assert ready['library_sha256']==p['library_sha256'] and ready['stage_profile'] is True and ready['profile_clock_domain']=='native_worker' and ready['CUDA_kernel_elapsed_claimed'] is ready['clean_throughput_result_claimed'] is False
        assert ready['backend_policy']==dict(resident_controls=True,serial_target_graph_reuse=True,serial_draft_graph_reuse=True,accepted_head_reuse=False)
        report['ready']=ready;save();server=ThreadingHTTPServer(('127.0.0.1',18082),BatchHandler);server.worker=worker;server.model='qwen38-27b-nvfp4';thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        def remember(key,body,prompt,ids,finish):
            assert key not in bodies;assert 'study_transport_padding' not in body and 'study_graph_snapshot' not in body;bodies[key]=body;responses[key]=dict(prompt=prompt,ids=ids,finish=finish)
            if body.get('study_profile') is True:profiled_ids.add(key)
        def check(label,key,body,barrier):
            wire=json.dumps(body,separators=(',',':')).encode();assert len(wire)<=2*1024*1024;(out/(label+'-request.json')).write_bytes(wire);barrier.wait(timeout=20)
            with urlopen(Request(url+'/v1/chat/completions',data=wire,headers={'Content-Type':'application/json'}),timeout=300) as response:raw=response.read()
            (out/(label+'-response.json')).write_bytes(raw);r=json.loads(raw);c=r['choices'][0]
            assert c['prompt_token_ids']==selection['inputs'][key]['prompt_ids'] and c['response_token_ids']==gold[key] and r['usage']==dict(prompt_tokens=3968,completion_tokens=128,total_tokens=4096)
            return dict(source_key=key,id=r['id'],body=body,prompt=c['prompt_token_ids'],ids=c['response_token_ids'],finish=c['finish_reason'])
        for phase in ('unprofiled','profiled'):
            report['status']=phase;report['controls'][phase]=dict(boundary=[],streams=[]);save()
            for index,group in enumerate(groups):
                barrier=threading.Barrier(len(group));values=[]
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(group)) as pool:
                    futures=[]
                    for row,key in enumerate(group):
                        body=copy.deepcopy(selection['inputs'][key]['body']);body['stream']=False
                        if phase=='profiled':body['study_profile']=True
                        futures.append(pool.submit(check,phase+'-boundary-'+str(index)+'-'+str(row),key,body,barrier))
                    values=[f.result() for f in futures]
                for row in values:remember(row['id'],row['body'],row['prompt'],row['ids'],row['finish'])
                report['controls'][phase]['boundary'].append(dict(index=index,requested_concurrency=len(group),results=values));save()
            for index,key in enumerate((keys[0],keys[4])):
                label=phase+'-stream-'+str(index);body=copy.deepcopy(selection['inputs'][key]['body']);body['stream']=True
                if phase=='profiled':body['study_profile']=True
                result=stream_request(url,body,out,label,128);assert result['status']=='completed' and result['generated_tokens']==128
                wire=[json.loads(n)['data'] for n in (out/(label+'-events.jsonl')).read_bytes().splitlines()];assert wire[-1]=='[DONE]';events_wire=[json.loads(n) for n in wire[:-1]];final=[e for e in events_wire if 'response_token_ids' in e];assert len(final)==1;final=final[0]
                assert [e['token_id'] for e in events_wire if 'token_id' in e]==gold[key] and final['response_token_ids']==gold[key] and final['prompt_token_ids']==selection['inputs'][key]['prompt_ids']
                remember(final['id'],body,final['prompt_token_ids'],final['response_token_ids'],final['choices'][0]['finish_reason']);report['controls'][phase]['streams'].append(dict(id=final['id'],source_key=key,label=label));save()
    except BaseException as error:report.update(status='failed',error=repr(error));save();raise
    finally:
        if server is not None:server.shutdown();server.server_close()
        if thread is not None:thread.join()
        if worker is not None:worker.close()
    # Close/flush the worker before parsing the last root-span end marker.
    try:
        raw=(out/'worker/worker-stdout.jsonl').read_bytes();native=events(raw,ready);assert len(native)==len(bodies)==62 and len(profiled_ids)==31
        for key,body in bodies.items():
            r=responses[key];match_native(native[key],body,r['prompt'],r['ids'],r['finish'])
        report['dispatch']=dispatches((out/'worker/batch-dispatch.jsonl').read_bytes(),bodies,native);report['host_stages']=admit(raw,profiled_ids)
        report['real_request_count']=len(bodies);report['status']='B4_resident_serial_host_stage_controls_collected_needs_independent_replay';save()
    except BaseException as error:report.update(status='failed',error=repr(error));save();raise
    return 0
if __name__=='__main__':raise SystemExit(main())
