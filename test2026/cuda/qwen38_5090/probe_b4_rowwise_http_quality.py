"""Actual B4 loopback protocol and adapted805 screening; no timing admission."""
from pathlib import Path
import argparse,base64,concurrent.futures,copy,hashlib,json,sys,threading,time,zipfile
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from native_batch_http_bridge import BatchWorker,BatchHandler,ThreadingHTTPServer
from probe_o128_http_quality import reference_quality,REFERENCE_SHA
sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();root=a.root;out=a.output;assert not out.exists();out.mkdir()
    sys.path.insert(0,str(root/'Garnet'));m=root/'Garnet/test2026/cuda/qwen38_5090';p=json.loads((m/'evidence/b4-rowwise-http-quality-preparation.json').read_bytes());corpus=root/'tools/mmmu-quality-corpus'
    from test2026.cuda.qwen38_5090.mmmu_http_quality import load_corpus,collect
    from test2026.cuda.qwen38_5090.verify_mmmu_http_quality import replay
    from test2026.cuda.qwen38_5090.b4_full4k_inputs import load_corpus as load_fixed
    from test2026.cuda.qwen38_5090.b4_padding_lifecycle_inputs import select_lifecycle
    from test2026.cuda.qwen38_5090.benchmark_stream import request as stream_request
    _,rows,gold=load_corpus(corpus);reference,expected=reference_quality(root,corpus)
    fixed=(m/'evidence/fixed-length-4096-corpus.zip').read_bytes();assert sha(fixed)==p['input_corpus_sha256'];selection=load_fixed(fixed);lifecycle=select_lifecycle(selection['inputs']);keys=list(selection['inputs'])
    ref='b4-full4k-exact-attempt1';archive=root/'results'/(ref+'-evidence.zip');assert sha(archive.read_bytes())==p['parents'][ref]['archive_sha256']
    with zipfile.ZipFile(archive) as z:r=json.loads(z.read('results/'+ref+'/reference/report.json'))
    assert r['status']=='eight_B1_full4k_oracles_collected' and r['inputs']==selection['inputs'];full_gold={row['keys'][0]:row['generated_ids'][0] for row in r['rows']};assert len(full_gold)==8
    cache=json.loads((root/'results/study-ssd-engine-cache/receipt.json').read_bytes())['cache_root']
    worker=BatchWorker([sys.executable,str(m/'launch_b4_rowwise_http_worker.py'),str(root),cache],out/'worker',4,10);server=thread=None
    report=dict(status='starting',library_sha256=p['library_sha256'],checkpoint_revision=p['checkpoint_revision'],model_file_manifest_sha256=p['model_file_manifest_sha256'],reference_archive_sha256=REFERENCE_SHA,reference_quality=reference,boundary_controls=[],stream_controls=[],negative_controls=[],adapted805_only=True,HTTP_quality_or_performance_qualified=False,two_times_target_achieved=False)
    def save():(out/'report.json').write_bytes(json.dumps(report,indent=2).encode())
    save();url='http://127.0.0.1:18082'
    try:
        deadline=time.monotonic()+1800
        while not worker.healthy():
            assert worker.process.poll() is None and worker.failure is None,worker.failure
            assert time.monotonic()<deadline,'Worker startup deadline exceeded';time.sleep(.25)
        ready=worker.ready_info;admission=ready['source_admission']
        assert (ready['batch'],ready['capacity'],ready['prefill_chunk'],ready['chunk'])==(4,4096,32,4)
        assert ready['generation_mode']=='verified_mtp' and ready['library_sha256']==p['library_sha256']
        assert admission['model_parents']==p['model_parents'] and admission['source_bundle_sha256']==p['bundle_sha256'] and admission['full_model_file_manifest_sha256']==p['model_file_manifest_sha256']
        report.update(status='boundary_controls',ready=ready);save()
        server=ThreadingHTTPServer(('127.0.0.1',18082),BatchHandler);server.worker=worker;server.model='qwen38-27b-nvfp4';thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        def check(control,key,body,barrier):
            wire=json.dumps(body,separators=(',',':')).encode();assert len(wire)<=2*1024*1024;(out/(control+'-request.json')).write_bytes(wire);barrier.wait(timeout=20)
            with urlopen(Request(url+'/v1/chat/completions',data=wire,headers={'Content-Type':'application/json'}),timeout=300) as response:raw=response.read()
            (out/(control+'-response.json')).write_bytes(raw);actual=json.loads(raw);choice=actual['choices'][0]
            assert choice['prompt_token_ids']==selection['inputs'][key]['prompt_ids'] and choice['response_token_ids']==full_gold[key]
            assert actual['usage']==dict(prompt_tokens=3968,completion_tokens=128,total_tokens=4096) and actual['study']['generation_mode']=='verified_mtp'
            return dict(control=control,source_key=key,id=actual['id'],prompt_and_output_IDs_exact=True,request_sha256=sha(wire),response_sha256=sha(raw),request_bytes=len(wire))
        groups=[g['real_keys'] for g in lifecycle['groups']]+[keys,keys]
        for index,group in enumerate(groups):
            barrier=threading.Barrier(len(group));large=index==len(groups)-1
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(group)) as executor:
                futures=[]
                for row,key in enumerate(group):
                    body=copy.deepcopy(selection['inputs'][key]['body']);body['stream']=False
                    if large:body['study_transport_padding']='x'*600000
                    futures.append(executor.submit(check,'boundary-'+str(index)+'-'+str(row),key,body,barrier))
                values=[f.result() for f in futures]
            report['boundary_controls'].append(dict(index=index,requested_concurrency=len(group),large_transport_fixture_only=large,results=values));save()
        # Two streaming controls exercise the same public wire through actual SSE.
        for index,key in enumerate((keys[0],keys[4])):
            body=copy.deepcopy(selection['inputs'][key]['body']);body['stream']=True;label='stream-'+str(index)
            result=stream_request(url,body,out,label,128);assert result['status']=='completed' and result['generated_tokens']==128
            events=[json.loads(line) for line in (out/(label+'-events.jsonl')).read_bytes().splitlines()]
            final=[json.loads(e['data']) for e in events if e.get('data')!='[DONE]' and 'response_token_ids' in json.loads(e['data'])];assert len(final)==1
            assert final[0]['prompt_token_ids']==selection['inputs'][key]['prompt_ids'] and final[0]['response_token_ids']==full_gold[key]
            report['stream_controls'].append(dict(label=label,source_key=key,id=final[0]['id'],full_IDs_exact=True));save()
        bad=copy.deepcopy(selection['inputs'][keys[0]]['body']);bad['ignore_eos']=1;bad['stream']=False;wire=json.dumps(bad).encode();(out/'invalid-ignore-eos-request.json').write_bytes(wire)
        try:
            with urlopen(Request(url+'/v1/chat/completions',data=wire,headers={'Content-Type':'application/json'}),timeout=30) as response:code=response.status;raw=response.read()
        except HTTPError as error:code=error.code;raw=error.read()
        (out/'invalid-ignore-eos-response.bin').write_bytes(raw);assert code==400 and worker.healthy();report['negative_controls'].append(dict(ignore_eos_nonboolean_rejected_HTTP400=True,worker_remained_healthy=True));save()
        report['status']='collecting805';save()
        identity=dict(engine='native',model_alias='qwen38-27b-nvfp4',generation_mode='verified_mtp',library_sha256=p['library_sha256'],ready=ready,physical_batch=4,private_padding=True,source_admission=admission,scope='Actual source91 B4/MTP3 private HTTP worker; adapted805 only, no general quality or timing admission.')
        collect(corpus,rows,gold,out/'mmmu-quality','native','qwen38-27b-nvfp4',url,identity,timeout=300)
        quality=replay(corpus,out/'mmmu-quality');assert quality['status']=='adapted_mmmu_http_quality_responses_replayed';report['quality']=quality;report['token_mismatches']=[]
        for row in rows:
            actual=json.loads((out/'mmmu-quality/cases'/row['id']/'response.bin').read_bytes())['choices'][0];control=expected[row['id']]
            if (actual['prompt_token_ids'],actual['response_token_ids'])!=(control['prompt_token_ids'],control['response_token_ids']):report['token_mismatches'].append(row['id'])
        save();assert not report['token_mismatches'],'B4/T32 screening ID differences retained'
        report['status']='B4_rowwise_HTTP_controls_and_adapted805_collected_needs_independent_replay';save()
    except BaseException as error:report.update(status='failed',error=repr(error));save();raise
    finally:
        if server is not None:server.shutdown();server.server_close()
        if thread is not None:thread.join()
        worker.close()
    return 0

if __name__=='__main__':raise SystemExit(main())
