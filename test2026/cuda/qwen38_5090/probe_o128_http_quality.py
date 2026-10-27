"""Actual loopback quality collection on the explicitly admitted O128 research worker."""
from pathlib import Path
import argparse,copy,hashlib,json,sys,tempfile,threading,time,zipfile
from urllib.request import Request,urlopen
from native_http_bridge import Worker,Handler,ThreadingHTTPServer
sha=lambda b:hashlib.sha256(b).hexdigest()
REFERENCE='tokenizer-nfc-http-study-attempt1'
REFERENCE_SHA='2ef0d7c0036ceb8e578ea64673131464d45191e0cebafaa68824583e396406c4'

def reference_quality(root,corpus):
    from test2026.cuda.qwen38_5090.verify_mmmu_http_quality import replay
    archive=root/'results'/(REFERENCE+'-evidence.zip');assert sha(archive.read_bytes())==REFERENCE_SHA
    prefix='results/'+REFERENCE+'/ordinary/mmmu-quality/'
    # Its enclosing old HTTP benchmark failed later. Only the independently
    # replayed complete805 quality collection supplies this numerical control.
    with zipfile.ZipFile(archive) as z,tempfile.TemporaryDirectory(prefix='o128-quality-reference-',dir=root/'tmp') as folder:
        manifest=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for n,h in manifest['files'].items():assert sha(z.read(n))==h
        dst=Path(folder)
        for n in manifest['files']:
            if n.startswith(prefix):
                target=dst/n.removeprefix(prefix);assert target.resolve().is_relative_to(dst.resolve());target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(n))
        quality=replay(corpus,dst);assert quality['status']=='adapted_mmmu_http_quality_responses_replayed'
        expected={n.removeprefix(prefix+'cases/').split('/')[0]:json.loads(z.read(n))['choices'][0] for n in manifest['files'] if n.startswith(prefix+'cases/') and n.endswith('/response.bin')}
        assert len(expected)==805
    return quality,expected

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args();root=args.root;out=args.output;assert not out.exists();out.mkdir()
    sys.path.insert(0,str(root/'Garnet'));m=root/'Garnet/test2026/cuda/qwen38_5090';corpus=root/'tools/mmmu-quality-corpus'
    from test2026.cuda.qwen38_5090.mmmu_http_quality import load_corpus,collect
    from test2026.cuda.qwen38_5090.verify_mmmu_http_quality import replay
    from test2026.cuda.qwen38_5090.native_o128_http_admission import require,LIBRARY,CERTIFICATES
    admission=require(root,'mtp',sha((root/'build/bin/garnet.dll').read_bytes()));_,rows,gold=load_corpus(corpus);reference,expected=reference_quality(root,corpus)
    cache=json.loads((root/'results/study-ssd-engine-cache/receipt.json').read_bytes())['cache_root'];worker=Worker(['cmd.exe','/d','/c',str(m/'serve_native_o128_worker.bat'),'mtp',cache],out/'worker');server=thread=None
    report=dict(status='starting',library_sha256=LIBRARY,source_admission=admission,reference_archive_sha256=REFERENCE_SHA,reference_enclosing_HTTP_attempt_passed=False,reference_quality=reference,boundary_controls=[],adapted805_only=True,broad_quality_or_performance_qualified=False,two_times_target_achieved=False)
    def save():(out/'report.json').write_bytes(json.dumps(report,indent=2).encode())
    save()
    try:
        deadline=time.monotonic()+1800
        while not worker.healthy():
            assert worker.process.poll() is None and worker.failure is None,worker.failure
            assert time.monotonic()<deadline,'Worker startup deadline exceeded';time.sleep(.25)
        ready=worker.ready_info;assert ready['prefill_chunk']==128 and ready['chunk']==4 and ready['batch']==1 and ready['capacity']==4096 and ready['generation_mode']=='mtp' and ready['library_sha256']==LIBRARY and ready['o128_admission']==admission
        report.update(status='boundary_controls',ready=ready);save();server=ThreadingHTTPServer(('127.0.0.1',18082),Handler);server.worker=worker;server.model='qwen38-27b-nvfp4';thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        # Two complete3968+128 text/image controls from the already validated
        # six-case corpus, with first and reset-repeat requests on this worker.
        full='o128-full4k-model-admission-attempt1';pinned=root/'results'/(full+'-verified.json');assert sha(pinned.read_bytes())==CERTIFICATES[full];proof=json.loads(pinned.read_bytes());archive=root/'results'/(full+'-evidence.zip');assert sha(archive.read_bytes())==proof['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            native=json.loads(z.read('results/'+full+'/native/report.json'));cases=json.loads(z.read('Garnet/test2026/cuda/qwen38_5090/evidence/native-host-stage-cases.json'))['cases']
        for index in (0,2,0,2):
            source=cases[index];body=copy.deepcopy(source['body']);body['stream']=False;key='boundary-'+str(len(report['boundary_controls']));wire=json.dumps(body,separators=(',',':')).encode();(out/(key+'-request.json')).write_bytes(wire)
            with urlopen(Request('http://127.0.0.1:18082/v1/chat/completions',data=wire,headers={'Content-Type':'application/json'}),timeout=300) as response:raw=response.read()
            (out/(key+'-response.json')).write_bytes(raw);result=json.loads(raw);choice=result['choices'][0];control=next(c for c in native['cases'] if c['key']==source['key']);assert choice['prompt_token_ids']==source['prompt_ids'] and choice['response_token_ids']==control['ordinary_ids']
            assert result['usage']==dict(prompt_tokens=3968,completion_tokens=128,total_tokens=4096);report['boundary_controls'].append(dict(key=key,source_key=source['key'],prompt_ids_exact=True,output_ids_exact=True));save()
        identity=dict(engine='native',model_alias='qwen38-27b-nvfp4',generation_mode='mtp',library_sha256=LIBRARY,ready=ready,physical_batch=1,worker_sha256=sha((m/'native_o128_stream_worker.py').read_bytes()),scope='Actual explicitly admitted60source/DLL841 O128 research worker; adapted805 only, no broad quality or timing admission.')
        report['status']='collecting805';save();collect(corpus,rows,gold,out/'mmmu-quality','native','qwen38-27b-nvfp4','http://127.0.0.1:18082',identity,timeout=300)
        quality=replay(corpus,out/'mmmu-quality');assert quality['status']=='adapted_mmmu_http_quality_responses_replayed';report['quality']=quality;report['token_mismatches']=[]
        for row in rows:
            actual=json.loads((out/'mmmu-quality/cases'/row['id']/'response.bin').read_bytes())['choices'][0];control=expected[row['id']]
            if (actual['prompt_token_ids'],actual['response_token_ids'])!=(control['prompt_token_ids'],control['response_token_ids']):report['token_mismatches'].append(row['id'])
        save();assert not report['token_mismatches'],'Preserved O128/T32 screening ID differences'
        report['status']='O128_HTTP_adapted805_and_four_full4k_controls_collected_needs_independent_replay';save()
    except BaseException as e:report.update(status='failed',error=repr(e));save();raise
    finally:
        if server is not None:server.shutdown();server.server_close()
        if thread is not None:thread.join()
        worker.close()
    return 0
if __name__=='__main__':raise SystemExit(main())
