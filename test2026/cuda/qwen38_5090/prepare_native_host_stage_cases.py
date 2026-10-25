"""Select four actual retained3968+128 text/image requests, never invent IDs."""
from pathlib import Path
import argparse,base64,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parent))
from benchmark_fixed_length_native import ARCHIVE_SHA256,MANIFEST_SHA256,final_ids
sha=lambda raw:hashlib.sha256(raw).hexdigest()
METHODS='Garnet/test2026/cuda/qwen38_5090/'


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();m=Path(__file__).resolve().parent;parent='fixed-length-http-study-attempt1'
    proof_path=args.proofs/(parent+'-verified.json');proof=json.loads(proof_path.read_bytes())
    assert proof==json.loads((args.proofs/(parent+'-remote-verified.json')).read_bytes()) and proof['status']=='fixed_length_http_study_replayed' and proof['native_fixed_4k_metrics_replayed']
    archive=args.proofs/(parent+'-evidence.zip');assert sha(archive.read_bytes())==proof['archive_sha256'];cases=[]
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for name,digest in manifest['files'].items():assert sha(z.read(name))==digest
        w=json.loads(z.read('results/'+parent+'/workflow.json'));assert w['exit_code']==0 and w['cleanup_finished'] and w['library_unchanged'] and w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after']
        derivation=json.loads((m/'evidence/native-host-stage-worker-derivation.json').read_bytes())
        assert sha(z.read(METHODS+'native_fixed_length_stream_worker.py'))==derivation['source_sha256']==sha((m/'native_fixed_length_stream_worker.py').read_bytes())
        assert sha(z.read(METHODS+'benchmark_fixed_length_native.py'))==sha((m/'benchmark_fixed_length_native.py').read_bytes())
        corpus_raw=z.read(METHODS+'evidence/fixed-length-4096-corpus.zip');assert sha(corpus_raw)==ARCHIVE_SHA256
        with zipfile.ZipFile(io.BytesIO(corpus_raw)) as corpus:
            raw=corpus.read('manifest.json');assert sha(raw)==MANIFEST_SHA256;frozen=json.loads(raw);rows={v['key']:v for v in frozen['rows']}
            for label in ('text','image'):
                cohort=next(c for c in frozen['cohorts'] if c['case']==label and c['concurrency']==1)
                for key in cohort['measured'][:2]:
                    row=rows[key];wire=corpus.read(row['request_file']);ids=corpus.read(row['expected_prompt_ids_file']);assert sha(wire)==row['request_sha256'] and sha(ids)==row['expected_prompt_ids_sha256']
                    body=json.loads(wire);prompt=json.loads(ids);assert body['ignore_eos'] is True and body['max_tokens']==128 and len(prompt)==3968
                    content=body['messages'][0]['content'];image_sha=sha(base64.b64decode(content[0]['image_url']['url'].split(',',1)[1],validate=True)) if label=='image' else None
                    item={'key':key,'case':label,'request_sha256':sha(wire),'body':body,'prompt_ids':prompt,'image_sha256':image_sha,'generated_ids_by_mode':{},'parent_client_keys':{},'parent_worker_request_ids':{}}
                    for mode in ('ordinary','mtp'):
                        prefix='results/'+parent+'/'+mode+'/';bench=prefix+'benchmark4k/';names=[n for n in z.namelist() if n.startswith(bench) and n.endswith('-manifest.json')];assert len(names)==1;benchmark=json.loads(z.read(names[0]));assert benchmark['status']=='completed'
                        client_key=benchmark['run_id']+'-'+key;assert z.read(bench+client_key+'-request.json')==wire
                        events=[json.loads(line) for line in z.read(bench+client_key+'-events.jsonl').splitlines()];final=final_ids(events,prompt)
                        complete=[]
                        for line in z.read(prefix+'worker/worker-stdout.jsonl').splitlines():
                            event=json.loads(json.loads(line)['line'])
                            if event.get('event')=='completed' and event['id']==final['id']:complete.append(event)
                        assert len(complete)==1;actual=complete[0];assert actual['prompt_ids']==prompt and actual['generated_ids']==final['response_token_ids'] and actual['image_sha256']==image_sha and actual['ignore_eos'] is True
                        item['generated_ids_by_mode'][mode]=actual['generated_ids'];item['parent_client_keys'][mode]=client_key;item['parent_worker_request_ids'][mode]=actual['id']
                    assert item['generated_ids_by_mode']['ordinary']==item['generated_ids_by_mode']['mtp'];cases.append(item)
    assert len(cases)==4 and len({r['image_sha256'] for r in cases if r['case']=='image'})==2
    result={'status':'native_host_stage_retained_cases_prepared_gpu_unrun','parent_archive_sha256':proof['archive_sha256'],'parent_proof_sha256':sha(proof_path.read_bytes()),'library_sha256':proof['library_sha256'],'corpus_archive_sha256':ARCHIVE_SHA256,'cases':cases,'orders':[[0,1,2,3],[2,0,3,1],[0,1,2,3]],'scope':'Four actual C1 measured requests selected before diagnostic execution, two text/two distinct image nonces,3968+128. Both actual prior ordinary/MTP worker and SSE IDs bound exactly; no new model or performance result. First round includes any first-use work; later rounds rotated/reset-repeat. Host intervals remain instrumented diagnostics.'}
    target=m/'evidence/native-host-stage-cases.json';raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'case_count':len(cases),'case_file_sha256':sha(raw),'bytes':len(raw),'gpu_executed':False}));return 0


if __name__=='__main__':raise SystemExit(main())
