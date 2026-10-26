"""Replay actual strict-failure controls/streams; never promote a partial benchmark."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parent))
from benchmark_stream import TokenEvents
from benchmark_fixed_length_reference import control_request,control_response
from benchmark_fixed_length_native import ARCHIVE_SHA256,MANIFEST_SHA256
from metrics import summarize
from reference_fixed4k_vllm_identity import validate

JOB='vllm-fixed4k-ordinary-attempt3'
METHODS='Garnet/test2026/cuda/qwen38_5090/'
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);args=a.parse_args()
    archive=args.proofs/(JOB+'-evidence.zip');proof=args.proofs/(JOB+'-verified.json');q=json.loads(proof.read_bytes())
    assert q['status']=='reference_fixed4k_failure_preserved' and q['engine']=='vllm' and q['mode']=='ordinary' and q['profiles']==[]
    assert q['replay_verifier_sha256']==q['archived_replay_verifier_sha256']==sha(Path(__file__).with_name('verify_vllm_fixed4k_launcher_completed.py').read_bytes())
    assert sha(archive.read_bytes())==q['archive_sha256']
    base='results/'+JOB+'/';root=base+'client/benchmark4k/';streams=[];controls={}
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('evidence-manifest.json'))
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for n,h in manifest['files'].items():assert sha(z.read(n))==h
        for n in ('benchmark_stream.py','benchmark_fixed_length_reference.py','benchmark_fixed_length_native.py','metrics.py','reference_fixed4k_vllm_identity.py'):
            assert sha(Path(__file__).with_name(n).read_bytes())==sha(z.read(METHODS+n))
        w=json.loads(z.read(base+'workflow.json'));p=json.loads(z.read(METHODS+'evidence/vllm-fixed4k-launcher-completed-preparation.json'))
        assert type(w['exit_code']) is int and w['exit_code']==1 and w['cleanup_finished'] is True and w['server_exit_code']==0
        assert w['source_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime_dependencies']
        for n,h in p['files'].items():assert sha(z.read(n))==h
        metadata=json.loads(z.read(base+'metadata.json'));settings=validate(json.loads(z.read(base+'server-info-raw.json')),
            json.loads(z.read(base+'server-version-raw.json')),z.read(base+'server.log').decode('utf8','replace'),'ordinary')
        for n,value in settings.items():assert metadata[n]==value
        raw=z.read(METHODS+'evidence/fixed-length-4096-corpus.zip');assert sha(raw)==ARCHIVE_SHA256
        with zipfile.ZipFile(io.BytesIO(raw)) as frozen:
            raw=frozen.read('manifest.json');assert sha(raw)==MANIFEST_SHA256;corpus=json.loads(raw)
            report=json.loads(z.read(root+'controls/report.json'));assert report['status']=='all124_nonstream_id_controls_collected' and len(report['cases'])==len(corpus['rows'])==124
            inputs={}
            for row,record in zip(corpus['rows'],report['cases']):
                key=row['key'];body=json.loads(frozen.read(row['request_file']));ids=json.loads(frozen.read(row['expected_prompt_ids_file']))
                wire=z.read(root+'controls/'+key+'-request.json');response=z.read(root+'controls/'+key+'-response.json')
                assert wire==json.dumps(control_request(body,'vllm'),separators=(',',':')).encode()
                assert sha(wire)==record['request_sha256'] and sha(response)==record['response_sha256'] and record['key']==key
                assert record['prompt_ids_exact'] is True and record['output_tokens']==128 and record['ordinary_ids_exact'] is False
                controls[key]=control_response(json.loads(response),'vllm',ids);inputs[key]=body
        names=[n for n in manifest['files'] if n.startswith(root) and n.endswith('-manifest.json')];assert len(names)==1
        bench=json.loads(z.read(names[0]));assert bench['status']=='failed' and bench['engine']=='vllm' and len(bench['cohorts'])==1
        assert bench['engine_metadata']==metadata and bench['corpus_manifest_sha256']==MANIFEST_SHA256
        run=bench['run_id'];prefix=run+'-';actual={}
        for name in sorted(n for n in manifest['files'] if n.startswith(root+prefix) and n.endswith('-events.jsonl')):
            key=name.removeprefix(root).removesuffix('-events.jsonl');short=key.removeprefix(prefix);body=inputs[short]
            wire=z.read(root+key+'-request.json');assert wire==json.dumps(body,separators=(',',':')).encode()
            saved=json.loads(z.read(root+key+'-result.json'));assert saved['key']==key and saved['request_sha256']==sha(wire)
            state=TokenEvents()
            for line in z.read(name).splitlines():
                event=json.loads(line)
                if event['data']!='[DONE]':
                    usage=json.loads(event['data']).get('usage')
                    if usage is not None:
                        assert type(usage['prompt_tokens']) is type(usage['completion_tokens']) is type(usage['total_tokens']) is int
                        assert usage['prompt_tokens']==3968 and 0<=usage['completion_tokens']<=128 and usage['total_tokens']==3968+usage['completion_tokens']
                        assert not(usage.get('prompt_tokens_details') or {}).get('cached_tokens',0)
                state.feed(event['data'],event['monotonic'])
            fresh=state.result(saved['submitted'],saved['completed'],128)
            for n,value in fresh.items():assert saved[n]==value
            assert saved['status']=='completed' and saved['prompt_tokens']==3968 and saved['generated_tokens']==128 and saved['finish_reason']=='length'
            current=saved['text'];previous=controls[short]['text'];exact=current==previous
            streams.append({'key':short,'prompt_tokens':3968,'output_tokens':128,'text_exact':exact,
                'serial_text_sha256':sha(previous.encode()),'stream_text_sha256':sha(current.encode()),
                'first_different_character':None if exact else next((i for i,(x,y) in enumerate(zip(current,previous)) if x!=y),min(len(current),len(previous))),
                'individual_streamed_ids_observed':False});actual[key]=saved
        assert any(not s['text_exact'] and s['key'].startswith('ctx4096-s3-text-c2-warmup-') for s in streams)
        first=bench['cohorts'][0];assert first['case']=='text' and first['concurrency']==1 and len(first['request_keys'])==8
        metrics=summarize([actual[key] for key in first['request_keys']])
        for n,value in metrics.items():assert first[n]==value
    result={'status':'vllm_fixed4k_attempt3_partial_controls_and_output_failure_replayed','archive_sha256':q['archive_sha256'],
        'original_failure_proof_sha256':sha(proof.read_bytes()),'files_verified':len(manifest['files']),
        'actual_nonstream_4k_id_controls':124,'raw_streams_replayed':len(streams),'stream_comparisons':streams,
        'retained_partial_timed_cohorts_recomputed':1,'metadata_settings_replayed':settings,
        'numerical_cause_or_harmlessness_proven':False,'broad_quality_qualified':False,'performance_admitted':False,
        'full_4k_benchmark_complete':False,'two_times_target_achieved':False,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),
        'scope':'Actual partial strict failure only. All124serial3968+128 prompt/completion IDs and every saved raw SSE/count/usage/text replayed. Concurrent warmup text differs; individual streamed IDs and numerical cause/equivalence are unproven. Original failure and complete first cohort retained; no full benchmark/chart/2x promotion.'}
    target=args.proofs/(JOB+'-partial-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'actual_nonstream_4k_id_controls':124,'raw_streams_replayed':len(streams),
        'changed_stream_texts':sum(not s['text_exact'] for s in streams),'proof_sha256':sha(raw)}));return 0

if __name__=='__main__':raise SystemExit(main())
