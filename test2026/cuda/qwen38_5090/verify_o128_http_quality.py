"""Independent raw O128 HTTP IDs, callback, adapted805 and restoration replay."""
from pathlib import Path
import argparse,hashlib,json,sys,tempfile,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.verify_mmmu_http_quality import replay
from test2026.cuda.qwen38_5090.mmmu_http_quality import load_corpus
from test2026.cuda.qwen38_5090.native_o128_http_admission import LIBRARY,CERTIFICATES
from test2026.cuda.qwen38_5090.probe_o128_http_quality import REFERENCE,REFERENCE_SHA
JOB='o128-http-quality-attempt1';P='Garnet/test2026/cuda/qwen38_5090/';sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
    a=argparse.ArgumentParser();a.add_argument('--proofs',type=Path,required=True);a.add_argument('--corpus',type=Path,required=True);a.add_argument('--scratch',type=Path,required=True);args=a.parse_args();archive=args.proofs/(JOB+'-evidence.zip')
    with zipfile.ZipFile(archive) as z:
        m=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
        for n,h in m['files'].items():assert sha(z.read(n))==h
        base='results/'+JOB+'/';doc=lambda n:json.loads(z.read(n));w=doc(base+'workflow.json');prep=z.read(P+'evidence/o128-http-quality-preparation.json');p=json.loads(prep);assert sha(prep)==w['preparation_sha256'] and sha(Path(__file__).read_bytes())==p['new_files'][P+Path(__file__).name]
        assert type(w['exit_code']) is int and w['cleanup_finished'] is w['baseline_source_restored'] is w['baseline_library_restored'] is True
        assert w['source_before_sha256']==w['source_after_sha256']==p['files'] and w['new_files_after']==w['new_files']==p['new_files']
        assert w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['checkpoint_after']==p['checkpoints']
        assert w['candidate_files']==p['candidate_files'] and w['candidate_base']==p['candidate_base'] and len(p['candidate_files'])==60 and w['candidate_library_sha256']==p['candidate_library_sha256']==LIBRARY and w['performance_admitted'] is False
        for n,h in {**p['new_files'],**p['audit_dependencies']}.items():assert sha(z.read(n))==h
        assert sha(z.read(p['source_bundle']))==p['source_bundle_sha256']
        for job,row in p['measurement_parents'].items():
            raw=z.read('results/'+job+'-verified.json');q=json.loads(raw);assert sha(raw)==row['proof_sha256'] and raw==(args.proofs/(job+'-verified.json')).read_bytes() and q['archive_sha256']==row['archive_sha256']==m['linked_parent_archives'][job]==sha((args.proofs/(job+'-evidence.zip')).read_bytes())
        def ledger(phase,expected):
            path=base+phase+'-ledger/commands.jsonl'
            if path not in m['files']:return None
            records=[json.loads(line) for line in z.read(path).splitlines()];assert len(records)==2 and records[0]['event']=='started' and records[1]['event']=='finished'
            assert records[0]['source']['owned_source_sha256']==records[1]['source_after']['owned_source_sha256']==expected
            return records[1]['exit_code']
        cpu=ledger('cpu',{**p['files'],**p['new_files']});assert type(cpu) is int
        candidate={**p['files'],**p['new_files'],**{'Garnet/'+n:h for n,h in p['candidate_files'].items()}};child=ledger('gpu',candidate)
        if child is not None:assert w['installed_source_sha256']==candidate and doc(base+'resources/resource-monitor.json')['child_exit_code']==child
        summary=dict(actual_adapted_quality_replayed=False,performance_qualified=False)
        if w['exit_code']==0:
            assert cpu==child==0 and w['status']=='O128_HTTP_quality_collected_needs_independent_replay'
            r=doc(base+'native/report.json');assert r['status']=='O128_HTTP_adapted805_and_four_full4k_controls_collected_needs_independent_replay' and not r['token_mismatches'] and r['reference_enclosing_HTTP_attempt_passed'] is False and r['library_sha256']==LIBRARY
            ready=r['ready'];assert ready['o128_admission']['certificate_proof_sha256']==CERTIFICATES and ready['prefill_chunk']==128 and ready['capacity']==4096 and ready['batch']==1 and ready['generation_mode']=='mtp' and ready['library_sha256']==LIBRARY
            _,rows,_=load_corpus(args.corpus);reference=args.proofs/(REFERENCE+'-evidence.zip');assert sha(reference.read_bytes())==REFERENCE_SHA
            with tempfile.TemporaryDirectory(prefix='o128-http-quality-replay-',dir=args.scratch) as temporary,zipfile.ZipFile(reference) as old:
                folder=Path(temporary);qualities={}
                for key,container,prefix in [('candidate',z,base+'native/mmmu-quality/'),('reference',old,'results/'+REFERENCE+'/ordinary/mmmu-quality/')]:
                    dst=folder/key;dst.mkdir()
                    for n in container.namelist():
                        if n.startswith(prefix):
                            target=dst/n.removeprefix(prefix);assert target.resolve().is_relative_to(dst.resolve());target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(container.read(n))
                    qualities[key]=replay(args.corpus,dst);assert qualities[key]['status']=='adapted_mmmu_http_quality_responses_replayed'
                assert qualities['candidate']==r['quality'] and qualities['reference']==r['reference_quality']
                outputs={};tokens={};native_ready=[]
                for line in z.read(base+'native/worker/worker-stdout.jsonl').splitlines():
                    try:event=json.loads(json.loads(line)['line'])
                    except ValueError:continue
                    if not isinstance(event,dict):continue
                    if event.get('event')=='ready':native_ready.append(event)
                    elif event.get('event')=='token':
                        key=event['id'];assert key not in outputs;tokens.setdefault(key,[]).append(event['token_id']);assert event['sampled_tokens']==len(tokens[key])
                    elif event.get('event')=='completed':
                        key=event['id'];assert key not in outputs and event['generated_ids']==tokens[key] and event['physical_batch']==event['real_rows']==1 and event['padded_rows']==0;outputs[key]=event
                    elif event.get('event')=='error':raise AssertionError(event)
                assert native_ready==[ready] and len(outputs)==809;used=set()
                def match(response):
                    key=response['id'];c=response['choices'][0];assert key not in used and outputs[key]['prompt_ids']==c['prompt_token_ids'] and outputs[key]['generated_ids']==c['response_token_ids'];used.add(key)
                for row in rows:
                    path='mmmu-quality/cases/'+row['id']+'/response.bin';response=doc(base+'native/'+path);control=json.loads(old.read('results/'+REFERENCE+'/ordinary/'+path))['choices'][0];c=response['choices'][0]
                    assert (c['prompt_token_ids'],c['response_token_ids'])==(control['prompt_token_ids'],control['response_token_ids']);match(response)
                job='o128-full4k-model-admission-attempt1'
                with zipfile.ZipFile(args.proofs/(job+'-evidence.zip')) as model:
                    cases=json.loads(model.read(P+'evidence/native-host-stage-cases.json'))['cases'];native=json.loads(model.read('results/'+job+'/native/report.json'))
                assert len(r['boundary_controls'])==4
                for i,index in enumerate((0,2,0,2)):
                    item=r['boundary_controls'][i];source=cases[index];key='boundary-'+str(i);request=doc(base+'native/'+key+'-request.json');expected={**source['body'],'stream':False};assert request==expected and item['source_key']==source['key']
                    response=doc(base+'native/'+key+'-response.json');c=response['choices'][0];original=next(c for c in native['cases'] if c['key']==source['key']);assert c['prompt_token_ids']==source['prompt_ids'] and c['response_token_ids']==original['ordinary_ids'] and response['usage']==dict(prompt_tokens=3968,completion_tokens=128,total_tokens=4096);match(response)
                assert len(used)==len(outputs)==809
                summary=dict(actual_adapted_quality_replayed=True,adapted_requests=805,all805_prompt_and_output_IDs_exact_T32=True,full4k_controls=4,actual_native_requests=809,quality=qualities['candidate'],performance_qualified=False)
        else:assert w['status']=='failed' and w.get('error')
        result=dict(status='actual_O128_HTTP_805_IDs_and_four_full4k_controls_replayed' if w['exit_code']==0 else 'actual_O128_HTTP_quality_failure_preserved',archive_sha256=sha(archive.read_bytes()),exit_code=w['exit_code'],error=w.get('error'),source_runtime_SDK_checkpoint_restored=True,candidate_library_sha256=LIBRARY,arrays=summary,official900_MMMU_qualified=False,broad_quality_or_performance_qualified=False,two_times_target_achieved=False)
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw;target.write_bytes(raw);print(json.dumps(dict(status=result['status'],proof_sha256=sha(raw),exit_code=w['exit_code'])))
if __name__=='__main__':main()
