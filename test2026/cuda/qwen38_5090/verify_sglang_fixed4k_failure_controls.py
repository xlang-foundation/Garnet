"""Replay actual pre-failure controls/screen; preserve126-token rejection and no timings.

Nonstreaming controls contain actual IDs. Common reference streaming exposes
usage and text; those streams cannot attest their unobserved individual IDs.
"""
from pathlib import Path
import argparse,base64,hashlib,io,json,statistics,sys,tempfile,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from run_sglang_fixed4k import job_path
from reference_fixed4k_sglang_identity import validate,CHECKPOINT
from benchmark_fixed_length_reference import control_request,control_response,paired_control
from benchmark_fixed_length_native import ARCHIVE_SHA256,MANIFEST_SHA256
from benchmark_stream import TokenEvents
from compare_http_runs import verify_run
from cohort_memory_replay import replay_cohort_memory
from prepare_vlm_reference import reference_tokens
from reference_mtp_equivalence import compare_pair
from test2026.cuda.qwen38_5090.verify_mmmu_http_quality import replay
from test2026.cuda.qwen38_5090.mmmu_http_quality import load_corpus,validate_response
sha=lambda raw:hashlib.sha256(raw).hexdigest()
GPU='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6'
METHODS='Garnet/test2026/cuda/qwen38_5090/'

def replay_stream(read,key,body,control):
    wire=read(key+'-request.json')
    assert wire==json.dumps(body,separators=(',',':')).encode(),'Timed frozen request bytes differ'
    saved=json.loads(read(key+'-result.json'));assert saved['request_sha256']==sha(wire) and saved['key']==key
    events=[json.loads(line) for line in read(key+'-events.jsonl').splitlines()];state=TokenEvents()
    for event in events:state.feed(event['data'],event['monotonic'])
    actual=state.result(saved['submitted'],saved['completed'],128)
    for name,value in actual.items():assert saved[name]==value,('Raw stream/result differs',key,name)
    assert actual['prompt_tokens']==3968 and actual['generated_tokens']==128 and actual['finish_reason']=='length'
    assert actual['text']==control['text'],'Timed stream text differs from actual same-engine ID control'
    return saved

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);a.add_argument('--job',required=True)
    a.add_argument('--corpus',type=Path,required=True);a.add_argument('--scratch',type=Path,required=True);args=a.parse_args()
    assert args.scratch.is_dir();job_path(Path('D:/CantorAI/5090-study-20261023'),args.job)
    archive=args.proofs/(args.job+'-evidence.zip');base='results/'+args.job+'/';profiles=[];prior=None;screening=None
    with zipfile.ZipFile(archive) as z:
        doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json');names=z.namelist()
        assert z.testzip() is None and len(names)==len(set(names))==len(manifest['files'])+1
        for name,digest in manifest['files'].items():assert sha(z.read(name))==digest
        w=doc(base+'workflow.json');praw=z.read(METHODS+'evidence/sglang-fixed4k-run-preparation.json');p=json.loads(praw)
        assert sha(praw)==w['preparation_sha256'] and p['status']=='sglang_fixed4k_run_prepared_unrun'
        assert w['engine']=='sglang' and w['mode'] in ('ordinary','mtp') and args.job.startswith('sglang-fixed4k-'+w['mode']+'-')
        assert w['cleanup_finished'] and w['source_sha256']==w['source_after_sha256']==p['files']
        assert w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime_dependencies'] and not w['performance_admitted']
        for name,digest in p['files'].items():assert sha(z.read(name))==digest
        # Local replay helpers must match the versions that actually collected.
        for name in ('benchmark_stream.py','metrics.py','compare_http_runs.py','cohort_memory_replay.py','benchmark_fixed_length_reference.py','benchmark_fixed_length_native.py','reference_fixed4k_sglang_identity.py','prepare_vlm_reference.py','reference_mtp_equivalence.py','mmmu_http_quality.py','mmmu_quality_protocol.py','verify_mmmu_http_quality.py'):
            assert sha(Path(__file__).with_name(name).read_bytes())==sha(z.read(METHODS+name)),('Replay helper differs',name)
        parent='fixed-length-http-study-attempt1';parent_raw=z.read('results/'+parent+'-verified.json');native=json.loads(parent_raw)
        assert native['status']=='fixed_length_http_study_replayed' and native['native_fixed_4k_metrics_replayed']
        assert sha(parent_raw)==p['native_proof_sha256']==w['native_proof_sha256']
        assert native['archive_sha256']==p['native_archive_sha256']==w['native_archive_sha256']==manifest['native_archive_sha256']
        assert sha((args.proofs/(parent+'-evidence.zip')).read_bytes())==native['archive_sha256']
        passed=w['exit_code']==0
        assert not passed and w['exit_code']==1
        if True:
            assert w['status']=='failed' and type(w['server_exit_code']) is int
            for ledger in ('cpu-ledger','metadata-ledger','client-ledger','stop-ledger'):
                records=[json.loads(line) for line in z.read(base+ledger+'/commands.jsonl').splitlines()]
                assert len(records)==2 and records[0]['event']=='started' and records[1]['event']=='finished' and records[1]['exit_code']==(1 if ledger=='client-ledger' else 0)
            assert b'Ran 31 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
            metadata=doc(base+'metadata.json');info=doc(base+'server-info-raw.json');expected=validate(info,w['mode'])
            for key,value in expected.items():assert metadata[key]==value
            launcher='serve_sglang_fixed4k.sh' if w['mode']=='ordinary' else 'serve_sglang_mtp3_fixed4k.sh'
            assert metadata['launcher_sha256']==sha(z.read(METHODS+launcher)) and metadata['source_after_startup_verified'] and metadata['owned_server_pid_verified']
            assert metadata['server_info_sha256']==sha(z.read(base+'server-info-raw.json')) and metadata['native_http_proof_sha256']==sha(parent_raw)
            identity=doc(base+'linux-identity.json');assert sha(z.read(base+'linux-identity.json'))==metadata['linux_identity_sha256']
            pins=doc(METHODS+'evidence/reference4k-source-pins.json');assert identity['source_pins_sha256']==sha(z.read(METHODS+'evidence/reference4k-source-pins.json'))
            assert identity['status']=='reference_fixed4k_installed_sources_and_owned_parent_verified' and identity['engine']=='sglang'
            assert identity['pid']==int(z.read(base+'server.pid').strip()) and identity['linux_proc_starttime_ticks']>0
            argv=identity['cmdline'];assert argv[0].startswith('/opt/cantorai5090/sglang/bin/') and CHECKPOINT in argv and argv[argv.index('--port')+1]=='18081' and argv[argv.index('--host')+1]=='127.0.0.1'
            assert sha(('\0'.join(argv)+'\0').encode())==identity['cmdline_sha256']
            assert set(identity['installed_sources'])==set(pins['linux_package_files']['sglang'])
            for name,row in identity['installed_sources'].items():assert row['sha256']==pins['linux_package_files']['sglang'][name]==sha(z.read(base+row['archived_copy']))
            assert identity['checkpoint_files']==pins['checkpoint_files'] and identity['checkpoint_revision']==metadata['checkpoint_revision']==pins['checkpoint_revision']
            for name,digest in pins['checkpoint_files'].items():assert sha(z.read('models/Qwen3.8-27B-NVFP4/'+name))==digest
            stops=[json.loads(line) for line in z.read(base+'stop-ledger/commands.log').splitlines() if line.startswith(b'{')]
            stops=[v for v in stops if v.get('signal')=='SIGTERM'];assert len(stops)==1 and stops[0]['pid']==identity['pid'] and stops[0]['cmdline_sha256']==identity['cmdline_sha256']
            report=doc(base+'client/report.json');assert report['status']=='failed' and report['engine']=='sglang' and report['mode']==w['mode'] and not report['performance_admitted']
            assert report['metadata_sha256']==sha(z.read(base+'metadata.json')) and report['native_http_proof_sha256']==sha(parent_raw)
            if w['mode']=='mtp':
                proof_names=[n for n in manifest['files'] if n.startswith('results/sglang-fixed4k-ordinary-') and n.endswith('-verified.json')];assert len(proof_names)==1
                raw=z.read(proof_names[0]);ordinary=json.loads(raw);assert sha(raw)==w['ordinary_proof_sha256'] and ordinary['status']=='reference_fixed4k_profile_replayed' and ordinary['mode']=='ordinary'
                ordinary_archive=args.proofs/(Path(proof_names[0]).name.removesuffix('-verified.json')+'-evidence.zip')
                assert sha(ordinary_archive.read_bytes())==ordinary['archive_sha256']==w['ordinary_archive_sha256']
                prior=zipfile.ZipFile(ordinary_archive);prior_base='results/'+ordinary_archive.name.removesuffix('-evidence.zip')+'/'
            greedy=doc(base+'client/greedy-controls/greedy/equivalence.json');assert greedy['status']=='passed' and greedy['engine']=='sglang' and len(greedy['cases'])==16
            assert greedy['mode']==('ordinary_capture' if w['mode']=='ordinary' else 'speculative_compare')
            for row in greedy['cases']:
                stem=base+'client/greedy-controls/greedy/'+row['key'];wire=z.read(stem+'-request.json');response=z.read(stem+'-response.json');body=json.loads(wire)
                assert sha(wire)==row['request_sha256'] and sha(response)==row['response_sha256']
                ids,completion=reference_tokens(json.loads(response),'sglang');assert ids==row['prompt_ids'] and completion==row['generated_ids']
                usage=json.loads(response)['usage'];assert usage['prompt_tokens']==len(ids) and usage['completion_tokens']==len(completion) and len(ids)==(41 if '-text-' in row['key'] else 104)
                body.pop('return_token_ids');body.pop('return_prompt_token_ids');assert sha(json.dumps(body,sort_keys=True,separators=(',',':')).encode())==row['semantic_request_sha256']
                if prior is not None:
                    previous=json.loads(prior.read(prior_base+'client/greedy-controls/greedy/equivalence.json'))
                    old=next(v for v in previous['cases'] if v['key']==row['key']);assert compare_pair(old,row) and row['exact_match']
            vlm=doc(base+'client/greedy-controls/correctness/oracle.json');assert len(vlm['cases'])==4 and vlm['grid']==[1,16,16]
            for row in vlm['cases']:
                stem=base+'client/greedy-controls/correctness/'+row['name'];wire=z.read(stem+'-request.json');raw=z.read(stem+'-response.json');response=json.loads(raw)
                assert sha(wire)==row['request_sha256'] and sha(raw)==row['response_sha256'] and sha(z.read(base+'client/greedy-controls/correctness/'+row['image_file']))==row['image_sha256']
                content=json.loads(wire)['messages'][0]['content'];assert content[1]['text']==row['question'] and sha(base64.b64decode(content[0]['image_url']['url'].split(',',1)[1],validate=True))==row['image_sha256']
                ids,completion=reference_tokens(response,'sglang');assert ids==row['prompt_ids'] and completion==row['reference_generated_ids'] and ids.count(248056)==64
                assert response['choices'][0]['message']['content'].strip()==row['reference_text'] and row['reference_text'].lower().rstrip('.')==row['expected_color']
            with tempfile.TemporaryDirectory(prefix='sglang4k-replay-',dir=args.scratch) as temporary:
                folder=Path(temporary).resolve();prefix=base+'client/mmmu-quality/'
                for name in manifest['files']:
                    if name.startswith(prefix):
                        target=folder/name.removeprefix(prefix);assert target.resolve().is_relative_to(folder);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
                screening=replay(args.corpus,folder);assert screening==doc(base+'client/mmmu-quality-replayed.json')==report['screening']
                assert screening['status']=='adapted_mmmu_http_quality_responses_replayed'
                if prior is not None:
                    _,rows,_=load_corpus(args.corpus)
                    for row in rows:
                        ids=json.loads((args.corpus/row['prompt_ids_file']).read_bytes());name='client/mmmu-quality/cases/'+row['id']+'/response.bin'
                        left=validate_response(prior.read(prior_base+name),'sglang','qwen38-27b-nvfp4',ids);right=validate_response(z.read(base+name),'sglang','qwen38-27b-nvfp4',ids);assert left['completion_token_ids']==right['completion_token_ids']
            # The failed request remains invalid under the unchanged128-ID gate.
            corpus_raw=z.read(METHODS+'evidence/fixed-length-4096-corpus.zip');assert sha(corpus_raw)==ARCHIVE_SHA256
            with zipfile.ZipFile(io.BytesIO(corpus_raw)) as corpus:
                raw=corpus.read('manifest.json');assert sha(raw)==MANIFEST_SHA256;frozen=json.loads(raw);row=frozen['rows'][0];key=row['key']
                root=base+'client/benchmark4k/controls/';request=z.read(root+key+'-request.json');raw=z.read(root+key+'-response.json');response=json.loads(raw)
                expected_prompt=json.loads(corpus.read(row['expected_prompt_ids_file']));body=json.loads(corpus.read(row['request_file']))
                assert request==json.dumps(control_request(body,'sglang'),separators=(',',':')).encode() and body['ignore_eos'] is True and body['max_tokens']==128
                choice=response['choices'][0];assert choice['prompt_token_ids']==expected_prompt and len(expected_prompt)==3968
                completion=choice['response_token_ids'];assert len(completion)==126 and all(type(v) is int and 0<=v<248320 for v in completion)
                assert choice['finish_reason']=='length' and response['usage']['prompt_tokens']==3968 and response['usage']['completion_tokens']==126 and response['usage']['total_tokens']==4094
                try:control_response(response,'sglang',expected_prompt)
                except ValueError as error:rejection=str(error)
                else:raise AssertionError('Unchanged strict128 gate incorrectly accepted126')
                controls=doc(root+'report.json');assert controls['status']=='failed' and controls['cases']==[]
                assert len([n for n in names if n.startswith(root) and n.endswith('-response.json')])==1
                assert not any(n.startswith(base+'client/benchmark4k/') and n.endswith('-events.jsonl') for n in names)
                failure={'key':key,'request_sha256':sha(request),'response_sha256':sha(raw),'prompt_ids_exact':True,'actual_prompt_ids':3968,'actual_completion_ids':126,'requested_completion_ids':128,'usage':response['usage'],'strict128_gate_rejection':rejection,'timed_streams':0}
        archived=sha(z.read(METHODS+'verify_sglang_fixed4k.py'))
        if prior is not None:prior.close()
    result={'status':'reference_fixed4k_failure_partial_controls_replayed','engine':'sglang','mode':w['mode'],'archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'screening':screening,'profiles':profiles,'failed_full4k_control':failure,'short_greedy_controls_replayed':16,'vlm_smoke_controls_replayed':4,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,'performance_admitted':False,'two_times_target_achieved':False,'scope':'Closed source/config/PID/lifecycle and actual raw request/response/control/SSE/metric/memory replay. Timed streams expose counts/text, no individual IDs reconstructed.805adapted screening and fourVLM controls are not broad/official quality. Strongest feasible tuning, repetitions, native comparison and2x remain open.'}
    target=args.proofs/(args.job+'-failure-controls-verified.json');data=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==data
    if not target.exists():target.write_bytes(data)
    print(json.dumps(result,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
