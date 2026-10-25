"""Replay a closed vLLM reference archive, never execute a model or infer IDs.

Nonstreaming controls contain actual IDs. Common reference streaming exposes
usage and text; those streams cannot attest their unobserved individual IDs.
"""
from pathlib import Path
import argparse,base64,hashlib,io,json,statistics,sys,tempfile,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from run_vllm_fixed4k import job_path
from reference_fixed4k_vllm_identity import validate,CHECKPOINT
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
        w=doc(base+'workflow.json');praw=z.read(METHODS+'evidence/vllm-fixed4k-run-preparation.json');p=json.loads(praw)
        assert sha(praw)==w['preparation_sha256'] and p['status']=='vllm_fixed4k_run_prepared_unrun'
        assert w['engine']=='vllm' and w['mode']=='ordinary' and args.job.startswith('vllm-fixed4k-'+w['mode']+'-')
        assert w['cleanup_finished'] and w['source_sha256']==w['source_after_sha256']==p['files']
        assert w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime_dependencies'] and not w['performance_admitted']
        for name,digest in p['files'].items():assert sha(z.read(name))==digest
        # Local replay helpers must match the versions that actually collected.
        for name in ('benchmark_stream.py','metrics.py','compare_http_runs.py','cohort_memory_replay.py','benchmark_fixed_length_reference.py','benchmark_fixed_length_native.py','reference_fixed4k_vllm_identity.py','prepare_vlm_reference.py','reference_mtp_equivalence.py','mmmu_http_quality.py','mmmu_quality_protocol.py','verify_mmmu_http_quality.py','reference_fixed4k_vllm_boundary.py','vllm_cache_control.py'):
            assert sha(Path(__file__).with_name(name).read_bytes())==sha(z.read(METHODS+name)),('Replay helper differs',name)
        parent='fixed-length-http-study-attempt1';parent_raw=z.read('results/'+parent+'-verified.json');native=json.loads(parent_raw)
        assert native['status']=='fixed_length_http_study_replayed' and native['native_fixed_4k_metrics_replayed']
        assert sha(parent_raw)==p['native_proof_sha256']==w['native_proof_sha256']
        assert native['archive_sha256']==p['native_archive_sha256']==w['native_archive_sha256']==manifest['native_archive_sha256']
        assert sha((args.proofs/(parent+'-evidence.zip')).read_bytes())==native['archive_sha256']
        for parent_name,parent_row in p['parents'].items():
            parent_raw=z.read('results/'+parent_name+'-verified.json');parent_proof=json.loads(parent_raw)
            assert parent_raw==(args.proofs/(parent_name+'-verified.json')).read_bytes() and sha(parent_raw)==parent_row['proof_sha256']
            assert parent_proof['status']==parent_row['status'] and parent_proof['archive_sha256']==parent_row['archive_sha256']==sha((args.proofs/(parent_name+'-evidence.zip')).read_bytes())
        passed=w['exit_code']==0
        if passed:
            assert w['status']=='reference_fixed4k_collected_needs_replay' and type(w['server_exit_code']) is int
            for ledger in ('cpu-ledger','metadata-ledger','client-ledger','stop-ledger'):
                records=[json.loads(line) for line in z.read(base+ledger+'/commands.jsonl').splitlines()]
                assert len(records)==2 and records[0]['event']=='started' and records[1]['event']=='finished' and records[1]['exit_code']==0
            assert ('Ran '+str(p['expected_cpu_tests'])+' tests').encode() in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
            metadata=doc(base+'metadata.json');info=doc(base+'server-info-raw.json');expected=validate(info,doc(base+'server-version-raw.json'),z.read(base+'server.log').decode('utf8','replace'),w['mode'])
            for key,value in expected.items():assert metadata[key]==value
            launcher='serve_vllm_fixed4k.sh' if w['mode']=='ordinary' else 'serve_vllm_mtp3_fixed4k.sh'
            assert metadata['launcher_sha256']==sha(z.read(METHODS+launcher)) and metadata['source_after_startup_verified'] and metadata['owned_server_pid_verified']
            assert metadata['server_info_sha256']==sha(z.read(base+'server-info-raw.json')) and metadata['native_http_proof_sha256']==sha(parent_raw)
            identity=doc(base+'linux-identity.json');assert sha(z.read(base+'linux-identity.json'))==metadata['linux_identity_sha256']
            pins=doc(METHODS+'evidence/reference4k-source-pins.json');assert identity['source_pins_sha256']==sha(z.read(METHODS+'evidence/reference4k-source-pins.json'))
            assert identity['status']=='reference_fixed4k_installed_sources_and_owned_parent_verified' and identity['engine']=='vllm'
            assert identity['pid']==int(z.read(base+'server.pid').strip()) and identity['linux_proc_starttime_ticks']>0
            argv=identity['cmdline'];assert argv[0].startswith('/opt/cantorai5090/vllm/bin/') and CHECKPOINT in argv and argv[argv.index('--port')+1]=='18080' and argv[argv.index('--host')+1]=='127.0.0.1'
            assert sha(('\0'.join(argv)+'\0').encode())==identity['cmdline_sha256']
            assert set(identity['installed_sources'])==set(pins['linux_package_files']['vllm'])
            for name,row in identity['installed_sources'].items():assert row['sha256']==pins['linux_package_files']['vllm'][name]==sha(z.read(base+row['archived_copy']))
            assert identity['checkpoint_files']==pins['checkpoint_files'] and identity['checkpoint_revision']==metadata['checkpoint_revision']==pins['checkpoint_revision']
            for name,digest in pins['checkpoint_files'].items():assert sha(z.read('models/Qwen3.8-27B-NVFP4/'+name))==digest
            stops=[json.loads(line) for line in z.read(base+'stop-ledger/commands.log').splitlines() if line.startswith(b'{')]
            stops=[v for v in stops if v.get('signal')=='SIGTERM'];assert len(stops)==1 and stops[0]['pid']==identity['pid'] and stops[0]['cmdline_sha256']==identity['cmdline_sha256']
            assert metadata['server_version_sha256']==sha(z.read(base+'server-version-raw.json'))
            config_identity=doc(base+'vllm-config-identity.json');schema=doc(METHODS+'evidence/vllm-fixed4k-config-schema-attempt1.json')
            assert sha(z.read(base+'vllm-config-identity.json'))==metadata['installed_config_identity_sha256']
            assert config_identity['status']=='installed_vllm_fixed4k_config_sources_verified' and config_identity['source_modified'] is config_identity['gpu_executed'] is False
            assert config_identity['pins_sha256']==sha(z.read(METHODS+'evidence/vllm-fixed4k-config-schema-attempt1.json')) and len(config_identity['files'])==5
            assert set(config_identity['files'])==set(schema['files'])
            for name,row in config_identity['files'].items():assert row['sha256']==schema['files'][name]['sha256']==sha(z.read(base+row['archived_copy']))
            report=doc(base+'client/report.json');assert report['status']=='reference_fixed4k_profile_collected' and report['engine']=='vllm' and report['mode']==w['mode'] and not report['performance_admitted']
            assert report['metadata_sha256']==sha(z.read(base+'metadata.json')) and report['native_http_proof_sha256']==sha(parent_raw)
            if w['mode']=='mtp':
                proof_names=[n for n in manifest['files'] if n.startswith('results/vllm-fixed4k-ordinary-') and n.endswith('-verified.json')];assert len(proof_names)==1
                raw=z.read(proof_names[0]);ordinary=json.loads(raw);assert sha(raw)==w['ordinary_proof_sha256'] and ordinary['status']=='reference_fixed4k_profile_replayed' and ordinary['mode']=='ordinary'
                ordinary_archive=args.proofs/(Path(proof_names[0]).name.removesuffix('-verified.json')+'-evidence.zip')
                assert sha(ordinary_archive.read_bytes())==ordinary['archive_sha256']==w['ordinary_archive_sha256']
                prior=zipfile.ZipFile(ordinary_archive);prior_base='results/'+ordinary_archive.name.removesuffix('-evidence.zip')+'/'
            from reference_fixed4k_vllm_boundary import replay as replay_boundary
            with tempfile.TemporaryDirectory(prefix='vllm4k-boundary-replay-',dir=args.scratch) as temporary:
                folder=Path(temporary).resolve();boundary_prefix=base+'client/boundary-controls/'
                for name in manifest['files']:
                    if name.startswith(boundary_prefix):
                        target=folder/name.removeprefix(boundary_prefix);assert target.resolve().is_relative_to(folder);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
                assert replay_boundary(Path(__file__).with_name('evidence')/'fixed-length-4096-corpus.zip',folder)==report['early_full4k_controls']
            greedy=doc(base+'client/greedy-controls/greedy/equivalence.json');assert greedy['status']=='passed' and greedy['engine']=='vllm' and len(greedy['cases'])==16
            assert greedy['mode']==('ordinary_capture' if w['mode']=='ordinary' else 'speculative_compare')
            for row in greedy['cases']:
                stem=base+'client/greedy-controls/greedy/'+row['key'];wire=z.read(stem+'-request.json');response=z.read(stem+'-response.json');body=json.loads(wire)
                assert sha(wire)==row['request_sha256'] and sha(response)==row['response_sha256']
                ids,completion=reference_tokens(json.loads(response),'vllm');assert ids==row['prompt_ids'] and completion==row['generated_ids']
                usage=json.loads(response)['usage'];assert usage['prompt_tokens']==len(ids) and usage['completion_tokens']==len(completion) and len(ids)==(41 if '-text-' in row['key'] else 104)
                assert 'return_prompt_token_ids' not in body;body.pop('return_token_ids');assert sha(json.dumps(body,sort_keys=True,separators=(',',':')).encode())==row['semantic_request_sha256']
                if prior is not None:
                    previous=json.loads(prior.read(prior_base+'client/greedy-controls/greedy/equivalence.json'))
                    old=next(v for v in previous['cases'] if v['key']==row['key']);assert compare_pair(old,row) and row['exact_match']
            vlm=doc(base+'client/greedy-controls/correctness/oracle.json');assert len(vlm['cases'])==4 and vlm['grid']==[1,16,16]
            for row in vlm['cases']:
                stem=base+'client/greedy-controls/correctness/'+row['name'];wire=z.read(stem+'-request.json');raw=z.read(stem+'-response.json');response=json.loads(raw)
                assert sha(wire)==row['request_sha256'] and sha(raw)==row['response_sha256'] and sha(z.read(base+'client/greedy-controls/correctness/'+row['image_file']))==row['image_sha256']
                content=json.loads(wire)['messages'][0]['content'];assert content[1]['text']==row['question'] and sha(base64.b64decode(content[0]['image_url']['url'].split(',',1)[1],validate=True))==row['image_sha256']
                ids,completion=reference_tokens(response,'vllm');assert ids==row['prompt_ids'] and completion==row['reference_generated_ids'] and ids.count(248056)==64
                assert response['choices'][0]['message']['content'].strip()==row['reference_text'] and row['reference_text'].lower().rstrip('.')==row['expected_color']
            with tempfile.TemporaryDirectory(prefix='vllm4k-replay-',dir=args.scratch) as temporary:
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
                        left=validate_response(prior.read(prior_base+name),'vllm','qwen38-27b-nvfp4',ids);right=validate_response(z.read(base+name),'vllm','qwen38-27b-nvfp4',ids);assert left['completion_token_ids']==right['completion_token_ids']
            corpus_raw=z.read(METHODS+'evidence/fixed-length-4096-corpus.zip');assert sha(corpus_raw)==ARCHIVE_SHA256
            with zipfile.ZipFile(io.BytesIO(corpus_raw)) as corpus:
                manifest_raw=corpus.read('manifest.json');assert sha(manifest_raw)==MANIFEST_SHA256;frozen=json.loads(manifest_raw)
                root=base+'client/benchmark4k/';benchmarks=[n for n in manifest['files'] if n.startswith(root) and n.endswith('-manifest.json')];assert len(benchmarks)==1;benchmark=doc(benchmarks[0]);run=benchmark['run_id']
                assert benchmark['status']=='completed' and benchmark['engine']=='vllm' and benchmark['engine_metadata']==metadata and benchmark['engine_metadata_sha256']==sha(z.read(base+'metadata.json'))
                assert benchmark['corpus_archive_sha256']==ARCHIVE_SHA256 and benchmark['corpus_manifest_sha256']==MANIFEST_SHA256 and benchmark['actual_input_tokens_required']==3968 and benchmark['actual_completion_tokens_required']==128 and benchmark['gpu']['uuid']==GPU
                assert benchmark['streamed_ids_observed'] is False and not benchmark['sampling_errors'] and not benchmark['performance_admitted']
                from vllm_cache_control import parse_metrics
                reset=doc(root+'cache-control/reset.json');assert reset['status']=='reset_acknowledged' and reset==benchmark['image_cache_reset']
                assert reset['idle_before']['running']==reset['idle_before']['waiting']==0
                assert [(r['endpoint'],r['status']) for r in reset['reset_responses']]==[('/reset_mm_cache',200),('/reset_encoder_cache',200)]
                before=parse_metrics(z.read(root+'cache-control/after-reset.prom'));after=parse_metrics(z.read(root+'cache-control/after-run.prom'))
                assert before==reset['after_reset'] and before['running']==before['waiting']==after['running']==after['waiting']==0
                assert before['process_start']==after['process_start']==reset['idle_before']['process_start']
                counters=doc(root+'cache-control/counter-delta.json');assert counters==benchmark['image_cache_counter_delta']
                assert counters['before']==before and counters['after']==after and counters['query_delta']==after['queries']-before['queries']>0 and counters['hit_delta']==after['hits']-before['hits']==0
                control_report=doc(root+'controls/report.json');assert control_report['status']=='all124_nonstream_id_controls_collected' and len(control_report['cases'])==124 and len(frozen['rows'])==124
                assert control_report['engine']=='vllm' and control_report['streamed_ids_observed'] is False
                assert benchmark['nonstream_controls_sha256']==sha(z.read(root+'controls/report.json'))
                actual_keys=set()
                for row,record in zip(frozen['rows'],control_report['cases']):
                    key=row['key'];body=json.loads(corpus.read(row['request_file']));ids=json.loads(corpus.read(row['expected_prompt_ids_file']));assert len(ids)==3968 and body['ignore_eos'] is True and record['key']==key
                    wire=z.read(root+'controls/'+key+'-request.json');raw=z.read(root+'controls/'+key+'-response.json')
                    assert wire==json.dumps(control_request(body,'vllm'),separators=(',',':')).encode() and sha(wire)==record['request_sha256'] and sha(raw)==record['response_sha256']
                    assert record['prompt_ids_exact'] is True and record['output_tokens']==128 and record['ordinary_ids_exact'] is (w['mode']=='mtp')
                    control=control_response(json.loads(raw),'vllm',ids)
                    if prior is not None:
                        previous=prior.read(prior_base+'client/benchmark4k/controls/'+key+'-response.json');paired_control(control,control_response(json.loads(previous),'vllm',ids))
                    replay_stream(lambda n:z.read(root+n),run+'-'+key,body,control);actual_keys.add(key)
                stream_names=[n for n in manifest['files'] if n.startswith(root) and n.endswith('-events.jsonl')];assert len(stream_names)==len(actual_keys)==124
                fresh=verify_run(benchmark,lambda n:z.read(root+n));assert len(fresh['cohorts'])==8 and fresh['transport']=={'scheme':'http','hostname':'127.0.0.1','path':'/'}
                assert [(c['case'],c['concurrency']) for c in benchmark['cohorts']]==[(c['case'],c['concurrency']) for c in frozen['cohorts']]
                samples=[json.loads(line) for line in z.read(root+run+'-vram.jsonl').splitlines()];memory={}
                for saved,cohort in zip(benchmark['cohorts'],frozen['cohorts']):
                    assert saved['request_keys']==[run+'-'+key for key in cohort['measured']] and saved['prompt_tokens']==[3968]*8 and saved['generated_tokens']==1024
                    phase=saved['case']+'-c'+str(saved['concurrency']);requests=[doc(root+key+'-result.json') for key in saved['request_keys']]
                    memory[phase]=replay_cohort_memory(samples,phase,saved,requests)
                resource=doc(base+'resources/resource-monitor.json');samples=[json.loads(line) for line in z.read(base+'resources/resource-vram.jsonl').splitlines()]
                peak=max(v['used_bytes'] for v in samples);assert resource['child_exit_code']==0 and resource['gpu']['uuid']==GPU and resource['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
                profiles.append({'raw_replay':fresh,'sampled_memory':memory,'whole_client_and_quality_sampled_peak_adapter_bytes':peak,'actual_nonstream_id_controls':124,'actual_timed_and_warmup_streams':124,'streamed_individual_ids_observed':False})
        archived=sha(z.read(METHODS+'verify_vllm_fixed4k.py'));assert archived==sha(Path(__file__).read_bytes())
        if prior is not None:prior.close()
    result={'status':'reference_fixed4k_profile_replayed' if passed else 'reference_fixed4k_failure_preserved','engine':'vllm','mode':w['mode'],'archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'screening':screening,'profiles':profiles,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,'performance_admitted':False,'two_times_target_achieved':False,'scope':'Closed source/config/PID/lifecycle and actual raw request/response/control/SSE/metric/memory replay. Timed streams expose counts/text, no individual IDs reconstructed.805adapted screening and fourVLM controls are not broad/official quality. Strongest feasible tuning, repetitions, native comparison and2x remain open.'}
    target=args.proofs/(args.job+'-verified.json');data=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==data
    if not target.exists():target.write_bytes(data)
    print(json.dumps({'status':result['status'],'engine':result['engine'],'archive_sha256':result['archive_sha256'],'files_verified':result['files_verified'],'proof_sha256':sha(data),'profiles':len(profiles)}));return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
