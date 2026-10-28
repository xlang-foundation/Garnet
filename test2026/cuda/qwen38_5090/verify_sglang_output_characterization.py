"""Replay closed output/overlap/color characterization, never infer streamed IDs.

Nonstreaming controls contain actual IDs. Common reference streaming exposes
usage and text; those streams cannot attest their unobserved individual IDs.
"""
from pathlib import Path
import argparse,base64,hashlib,io,json,statistics,sys,tempfile,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from run_sglang_output_characterization import job_path
from reference_fixed4k_sglang_reserved_identity import validate,CHECKPOINT
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
    a.add_argument('--scratch',type=Path,required=True);args=a.parse_args()
    assert args.scratch.is_dir();job_path(Path('D:/CantorAI/5090-study-20261023'),args.job)
    archive=args.proofs/(args.job+'-evidence.zip');base='results/'+args.job+'/';profiles=[];prior=None;screening=None
    with zipfile.ZipFile(archive) as z:
        doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json');names=z.namelist()
        assert z.testzip() is None and len(names)==len(set(names))==len(manifest['files'])+1
        for name,digest in manifest['files'].items():assert sha(z.read(name))==digest
        w=doc(base+'workflow.json');praw=z.read(METHODS+'evidence/sglang-output-characterization-run-preparation.json');p=json.loads(praw)
        assert sha(praw)==w['preparation_sha256'] and p['status']=='sglang_output_characterization_prepared_unrun'
        assert w['engine']=='sglang' and w['mode']=='ordinary' and args.job.startswith('sglang-output-characterization-attempt')
        assert w['cleanup_finished'] and w['source_sha256']==w['source_after_sha256']==p['files']
        assert w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime_dependencies'] and not w['performance_admitted']
        for name,digest in p['files'].items():assert sha(z.read(name))==digest
        # Local replay helpers must match the versions that actually collected.
        for name in ('benchmark_stream.py','metrics.py','compare_http_runs.py','cohort_memory_replay.py','benchmark_fixed_length_reference.py','benchmark_fixed_length_native.py','reference_fixed4k_sglang_identity.py','prepare_vlm_reference.py','reference_mtp_equivalence.py','reference_fixed4k_sglang_reserved_identity.py','reference_concurrency_characterization.py','replay_reference_concurrency_characterization.py','reference_fixed4k_boundary.py','mmmu_http_quality.py','mmmu_quality_protocol.py','verify_mmmu_http_quality.py'):
            assert sha(Path(__file__).with_name(name).read_bytes())==sha(z.read(METHODS+name)),('Replay helper differs',name)
        parent='fixed-length-http-study-attempt1';parent_raw=z.read('results/'+parent+'-verified.json');native=json.loads(parent_raw)
        assert native['status']=='fixed_length_http_study_replayed' and native['native_fixed_4k_metrics_replayed']
        assert sha(parent_raw)==p['native_proof_sha256']==w['native_proof_sha256']
        assert native['archive_sha256']==p['native_archive_sha256']==w['native_archive_sha256']==manifest['native_archive_sha256']
        assert sha((args.proofs/(parent+'-evidence.zip')).read_bytes())==native['archive_sha256']
        passed=w['exit_code']==0
        if passed:
            assert w['status']=='sglang_output_characterization_collected_needs_replay' and type(w['server_exit_code']) is int
            for ledger in ('cpu-ledger','metadata-ledger','client-ledger','stop-ledger'):
                records=[json.loads(line) for line in z.read(base+ledger+'/commands.jsonl').splitlines()]
                assert len(records)==2 and records[0]['event']=='started' and records[1]['event']=='finished' and records[1]['exit_code']==0
            assert b'Ran 33 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
            metadata=doc(base+'metadata.json');info=doc(base+'server-info-raw.json');expected=validate(info,w['mode'])
            for key,value in expected.items():assert metadata[key]==value
            launcher='serve_sglang_reserved_fixed4k.sh' if w['mode']=='ordinary' else 'serve_sglang_reserved_mtp3_fixed4k.sh'
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
            context_pins=doc(METHODS+'evidence/sglang-context-limit-source-pins.json');context_identity=doc(base+'context-limit-identity.json')
            assert sha(z.read(base+'context-limit-identity.json'))==metadata['installed_context_limit_identity_sha256']
            assert context_identity['status']=='installed_context_limit_sources_verified' and context_identity['source_modified'] is False
            assert context_identity['pins_sha256']==sha(z.read(METHODS+'evidence/sglang-context-limit-source-pins.json'))
            assert context_identity['configured_engine_context_capacity']==4098 and context_identity['actual_workload_context_capacity']==4096
            assert set(context_identity['files'])==set(context_pins['files']) and len(context_pins['files'])==19
            for name,row in context_identity['files'].items():
                assert row['sha256']==context_pins['files'][name]==sha(z.read(base+row['archived_copy']))==sha(z.read(METHODS+'evidence/sglang-context-limit-source/'+name))
            prior_raw=z.read('results/sglang-fixed4k-ordinary-attempt2-partial-verified.json');prior_proof=json.loads(prior_raw)
            assert sha(prior_raw)==p['prior_partial_proof_sha256']==w['prior_partial_proof_sha256']
            assert prior_proof['status']=='reference_fixed4k_attempt2_failure_partial_evidence_replayed' and prior_proof['archive_sha256']==p['prior_archive_sha256']==w['prior_archive_sha256']
            prior_archive=args.proofs/'sglang-fixed4k-ordinary-attempt2-evidence.zip';assert sha(prior_archive.read_bytes())==prior_proof['archive_sha256']
            native_closed=w['closed_native_admission'];assert native_closed['job']=='bf16-batch8-model-admission-attempt1' and native_closed['cleanup_finished'] is True
            native_archive=args.proofs/(native_closed['job']+'-evidence.zip');assert sha(native_archive.read_bytes())==native_closed['archive_sha256']
            with zipfile.ZipFile(native_archive) as previous:
                closed=json.loads(previous.read('results/'+native_closed['job']+'/workflow.json'));assert closed==doc('results/'+native_closed['job']+'/workflow.json') and closed['exit_code']==native_closed['exit_code'] and closed['cleanup_finished'] is True and closed['library_unchanged'] is True
                assert closed['source_sha256']==closed['source_after_sha256'] and closed['runtime_dependencies']==closed['runtime_dependencies_after']
            from reference_concurrency_characterization import load_cases
            from replay_reference_concurrency_characterization import replay as replay_characterization
            def read_case(name):
                if name=='results/sglang-fixed4k-ordinary-attempt2-evidence.zip':return prior_archive.read_bytes()
                return z.read(name)
            groups=load_cases(read=read_case);report=doc(base+'client/report.json');assert report['metadata']==metadata
            client_manifest=doc(base+'client/file-manifest.json');assert {n.removeprefix(base+'client/') for n in manifest['files'] if n.startswith(base+'client/')}==set(client_manifest['files'])|{'file-manifest.json'}
            replayed=replay_characterization(lambda name:z.read(base+'client/'+name),groups)
            resource=doc(base+'resources/resource-monitor.json');samples=[json.loads(line) for line in z.read(base+'resources/resource-vram.jsonl').splitlines()];peak=max(v['used_bytes'] for v in samples)
            assert resource['child_exit_code']==0 and resource['gpu']['uuid']==GPU and resource['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
            profiles.append({'raw_characterization':replayed,'correctness_sampled_peak_adapter_used_bytes':peak})
        archived=sha(z.read(METHODS+'verify_sglang_output_characterization.py'))
        if prior is not None:prior.close()
    result={'status':'sglang_output_characterization_replayed' if passed else 'sglang_output_characterization_failure_preserved','engine':'sglang','mode':w['mode'],'configured_engine_context_capacity':4098,'actual_workload_context_capacity':4096,'engine_context_reserved_slots':2,'archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'screening':screening,'profiles':profiles,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,'performance_admitted':False,'two_times_target_achieved':False,'scope':'Closed owned source/config/PID/lifecycle and raw288request/135overlap-wave output/isolation replay. All text/ID variants retained; short color bursts do not require measurable decode intervals. No physical GPU batch, numerical cause/harmlessness/equivalence, broad quality or performance/2x admission.'}
    target=args.proofs/(args.job+'-verified.json');data=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==data
    if not target.exists():target.write_bytes(data)
    print(json.dumps(result,indent=2));return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
