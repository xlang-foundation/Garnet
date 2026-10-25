"""Freeze the reference controller only after the full native proof pair exists."""
from pathlib import Path
import argparse,hashlib,json,zipfile

sha=lambda raw:hashlib.sha256(raw).hexdigest()
METHODS=(
    'run_sglang_fixed4k.py','prepare_sglang_fixed4k_run.py',
    'prepare_sglang_fixed4k_metadata.py','reference_fixed4k_sglang_identity.py',
    'inspect_reference_fixed4k_linux.py','stop_reference_server.py',
    'probe_reference_fixed4k.py','serve_sglang_fixed4k.sh','serve_sglang_mtp3_fixed4k.sh',
    'run_reference_greedy_candidate.py','prepare_vlm_reference.py',
    'run_bf16_low_batch_native_build.py','run_logged.py',
    'monitor_native_probe.py','sample_vram.py','metrics.py','benchmark_stream.py',
    'reference_mtp_equivalence.py','benchmark_fixed_length_reference.py',
    'benchmark_fixed_context_reference.py','compare_http_runs.py',
    'test_reference_fixed4k_identity.py','test_sglang_fixed4k_owner.py',
    'test_fixed_length_reference.py','test_fixed_context_reference.py',
    'test_vllm_cache_control.py','test_reference_mtp_equivalence.py',
    'evidence/reference4k-source-pins.json',
    'evidence/reference4k-sglang-identity-fixture.json',
)
TEST_MODULES=tuple('test2026.cuda.qwen38_5090.'+name for name in (
    'test_reference_fixed4k_identity','test_sglang_fixed4k_owner',
    'test_fixed_length_reference','test_vllm_cache_control',
    'test_reference_mtp_equivalence',
))

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);a.add_argument('--plan',type=Path,required=True);args=a.parse_args()
    m=Path(__file__).resolve().parent;repo=m.parents[2];name='fixed-length-http-study-attempt1'
    required=[args.proofs/(name+suffix) for suffix in ('-verified.json','-remote-verified.json','-evidence.zip')]
    missing=[str(p) for p in required if not p.is_file()]
    if missing:
        print(json.dumps({'status':'sglang_fixed4k_native_proof_pair_missing_no_preparation','missing':missing}));return 2
    proof_raw=required[0].read_bytes();q=json.loads(proof_raw)
    assert q==json.loads(required[1].read_bytes()) and q['status']=='fixed_length_http_study_replayed' and q['native_fixed_4k_metrics_replayed']
    assert sha(required[2].read_bytes())==q['archive_sha256']
    with zipfile.ZipFile(required[2]) as z:
        w=json.loads(z.read('results/'+name+'/workflow.json'))
        assert w['exit_code']==0 and w['cleanup_finished'] and w['library_unchanged']
        assert w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after']
        files=dict(w['source_sha256']);runtime=w['runtime_dependencies']
    for method in METHODS:files['Garnet/test2026/cuda/qwen38_5090/'+method]=sha((m/method).read_bytes())
    pins=json.loads((m/'evidence/reference4k-source-pins.json').read_bytes())
    files.update({'models/Qwen3.8-27B-NVFP4/'+n:h for n,h in pins['checkpoint_files'].items()})
    for relative,digest in files.items():
        if relative.startswith('Garnet/'):
            assert sha((repo/relative.removeprefix('Garnet/')).read_bytes())==digest,('Local qualified source drift',relative)
    files['results/'+name+'-verified.json']=sha(proof_raw)
    scope='One exclusive owned SGLang0.5.22 ordinary or NEXTN3/EAGLE attempt; existing checkpoint/cache, context4096/BF16KV/F32SSM and disabled image/prefix caches. Real resolved server/PID/source checks, fourVLM/16greedy/805adapted screening,124actual nonstream ID controls then124identical frozen3968+128 HTTP streams. Common reference streamed IDs unavailable; exact same-engine text/counts checked. MTP requires closed replayed ordinary full-ID controls. SIGTERM only the verified owned parent, observe wrapper terminal and adapter idle, preserve all failures/raw bytes/source hashes. Whole-probe monitor is client/quality scope while cohort peaks are serving scope. Independent archive replay, strongest feasible reference tuning, broad quality, repetitions, INT4/4B and2x remain open.'
    p={'status':'sglang_fixed4k_run_prepared_unrun','files':files,'runtime_dependencies':runtime,
       'native_proof_sha256':sha(proof_raw),'native_archive_sha256':q['archive_sha256'],
       'test_modules':TEST_MODULES,'scope':scope}
    target=m/'evidence/sglang-fixed4k-run-preparation.json';data=json.dumps(p,indent=2).encode()
    assert not target.exists() or target.read_bytes()==data
    if not target.exists():target.write_bytes(data)
    uploads=[{'local':str(m/n),'target':'Garnet/test2026/cuda/qwen38_5090/'+n,'sha256':sha((m/n).read_bytes())} for n in METHODS]
    uploads.append({'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(data)})
    plan=json.dumps({'uploads':uploads,'scope':scope},indent=2).encode();assert not args.plan.exists() or args.plan.read_bytes()==plan
    if not args.plan.exists():args.plan.write_bytes(plan)
    print(json.dumps({'status':p['status'],'preparation_sha256':sha(data),'upload_targets':len(uploads),'pinned_study_files':len(files)}));return 0

if __name__=='__main__':raise SystemExit(main())
