"""Prepare four serial native engines for frozen2K/C1,C4 serving cells."""
from pathlib import Path
import hashlib,json,shutil,socket,sys
package=Path(__file__).resolve().parent
root=Path(sys.argv[1]).resolve()
root.mkdir(parents=True,exist_ok=False)
shutil.copyfile(package/'run1.py',root/'run1.py')
workspace=root.parents[3];m=root.parent
old=m/'mistral-qo-scheduled-http1';native=m/'mistral-half-attention-native1';quality=m/'mistral-half-coding-quality1'
load=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
for folder,digest in [('mistral-half-prefix-http1','895758ad0fef87a35b4d0749fb5179b824e964a7153f88fd85f12194fe42f7c1'),('mistral-half-prefix-pressure1','14e338f9565f86cf3f8b4e566fe6b85afa08db782723942f6b8d6fb1f2830e9f'),('mistral-half-coding-quality1','78b2a8d7b1d95c093cfa042f2377aee37a5c77f6ee3cb0ab7241ed1da574de4e')]:assert sha(m/folder/'provenance1.json')==digest
assert load(quality/'collection-audit1.json')['passed'] and not load(quality/'collection-audit1.json')['quality_gate_passed']
tools=root/'tools1';assert not tools.exists();tools.mkdir()
frozen=load(old/'provenance1.json')['files_sha256']
for p in sorted((old/'http-tools1').iterdir()):
    if not p.is_file() or p.suffix!='.py':continue
    assert sha(p)==frozen[str(p.relative_to(old))],p.name
    shutil.copy2(p,tools/p.name)
worker=tools/'local_coder_batch_worker.py';worker.chmod(0o644);shutil.copyfile(quality/'tools1/local_coder_batch_worker.py',worker)
assert all(token not in worker.read_text() for token in ['verification_f32_sha256','native_survivor_restore_witness','native_request_state_witness','prepare_gpu_counters'])
# Preserve cold/warm lifecycle and metric/client code. Generalize only the
# frozen serving-cell selection. All hashes/source copies move to preflight.
p=tools/'run_mistral_cold_prefix.py';text=p.read_text()
changes=[
 ("    args=parser.parse_args()","    parser.add_argument('--case-id',required=True,choices=['p2048-c1-o128','p2048-c4-o128'])\n    args=parser.parse_args()\n    concurrency=int(args.case_id.split('-c')[1].split('-')[0])"),
 ("    plan=plans['p2048-c4-o128'];path=Path(plan['cases'])","    plan=plans[args.case_id];path=Path(plan['cases'])"),
 ("    assert hashlib.sha256(path.read_bytes()).hexdigest()==plan['cases_sha256']","    assert str(path)==spec['frozen_case_file'] # SHA checked before the campaign and after terminal"),
 ("    originals=json.loads(path.read_text());assert len(originals)==4","    originals=json.loads(path.read_text());assert len(originals)==concurrency"),
 ("    coding=[dict(id='p2048-c4-o128-'+str(i)) for i in range(5)]","    coding=[dict(id=args.case_id+'-'+str(i)) for i in range(concurrency)]"),
 ("    bodies={coding[i]['id']:originals[i] for i in range(4)}","    bodies={coding[i]['id']:originals[i] for i in range(concurrency)}"),
 ("    replacement=copy.deepcopy(originals[0]);content=replacement['messages'][0]['content']\n    replacement['messages'][0]['content']=content.replace('job 0.','job 9.',1)\n    assert replacement['messages'][0]['content']!=content\n    bodies[coding[4]['id']]=replacement\n",''),
 ("enumerate(['A','B','C','D'])","enumerate(['A','B','C','D'][:concurrency])"),
 ("max_workers=4","max_workers=concurrency"),
 ("threading.Barrier(4)","threading.Barrier(concurrency)"),
 ("    frozen=args.output/'frozen';frozen.mkdir()\n    for name in ('run_mistral_cold_prefix.py','local_coder_stream_benchmark.py','run_local_coder_baseline.py'):\n        shutil.copyfile(Path(__file__).with_name(name),frozen/name)\n",''),
 ("        report['fresh_control_reference']='/Users/shawn/CantorAI/out/research/mistral7b-local-coder/mistral-long-prefix-budget1/http2/campaign/fresh'","        report['fresh_control_reference']=spec['saved_native_control']"),
]
for a,b in changes:assert text.count(a)==1,(a[:100],text.count(a));text=text.replace(a,b)
compile(text,str(p),'exec');p.chmod(0o644);p.write_text(text)
assert 'hashlib.sha256(' not in text
binding=load(native/'model-bindings7.json');library=binding['libraries']['on']['path']
original_manifest=m/'mistral-profile-memory1/admission1/manifest.json';manifest=load(original_manifest)
cells=root/'cases1';cells.mkdir();plans=[]
for entry in manifest['plans']:
    if entry['id'] not in ['p2048-c1-o128','p2048-c4-o128']:continue
    source=Path(entry['cases']);assert sha(source)==entry['cases_sha256']
    dest=cells/(entry['id']+'.json');shutil.copy2(source,dest)
    plans.append(dict(entry,cases=str(dest)))
assert len(plans)==2
(root/'serving-manifest1.json').write_text(json.dumps(dict(scope='Unchanged selected frozen serving cells; only local paths differ.',plans=plans),indent=2)+'\n')
shutil.copy2(m/'mistral-packed-http1/fixture1.json',root/'fixture1.json')
base=load(old/'wave-1-ordinary/profile.json');assert base['trace_stage_latency'] is False and base['prefix_entries']==base['prefix_host_bytes']==0 and not base['prefix_cache']
base.update(half_attention_policy=True,prefix_capture_policy='prefill_only_v1',scope='Serial fresh native all-half-prefill engines,2K/C1 and2K/C4,128 outputs, cold/warm1/warm2, no prefix retention or state/GPU profiling. Coding remains0/8; observational comparison against frozen Apple references, not production qualification.')
ports=[];held=[]
for i in range(4):
    sock=socket.socket();sock.bind(('127.0.0.1',0));held.append(sock);ports.append(sock.getsockname()[1])
waves=[]
for index,case in enumerate(['p2048-c1-o128','p2048-c4-o128','p2048-c4-o128','p2048-c1-o128']):
    label=f'wave-{index+1}-'+case;folder=root/label;folder.mkdir();profile=dict(base,profile_id='mistral-half-serving-'+label)
    (folder/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    spec=load(old/'wave-1-ordinary/engine.json');argv=spec['argv'];argv[0]=str(workspace/'out/research/qwen-local-coder/asset-env/bin/python');argv[2]=str(tools/'local_coder_batch_http.py')
    argv[argv.index('--log-root')+1]=str(folder/'server-template');argv[argv.index('--port')+1]=str(ports[index]);at=argv.index('--')
    argv[at+1:]=[binding['xlang'],'-B',str(worker),library,binding['source'],binding['model'],str(folder/'profile.json')]
    spec.update(engine='garnet-native-half-'+label,url='http://127.0.0.1:'+str(ports[index]),argv=argv,
        native_library_sha256=binding['libraries']['on']['sha256'],source_manifest_sha256=binding['source_audit_sha256'],profile_sha256=sha(folder/'profile.json'),
        source_commit='644b5bb6d7a4d3186b03a0dcfd22c57ac1edffc0',scope=base['scope'],
        precision_scope='FP32 canonical KV/decode, private half-QKV/P16 prefill, FP16 affine operands/coefficients and FP32 accumulators. References retain their frozen FP16 state policy; no universal numerical equivalence claim.',
        frozen_case_file=next(e['cases'] for e in plans if e['id']==case),saved_native_control=str(m/'mistral-profile-memory1/admission1/server-1/worker-stdout.jsonl'))
    (folder/'engine.json').write_text(json.dumps(spec,indent=2)+'\n')
    cmd=[str(workspace/'out/research/qwen-local-coder/asset-env/bin/python'),'-B',str(tools/'run_mistral_cold_prefix.py'),'--engine-spec',str(folder/'engine.json'),'--fixture',str(root/'fixture1.json'),'--serving-manifest',str(root/'serving-manifest1.json'),'--output',str(folder/'campaign'),'--case-id',case]
    waves.append(dict(label=label,case=case,command=cmd,engine_sha256=sha(folder/'engine.json'),profile_sha256=sha(folder/'profile.json')))
for sock in held:sock.close()
plan=dict(scope=base['scope'],waves=waves,rounds=['cold','warm1','warm2'],fresh_engines_per_cell=2,hardware_cold_claim=False,
    single_request_decode_cell='p2048-c1-o128',aggregate_cell='p2048-c4-o128',targets=dict(single_decode=2.,aggregate_throughput=2.,TTFT_ratio=.5),
    native_model_binding=str(native/'model-bindings7.json'),model_files_sha256=binding['model_files_sha256'],model=binding['model'],library=library,library_sha256=binding['libraries']['on']['sha256'],source=binding['source'],
    original_manifest_sha256=sha(original_manifest),original_collector_sha256=sha(old/'http-tools1/run_mistral_cold_prefix.py'),unprofiled_worker_sha256=sha(worker),
    frozen_reference_provenance=str(m/'mistral-repeated-reference-provenance1.json'),goal_complete=False)
(root/'plan1.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps(dict(prepared=True,waves=len(waves),serving_cells=[e['id'] for e in plans],no_reference_inference=True)))
