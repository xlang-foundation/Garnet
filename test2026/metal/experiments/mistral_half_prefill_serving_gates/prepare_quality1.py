"""Freeze the original eight-task coding check for the half-prefill engine."""
import sys
from pathlib import Path
import hashlib,json,shutil
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
old=root.parent/'mistral-current-coding-quality1'
native=root.parent/'mistral-half-attention-native1'
load=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
assert load(root.parent/'mistral-half-prefix-http1/half-http-audit1.json')['passed']
frozen=load(old/'provenance1.json')['files_sha256']
tools=root/'tools1';assert not tools.exists();tools.mkdir()
for p in sorted((old/'tools1').iterdir()):
    if p.is_dir():continue
    assert sha(p)==frozen[str(p.relative_to(old))],str(p)
    shutil.copy2(p,tools/p.name)
for name in ['fixture1.json','profile1.json','run_quality1.py']:
    assert sha(old/name)==frozen[name];shutil.copy2(old/name,root/name)
worker=tools/'local_coder_batch_worker.py';s=worker.read_text()
def edit(a,b):
    global s
    assert s.count(a)==1,(a[:70],s.count(a));s=s.replace(a,b)
edit('    backend=garnet.backend_info()', '''    backend=garnet.backend_info()
    half_policy=profile.get('half_attention_policy',False)
    if type(half_policy) is not bool or half_policy!=('fp16_prefill_all_p16_gpupack_v1' in backend):raise ValueError('Half-prefill profile differs from compiled engine')
    if 'committed_kv_mps_fp32_v1' in backend:raise ValueError('Quality study excludes committed-MPS policy')''')
edit("                if attention_buckets:expected=profile['expected_chunk_attention_dispatches'] if len(ids)>batch and attention_valid<=8 else 0", "                if attention_buckets:expected=profile['expected_chunk_attention_dispatches'] if len(ids)>batch and attention_valid<=8 and not half_policy else 0")
edit('                if attention_buckets and attention_valid<=8:expected=0', '''                if attention_buckets and attention_valid<=8:expected=0
                if half_policy:
                    half_count=decoder.session.chunk_half_attention_dispatches
                    half_expected=profile['expected_chunk_attention_dispatches'] if len(ids)>batch else 0
                    emit(dict(event='native_chunk_half_attention',stage='prefill' if len(ids)>batch else 'decode',dispatches=half_count,active_tokens=attention_valid))
                    if half_count!=half_expected:raise ValueError('Half-prefill route coverage differs')
                    expected=0''')
edit('                expected_tail=expected if attention_valid<=8 else 0;expected_extent=0', '                expected_tail=expected if attention_valid<=8 and not half_policy else 0;expected_extent=0')
edit('        if packed:\n            decoder.session.prefill_set_packed_affine(True)', '''        half_scratch=decoder.session.chunk_half_attention_scratch_bytes
        if half_scratch!=(71303552 if half_policy else 0):raise ValueError('Half-prefill scratch differs')
        emit(dict(event='native_chunk_half_attention_prepared',enabled=half_policy,scratch_bytes=half_scratch))
        if packed:
            decoder.session.prefill_set_packed_affine(True)''')
compile(s,str(worker),'exec');worker.chmod(0o644);worker.write_text(s)
profile=load(root/'profile1.json');assert profile['prefix_cache'] is False
profile.update(profile_id='mistral-half-prefill-coding-quality1',half_attention_policy=True,scope='Eight unchanged coding tasks and strict evaluator, candidate half-prefill engine, no reference rerun. Collection validity is distinct from coding-quality success. No performance claim.')
(root/'profile1.json').chmod(0o644);(root/'profile1.json').write_text(json.dumps(profile,indent=2)+'\n')
oldb=load(old/'bindings1.json');nb=load(native/'model-bindings7.json');b=dict(nb)
b.update(library=nb['libraries']['on']['path'],library_sha256=nb['libraries']['on']['sha256'],worker_sha256=sha(worker),profile_sha256=sha(root/'profile1.json'),
    fixture_sha256=sha(root/'fixture1.json'),suite_sha256=oldb['suite_sha256'],evaluator_sha256=oldb['evaluator_sha256'],old_quality_provenance_sha256=sha(old/'provenance1.json'),scope=profile['scope'])
assert sha(tools/'suite.json')==b['suite_sha256'] and sha(tools/'evaluate_local_coder_quality.py')==b['evaluator_sha256']
(root/'bindings1.json').write_text(json.dumps(b,indent=2)+'\n')
print(json.dumps(dict(prepared=True,unchanged_tasks=8,worker_sha256=b['worker_sha256'],evaluator_unchanged=True)))
