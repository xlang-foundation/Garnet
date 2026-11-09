"""Compile the helper's existing bounded prefill API at several upper bounds, retaining native KV ownership."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;M=R.parent;V=M/'mistral-fused-verify-gqa-profile1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert read(V/'qualification-audit1.json')['passed'] and read(V/'measurement-audit1.json')['passed']
s=(V/'b1_worker1.py').read_text();assert s.count('assert len(sys.argv)==6')==1
s=s.replace('assert len(sys.argv)==6',"assert len(sys.argv)==7\nhelper_prefill_maximum=int(sys.argv[6]);assert helper_prefill_maximum in (16,32,64,128,256)")
s=s.replace('gqa_values_enabled=value_enabled)', 'gqa_values_enabled=value_enabled,helper_prefill_maximum=helper_prefill_maximum)')
s=s.replace('offset=start;last=None;plans=[]','offset=start;last=None;plans=[];profile_samples=[]')
s=s.replace('  chunk=16\n', '  chunk=helper_prefill_maximum\n')
s=s.replace('while chunk<remaining and chunk<16:chunk*=2','while chunk<remaining and chunk<helper_prefill_maximum:chunk*=2')
s=s.replace('  if chunk==1:\n   last=draft_step', '  prefill_started=time.monotonic()\n  if chunk==1:\n   last=draft_step')
s=s.replace('   offset+=1;plans.append(1);continue', '''   if experiment=='qualify':profile_samples.append(dict(width=1,actual=1,offset=offset,tracked=draft.session.diagnostic_tracked_token_count(0,epoch),wall_seconds=time.monotonic()-prefill_started,gpu_ms=draft.session.gpu_ms,mpp=draft.session.mpp_affine_dispatches,indexed=draft.session.indexed_decode_affine_dispatches))
   offset+=1;plans.append(1);continue''')
s=s.replace('  offset+=length;plans.append(chunk)', '''  if experiment=='qualify':profile_samples.append(dict(width=chunk,actual=length,offset=offset,tracked=draft.session.diagnostic_tracked_token_count(0,epoch),wall_seconds=time.monotonic()-prefill_started,gpu_ms=draft.session.gpu_ms,mpp=draft.session.mpp_affine_dispatches,indexed=draft.session.indexed_decode_affine_dispatches))
  offset+=length;plans.append(chunk)''')
s=s.replace(' return last,plans',' return last,plans,profile_samples')
s=s.replace('predict,plans=tracked_prefill(prompt,cached,epoch,policy)','predict,plans,profile_samples=tracked_prefill(prompt,cached,epoch,policy)')
s=s.replace('prefill_chunks=plans,','prefill_chunks=plans,prefill_samples=profile_samples,')
s=s.replace('draft.enable_prefill(16);draft.prefill_compile_ms=0','draft.enable_prefill(helper_prefill_maximum);draft.prefill_compile_ms=0')
s=s.replace('for width in (2,4,8,16):draft._resident_prefill_graphs(width)','for width in (2,4,8,16,32,64,128,256):\n  if width<=helper_prefill_maximum:draft._resident_prefill_graphs(width)')
s=s.replace('prepared=dict(seconds=', 'prepared=dict(helper_prefill_maximum=helper_prefill_maximum,seconds=')
assert 'prefill_samples=profile_samples' in s and 'while chunk<remaining and chunk<16' not in s
(R/'b1_worker1.py').write_text(s);compile(s,'worker','exec');(R/'input1.json').write_bytes((V/'input1.json').read_bytes())
plan=dict(scope='Native real causal-helper profile upper bound; full target and helper model weights unchanged.',library=str(V/'build-on1'),source=str(V/'source1'),library_sha256=read(V/'build-audit1.json')['candidate_library_sha256'],source_bindings_sha256=sha(V/'source-bindings1.json'),worker_base_sha256=sha(V/'b1_worker1.py'),worker_sha256=sha(R/'b1_worker1.py'),profile_caps=[16,32,64,128,256],initial_qualification_caps=[16,256],target_profile_unchanged=True,helper_capacity=512,initial_actual_prompt_tokens=222,helper_kv_ownership='Existing native leased row and reconcile_dense_tokens; no host KV snapshots.',all_helper_graph_families_precompiled=True,no_xmodel_operator_changes=True,no_native_library_changes=True,no_comparator_execution=True,goal_complete=False)
(R/'plan1.json').write_text(json.dumps(plan,indent=2)+'\n');print('Prepared helper profiles16..256 using the existing native bounded API and compiled library.')
