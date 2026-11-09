"""Freeze bounded-context and split-score experimental evidence after terminal audits."""
from pathlib import Path
import hashlib,json,sys
R=Path(__file__).resolve().parent;M=R.parent;B=M/'mistral-bounded-draft-long1';V=M/'mistral-verification-split-score-screen1';N=M/'mistral-fused-q4-decode-native1';W=M.parents[2]
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 assert not p.exists(),str(p);p.write_text(json.dumps(d,indent=2)+'\n')
sys.path.insert(0,str(N/'tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes()
roots=[B,V,R,M/'mistral-verification-split-score-source-replay1',M/'mistral-verification-split-score-audit-replay1',M/'mistral-verification-split-score-audit-replay2']
assert all(not(p/'provenance1.json').exists() for p in roots)
for n,h in read(M/'mistral-half-attention-native1/source-audit7.json')['original_dirty_sha256'].items():assert sha(W/'Garnet'/n)==h,n
sg=M/'mistral-fused-mlp-serving1';assert sha(sg/'provenance1.json')=='0ae9d716752330deee4f5d5ebc4eabf45825e5f03963ff0ad8fc14262d265a9a'
for n,h in read(sg/'provenance1.json')['files_sha256'].items():assert sha(sg/n)==h,n
for root,names in [(B,['qualification-audit1.json','measurement-audit1.json','counter-audit1.json']),(V,['audit1.json']),(R,['build-audit1.json','qualification-audit1.json','measurement-audit1.json','source-audit1.json'])]:
 for name in names:assert read(root/name)['passed'],name
assert read(roots[-1]/'replay-proof1.json')['passed']
assert len(read(roots[-1]/'replay-proof1.json')['reports'])==6
for x in read(roots[-1]/'replay-proof1.json')['reports']:assert x['complete_report_exact']
for name,h in read(R/'source-bindings2.json')['files_sha256'].items():assert sha(R/'source2'/name)==h,name
for mode,build in [('on','build-on1'),('off','build-default-off1')]:
 assert sha(R/build/'libgarnet.dylib')==read(R/'build-audit1.json')['library_sha256'][mode]
 assert '100% tests passed out of 53' in (R/('suite-'+('on1' if mode=='on' else 'default-off1')+'.stdout')).read_text()
 cache=(R/build/'CMakeCache.txt').read_text();assert 'GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES:BOOL='+('ON' if mode=='on' else 'OFF') in cache
 assert 'GARNET_METAL_FUSED_Q4_DECODE:BOOL=OFF' in cache and 'GARNET_METAL_NATIVE_BFLOAT_BODY:BOOL=OFF' in cache
save(B/'decision1.json',dict(status='bounded_helper2k_context_native_stage_qualified',previous_goal_turn_classification='no progress: preceding response explained existing comparison; resumed with operator and native implementation evidence',target_prompt_unchanged=True,helper_base_tokens=221,helper_capacity=512,native_stage_gate=read(B/'measurement-audit1.json')['declared_native_stage_gate_passed'],measurements=read(B/'measurement-audit1.json')['comparisons'],counter_priority=dict(score_percent=read(B/'counter-audit1.json')['verification_score_percent'],value_percent=read(B/'counter-audit1.json')['verification_value_percent']),decision='Keep fixed helper-context policy experimental; integrate only after broader quality and serving lifecycle qualification.',serving_speed_claim=False,coding_quality_qualified=False,goal_complete=False))
save(V/'decision1.json',dict(status='same_word_split_score_component_gate_passed',audit_sha256=sha(V/'audit1.json'),component_gate=read(V/'audit1.json')['declared_component_gate_passed'],parent_word_pairs=read(V/'audit1.json')['parent_word_pairs'],no_additional_gpu_workspace=True,decision='Proceed to native model qualification; no serving speed claim.',goal_complete=False))
a=read(R/'measurement-audit1.json')
save(R/'decision1.json',dict(status='default_off_native_split_scores_correct_and_repeatable_stage_gain',source_audit_sha256=sha(R/'source-audit1.json'),operator_audit_sha256=sha(V/'audit1.json'),native_unit_tests_on=53,native_unit_tests_default_off=53,native_model_state_digest_pairs=260+260,decode_measurements=a['rows'],paired_ratios=a['comparisons'],declared_native_stage_gate_passed=a['declared_native_stage_gate_passed'],new_gpu_workspace_bytes=0,new_weight_copy_bytes=0,original_dirty_unchanged=True,selected_serving_baseline_unchanged=True,all_owned_processes_terminal=True,no_comparator_execution=True,default_option_on=False,prefix_cache_lifecycle_qualified_for_new_build=False,matched_serving_integrated=False,coding_quality_qualified=False,production_runtime_integrated=False,goal_complete=False,next_action='Integrate bounded speculation into serving with native request ownership and prefix lifecycle; measure matched C1/C4 against frozen references. Further target grouped verification QK/value reuse at long context.'))
result=[]
for root in roots:
 files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
 save(root/'provenance1.json',dict(scope='Experimental evidence; default off, no new serving/comparator speed admission or goal completion',files_sha256=files,all_owned_processes_terminal=True,no_comparator_execution=True,goal_complete=False))
 for p in [root]+list(root.rglob('*')):
  if not p.is_symlink():p.chmod(p.stat().st_mode&~0o222)
 result.append(dict(root=str(root),files=len(files),provenance_sha256=sha(root/'provenance1.json')))
print(json.dumps(dict(passed=True,frozen=result),indent=2))
