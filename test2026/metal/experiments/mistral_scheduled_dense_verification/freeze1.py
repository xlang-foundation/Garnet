"""Freeze default-off native scheduler bridge and diagnostic B1 serving evidence."""
from pathlib import Path
import hashlib,json,sys
M=Path(__file__).resolve().parent;W=M.parents[2];C=M/'mistral-scheduled-dense-verification1';H=M/'mistral-scheduled-dense-http1';O=M/'mistral-split-score-ownership1'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):assert not p.exists(),str(p);p.write_text(json.dumps(d,indent=2)+'\n')
sys.path.insert(0,str(M/'mistral-fused-q4-decode-native1/tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes()
for n,h in read(M/'mistral-half-attention-native1/source-audit7.json')['original_dirty_sha256'].items():assert sha(W/'Garnet'/n)==h,n
for p in [O/'audit2.json',C/'build-audit1.json',C/'qualification-audit2.json',C/'source-audit1.json',H/'audit2.json',M/'mistral-scheduled-dense-audit-replay1/replay-proof1.json']:assert read(p)['passed'],str(p)
for n,h in read(C/'source-bindings1.json')['files_sha256'].items():assert sha(C/'source1'/n)==h,n
for mode,build in [('on','build-on1'),('off','build-default-off1')]:
 assert sha(C/build/'libgarnet.dylib')==read(C/'build-audit1.json')['library_sha256'][mode]
 assert 'GARNET_METAL_DENSE_VERIFY_SCHEDULER:BOOL='+('ON' if mode=='on' else 'OFF') in (C/build/'CMakeCache.txt').read_text()
 assert '100% tests passed out of 53' in (C/('suite-on1.stdout' if mode=='on' else 'suite-default-off1.stdout')).read_text()
sg=M/'mistral-fused-mlp-serving1';assert sha(sg/'provenance1.json')=='0ae9d716752330deee4f5d5ebc4eabf45825e5f03963ff0ad8fc14262d265a9a'
for n,h in read(sg/'provenance1.json')['files_sha256'].items():assert sha(sg/n)==h,n
save(O/'decision1.json',dict(status='new_b1t4_abort_foreign_writer_recovery_passed',native_binary_cases=24,native_rejections=169,native_driver_count_expectation_repaired_without_inference_rerun=True,scope='Native unit ownership proof; no speed admission',goal_complete=False))
save(C/'decision1.json',dict(status='native_scheduled_dense_verification_correct_for_qualified_b1_profile',previous_goal_turn_classification='progress: split kernel/model measurements and checkpoint8408a0641',current_goal_turn_classification='progress: native scheduler target verification/span commit, ownership qualification and real HTTP diagnostics',native_api='continuous_scheduler.verify_decode_json(hidden,positions,slots,active,tokens,stops)',native_owns=['known token','remaining output budget','row epoch and slots','verification/accepted rollback','token ledger proof','prefix capture then terminal release','scheduler multi-token commit'],native_units_on=53,native_units_default_off=53,full_current_ordinary_state_digest_pairs=65,native_output_budget129_for_128_token_witness=True,witness_cancelled_after_capture=True,separate_actual_budget2_terminal_release_proved=True,native_api_rejections=12,audit_negative_controls=12,physical_rows_limit4_unchanged=True,c4t4_qualified=False,compact_prefill_verification_qualified=False,prefix_lifecycle_for_new_bridge_qualified=False,default_option_on=False,production_runtime_integrated=False,goal_complete=False))
a=read(H/'audit2.json');save(H/'decision1.json',dict(status='diagnostic_b1_scheduled_speculation_http_qualified_no_full_profile_promotion',requests=6,output_ids_exact_native=768,native_target_calls=a['native_target_calls'],split_layer_dispatches=a['split_layer_dispatches'],warmed_median=a['warmed_median'],diagnostic_ratios_vs_frozen_references=a['diagnostic_ratios_vs_frozen_references'],active_rows=1,reference_max_sequences=4,maximum_concurrency_profiles_match=False,c4_measured=False,http_full_kv_hidden_observed=False,separate_native_state_gate=str(C/'qualification-audit2.json'),prefix_cache_enabled=False,selected_full_serving_comparison_promoted=False,no_comparator_execution=True,all_owned_engines_terminal=True,coding_quality_qualified=False,goal_complete=False,next_action='Qualify C4 native scheduled verification, compact owner prefill coexistence and up-to16 verification token rows before C4 speculation. Add new-build returning/append/late/early prefix lifecycle; optimize larger prefill profiles and shared GQA KV reads.'))
roots=[O,C,H,M/'mistral-scheduled-dense-source-replay1',M/'mistral-scheduled-dense-audit-replay1'];results=[]
for root in roots:
 files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()};save(root/'provenance1.json',dict(scope='Experimental native ownership/managed verification or B1 transport diagnostics; default off, no full C4/profile/goal admission',files_sha256=files,all_owned_processes_terminal=True,no_comparator_execution=True,goal_complete=False))
 for p in [root]+list(root.rglob('*')):
  if not p.is_symlink():p.chmod(p.stat().st_mode&~0o222)
 results.append(dict(root=str(root),files=len(files),provenance_sha256=sha(root/'provenance1.json')))
for name in ['replay_scheduled_dense_checkpoint1.py','prepare_scheduled_http1.py']:p=M/name;p.chmod(p.stat().st_mode&~0o222)
print(json.dumps(dict(passed=True,frozen=results),indent=2))
