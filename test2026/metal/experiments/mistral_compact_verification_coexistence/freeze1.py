"""Freeze native C4 ownership and compact/verification coexistence qualifications."""
from pathlib import Path
import hashlib,json,sys
M=Path(__file__).resolve().parent;W=M.parents[2];R=M/'mistral-compact-verification-coexist1';L=M/'mistral-scheduled-c4-ownership2';F=M/'mistral-scheduled-c4-ownership1'
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
for p in [L/'audit1.json',R/'build-audit1.json',R/'qualification-audit1.json',R/'probe-audit1.json',R/'source-audit1.json',M/'mistral-compact-verification-audit-replay1/replay-proof1.json']:assert read(p)['passed'],str(p)
for n,h in read(R/'source-bindings1.json')['files_sha256'].items():assert sha(R/'source1'/n)==h,n
for mode,build in [('on','build-on1'),('off','build-default-off1')]:
 assert sha(R/build/'libgarnet.dylib')==read(R/'build-audit1.json')['library_sha256'][mode]
 assert 'GARNET_METAL_COMPACT_VERIFY_COEXISTENCE:BOOL='+('ON' if mode=='on' else 'OFF') in (R/build/'CMakeCache.txt').read_text()
 assert '100% tests passed out of 53' in (R/('suite-on1.stdout' if mode=='on' else 'suite-default-off1.stdout')).read_text()
sg=M/'mistral-fused-mlp-serving1';assert sha(sg/'provenance1.json')=='0ae9d716752330deee4f5d5ebc4eabf45825e5f03963ff0ad8fc14262d265a9a'
for n,h in read(sg/'provenance1.json')['files_sha256'].items():assert sha(sg/n)==h,n
save(F/'decision1.json',dict(status='correct_where_exercised_but_full_c4_decode_coverage_missing',all_shared_lifecycle_state_differences=0,output_id_differences=0,actual_decode_occupancies=[1,2,3],required_full_occupancy4_unmet=True,independent_audit_passed=False,superseded_by_changed_budget_fixture=str(L),no_comparator_execution=True,performance_measured=False,goal_complete=False))
save(L/'decision1.json',dict(status='native_c4t1_scheduler_lifecycle_qualified',previous_goal_turn_classification='progress: native scheduler span API and B1 HTTP checkpoint820af38ed',current_goal_turn_classification='progress: real C4 multi-owner lifecycle proof and native compact profile coexistence extension',summary=read(L/'audit1.json')['summary'],full_occupancy4_exercised=True,source_runtime_unchanged_from_prior_bridge=True,c4t4_qualified=False,performance_measured=False,no_comparator_execution=True,goal_complete=False))
a=read(R/'qualification-audit1.json');p=read(R/'probe-audit1.json')
save(R/'decision1.json',dict(status='default_off_native_b4t1_compact_verification_coexistence_qualified',native_option='GARNET_METAL_COMPACT_VERIFY_COEXISTENCE',option_default_on=False,native_on_tests=53,native_default_off_tests=53,preparation_orders=a['preparation_orders'],primary_model_full_state_digest_pairs=a['full_four_row_state_digest_pairs'],preparation_guard_state_digest_pairs=a['preparation_guard_state_digest_pairs'],primary_rejected_api_state_digest_pairs=a['rejected_api_state_digest_pairs'],b2_probe_full_state_digest_pairs=p['summary']['full_four_row_state_digest_pairs'],b2_probe_rejected_api_state_digest_pairs=p['summary']['rejected_call_state_digest_pairs'],all_output_ids_and_full_state_exact_current_ordinary=True,initial_full_batch_state_exact_prior_noncompact=True,compiled_owner_families=17,prepared_maximum_chunk512=True,actual_b1_routes=a['actual_profile_routes'],actual_b2_routes=p['summary']['actual_routes'],no_request_time_compilation=True,managed_new_preparation_rejected=True,unsupported_capacity_rejected=True,flag_off_combinations_rejected=True,shader_and_xmodel_operator_content_unchanged=True,prefix_retention_tested=False,tracking_fixture_identity_legacy_t128_label='Tracking-only fixture: no retained cache. Future prefix-retention profile identity must name actual prepared512 family and verifier geometry.',c4t1_qualified=True,c4t4_qualified=False,performance_measured=False,new_serving_metrics=False,production_promotion=False,no_comparator_execution=True,all_owned_engines_terminal=True,goal_complete=False,next_action='Implement explicit native FP32 selected-row affine/head profiles up to16 token rows for B4/T4; do not merely widen bounds or fall into FP16 MPP. Preserve native rollback, per-row budgets and compact owner views, then qualify real speculation/prefix lifecycle and matched C1/C4 serving.'))
roots=[F,L,R,M/'mistral-compact-verification-source-replay1',M/'mistral-compact-verification-audit-replay1'];frozen=[]
for root in roots:
 files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()};save(root/'provenance1.json',dict(scope='Experimental C4/T1 ownership or native compact-profile coexistence; no C4/T4 or new speed admission',files_sha256=files,all_owned_processes_terminal=True,no_comparator_execution=True,goal_complete=False))
 for p in [root]+list(root.rglob('*')):
  if not p.is_symlink():p.chmod(p.stat().st_mode&~0o222)
 frozen.append(dict(root=str(root),files=len(files),provenance_sha256=sha(root/'provenance1.json')))
p=M/'replay_compact_verification1.py';p.chmod(p.stat().st_mode&~0o222)
print(json.dumps(dict(passed=True,frozen=frozen),indent=2))
