"""Freeze completed operator/native evidence without changing symlink targets."""
from pathlib import Path
import hashlib,json,stat
R=Path(__file__).resolve().parent;P=R.parent/'mistral-tensor-view-prefill-screen1'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 assert not p.exists();p.write_text(json.dumps(d,indent=2)+'\n')
for n in ['qualification-audit1.json','measurement-audit1.json','source-audit1.json','patch-audit1.json','replay-audit1.json']:assert read(R/n)['passed']
for n in ['source-audit2.json','replay-audit2.json','timing-audit3.json']:assert read(P/n)['passed']
meas=read(R/'measurement-audit1.json');summaries=[]
for c in meas['cohorts']:
 summaries.append(dict(order=c['order'],disabled_first_prediction_seconds=c['disabled']['native_first_prediction_seconds'],enabled_first_prediction_seconds=c['enabled']['native_first_prediction_seconds'],first_prediction_ratio=c['disabled']['native_first_prediction_seconds']/c['enabled']['native_first_prediction_seconds'],prefill_burst_ratio=c['disabled']['burst_seconds']/c['enabled']['burst_seconds']))
save(R/'decision1.json',dict(passed=True,decision='Retain as a default-off bounded native candidate. Exact full-model/lifecycle qualification passed; native prefill gain is small and the 2x target remains unmet. No production-default or HTTP admission.',native_tests_per_build=56,native_builds=['on','default-off'],exact_qualification_output_ids=494,full_state_digest_pairs=40820,measured_first_prediction_ids=96,measured_bursts=16,warmup_bursts=8,cohorts=summaries,operator_down_ratios=[1.0677809027337752,1.072527051536475],extra_gpu_buffers=0,xmodel_operator_changes=False,reference_benchmarks_rerun=False,http_comparison_unchanged=True,no_ane_claim=True,next_step='Screen direct tensor operands with a larger row tile to reuse each unpacked weight over more actual token rows, before any new model integration. Keep current lower-cost fallback for partial profiles.',goal_complete=False))
save(P/'decision1.json',dict(passed=True,decision='Reject global per-call half-weight expansion and alternate threadgroup ordering. Existing bounded shared Q4 tile plus direct tensor operands shows modest component gains and justified native qualification.',trained_fixture_files=10,full_parent_comparison_words=94372480,fp64_samples=1280,unchanged_numerical_bound=True,writer_map_qualified=False,model_or_http_speed_claim=False,no_comparator_execution=True,goal_complete=False))
results={}
for root in [P,R]:
 files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
 provenance=dict(passed=True,root=str(root),files_sha256=files,all_owned_gpu_processes_terminal=True,no_comparator_execution=True,goal_complete=False)
 save(root/'provenance1.json',provenance);results[str(root)]=dict(files=len(files),provenance_sha256=sha(root/'provenance1.json'))
 for p in sorted(root.rglob('*'),key=lambda p:len(p.parts),reverse=True):
  if not p.is_symlink():p.chmod(p.stat().st_mode&~0o222)
 root.chmod(root.stat().st_mode&~0o222)
save(R.parent/'mistral-down-tensor-freeze1.json',dict(passed=True,roots=results,symlink_targets_not_chmodded=True,goal_complete=False));print(json.dumps(results,indent=2))
