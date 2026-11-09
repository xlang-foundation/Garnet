"""Bind successful and failed attempts, native source/build, qualification and timing."""
from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;M=R.parent;P=M/'mistral-causal-batch4-measure1';W=Path('/Users/shawn/CantorAI');G=W/'Garnet'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,v):
 p=R/n;assert not p.exists(),n;p.write_text(json.dumps(v,indent=2)+'\n')
assert all(read(R/n)['passed'] for n in ['build-audit1.json','helper-audit1.json','productive-audit1.json','source-audit1.json']) and read(P/'audit1.json')['passed']
measurement=read(P/'audit1.json')
save('decision1.json',dict(passed=True,decision='Retain as an experimental qualified B4 helper path; do not enable unconditional helper drafting as a production default.',reason='Native decode improved modestly, but paired aggregate ratios disagree and do not establish a consistent throughput benefit.',native_warm_medians=measurement['warm_medians'],paired_ratios=measurement['paired_ratios'],fixed_vllm_sglang_results_unchanged=True,matched_http_admitted=False,two_times_target_met=False,tokenizer_or_xmodel_operator_changes=False,ane_used=False,quality_suite_passed=False,next_work='Screen lower-cost causal proposals and inspect compiled prefill/memory/kernel profiles before another serving measurement.',goal_complete=False))
diff=subprocess.check_output(['/usr/bin/git','-C',str(G),'diff','--binary']);status=subprocess.check_output(['/usr/bin/git','-C',str(G),'status','--short'],text=True)
assert not subprocess.check_output(['/usr/bin/git','-C',str(G),'diff','--cached','--name-only'])
save('workspace-before-checkpoint1.json',dict(head=subprocess.check_output(['/usr/bin/git','-C',str(G),'rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['/usr/bin/git','-C',str(G),'branch','--show-current'],text=True).strip(),tracked_wip_diff_sha256=hashlib.sha256(diff).hexdigest(),status=status,untracked_preserved_sha256={n:sha(G/n) for n in ['test2026/metal/affine_decode_block_benchmark.cpp','test2026/metal/audit_affine_decode_block_benchmark.py']}))
root_files=sorted(p.name for p in R.iterdir() if p.is_file());measurement_files=sorted(p.name for p in P.iterdir() if p.is_file())
paths=[R/n for n in root_files]+[P/n for n in measurement_files]+[R/'build-on1'/n for n in ['libgarnet.dylib','CMakeCache.txt','CTestTestfile.cmake','Testing/Temporary/LastTest.log']]
save('provenance1.json',dict(passed=True,files_sha256={str(p):sha(p) for p in paths},root_files=root_files,measurement_root_files=measurement_files,base_native=str(M/'mistral-gqa16-value-native1'),qualified_source=str(R/'source1'),source_bindings_sha256=sha(R/'source-bindings1.json'),build_library_sha256=sha(R/'build-on1/libgarnet.dylib'),collection_serial=True,native_source_tree_or_model_hashing_inside_timed_workers=False,diagnostic_full_state_hashes_inside_timed_workers=False,no_comparator_execution=True,goal_complete=False))
print('Bound helper qualification,128-output ABBA collection and preserved workspace WIP.')
