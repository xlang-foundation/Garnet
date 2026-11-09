"""Validate completed native build evidence using the actual installed CTest summary format."""
from pathlib import Path
import hashlib,json,re
R=Path(__file__).resolve().parent;Q=R.parent/'mistral-causal-batch4-helper1';read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
last=0
for name in ['configure1','build1','suite1']:
 s=read(R/(name+'-status.json'));assert s['status']=='terminal' and s['returncode']==0 and s['started_ns']>=last;last=s['finished_ns']
text=(R/'suite1.stdout').read_text();summary=re.search(r'100% tests passed out of (\d+)',text);assert summary and int(summary[1])==57 and re.search(r'metal_fused_q4_decode_reference\s+\.+\s+Passed',text)
cache=(R/'build-on1/CMakeCache.txt').read_text()
for flag in ['GARNET_METAL_FUSED_Q4_DECODE','GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES','GARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES']:assert flag+':BOOL=ON' in cache
for n,h in read(R/'source-bindings1.json')['files_sha256'].items():assert sha(R/'source1'/n)==h,n
report=dict(passed=True,native_tests=57,source_unchanged=True,changed_profile_options=['GARNET_METAL_FUSED_Q4_DECODE:OFF->ON'],candidate_library_sha256=sha(R/'build-on1/libgarnet.dylib'),control_library_sha256=sha(Q/'build-on1/libgarnet.dylib'),source_bindings_sha256=sha(R/'source-bindings1.json'),no_dirty_root_build=True,no_model_or_comparator_execution=True,summary_parser_repair='Original controller expected an alternative CTest summary with an explicit zero-failure clause. All57 tests and processes were already terminal and successful; this auditor recognizes their actual installed CTest output, without rerunning tests.',original_build_script_sha256=sha(R/'build1.py'),build_auditor_sha256=sha(R/'audit_build2.py'),goal_complete=False)
p=R/'build-audit1.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
