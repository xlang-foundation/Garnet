"""Bind operator sources and trained assets to their frozen native and capture provenance."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;M=R.parent;Q=M/'mistral-causal-batch4-helper1';B=M/'mistral-tiled-half-weight-screen1';F=M/'mistral-fused-mlp-trained1'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def need(v,why):
 if not v:raise ValueError(why)
checks={}
def bound(p,h=None):
 actual=sha(p)
 if h is not None:need(actual==h,'hash mismatch '+str(p))
 checks[str(p)]=actual
for name in ['pilot2','pilot4','timing1','timing2']:
 pre=R/(name+'-preflight.json');post=R/(name+'-postflight.json');d=read(pre);need(d['passed'] and read(post)['passed'],'collection pre/post failed')
 for n,h in d['files_sha256'].items():bound(Path(n),h)
 bound(pre);bound(post)
# Verify actual learned weights/activations against the older capture provenance,
# beyond merely trusting the new operator collection's own hashes.
provenance=read(F/'provenance1.json');used=[Path(n) for n in checks if n.startswith(str(F)+'/')]
need(len(used)>=10,'trained fixture coverage')
for p in used:
 relative=str(p.relative_to(F));need(relative in provenance['files_sha256'],'trained asset absent from original manifest '+relative);bound(p,provenance['files_sha256'][relative])
bound(F/'provenance1.json')
base=(B/'combined1.metal').read_text();native=(Q/'source1/src/backends/metal/packed_affine_shader.h').read_text().split('R"METAL(\n',1)[1].split(')METAL"',1)[0]
need(native in base,'original packed arithmetic absent');bound(B/'combined1.metal');bound(Q/'source1/src/backends/metal/packed_affine_shader.h')
need((R/'combined1.metal').read_text()==base+'\n'+(R/'direct1.metal').read_text(),'initial shader composition')
need((R/'combined2.metal').read_text()==base+'\n'+(R/'direct2.metal').read_text(),'corrected shader composition')
need((R/'combined3.metal').read_text()==(R/'combined2.metal').read_text()+'\n'+(R/'shared3.metal').read_text(),'shared shader composition')
need((R/'combined4.metal').read_text()==(R/'combined3.metal').read_text()+'\n'+(R/'walk4.metal').read_text(),'walk shader composition')
for n in ['pilot2-audit.json','pilot4-audit.json','walk-source-audit1.json','timing-audit1.json','timing-audit3.json']:
 d=read(R/n);need(d['passed'],'audit failed '+n)
 for p,h in d.get('files_sha256',{}).items():bound(Path(p) if Path(p).is_absolute() else R/p,h)
 bound(R/n)
for n in ['audit_pilot2.py', 'audit_pilot4.py', 'audit_source1.py', 'audit_timing1.py', 'audit_timing2.py', 'audit_timing3.py', 'audit_walk_source1.py', 'combined1.metal', 'combined2.metal', 'combined3.metal', 'combined4.metal', 'direct1.metal', 'direct2.metal', 'pilot1.mm', 'pilot2.mm', 'pilot3-host.patch', 'pilot3.mm', 'pilot4-host.patch', 'pilot4.mm', 'prepare1.py', 'prepare_timing1.py', 'prepare_timing2.py', 'run_pilot1.py', 'run_pilot2.py', 'run_pilot3.py', 'run_pilot4.py', 'run_timing1.py', 'run_timing2.py', 'shared3.metal', 'timing-instrumentation1.patch', 'timing-instrumentation2.patch', 'timing1.mm', 'timing2.mm', 'walk4.metal']:bound(R/n)
bound(Path(__file__))
report=dict(passed=True,trained_fixture_files=len(used),unchanged_native_control=True,shader_compositions_checked=4,source_hashes_during_timed_collection=False,all_gpu_processes_terminal=True,native_model_admission=False,writer_map_qualified=False,no_comparator_execution=True,goal_complete=False,files_sha256=checks)
out=R/'source-audit2.json';need(not out.exists(),'report already exists');out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='files_sha256'},indent=2))
