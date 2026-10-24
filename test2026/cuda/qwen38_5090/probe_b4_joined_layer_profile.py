"""Separate-process first-use TensorRT layer diagnostic; never serving timing."""
from pathlib import Path
import hashlib,json,math,os,runpy,sys
root,out=map(Path,sys.argv[1:3]);out.mkdir(exist_ok=True);sha=lambda b:hashlib.sha256(b).hexdigest()
methods=root/'Garnet/test2026/cuda/qwen38_5090';path=out/'layer-profile-report.json';assert not path.exists()
parent='b4-joined-delay10-http-pilot-attempt1';workflow=json.loads((root/'results'/parent/'workflow.json').read_bytes())
events=[json.loads(l) for l in (root/'results'/parent/'ledger/commands.jsonl').read_text().splitlines()]
assert events[-1]['event']=='finished' and events[-1]['exit_code']==workflow['exit_code']
assert (root/'results'/(parent+'-evidence.zip')).is_file()
library_sha=sha((root/'build/bin/garnet.dll').read_bytes());assert library_sha==workflow['library_sha256']
preflight=json.loads((out/'preflight/report.json').read_bytes())
assert preflight['status']=='passed' and preflight['startup_profile'] and preflight['library_sha256']==library_sha
profile=out/'layers.tsv';assert Path(os.environ.get('GARNET_PROFILE_DECODE_LAYERS',''))==profile and not profile.exists()
for n,h in workflow['workspace_source_sha256'].items():assert sha((root/'Garnet'/n).read_bytes())==h
policy=json.loads((methods/'evidence/b4-joined-mlp-source-policy.json').read_bytes());assert sha((root/'Garnet'/policy['file']).read_bytes())==policy['new_sha256']
report={'status':'running','library_sha256':library_sha,'scope':'First-use eager TensorRT IProfiler enqueue per engine with startup opt-in. Instrumented context retained; not warm CUDA Graph latency, isolated device-active work, HTTP throughput or speedup. Full retained B4 joined correctness probe runs afterward.','source_sha256':{n:sha((methods/n).read_bytes()) for n in ['probe_b4_joined_layer_profile.py','probe_trt_layer_profiler.py','probe_b4_host_joined_mtp.py']}}
path.write_text(json.dumps(report,indent=2))
try:
 sys.argv=[str(methods/'probe_b4_host_joined_mtp.py'),str(root),'4',str(out/'correctness'),'host']
 runpy.run_path(sys.argv[0],run_name='__main__')
 assert profile.exists();rows=[]
 for line in profile.read_text().splitlines():
  fields=line.split('\t')
  if fields==['engine','layer','milliseconds']:continue
  assert len(fields)==3;milliseconds=float(fields[2]);assert math.isfinite(milliseconds) and milliseconds>=0
  rows.append({'engine':fields[0],'layer':fields[1],'milliseconds':milliseconds})
 assert rows;report.update(status='correctness_and_layer_capture_finished_needs_replay',layers=rows,profile_sha256=sha(profile.read_bytes()))
except BaseException as failure:report.update(status='failed',error=repr(failure));raise
finally:path.write_text(json.dumps(report,indent=2))
