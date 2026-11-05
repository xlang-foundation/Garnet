"""Pinned B2 scheduled MTP: eight coding tasks and two real2K prompts.

Args: runtime, build, source, asset manifest, auxiliary, qualified B1 coding,
admitted B1 HTTP root, NEW output. No model speed admission.
"""
import hashlib,json,subprocess,sys
from pathlib import Path
runtime,build,source,assets,auxiliary,qualified,admission,destination=sys.argv[1:9]
out=Path(destination);out.mkdir(exist_ok=False);qualified=Path(qualified);admission=Path(admission)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(auxiliary)=='dec6630788d1ba4557230ca02b5df2db239f2b2bbcfff1c6742776e231e7540b'
coding=json.loads((qualified/'coding.json').read_text())
assert len(coding['plans'])==8
plans=[dict(id=p['id'],prompt=p['prompt_ids'],max_output=p['max_output'],expected=p['reference_ids'],coding=True) for p in coding['plans']]
canonical=json.loads((admission/'sources/canonical.json').read_text());long=next(p for p in canonical if p['id']=='p2048-c2-o128')
workload=json.loads(Path(long['workload']).read_text())
for row,request in enumerate(workload['requests']):plans.append(dict(id='long-'+str(row),prompt=request['token_ids'],max_output=128,expected=long['expected_ids'][row],coding=False))
fixture=out/'tasks.json';fixture.write_text(json.dumps(dict(plans=plans,responses=str(out/'responses')))+'\n')
argv=[sys.executable,str(Path(__file__).with_name('test_trained_verification_forward.py')),runtime,build,source,assets,'4096',str(out/'engine'),'run_scheduled_mtp_coding.py',auxiliary,str(fixture)]
files=[Path(__file__),Path(__file__).with_name('run_scheduled_mtp_coding.py'),fixture,Path(build)/'libgarnet.dylib']
files += [Path(source)/p for p in ['xModel/qwen38/vl_27b/metal_mtp_study.py','src/backends/metal/continuous_scheduler.cpp','src/backends/metal/mtp_generation.cpp','src/backends/metal/mtp_session.cpp','src/backends/metal/decoder_session.cpp','src/runtime/scheduler/continuous_batch_scheduler.cpp']]
files += [Path(source)/p for p in ['src/backends/metal/compiled_graph.cpp','src/backends/metal/compiled_graph.h','src/entry/metal_main.cpp','cmake/GarnetMetal.cmake','xModel/qwen38/vl_27b/metal_mtp.py']]
(out/'provenance.json').write_text(json.dumps(dict(argv=argv,sha256={str(p):sha(p) for p in files},benchmark=False),indent=2)+'\n')
subprocess.run(argv,check=True)
