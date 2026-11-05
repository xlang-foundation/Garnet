"""Two frozen C1 requests using immutable qualified B1 code and existing counters.

Args: runtime, build, qualified source archive, admitted HTTP root, NEW output.
Native phase durations are diagnostics, never a repeated performance admission.
"""
import hashlib,json,subprocess,sys
from pathlib import Path
runtime,build,archive,admission,destination=sys.argv[1:6]
archive=Path(archive);admission=Path(admission);out=Path(destination);out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((archive/'manifest.json').read_text())
assert sha(Path(build)/'libgarnet.dylib')==manifest['builds']['native-mtp-profiles-build']['libgarnet.dylib']
for name,wanted in manifest['sha256'].items():assert sha(archive/name)==wanted
profile=json.loads((admission/'sources/profile.json').read_text());plans=[]
for plan in json.loads((admission/'sources/canonical.json').read_text()):
 if plan['id'] not in ['p256-c1-o128','p2048-c1-o128']:continue
 workload=json.loads(Path(plan['workload']).read_text());plans.append(dict(id=plan['id'],prompt_ids=workload['requests'][0]['token_ids'],expected=plan['expected_ids'][0],max_output=128))
assert len(plans)==2
command=json.loads((admission/'command.json').read_text());checkpoint=command[command.index('--checkpoint')+1]
assets=json.loads((admission.parent/'27b-asset-hashes.json').read_text())
for entry in assets['files']:
 p=Path(assets['root'])/entry['name'];h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 assert p.stat().st_size==entry['bytes'] and h.hexdigest()==entry['sha256']
assert sha(profile['auxiliary_checkpoint'])=='dec6630788d1ba4557230ca02b5df2db239f2b2bbcfff1c6742776e231e7540b'
fixture=out/'fixture.json';fixture.write_text(json.dumps(dict(checkpoint=checkpoint,auxiliary=profile['auxiliary_checkpoint'],plans=plans,output=str(out/'result.json')))+'\n')
runner=Path(__file__).with_name('run_mtp_phase_costs.py');argv=[runtime,'-B',str(runner),str(fixture),build,str(archive)]
(out/'provenance.json').write_text(json.dumps(dict(argv=argv,source_manifest_sha256=sha(archive/'manifest.json'),library_sha256=sha(Path(build)/'libgarnet.dylib'),controller_sha256=sha(__file__),runner_sha256=sha(runner),benchmark=False),indent=2)+'\n')
subprocess.run(argv,check=True)
