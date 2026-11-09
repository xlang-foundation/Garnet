from pathlib import Path
import hashlib,json,sys
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert len(sys.argv)==2,'FRESH_OUTPUT_DIRECTORY required'
out=Path(sys.argv[1]).resolve();assert not out.exists();out.mkdir()
index=read(R/'qualification.json')
for n,h in index['files_sha256'].items():assert sha(R/n)==h,n
for item in index['frozen_roots']:
 root=Path(item['root']);provenance=root/'provenance1.json';assert sha(provenance)==item['provenance_sha256']
 for name,h in read(provenance)['files_sha256'].items():assert sha(root/name)==h,str(root/name)
results=[]
for job in index['audit_replay']:
 script=Path(job['script']);report=Path(job['report']);assert sha(script)==job['script_sha256'] and sha(report)==job['report_sha256']
 text=script.read_text();needle="R/'"+report.name+"'";assert text.count(needle)==1
 target=out/report.name;text=text.replace(needle,'Path('+repr(str(target))+')')
 exec(compile(text,str(script),'exec'),{'__file__':str(script),'__name__':'__main__'})
 assert read(target)==read(report)
 results.append(dict(report=str(report),complete_report_exact=True))
(out/'replay.json').write_text(json.dumps(dict(passed=True,cpu_only=True,package_bindings_verified=True,raw_bindings_verified=True,results=results),indent=2)+'\n')
