"""Validate curated bytes and replay eleven CPU reports in isolated per-root interpreters."""
from pathlib import Path
import hashlib,json,subprocess,sys
D=Path(__file__).resolve().parent;assert len(sys.argv)==2,'FRESH_OUTPUT_DIRECTORY';F=Path(sys.argv[1]).resolve();assert not F.exists();F.mkdir(parents=True)
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,e in read(D/'package-contents1.json')['files'].items():assert sha(D/n)==e['package_sha256'] and sha(Path(e['raw']))==e['raw_sha256'],n
refs=read(D/'source-reference1.json')['roots']
for r in refs.values():assert sha(Path(r['raw_root'])/'provenance1.json')==r['provenance_sha256']
bootstrap='''from pathlib import Path
import hashlib,json,runpy,sys
R=Path(sys.argv[1]);F=Path(sys.argv[2]);F.mkdir();reports=json.loads(sys.argv[3]);w=Path.write_text;e=Path.exists;active=None;results={}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,s,*a,**k):
 if p==R/active:return w(F/active,s,*a,**k)
 raise AssertionError('unexpected replay write '+str(p))
def exists(p):return False if p==R/active else e(p)
for script,report in reports:
 active=report;before=sha(R/report);Path.write_text=write;Path.exists=exists
 try:runpy.run_path(str(R/script),run_name='__main__')
 finally:Path.write_text=w;Path.exists=e
 assert sha(R/report)==before and (R/report).read_bytes()==(F/report).read_bytes(),report
 results[report]=before
(F/'audit1.json').write_text(json.dumps(dict(passed=True,identical_reports=results,no_gpu_or_model_execution=True),indent=2)+'\\n')
'''
groups={
'row_tiles':[('audit_pilot1.py','pilot-audit1.json'),('audit_timing1.py','timing-audit1.json')],
'fused_tensor':[('audit_pilot1.py','pilot-audit1.json'),('audit_timing1.py','timing-audit1.json')],
'fused_verifier':[('audit_qualification1.py','qualification-audit1.json'),('audit_measurement1.py','measurement-audit1.json')],
'helper_profiles':[('audit_qualification1.py','qualification-audit1.json'),('audit_qualification2.py','qualification-audit2.json'),('audit_measurement1.py','measurement-audit1.json'),('audit_measurement2.py','measurement-audit2.json'),('audit_identity3.py','identity-audit3.json')]}
result={}
for name,reports in groups.items():
 cmd=[sys.executable,'-B','-c',bootstrap,refs[name]['raw_root'],str(F/name),json.dumps(reports)]
 with (F/(name+'.stdout')).open('x') as o,(F/(name+'.stderr')).open('x') as e:r=subprocess.run(cmd,stdout=o,stderr=e)
 assert r.returncode==0,(F/(name+'.stderr')).read_text();result[name]=read(F/name/'audit1.json')
need=sum(len(r['identical_reports']) for r in result.values());assert need==11
(F/'audit1.json').write_text(json.dumps(dict(passed=True,identical_report_count=need,isolated_root_replays=result,no_gpu_or_model_execution=True,no_comparator_execution=True,goal_complete=False),indent=2)+'\n');print('Package CPU replay passed:',need,'identical reports in four isolated interpreters')
