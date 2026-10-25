"""Complete the promised CPU suite; preserve original 49-test build evidence."""
from pathlib import Path
import hashlib,json,os,sys,unittest,zipfile
root=Path('D:/CantorAI/5090-study-20261023');repo=root/'Garnet';results=root/'results';methods=repo/'test2026/cuda/qwen38_5090'
sys.path.insert(0,str(repo));sys.path.insert(0,str(methods));from run_logged import run
sha=lambda b:hashlib.sha256(b).hexdigest();name='bf16-float-reference-cpu-completion-attempt1'
prep_path=methods/'evidence/bf16-float-reference-cpu-completion-preparation.json';p=json.loads(prep_path.read_bytes())
for n,h in p['methods'].items():assert sha((methods/n).read_bytes())==h
build=results/'bf16-float-reference-model-build-attempt1-evidence.zip';assert sha(build.read_bytes())==p['build_archive_sha256']
for n,h in p['product_source_sha256'].items():assert sha((repo/n).read_bytes())==h,('Product source drift',n)
library=root/'build/bin/garnet.dll';assert sha(library.read_bytes())==p['library_sha256']
inventory=results/'bf16-float-reference-cpu-inventory.json';old=json.loads(inventory.read_bytes());assert sha(inventory.read_bytes())==p['inventory_sha256']
for n,h in old['source_sha256'].items():assert sha((repo/n).read_bytes())==h
target=repo/p['target'];replacement=results/'bf16-float-reference-joined-profile-test.py.txt';new=replacement.read_bytes();assert sha(new)==p['target_sha256']
out=results/name;assert not out.exists();out.mkdir();(out/'original-test.py.txt').write_bytes(target.read_bytes());os.chdir(repo)
w={'status':'running','preparation_sha256':sha(prep_path.read_bytes()),'original_cpu_tests':49,'build_archive_sha256':p['build_archive_sha256'],'library_sha256':p['library_sha256'],'scope':p['scope']}
def save():(out/'workflow.json').write_text(json.dumps(w,indent=2))
save();code=1
try:
 target.write_bytes(new)
 for n,h in p['test_source_sha256'].items():assert sha((repo/n).read_bytes())==h
 def names(s):
  for x in s:
   if isinstance(x,unittest.TestSuite):yield from names(x)
   else:yield x.id()
 w['test_ids']=list(names(unittest.defaultTestLoader.loadTestsFromNames(p['cpu_tests'])));assert w['test_ids']==p['test_ids'] and len(w['test_ids'])==50;save()
 code=run([sys.executable,'-m','unittest']+p['cpu_tests'],out/'cpu-ledger','correctness','Complete exact pinned 50-test CPU inventory without rebuilding DLL',[prep_path,library]+[repo/n for n in p['test_source_sha256']]+[repo/n for n in p['product_source_sha256']])
 if code:raise RuntimeError('Completed CPU suite failed')
 w['status']='bf16_float_reference_cpu_completion_finished_needs_replay'
except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1;raise
finally:
 w.update(exit_code=code,library_unchanged=sha(library.read_bytes())==p['library_sha256'],test_source_after_sha256={n:sha((repo/n).read_bytes()) for n in p['test_source_sha256']},product_source_after_sha256={n:sha((repo/n).read_bytes()) for n in p['product_source_sha256']});save()
 files=[f for f in out.rglob('*') if f.is_file()]+[prep_path,inventory,results/'bf16-float-reference-cpu-inventory-source.zip']+[methods/n for n in p['methods']]+[repo/n for n in p['test_source_sha256']]
 raw={f.relative_to(root).as_posix():f.read_bytes() for f in files};manifest={'files':{n:sha(v) for n,v in raw.items()},'linked_archives':{build.name:p['build_archive_sha256']},'scope':p['scope']};archive=results/(name+'-evidence.zip');assert not archive.exists()
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for n,v in raw.items():z.writestr(n,v)
  z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
 print(json.dumps({'status':w['status'],'archive_sha256':sha(archive.read_bytes()),'bytes':archive.stat().st_size,'files':len(raw)}))
raise SystemExit(code)
