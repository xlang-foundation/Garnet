"""Own an untimed B8 model admission after an explicitly closed reference job.

No source/runtime replacement, server startup, MTP or performance collection.
"""
from pathlib import Path
import argparse,hashlib,json,os,re,shutil,sys,time,zipfile
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def closed_reference(root,name,digest):
 if not re.fullmatch(r'sglang-fixed4k-(ordinary|mtp)-attempt[1-9][0-9]*',name):raise ValueError('A named closed reference attempt is required')
 if not re.fullmatch(r'[0-9a-f]{64}',digest):raise ValueError('Actual closed archive SHA256 required')
 path=root/'results'/(name+'-evidence.zip');assert sha(path)==digest
 with zipfile.ZipFile(path) as z:
  manifest=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None
  for relative,expected in manifest['files'].items():assert hashlib.sha256(z.read(relative)).hexdigest()==expected
  w=json.loads(z.read('results/'+name+'/workflow.json'))
 assert w==json.loads((root/'results'/name/'workflow.json').read_bytes())
 assert type(w['exit_code']) is int and w['cleanup_finished'] is True and type(w['server_exit_code']) is int
 assert w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after']
 return {'job':name,'archive_sha256':digest,'exit_code':w['exit_code'],'cleanup_finished':True}

def retire(cache,root,before,started):
 """Same bounded owned-cache policy as the closed native HTTP controller."""
 resolved=cache.resolve(strict=True);assert json.loads((cache/'study-owner.json').read_bytes())['study']==str(root);retired=[]
 for target in cache.iterdir():
  if target.name in before or not target.name.startswith('garnet-cuda-graph-'):continue
  st=target.lstat();assert not(getattr(st,'st_file_attributes',0)&0x400) and target.resolve(strict=True).parent==resolved and st.st_ctime>=started-2
  entries=list(target.iterdir());assert all(f.name in ('model.engine','model.engine.tmp') for f in entries)
  for f in entries:
   fs=f.lstat();assert f.is_file() and not(getattr(fs,'st_file_attributes',0)&0x400) and f.resolve(strict=True).parent==target.resolve(strict=True)
   h=hashlib.sha256()
   with f.open('rb') as stream:
    for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
   retired.append({'path':str(f),'bytes':fs.st_size,'sha256':h.hexdigest()})
  for f in entries:f.unlink()
  target.rmdir()
 return retired

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--closed-reference-job',required=True);a.add_argument('--closed-reference-archive-sha256',required=True);args=a.parse_args()
 root=Path('D:/CantorAI/5090-study-20261023');m=Path(__file__).resolve().parent;assert m==(root/'Garnet/test2026/cuda/qwen38_5090').resolve()
 sys.path.insert(0,str(m));from run_logged import run
 from run_bf16_low_batch_native_build import idle
 prep=m/'evidence/bf16-batch8-admission-run-preparation.json';p=json.loads(prep.read_bytes());assert p['status']=='bf16_batch8_model_admission_prepared_gpu_unrun'
 for name,digest in p['files'].items():assert sha(root/name)==digest,('B8 input drift',name)
 for name,digest in p['runtime_dependencies'].items():assert sha(root/'build/bin'/name)==digest,('Runtime drift',name)
 reference=closed_reference(root,args.closed_reference_job,args.closed_reference_archive_sha256)
 for name,row in p['parents'].items():
  proof=root/'results'/(name+'-verified.json');q=json.loads(proof.read_bytes());assert sha(proof)==row['proof_sha256'] and q['status']==row['status'] and sha(root/'results'/(name+'-evidence.zip'))==q['archive_sha256']==row['archive_sha256']
  closed=json.loads((root/'results'/name/'workflow.json').read_bytes());assert closed['exit_code']==0 and closed['cleanup_finished'] and closed['library_unchanged']
 receipt=root/'results/study-ssd-engine-cache/receipt.json';scratch=json.loads(receipt.read_bytes());cache=Path(scratch['cache_root'])
 assert scratch['status']=='ssd_engine_scratch_prepared' and cache.resolve().drive.upper()=='C:' and json.loads((cache/'study-owner.json').read_bytes())['study']==str(root) and shutil.disk_usage(cache).free>=96*1024**3
 name='bf16-batch8-model-admission-attempt1';out=root/'results'/name;archive=root/'results'/(name+'-evidence.zip');assert not out.exists() and not archive.exists()
 idle();out.mkdir();os.chdir(root/'Garnet');code=1;before={f.name for f in cache.iterdir()};started=time.time()
 library=root/'build/bin/garnet.dll';common=[root/n for n in p['files']]+[prep,receipt,root/'results'/args.closed_reference_job/'workflow.json']+[root/'results'/(n+'-verified.json') for n in p['parents']]
 w={'status':'running','preparation_sha256':sha(prep),'source_sha256':p['files'],'runtime_dependencies':p['runtime_dependencies'],'library_sha256':sha(library),'parents':p['parents'],'closed_reference':reference,'cleanup_finished':False,'performance_admitted':False,'scope':p['scope']}
 def save():(out/'workflow.json').write_text(json.dumps(w,indent=2),encoding='utf8')
 save()
 try:
  code=run([sys.executable,'-m','unittest',*p['test_modules']],out/'cpu-ledger','correctness','Strict full-row state/permutation/profile and closed-reference/cache ownership controls',common)
  if code:raise RuntimeError('B8 CPU admission controls failed')
  command=[sys.executable,str(m/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--','cmd.exe','/d','/c',str(m/'test_bf16_batch8_admission.bat'),str(out/'model'),str(cache)]
  code=run(command,out/'gpu-ledger','correctness','Actual B8 construction/vision/short mixed and full3968+128 text/image, three row permutations/reset; full-state/raw-vector witnesses, no timing',common)
  if code:raise RuntimeError('B8 model admission failed')
  report=json.loads((out/'model/report.json').read_bytes());assert report['status']=='bf16_batch8_ordinary_admission_collected_needs_replay'
  w['status']='bf16_batch8_model_admission_collected_needs_replay'
 except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
 finally:
  w.update(exit_code=code,source_after_sha256={n:sha(root/n) for n in p['files']},runtime_dependencies_after={n:sha(root/'build/bin'/n) for n in p['runtime_dependencies']},library_unchanged=sha(library)==w['library_sha256']);save()
  idle();w['retired_private_engine_files']=retire(cache,root,before,started);w['cleanup_finished']=True;save()
  paths=[f for f in out.rglob('*') if f.is_file()]+common;raw={f.relative_to(root).as_posix():f.read_bytes() for f in paths}
  manifest={'files':{n:hashlib.sha256(data).hexdigest() for n,data in raw.items()},'scope':p['scope']}
  with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
   for n,data in raw.items():z.writestr(n,data)
   z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
  print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(raw)}),flush=True)
 return code
if __name__=='__main__':raise SystemExit(main())
