"""Standalone correctness only, after the original model worker is terminal."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,zipfile

root=Path('D:/CantorAI/5090-study-20261023');repo=root/'Garnet';results=root/'results'
methods=repo/'test2026/cuda/qwen38_5090';sys.path.insert(0,str(methods))
from run_logged import run

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
name='parallel-batch-top1-correctness-attempt1';out=results/name
assert not out.exists() and not (results/(name+'-evidence.zip')).exists()
prep_path=methods/'evidence/parallel-batch-top1-preparation.json';p=json.loads(prep_path.read_bytes())
for n,h in p['files'].items():assert sha(root/n)==h,('Source/method/input drift',n)
for n,h in p['inherited_model_sources'].items():assert sha(repo/n)==h,('Inherited source drift',n)
for n,h in p['tools'].items():assert sha(Path(n))==h,('Compiler/tool drift',n)
parent=p['parent'];proofpath=results/(parent+'-verified.json');proof=json.loads(proofpath.read_bytes())
assert sha(proofpath)==p['parent_proof_sha256'] and sha(results/(parent+'-evidence.zip'))==proof['archive_sha256']==p['parent_archive_sha256']
closed=json.loads((results/parent/'workflow.json').read_bytes())
assert closed['cleanup_finished'] and closed['exit_code'] in (0,1) and closed['source_sha256']==closed['source_after_sha256']==p['inherited_model_sources']
expected_status='bf16_padded_lifecycle_replayed' if closed['exit_code']==0 else 'bf16_padded_lifecycle_failure_preserved'
assert proof['status']==expected_status and sha(methods/'verify_bf16_padded_lifecycle.py')==proof['replay_verifier_sha256']
if closed['exit_code']==0:assert len(proof['cases'])==14 and proof['raw_hidden_and_normalized_arrays']==102
library=root/'build/bin/garnet.dll';digest=sha(library)
assert digest==p['observed_unused_library_sha256']==proof['library_sha256']
def idle():
 assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
idle()
# All compiler outputs are private to one fixed component folder. Never stage
# a runtime DLL, overwrite an existing executable, or clean unrelated files.
scratch=root/'tools/parallel-batch-top1';assert not scratch.exists()
assert scratch.parent.resolve(strict=True)==(root/'tools').resolve(strict=True)
scratch.mkdir();(scratch/'study-owner.json').write_text(json.dumps({'study':str(root),'component':'parallel-batch-top1','scope':'Standalone compiler outputs only; no runtime replacement.'},indent=2))
assert scratch.resolve(strict=True).parent==(root/'tools').resolve(strict=True)
out.mkdir();os.chdir(repo);code=1;exe=scratch/'probe.exe'
w={'status':'running','source_sha256':p['files'],'inherited_model_sources':p['inherited_model_sources'],
 'preparation_sha256':sha(prep_path),'parent':parent,'parent_status':proof['status'],'parent_proof_sha256':sha(proofpath),
 'parent_archive_sha256':proof['archive_sha256'],'observed_unused_library_sha256':digest,'tools':p['tools'],
 'cleanup_finished':False,'runtime_loaded':False,'scope':p['scope']}
def save():(out/'workflow.json').write_text(json.dumps(w,indent=2),encoding='utf8')
common=[root/n for n in p['files']]+[repo/n for n in p['inherited_model_sources']]+[prep_path,proofpath]
save()
try:
 code=run([sys.executable,'-m','unittest','test2026.cuda.qwen38_5090.test_parallel_top1_oracle'],out/'cpu-ledger','correctness','Independent top1 fixture oracle controls',common)
 if code:raise RuntimeError('CPU top1 oracle controls failed')
 code=run(['cmd.exe','/d','/c',str(methods/'compile_parallel_batch_top1.bat'),p['vcvars'],p['nvcc'],str(scratch),str(Path(p['host_compiler']).parent)],out/'build-ledger','build','Standalone sm120 candidate plus unchanged legacy sampler',common+[Path(n) for n in p['tools']])
 if code:raise RuntimeError('Standalone top1 compiler/linker failed')
 assert Path((scratch/'host-compiler.txt').read_text().splitlines()[0]).samefile(Path(p['host_compiler']))
 assert exe.is_file();w['executable_sha256']=sha(exe);save()
 code=run([sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--',str(exe),str(out/'observations.jsonl')],out/'gpu-ledger','correctness','330top1 cases/1000raw observations against serial CPU and unchanged legacy GPU sampler',common+[exe])
 if code:raise RuntimeError('Standalone parallel top1 controls failed')
 w['status']='parallel_batch_top1_correctness_finished_needs_replay'
except BaseException as error:
 w.update(status='failed',error=repr(error));code=code or 1;raise
finally:
 w.update(exit_code=code,library_unchanged=sha(library)==digest,
  source_after_sha256={n:sha(root/n) for n in p['files']},
  inherited_model_sources_after={n:sha(repo/n) for n in p['inherited_model_sources']},
  tools_after={n:sha(Path(n)) for n in p['tools']})
 if exe.exists():w['executable_after_sha256']=sha(exe)
 save();idle();w['cleanup_finished']=True;save()
 files=[f for f in out.rglob('*') if f.is_file()]+common+[scratch/'study-owner.json']
 if (scratch/'host-compiler.txt').exists():files.append(scratch/'host-compiler.txt')
 if exe.exists():files.append(exe)
 raw={f.relative_to(root).as_posix():f.read_bytes() for f in files}
 manifest={'files':{n:hashlib.sha256(data).hexdigest() for n,data in raw.items()},'parent_archive_sha256':proof['archive_sha256'],'scope':p['scope']}
 archive=results/(name+'-evidence.zip')
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for n,data in raw.items():z.writestr(n,data)
  z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
 print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(raw)}))
raise SystemExit(code)
