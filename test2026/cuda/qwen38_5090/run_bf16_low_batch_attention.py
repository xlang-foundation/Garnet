"""Isolated B1 numerical correctness only, after the full model worker is terminal/replayed."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,zipfile

root=Path('D:/CantorAI/5090-study-20261023');repo=root/'Garnet';results=root/'results'
methods=repo/'test2026/cuda/qwen38_5090';sys.path.insert(0,str(methods))
from run_logged import run

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
name='bf16-low-batch-attention-native-attempt1';out=results/name
assert not out.exists() and not (results/(name+'-evidence.zip')).exists()
prep_path=methods/'evidence/bf16-low-batch-attention-preparation.json';p=json.loads(prep_path.read_bytes())
for n,h in p['files'].items():assert sha(root/n)==h,('Source/method/input drift',n)
for n,h in p['inherited_model_sources'].items():assert sha(repo/n)==h,('Inherited source drift',n)
for n,h in p['tools'].items():assert sha(Path(n))==h,('Compiler/tool drift',n)
parent=p['parent'];proofpath=results/(parent+'-verified.json');proof=json.loads(proofpath.read_bytes())
assert sha(proofpath)==p['parent_proof_sha256'] and sha(results/(parent+'-evidence.zip'))==proof['archive_sha256']==p['parent_archive_sha256']
closed=json.loads((results/parent/'workflow.json').read_bytes())
assert closed['cleanup_finished'] and closed['exit_code']==0 and closed['source_sha256']==closed['source_after_sha256']==p['inherited_model_sources']
expected_status='bf16_padded_lifecycle_replayed'
assert proof['status']==expected_status and sha(methods/'verify_bf16_padded_lifecycle_scope_fixed.py')==proof['replay_verifier_sha256']
if closed['exit_code']==0:assert len(proof['cases'])==14 and proof['raw_hidden_and_normalized_arrays']==102
library=root/'build/bin/garnet.dll';digest=sha(library)
assert digest==p['observed_unused_library_sha256']==proof['library_sha256']
def idle():
 assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
idle()
# All compiler outputs are private to one fixed component folder. Never stage
# a runtime DLL, overwrite an existing executable, or clean unrelated files.
scratch=root/'tools/bf16-kv-native';assert scratch.is_dir()
assert scratch.parent.resolve(strict=True)==(root/'tools').resolve(strict=True)
assert not (scratch/'low-batch-study-owner.json').exists()
(scratch/'low-batch-study-owner.json').write_text(json.dumps({'study':str(root),'component':'bf16-low-batch-attention','scope':'Standalone compiler outputs only; no runtime replacement.'},indent=2))
assert scratch.resolve(strict=True).parent==(root/'tools').resolve(strict=True)
assert not (scratch/'bf16-low-batch-attention.exe').exists() and not (scratch/'low-batch-policy.exe').exists()
assert sha(scratch/'bf16_kv_fixture_layout.h')==p['fixture_header_sha256']
os.environ['CUDA_VISIBLE_DEVICES']='0'
out.mkdir();os.chdir(repo);code=1;exe=scratch/'bf16-low-batch-attention.exe'
w={'status':'running','source_sha256':p['files'],'inherited_model_sources':p['inherited_model_sources'],
 'preparation_sha256':sha(prep_path),'parent':parent,'parent_status':proof['status'],'parent_proof_sha256':sha(proofpath),
 'parent_archive_sha256':proof['archive_sha256'],'observed_unused_library_sha256':digest,'tools':p['tools'],
 'cleanup_finished':False,'runtime_loaded':False,'scope':p['scope']}
def save():(out/'workflow.json').write_text(json.dumps(w,indent=2),encoding='utf8')
common=[root/n for n in p['files']]+[repo/n for n in p['inherited_model_sources']]+[prep_path,proofpath]
save()
try:
 code=run([sys.executable,'-m','unittest','test2026.cuda.qwen38_5090.test_bf16_low_batch_candidate','test2026.cuda.qwen38_5090.test_bf16_kv_overlay_oracle'],out/'cpu-ledger','correctness','B1 numerical profile isolation, full-chunk/boundary coverage and independent FP64 controls',common)
 if code:raise RuntimeError('CPU top1 oracle controls failed')
 code=run(['cmd.exe','/d','/c',str(methods/'compile_bf16_low_batch_attention.bat'),p['vcvars'],p['nvcc'],str(scratch),str(Path(p['host_compiler']).parent)],out/'build-ledger','build','CPU-only33policy controls/import guard, plus separate sm120 B1 attention numerical probe',common+[Path(n) for n in p['tools']])
 if code:raise RuntimeError('Low-batch attention compiler/linker or CPU policy failed')
 assert Path((scratch/'low-batch-host-compiler.txt').read_text().splitlines()[0]).samefile(Path(p['host_compiler']))
 assert exe.is_file();w['executable_sha256']=sha(exe);save()
 code=run([sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--',str(exe),str(out/'witnesses.f32')],out/'gpu-ledger','correctness','44B1 attention cases/119040selected scalars; independent FP64 replay mandatory,3captures/104negative/40cache-equivalence checks',common+[exe])
 if code:raise RuntimeError('Low-batch attention collection failed')
 w['status']='bf16_low_batch_attention_collected_needs_fp64_replay'
except BaseException as error:
 w.update(status='failed',error=repr(error));code=code or 1;raise
finally:
 w.update(exit_code=code,library_unchanged=sha(library)==digest,
  source_after_sha256={n:sha(root/n) for n in p['files']},
  inherited_model_sources_after={n:sha(repo/n) for n in p['inherited_model_sources']},
  tools_after={n:sha(Path(n)) for n in p['tools']})
 if exe.exists():w['executable_after_sha256']=sha(exe)
 save();idle();w['cleanup_finished']=True;save()
 files=[f for f in out.rglob('*') if f.is_file()]+common+[scratch/'low-batch-study-owner.json',scratch/'bf16_kv_fixture_layout.h']
 if (scratch/'low-batch-host-compiler.txt').exists():files.append(scratch/'low-batch-host-compiler.txt')
 for extra in (scratch/'low-batch-policy.exe',scratch/'low-batch-policy-imports.txt'):
  if extra.exists():files.append(extra)
 if exe.exists():files.append(exe)
 raw={f.relative_to(root).as_posix():f.read_bytes() for f in files}
 manifest={'files':{n:hashlib.sha256(data).hexdigest() for n,data in raw.items()},'parent_archive_sha256':proof['archive_sha256'],'scope':p['scope']}
 archive=results/(name+'-evidence.zip')
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for n,data in raw.items():z.writestr(n,data)
  z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
 print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(raw)}))
raise SystemExit(code)
