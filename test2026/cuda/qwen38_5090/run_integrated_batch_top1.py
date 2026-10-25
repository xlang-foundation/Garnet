"""Untimed production-API controls, after full lifecycle closure and source installation.

This runner never installs source or builds/loads/replaces garnet.dll. Its
preparation must be materialized from actual closed parent proofs/tool hashes.
"""
from pathlib import Path
import hashlib,json,os,subprocess,sys,zipfile

root=Path('D:/CantorAI/5090-study-20261023');repo=root/'Garnet';results=root/'results'
m=repo/'test2026/cuda/qwen38_5090';sys.path.insert(0,str(m));from run_logged import run
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
name='integrated-batch-top1-correctness-attempt1';out=results/name
prep_path=m/'evidence/integrated-batch-top1-preparation.json';p=json.loads(prep_path.read_bytes())
assert not out.exists() and not (results/(name+'-evidence.zip')).exists()
for n,h in p['files'].items():assert sha(root/n)==h,('Candidate/input drift',n)
for n,h in p['tools'].items():assert sha(Path(n))==h,('Tool drift',n)
for n,h in p['runtime_dependencies'].items():assert sha(root/'build/bin'/n)==h,('Runtime drift',n)
parent='bf16-padded-lifecycle-scope-fixed-attempt2';proofpath=results/(parent+'-verified.json');proof=json.loads(proofpath.read_bytes())
assert sha(proofpath)==p['parent_proof_sha256'] and proof['status']=='bf16_padded_lifecycle_replayed'
assert len(proof['cases'])==14 and proof['raw_hidden_and_normalized_arrays']==102 and not proof['performance_admitted']
assert sha(results/(parent+'-evidence.zip'))==p['parent_archive_sha256']==proof['archive_sha256']
closed=json.loads((results/parent/'workflow.json').read_bytes());assert closed['exit_code']==0 and closed['cleanup_finished'] and closed['library_unchanged']
assert closed['source_sha256']==closed['source_after_sha256']==p['qualified_parent_source_sha256']
assert sha(m/'verify_bf16_padded_lifecycle_scope_fixed.py')==proof['replay_verifier_sha256']
operator='parallel-batch-top1-correctness-attempt1';opproofpath=results/(operator+'-verified.json');opproof=json.loads(opproofpath.read_bytes())
assert sha(opproofpath)==p['operator_proof_sha256'] and opproof['status']=='parallel_batch_top1_correctness_replayed'
assert opproof['cases_checked']==330 and opproof['raw_observations_checked']==1000 and not opproof['performance_admitted']
assert sha(results/(operator+'-evidence.zip'))==p['operator_archive_sha256']==opproof['archive_sha256']
library=root/'build/bin/garnet.dll';assert sha(library)==p['observed_unused_library_sha256']==proof['library_sha256']
scratch=root/'tools/parallel-batch-top1';owner=scratch/'study-owner.json'
assert scratch.resolve(strict=True).parent==(root/'tools').resolve(strict=True)
assert json.loads(owner.read_bytes())['study']==str(root) and json.loads(owner.read_bytes())['component']=='parallel-batch-top1'
legacy_exe=scratch/'probe.exe';assert sha(legacy_exe)==p['preserved_component_executable_sha256']
exe=scratch/'integrated-probe.exe';policy=scratch/'integrated-policy.exe'
assert not exe.exists() and not policy.exists()
def idle():
 assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
idle();out.mkdir();os.chdir(repo)
w={'status':'running','preparation_sha256':sha(prep_path),'source_sha256':p['files'],'tools':p['tools'],
 'runtime_dependencies':p['runtime_dependencies'],'observed_unused_library_sha256':sha(library),
 'parent_proof_sha256':sha(proofpath),'parent_archive_sha256':proof['archive_sha256'],
 'operator_proof_sha256':sha(opproofpath),'operator_archive_sha256':opproof['archive_sha256'],
 'runtime_loaded':False,'cleanup_finished':False,'variants':[],'scope':p['scope']}
def save():(out/'workflow.json').write_text(json.dumps(w,indent=2),encoding='utf8')
common=[root/n for n in p['files']]+[prep_path,proofpath,opproofpath]
save();code=1;prior_env=os.environ.get('GARNET_PARALLEL_BATCH_TOP1')
try:
 code=run([sys.executable,'-m','unittest','test2026.cuda.qwen38_5090.test_parallel_top1_oracle','test2026.cuda.qwen38_5090.test_integrated_top1_oracle'],out/'cpu-ledger','correctness','Independent original and extended fixed fixture oracle controls',common)
 if code:raise RuntimeError('CPU fixture controls failed')
 code=run(['cmd.exe','/d','/c',str(m/'compile_integrated_batch_top1.bat'),p['vcvars'],p['nvcc'],str(scratch),str(Path(p['host_compiler']).parent)],out/'build-ledger','build','Production dispatch plus frozen legacy GPU sampler and CPU policy executable',common+[Path(n) for n in p['tools']])
 if code:raise RuntimeError('Production API probe build failed')
 assert Path((scratch/'integrated-host-compiler.txt').read_text().splitlines()[0]).samefile(Path(p['host_compiler']))
 w.update(executable_sha256=sha(exe),policy_executable_sha256=sha(policy));save()
 code=run([str(policy)],out/'policy-ledger','correctness','Exact opt-in and all CPU fallback boundaries',common+[policy])
 if code:raise RuntimeError('CPU dispatch policy failed')
 for label,value,enabled in [('disabled','0',False),('enabled','1',True),('unset',None,False),('malformed','01',False)]:
  idle()
  if value is None:os.environ.pop('GARNET_PARALLEL_BATCH_TOP1',None)
  else:os.environ['GARNET_PARALLEL_BATCH_TOP1']=value
  entry={'label':label,'environment_value':value,'expected_enabled':enabled,'finished':False};w['variants'].append(entry);save()
  code=run([sys.executable,str(m/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/(label+'-resources')),'--',str(exe),str(out/(label+'-observations.jsonl')),'1' if enabled else '0'],out/(label+'-ledger'),'correctness','352production API cases/1066raw observations; mode '+label,common+[exe])
  entry.update(exit_code=code,finished=True);save()
  if code:raise RuntimeError('Production API controls failed: '+label)
 w['status']='integrated_batch_top1_correctness_finished_needs_replay'
except BaseException as error:
 w.update(status='failed',error=repr(error));code=code or 1;raise
finally:
 if prior_env is None:os.environ.pop('GARNET_PARALLEL_BATCH_TOP1',None)
 else:os.environ['GARNET_PARALLEL_BATCH_TOP1']=prior_env
 w.update(exit_code=code,source_after_sha256={n:sha(root/n) for n in p['files']},tools_after={n:sha(Path(n)) for n in p['tools']},
  runtime_dependencies_after={n:sha(root/'build/bin'/n) for n in p['runtime_dependencies']},library_unchanged=sha(library)==p['observed_unused_library_sha256'],
  preserved_component_executable_unchanged=sha(legacy_exe)==p['preserved_component_executable_sha256'])
 if exe.exists():w['executable_after_sha256']=sha(exe)
 if policy.exists():w['policy_executable_after_sha256']=sha(policy)
 save();idle();w['cleanup_finished']=True;save()
 files=[f for f in out.rglob('*') if f.is_file()]+common+[owner]
 files+=[f for f in (exe,policy,scratch/'integrated-host-compiler.txt') if f.exists()]
 raw={f.relative_to(root).as_posix():f.read_bytes() for f in files}
 manifest={'files':{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()},'parent_archive_sha256':proof['archive_sha256'],'operator_archive_sha256':opproof['archive_sha256'],'scope':p['scope']}
 archive=results/(name+'-evidence.zip')
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for n,b in raw.items():z.writestr(n,b)
  z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
 print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(raw)}))
raise SystemExit(code)
