"""Bind both actual build/lifecycle proofs and freeze complete model methods."""
from pathlib import Path
import argparse,ast,hashlib,json,zipfile

sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
METHODS=('probe_integrated_batch_top1_model.py','integrated_top1_model_replay.py','test_integrated_top1_model_replay.py',
 'test_integrated_top1_model.bat','run_integrated_batch_top1_model.py','verify_integrated_batch_top1_model.py','prepare_integrated_top1_model.py',
 'verify_integrated_batch_top1_native_build_fixed.py','evidence/integrated-top1-native-build-verifier-correction.json',
 'evidence/integrated-top1-model-probe-derivation.json',
 'evidence/integrated-batch-top1-native-build-attempt1-verified.json','evidence/integrated-batch-top1-native-build-attempt1-remote-verified.json',
 'evidence/integrated-top1-model-checker-cpu-ledger/commands.jsonl','evidence/integrated-top1-model-checker-cpu-ledger/commands.log')

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--proofs',type=Path,required=True);parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
 m=Path(__file__).resolve().parent;repo=m.parents[2];e=m/'evidence';parents={};proofs={}
 for name,status in [('bf16-padded-lifecycle-scope-fixed-attempt2','bf16_padded_lifecycle_replayed'),('integrated-batch-top1-native-build-attempt1','integrated_batch_top1_native_build_replayed')]:
  a=json.loads((args.proofs/(name+'-verified.json')).read_bytes());raw=(args.proofs/(name+'-remote-verified.json')).read_bytes();b=json.loads(raw)
  assert a==b and b['status']==status and not b['performance_admitted'] and sha(args.proofs/(name+'-evidence.zip'))==b['archive_sha256']
  parents[name]={'status':status,'proof_sha256':hashlib.sha256(raw).hexdigest(),'archive_sha256':b['archive_sha256']};proofs[name]=b
 oldname='bf16-padded-lifecycle-scope-fixed-attempt2';buildname='integrated-batch-top1-native-build-attempt1';old=proofs[oldname];built=proofs[buildname]
 assert len(old['cases'])==14 and old['raw_hidden_and_normalized_arrays']==102
 assert built['cpu_tests']==63 and built['policy_settings']==9 and built['policy_boundaries']==112 and built['runtime_dependencies_preserved']==5 and not built['gpu_model_executed']
 assert built['replay_verifier_sha256']==sha(m/'verify_integrated_batch_top1_native_build_fixed.py')
 correction=e/'integrated-top1-native-build-verifier-correction.json';c=json.loads(correction.read_bytes())
 assert sha(correction)==built['checker_correction_sha256'] and c['result_sha256']==built['replay_verifier_sha256'] and c['original_sha256']==built['archived_replay_verifier_sha256']
 with zipfile.ZipFile(args.proofs/(oldname+'-evidence.zip')) as z:
  oldp=json.loads(z.read('Garnet/test2026/cuda/qwen38_5090/evidence/bf16-padded-lifecycle-scope-fixed-preparation.json'))
  files={'Garnet/'+n:h for n,h in oldp['source_sha256'].items()};files.update({'Garnet/test2026/cuda/qwen38_5090/'+n:h for n,h in oldp['methods'].items()});files.update(oldp['pinned_inputs'])
  files['results/study-ssd-engine-cache/receipt.json']=hashlib.sha256(z.read('results/study-ssd-engine-cache/receipt.json')).hexdigest()
  files['results/bf16-kv-metadata-fixture.json']=oldp['fixture_sha256']
  recipe=json.loads((e/'integrated-top1-model-probe-derivation.json').read_bytes());source=z.read('Garnet/test2026/cuda/qwen38_5090/'+recipe['source']);assert hashlib.sha256(source).hexdigest()==recipe['source_sha256']
  text=source.decode().replace('\r\n','\n')
  for before,after in recipe['changes']:assert text.count(before)==1;text=text.replace(before,after,1)
  assert text.encode()==(m/'probe_integrated_batch_top1_model.py').read_bytes() and sha(m/'probe_integrated_batch_top1_model.py')==recipe['result_sha256']
 with zipfile.ZipFile(args.proofs/(buildname+'-evidence.zip')) as z:
  bp=json.loads(z.read('Garnet/test2026/cuda/qwen38_5090/evidence/integrated-batch-top1-native-build-preparation.json'));w=json.loads(z.read('results/'+buildname+'/workflow.json'))
  assert w['exit_code']==0 and w['cleanup_finished'] and w['source_sha256']==w['source_after_sha256']==bp['files']
  assert w['library_sha256']==built['library_sha256'] and w['retained_original_library_sha256']==old['library_sha256'];files.update(bp['files'])
 files.update({'Garnet/test2026/cuda/qwen38_5090/'+n:sha(m/n) for n in METHODS})
 for n,h in files.items():
  if n.startswith('Garnet/'):assert sha(repo/n.removeprefix('Garnet/'))==h,('Local method/source drift',n)
 for n in METHODS:
  if n.endswith('.py'):ast.parse((m/n).read_text())
 ledger=[json.loads(line) for line in (e/'integrated-top1-model-checker-cpu-ledger/commands.jsonl').read_bytes().splitlines()]
 assert len(ledger)==2 and ledger[-1]['exit_code']==0 and b'Ran 7 tests' in (e/'integrated-top1-model-checker-cpu-ledger/commands.log').read_bytes()
 deps=dict(bp['runtime_dependencies']);deps['garnet.dll']=built['library_sha256']
 p={'status':'complete_two_mode_model_corpus_prepared_unrun','files':files,'parents':parents,'runtime_dependencies':deps,
  'scope':'Full unchanged14case/102raw-vector BF16 lifecycle corpus in separate disabled then enabled native model processes, seven groups/two repeats including1/2/3/4real rows, ordinary/MTP, unequal-prefix commit/rollback/reset/pending and retirement/callback controls. Require exact full report and all raw vectors versus prior passing97c parent, exact owned model allocations, positive actual selected-route host counters and unchanged other route. Before enabled process, disabled corpus must fully pass raw parent parity.7synthetic checker controls and11unchanged state CPU controls. Keep original97c and current5ac DLLs/runtimes/sources unchanged throughout. Not broad quality, B8, actual4K, streaming/HTTP, serving memory, speed or2x qualification.'}
 target=e/'integrated-top1-model-preparation.json';data=json.dumps(p,indent=2).encode()
 if target.exists():assert target.read_bytes()==data
 else:target.write_bytes(data)
 plan={'status':'complete_model_method_upload_plan_prepared_unrun','uploads':[{'local':str(m/n),'target':'Garnet/test2026/cuda/qwen38_5090/'+n,'sha256':sha(m/n)} for n in METHODS],
  'preparation':{'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(target)},
  'scope':'Read-only source/runtime/parent/cache preflight; reject unknown target bytes, upload absent methods/receipts only and acknowledge each. Start exactly one original model controller, never replace source/runtime while it lives.'}
 data=json.dumps(plan,indent=2).encode()
 if args.plan.exists():assert args.plan.read_bytes()==data
 else:args.plan.write_bytes(data)
 print(json.dumps({'status':p['status'],'source_method_input_pins':len(files),'methods':len(METHODS),'preparation_sha256':sha(target)}));return 0

if __name__=='__main__':raise SystemExit(main())
