"""Prepare a fresh full model attempt after preserving the typed-checker failure."""
from pathlib import Path
import argparse,hashlib,json,zipfile
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);a.add_argument('--plan',type=Path,required=True);args=a.parse_args()
 m=Path(__file__).resolve().parent;repo=m.parents[2];e=m/'evidence';name='bf16-low-batch-native-build-attempt1';failed='bf16-low-batch-model-consistency-attempt1'
 required=[args.proofs/(n+s) for n in (name,failed) for s in ('-verified.json','-remote-verified.json','-evidence.zip')]
 if not all(p.is_file() for p in required):print(json.dumps({'status':'b1_model_fixed_prerequisites_missing'}));return 2
 parents={}
 for n,status,exitcode in ((name,'bf16_low_batch_native_build_replayed',0),(failed,'bf16_low_batch_model_consistency_failure_preserved',1)):
  raw=(args.proofs/(n+'-remote-verified.json')).read_bytes();q=json.loads(raw);assert q==json.loads((args.proofs/(n+'-verified.json')).read_bytes()) and q['status']==status and not q['performance_admitted']
  assert sha(args.proofs/(n+'-evidence.zip'))==q['archive_sha256']
  with zipfile.ZipFile(args.proofs/(n+'-evidence.zip')) as z:
   closed=json.loads(z.read('results/'+n+'/workflow.json'));assert closed['exit_code']==exitcode and closed['cleanup_finished']
   if n==name:
    native=closed;files=dict(closed['installed_source']);files.update(closed['methods']);parents.update(closed['parents']);runtime=closed['runtime_after']
    assert q['replay_verifier_sha256']==q['archived_replay_verifier_sha256']==sha(m/'verify_bf16_low_batch_native_build.py')
   else:
    assert closed['library_unchanged'] and closed['source_sha256']==closed['source_after_sha256'] and closed['runtime_dependencies']==closed['runtime_dependencies_after']==runtime
    report=json.loads(z.read('results/'+n+'/b4-smoke/report.json'));assert report['status']=='failed' and len(report['cases'])==1 and 'Unqualified same-precision state representation' in report['error']
    assert q['replay_verifier_sha256']==q['archived_replay_verifier_sha256']==sha(m/'verify_bf16_low_batch_model.py')
  parents[n]={'status':status,'proof_sha256':hashlib.sha256(raw).hexdigest(),'archive_sha256':q['archive_sha256']}
 replay=json.loads((e/'bf16-low-batch-model-attempt1-packed-reanalysis.json').read_bytes());assert replay['status']=='failed_attempt_preserved_first_observed_packed_row_reanalysis_passed' and replay['archive_sha256']==parents[failed]['archive_sha256'] and replay['library_sha256']==runtime['garnet.dll'] and replay['original_failure_preserved'] and all(replay['checks'].values()) and replay['packed_gate_sha256']==sha(m/'bf16_packed_state_gate.py')
 derivation=json.loads((e/'bf16-low-batch-model-fixed-method-derivation.json').read_bytes());assert derivation['parent_failed_archive_sha256']==parents[failed]['archive_sha256']
 with zipfile.ZipFile(args.proofs/(failed+'-evidence.zip')) as z:
  for target,row in derivation['files'].items():
   raw=z.read(row['source']);assert hashlib.sha256(raw).hexdigest()==row['source_sha256'];text=raw.decode()
   for before,after,count in row['changes']:assert text.count(before)==count;text=text.replace(before,after)
   assert text.encode()==(m/target).read_bytes() and sha(m/target)==row['result_sha256']
 parent='integrated-batch-top1-model-lifecycle-attempt1'
 with zipfile.ZipFile(args.proofs/(parent+'-evidence.zip')) as z:
  for suffix in ['report.json']+['group-'+str(g)+'-repeat-'+str(r)+'-ordinary-last-'+k+'.f32' for g in (0,1) for r in (0,1) for k in ('hidden','normalized')]:
   n='results/'+parent+'/enabled/'+suffix;files[n]=hashlib.sha256(z.read(n)).hexdigest()
 methods=('probe_bf16_low_batch_model_fixed.py','test_bf16_low_batch_model_fixed.bat','run_bf16_low_batch_model_fixed.py','verify_bf16_low_batch_model_fixed.py','prepare_bf16_low_batch_model_fixed.py','bf16_packed_state_gate.py','test_bf16_packed_state_gate.py','test_bf16_model_state_witness.py','test_bf16_same_precision_gate.py','test_bf16_padded_state_gate.py','evidence/bf16-low-batch-model-attempt1-packed-reanalysis.json','evidence/bf16-low-batch-model-fixed-method-derivation.json')
 files.update({'Garnet/test2026/cuda/qwen38_5090/'+n:sha(m/n) for n in methods})
 for n,h in files.items():
  if n.startswith('Garnet/'):assert sha(repo/n.removeprefix('Garnet/'))==h,('Local source drift',n)
 p={'status':'b1_model_fixed_consistency_prepared_unrun','files':files,'parents':parents,'runtime_dependencies':runtime,'scope':'Fresh full unchanged4B4smoke/8B1case plan after immutable first-attempt typed-predicate failure and actual partial-data reanalysis. New packed-versus-packed gate requires real packed geometry/flags plus complete physical/logical/component/slot hashes; old packed-versus-FP32-reference gate untouched.18CPU controls precede fresh B4/B1 model collection. No product math/native DLL/tolerance change, broad quality, actual4K, HTTP timing or2x admission.'}
 target=e/'bf16-low-batch-model-preparation-attempt2.json';data=json.dumps(p,indent=2).encode()
 if target.exists():assert target.read_bytes()==data
 else:target.write_bytes(data)
 plan={'uploads':[{'local':str(m/n),'target':'Garnet/test2026/cuda/qwen38_5090/'+n,'sha256':sha(m/n)} for n in methods]+[{'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(target)}],'scope':p['scope']};data=json.dumps(plan,indent=2).encode()
 if args.plan.exists():assert args.plan.read_bytes()==data
 else:args.plan.write_bytes(data)
 print(json.dumps({'status':p['status'],'files':len(files),'methods':len(methods),'preparation_sha256':sha(target),'library_sha256':runtime['garnet.dll']}));return 0
if __name__=='__main__':raise SystemExit(main())
