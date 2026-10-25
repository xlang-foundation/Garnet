"""Pin real model consistency collection to the closed native build and parents."""
from pathlib import Path
import argparse,hashlib,json,zipfile
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);a.add_argument('--plan',type=Path,required=True);args=a.parse_args()
 m=Path(__file__).resolve().parent;repo=m.parents[2];e=m/'evidence';name='bf16-low-batch-native-build-attempt1'
 required=[args.proofs/(name+s) for s in ('-verified.json','-remote-verified.json','-evidence.zip')]
 if not all(f.is_file() for f in required):print(json.dumps({'status':'b1_model_native_build_prerequisite_missing'}));return 2
 raw=(args.proofs/(name+'-remote-verified.json')).read_bytes();q=json.loads(raw);assert q==json.loads((args.proofs/(name+'-verified.json')).read_bytes()) and q['status']=='bf16_low_batch_native_build_replayed' and not q['performance_admitted']
 assert sha(args.proofs/(name+'-evidence.zip'))==q['archive_sha256'] and q['replay_verifier_sha256']==q['archived_replay_verifier_sha256']==sha(m/'verify_bf16_low_batch_native_build.py')
 with zipfile.ZipFile(args.proofs/(name+'-evidence.zip')) as z:
  w=json.loads(z.read('results/'+name+'/workflow.json'));assert w['exit_code']==0 and w['cleanup_finished'] and w['library_sha256']==q['library_sha256']
  files=dict(w['installed_source']);files.update(w['methods']);runtime=w['runtime_after'];parents=dict(w['parents'])
 parents[name]={'status':q['status'],'proof_sha256':hashlib.sha256(raw).hexdigest(),'archive_sha256':q['archive_sha256']}
 parent='integrated-batch-top1-model-lifecycle-attempt1'
 with zipfile.ZipFile(args.proofs/(parent+'-evidence.zip')) as z:
  for suffix in ['report.json']+['group-'+str(g)+'-repeat-'+str(r)+'-ordinary-last-'+k+'.f32' for g in (0,1) for r in (0,1) for k in ('hidden','normalized')]:
   n='results/'+parent+'/enabled/'+suffix;files[n]=hashlib.sha256(z.read(n)).hexdigest()
 methods=('probe_bf16_low_batch_model.py','test_bf16_low_batch_model.bat','run_bf16_low_batch_model.py','verify_bf16_low_batch_model.py','prepare_bf16_low_batch_model.py')
 files.update({'Garnet/test2026/cuda/qwen38_5090/'+n:sha(m/n) for n in methods})
 for n,h in files.items():
  if n.startswith('Garnet/'):assert sha(repo/n.removeprefix('Garnet/'))==h,('Local source drift',n)
 p={'status':'b1_model_consistency_prepared_unrun','files':files,'parents':parents,'runtime_dependencies':runtime,'scope':'Sequential independent native B4 text/image row0 smoke then true B1 four-request/two-repeat ordinary/MTP/full-state/prefix0,1,2,4/reset/callback controls. Preserve old DLLs, current source/runtime and raw failures. Reuse owned SSD scratch and retire only observed private engine files after terminal child. B1-vs-B4 numerical/token differences retained explicitly. Not a full legacy14case regression, broad quality, actual4K, HTTP timing, serving memory or2x qualification.'}
 target=e/'bf16-low-batch-model-preparation.json';data=json.dumps(p,indent=2).encode()
 if target.exists():assert target.read_bytes()==data
 else:target.write_bytes(data)
 plan={'uploads':[{'local':str(m/n),'target':'Garnet/test2026/cuda/qwen38_5090/'+n,'sha256':sha(m/n)} for n in methods]+[{'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(target)}],'scope':p['scope']};data=json.dumps(plan,indent=2).encode()
 if args.plan.exists():assert args.plan.read_bytes()==data
 else:args.plan.write_bytes(data)
 print(json.dumps({'status':p['status'],'files':len(files),'methods':len(methods),'preparation_sha256':sha(target),'library_sha256':runtime['garnet.dll']}));return 0
if __name__=='__main__':raise SystemExit(main())
