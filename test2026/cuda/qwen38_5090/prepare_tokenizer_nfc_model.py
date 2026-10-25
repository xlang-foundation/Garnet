"""Prepare fresh unchanged model probes after the actual tokenizer native build closes."""
from pathlib import Path
import argparse,hashlib,json,zipfile
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);a.add_argument('--plan',type=Path,required=True);args=a.parse_args();m=Path(__file__).resolve().parent;repo=m.parents[2];build='tokenizer-nfc-native-build-attempt1';old='bf16-low-batch-model-consistency-attempt2';parents={}
 required=[args.proofs/(n+s) for n in (build,old) for s in ('-verified.json','-remote-verified.json','-evidence.zip')]
 if not all(f.is_file() for f in required):print(json.dumps({'status':'tokenizer_model_prerequisites_missing_no_preparation'}));return 2
 for name,status in ((build,'tokenizer_nfc_native_build_replayed'),(old,'bf16_low_batch_model_consistency_replayed')):
  raw=(args.proofs/(name+'-remote-verified.json')).read_bytes();q=json.loads(raw);assert q==json.loads((args.proofs/(name+'-verified.json')).read_bytes()) and q['status']==status and not q['performance_admitted'] and sha(args.proofs/(name+'-evidence.zip'))==q['archive_sha256']
  with zipfile.ZipFile(args.proofs/(name+'-evidence.zip')) as z:
   w=json.loads(z.read('results/'+name+'/workflow.json'));assert w['exit_code']==0 and w['cleanup_finished']
   if name==build:
    assert q['native_cpu_tokenizations_exact']==955 and q['cpu_regressions_passed']==49 and q['replay_verifier_sha256']==q['archived_replay_verifier_sha256']==sha(m/'verify_tokenizer_nfc_native_build.py');files=dict(w['installed_source']);files.update(w['methods_after']);runtime=w['runtime_after'];retained=q['retained_library_sha256']
   else:
    assert q['library_sha256']==retained and w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after'];assert [(v['label'],v['cases']) for v in q['variants']]==[('b4-smoke',4),('b1',8)]
    for n in z.namelist():
     if any(n.startswith('results/'+old+'/'+label+'/') for label in ('b4-smoke','b1')):files[n]=hashlib.sha256(z.read(n)).hexdigest()
  parents[name]={'status':status,'proof_sha256':hashlib.sha256(raw).hexdigest(),'archive_sha256':q['archive_sha256']}
 recipe=json.loads((m/'evidence/tokenizer-nfc-model-method-derivation.json').read_bytes());assert recipe['parent_archive_sha256']==parents[old]['archive_sha256']
 with zipfile.ZipFile(args.proofs/(old+'-evidence.zip')) as z:
  for n,row in recipe['files'].items():
   raw=z.read(row['source']);assert hashlib.sha256(raw).hexdigest()==row['source_sha256'];text=raw.decode()
   for before,after,count in row['changes']:assert text.count(before)==count;text=text.replace(before,after)
   assert text.encode()==(m/n).read_bytes() and sha(m/n)==row['result_sha256']
 methods=('run_tokenizer_nfc_model.py','verify_tokenizer_nfc_model.py','prepare_tokenizer_nfc_model.py','compare_tokenizer_model_parent.py','test_tokenizer_model_parent.py','evidence/tokenizer-nfc-model-method-derivation.json');files.update({'Garnet/test2026/cuda/qwen38_5090/'+n:sha(m/n) for n in methods})
 for n,h in files.items():
  if n.startswith('Garnet/'):assert sha(repo/n.removeprefix('Garnet/'))==h,('Local qualified source drift',n)
 p={'status':'tokenizer_nfc_model_consistency_prepared_unrun','files':files,'parents':parents,'runtime_dependencies':runtime,'scope':'Fresh unchanged4B4/8B1 text/image/color ordinary/MTP/callback/prefix/reset corpus on rebuilt tokenizer DLL;24CPU controls. Original probe/launcher and thresholds unchanged. Additionally compare every captured full logical state and all raw vectors/IDs/callbacks to retained8ae model evidence. Both old failures remain preserved. Source/runtime and owned cache freeze/cleanup; not broad quality, actual4K/HTTP/performance or2x.'};target=m/'evidence/tokenizer-nfc-model-preparation.json';data=json.dumps(p,indent=2).encode();assert not target.exists() or target.read_bytes()==data
 if not target.exists():target.write_bytes(data)
 rows=[{'local':str(m/n),'target':'Garnet/test2026/cuda/qwen38_5090/'+n,'sha256':sha(m/n)} for n in methods]+[{'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(target)}];data=json.dumps({'uploads':rows,'scope':p['scope']},indent=2).encode();assert not args.plan.exists() or args.plan.read_bytes()==data
 if not args.plan.exists():args.plan.write_bytes(data)
 print(json.dumps({'status':p['status'],'files':len(files),'new_methods':len(methods),'preparation_sha256':sha(target),'library_sha256':runtime['garnet.dll']}));return 0
if __name__=='__main__':raise SystemExit(main())
