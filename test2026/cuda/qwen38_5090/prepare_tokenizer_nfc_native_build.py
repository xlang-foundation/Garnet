"""Bind the actual closed model/failure and CPU normalization proof for a rebuild."""
from pathlib import Path
import argparse,hashlib,json,zipfile
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
PRODUCT=('src/tokenizer/qwen_tokenizer.cpp','src/tokenizer/qwen_tokenizer.h','src/tokenizer/qwen_unicode_categories.h','src/tokenizer/qwen_unicode_normalization.h','src/tokenizer/qwen_unicode_normalization_data.h')
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);a.add_argument('--work',type=Path,required=True);args=a.parse_args();m=Path(__file__).resolve().parent;repo=m.parents[2];parents={}
 for name,status,code in (('bf16-low-batch-model-consistency-attempt2','bf16_low_batch_model_consistency_replayed',0),('bf16-low-batch-http-study-attempt1','bf16_low_batch_http_study_failure_preserved',1)):
  raw=(args.proofs/(name+'-remote-verified.json')).read_bytes();q=json.loads(raw);assert q==json.loads((args.proofs/(name+'-verified.json')).read_bytes()) and q['status']==status and sha(args.proofs/(name+'-evidence.zip'))==q['archive_sha256']
  with zipfile.ZipFile(args.proofs/(name+'-evidence.zip')) as z:
   w=json.loads(z.read('results/'+name+'/workflow.json'));assert w['exit_code']==code and w['cleanup_finished'] and w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after']
   if code==0:base=w['source_sha256'];runtime=w['runtime_dependencies']
  parents[name]={'status':status,'proof_sha256':hashlib.sha256(raw).hexdigest(),'archive_sha256':q['archive_sha256'],'exit_code':code}
 cpu=args.proofs/'tokenizer-nfc-cpu-qualified-verified.json';c=json.loads(cpu.read_bytes());assert c['status']=='tokenizer_nfc_cpu_evidence_replayed' and c['corpus_cases_exact']==955 and c['normalization_scalar_cases_exact']==1112064 and c['normalization_sequence_cases_exact']==30012 and c['full_encode_cases_exact']==5027 and c['replay_verifier_sha256']==sha(m/'verify_tokenizer_nfc_cpu.py')
 archive=args.proofs/'tokenizer-nfc-cpu-qualified-evidence.zip';assert sha(archive)==c['archive_sha256'];oracle=args.work/'tokenizer-nfc-cpu-oracles.zip';assert sha(oracle)==c['oracle_archive_sha256'];candidate={n:sha(repo/n) for n in PRODUCT};assert all(c['source_sha256'][n]==h for n,h in candidate.items())
 inventory=args.work/'tokenizer-nfc-native-build-attempt1-inventory.json';i=json.loads(inventory.read_bytes());assert i['status']=='tokenizer_native_prebuild_inventory_passed' and i['request']['base_files']==base and i['request']['runtime']==runtime and i['cuda_objects'];base=dict(base);base['Garnet/src/tokenizer/tokenizer.h']=i['tokenizer_base']['src/tokenizer/tokenizer.h'];baseline={n:i['tokenizer_base'][n] for n in PRODUCT}
 for n,h in base.items():
  if n.startswith('Garnet/'):assert sha(repo/n.removeprefix('Garnet/'))==h
 previous=json.loads((m/'evidence/bf16-low-batch-native-build-preparation.json').read_bytes());assert i['request']['environment']==previous['environment_files'];bundle=args.work/'tokenizer-nfc-native-source.zip';assert not bundle.exists()
 with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
  for n in PRODUCT:z.writestr(n,(repo/n).read_bytes())
 names=('prepare_tokenizer_nfc_native_build.py','run_tokenizer_nfc_native_build.py','verify_tokenizer_nfc_native_build.py','probe_native_tokenizer_frozen_cpu.py','verify_tokenizer_nfc_cpu.py');methods={'Garnet/test2026/cuda/qwen38_5090/'+n:sha(m/n) for n in names}
 uploads=[{'local':str(m/n),'target':'Garnet/test2026/cuda/qwen38_5090/'+n,'sha256':sha(m/n)} for n in names]
 inputs={'results/'+f.name:sha(f) for f in (cpu,archive,oracle,inventory,bundle)};inputs.update({'tools/'+n:sha(args.work/n) for n in ('inspect-tokenizer-native-build.py','tokenizer-native-inventory-input.json')})
 p={'status':'tokenizer_nfc_native_build_prepared_unrun','parents':parents,'cpu_proof':c,'base_files':base,'runtime':runtime,'environment':i['request']['environment'],'candidate_files':candidate,'candidate_base':baseline,'methods':methods,'inputs':inputs,'source_zip':'results/'+bundle.name,'cache_before_sha256':i['cache_sha256'],'cuda_objects_before':i['cuda_objects'],'vcvars':previous['vcvars'],'test_modules':previous['test_modules'],'scope':'Preserve8ae DLL and2replaced tokenizer files, install5CPU tokenizer files, reuse existing Release/MT/sm120/default-off native build and preserved SDK.49unchanged CPU profile regressions plus955actual rebuilt native C API tokenizations. Record CUDA object hashes; no automatic GPU model, HTTP, quality, speed or2x admission. Failure attempts guarded source/library restore.'}
 target=m/'evidence/tokenizer-nfc-native-build-preparation.json';assert not target.exists();target.write_text(json.dumps(p,indent=2),encoding='utf8');uploads.extend([{'local':str(bundle),'target':'results/'+bundle.name,'sha256':sha(bundle)},{'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(target)}]);(args.work/'tokenizer-nfc-native-build-upload-plan.json').write_text(json.dumps({'uploads':uploads},indent=2),encoding='utf8');print(json.dumps({'status':p['status'],'source_files':5,'replaced_files':2,'methods':len(names),'preparation_sha256':sha(target)}))
if __name__=='__main__':main()
