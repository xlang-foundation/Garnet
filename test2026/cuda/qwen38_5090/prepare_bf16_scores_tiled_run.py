"""Prepare isolated GPU correctness using the actual privately compiled binary."""
from pathlib import Path
import argparse,hashlib,json
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
METHODS=('prepare_bf16_scores_tiled_run.py','run_bf16_scores_tiled.py','verify_bf16_scores_tiled.py',
 'bf16_low_batch_scores_tiled.cu','bf16_low_batch_scores_tiled.cuh','probe_bf16_scores_tiled.cu',
 'bf16_low_batch_attention_oracle.py','bf16_kv_overlay_oracle.py','monitor_native_probe.py','sample_vram.py',
 'run_logged.py','run_bf16_low_batch_native_build.py','test_bf16_low_batch_candidate.py','test_bf16_kv_overlay_oracle.py',
 'evidence/bf16-score-tiling-candidate.json','evidence/bf16-score-tiling-probe-preparation.json',
 'evidence/bf16-score-tiling-private-build.json','evidence/bf16-score-tiling-private-inputs.json',
 'evidence/bf16-score-tiling-private-compile-only.bat',
 'evidence/bf16-score-tiling-private-build-ledger/commands.jsonl','evidence/bf16-score-tiling-private-build-ledger/commands.log','evidence/nvfp4-config.json')

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);a.add_argument('--executable',type=Path,required=True);a.add_argument('--plan',type=Path,required=True);args=a.parse_args();m=Path(__file__).resolve().parent;repo=m.parents[2]
 parent='tokenizer-nfc-model-consistency-attempt1';proofpath=args.proofs/(parent+'-verified.json');proof=json.loads(proofpath.read_bytes())
 assert proof==json.loads((args.proofs/(parent+'-remote-verified.json')).read_bytes()) and proof['status']=='tokenizer_nfc_model_consistency_replayed'
 assert sha(args.proofs/(parent+'-evidence.zip'))==proof['archive_sha256']
 build=json.loads((m/'evidence/bf16-score-tiling-private-build.json').read_bytes());assert build['status']=='bf16_score_tiling_private_cpu_compile_and_link_passed' and not build['probe_executed']
 assert sha(args.executable)==build['output_sha256']['bf16-scores-tiled.exe']
 files={'Garnet/test2026/cuda/qwen38_5090/'+n:sha(m/n) for n in METHODS}
 files['Garnet/xModel/qwen35/vl_0_8b/bf16_kv_profile.py']=sha(repo/'xModel/qwen35/vl_0_8b/bf16_kv_profile.py')
 for source,digest in build['input_sha256'].items():
  local=Path(source)
  if local.is_relative_to(repo):
   assert sha(local)==digest;files['Garnet/'+local.relative_to(repo).as_posix()]=digest
  elif local.name=='bf16_kv_fixture_layout.h':files['tools/bf16-kv-native/bf16_kv_fixture_layout.h']=digest
 files['tools/bf16-kv-native/bf16-scores-tiled.exe']=sha(args.executable);files['results/'+parent+'-verified.json']=sha(proofpath)
 p={'status':'bf16_scores_tiled_gpu_correctness_prepared_unrun','files':files,'parent':parent,'parent_proof_sha256':sha(proofpath),'parent_archive_sha256':proof['archive_sha256'],'observed_unused_library_sha256':proof['library_sha256'],
    'scope':'Standalone existing SM120 binary compiled from pinned production baseline and separate tiled-score candidate.44actual cases compare all output/workspace bytes,3candidate graph replays,40prefix/cache checks,104malformed input cases rejected by both implementations, readonly/canaries.119040selected scalars need independent FP64 replay. Garnet DLL is hashed but unused. Original HTTP supervisor must first be observed terminal; actual closed HTTP archive SHA supplied at launch. No kernel/model timing, deployment, MTP, B8, broad quality or2x admission.'}
 target=m/'evidence/bf16-scores-tiled-run-preparation.json';raw=json.dumps(p,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
 if not target.exists():target.write_bytes(raw)
 uploads=[{'local':str(m/n),'target':'Garnet/test2026/cuda/qwen38_5090/'+n,'sha256':sha(m/n)} for n in METHODS]
 uploads += [{'local':str(args.executable),'target':'tools/bf16-kv-native/bf16-scores-tiled.exe','sha256':sha(args.executable)},{'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(target)}]
 raw=json.dumps({'uploads':uploads,'scope':p['scope']},indent=2).encode();assert not args.plan.exists() or args.plan.read_bytes()==raw
 if not args.plan.exists():args.plan.write_bytes(raw)
 print(json.dumps({'status':p['status'],'preparation_sha256':sha(target),'upload_targets':len(uploads),'gpu_executed':False}));return 0
if __name__=='__main__':raise SystemExit(main())
