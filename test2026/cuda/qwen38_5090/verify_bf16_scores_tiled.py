"""Independent FP64/raw/source replay for the unwired score-tiling experiment."""
from pathlib import Path
import argparse,hashlib,json,zipfile
from bf16_low_batch_attention_oracle import ATOL,RTOL,case_plan,sample_triples,compare_witness
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();name='bf16-scores-tiled-correctness-attempt1';archive=args.proofs/(name+'-evidence.zip');base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/';numerical=[];failures=[]
 with zipfile.ZipFile(archive) as z:
  doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json');assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
  for n,digest in manifest['files'].items():assert sha(z.read(n))==digest
  w=doc(base+'workflow.json');raw=z.read(methods+'evidence/bf16-scores-tiled-run-preparation.json');p=json.loads(raw);assert sha(raw)==w['preparation_sha256']
  assert w['source_sha256']==w['source_after_sha256']==p['files'] and w['library_unchanged'] and w['cleanup_finished'] and not w['runtime_loaded'] and not w['performance_admitted']
  for n,digest in p['files'].items():assert sha(z.read(n))==digest
  for helper in ('bf16_low_batch_attention_oracle.py','bf16_kv_overlay_oracle.py'):assert sha(Path(__file__).with_name(helper).read_bytes())==sha(z.read(methods+helper))
  proof=doc('results/'+p['parent']+'-verified.json');assert proof['status']=='tokenizer_nfc_model_consistency_replayed' and proof['library_sha256']==p['observed_unused_library_sha256']==w['observed_unused_library_sha256']
  assert sha(z.read('results/'+p['parent']+'-verified.json'))==p['parent_proof_sha256']==w['parent_proof_sha256']
  assert proof['archive_sha256']==p['parent_archive_sha256']==w['parent_archive_sha256']==sha((args.proofs/(p['parent']+'-evidence.zip')).read_bytes())
  closed=doc('results/fixed-length-http-study-attempt1/workflow.json');assert closed['exit_code']==0 and closed['cleanup_finished'] and closed['library_unchanged']
  assert w['closed_http_archive_sha256']==sha((args.proofs/'fixed-length-http-study-attempt1-evidence.zip').read_bytes())
  build=doc(methods+'evidence/bf16-score-tiling-private-build.json');assert sha(z.read('tools/bf16-kv-native/bf16-scores-tiled.exe'))==build['output_sha256']['bf16-scores-tiled.exe']
  passed=w['exit_code']==0
  if passed:
   assert w['status']=='bf16_scores_tiled_collected_needs_fp64_replay'
   for ledger in ('cpu-ledger','gpu-ledger'):
    events=[json.loads(line) for line in z.read(base+ledger+'/commands.jsonl').splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and events[1]['exit_code']==0
   records=[]
   for line in z.read(base+'gpu-ledger/commands.log').splitlines():
    if line.startswith(b'{'):
     value=json.loads(line)
     if 'event' in value or 'status' in value:records.append(value)
   assert len(records)==90 and records[0]['event']=='gpu_identity' and records[0]['sm']==120 and records[0]['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6'
   assert records[-1]=={'status':'bf16_scores_tiled16_collected_needs_fp64_replay','cases':44,'witness_bytes':119040*4,'performance_admitted':False,'model_qualified':False}
   binary=z.read(base+'witnesses.f32');assert len(binary)==119040*4;cursor=0
   for i,case in enumerate(case_plan()):
    started,row=records[1+2*i:3+2*i];assert started=={'event':'case_started',**case} and row['event']=='case_collected' and all(row[k]==v for k,v in case.items())
    triples=sample_triples(case);length=len(triples)*256*4;assert row['witness_offset']==cursor and row['witness_bytes']==length and row['samples']==[list(t) for t in triples]
    assert row['readonly_exact'] and row['canaries_exact'] and row['prefix_cache_exact']==bool(case['end']) and row['graph_replay_exact']==(case['id'] in (1,15,63)) and row['rejected_calls']==(52 if case['id'] in (0,14) else 0)
    try:numerical.append({'id':case['id'],**compare_witness(case,binary[cursor:cursor+length])})
    except ValueError as error:failures.append({'id':case['id'],'error':str(error)})
    cursor+=length
   assert cursor==len(binary)
   resource=doc(base+'resources/resource-monitor.json');samples=[json.loads(line) for line in z.read(base+'resources/resource-vram.jsonl').splitlines()];peak=max(row['used_bytes'] for row in samples)
   assert resource['child_exit_code']==0 and resource['gpu']['uuid']==records[0]['uuid'] and resource['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
  else:peak=None
  archived=sha(z.read(methods+'verify_bf16_scores_tiled.py'))
 passed=passed and not failures
 result={'status':'bf16_scores_tiled_fp64_replayed' if passed else 'bf16_scores_tiled_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'cases':numerical,'numerical_failures':failures,'atol':ATOL,'rtol':RTOL,'correctness_sampled_peak_adapter_bytes':peak,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,'model_integrated':False,'performance_admitted':False,'two_times_target_achieved':False,'scope':p['scope']}
 target=args.proofs/(name+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
 if not target.exists():target.write_bytes(raw)
 print(json.dumps(result,indent=2));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
