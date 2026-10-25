"""Replay explicit prefill16 short target-state admission; no speed/4K claim."""
from pathlib import Path
import argparse,hashlib,json,math,struct,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parent))
from compare_tokenizer_model_parent import compare_reports
from bf16_packed_state_gate import same_packed_bf16_row
sha=lambda raw:hashlib.sha256(raw).hexdigest()
METHODS='Garnet/test2026/cuda/qwen38_5090/'

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();name='mtp-prefill16-model-consistency-attempt1';archive=args.proofs/(name+'-evidence.zip');base='results/'+name+'/';summaries=[];peak=None
 with zipfile.ZipFile(archive) as z:
  doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json');assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
  for n,digest in manifest['files'].items():assert sha(z.read(n))==digest
  w=doc(base+'workflow.json');praw=z.read(METHODS+'evidence/mtp-prefill16-model-run-preparation.json');p=json.loads(praw)
  assert sha(praw)==w['preparation_sha256'] and p['status']=='mtp_prefill16_model_prepared_gpu_unrun'
  assert w['cleanup_finished'] is True and w['library_unchanged'] is True and w['source_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime_dependencies'] and w['library_sha256']==p['runtime_dependencies']['garnet.dll'] and w['parents']==p['parents']
  for n,digest in p['files'].items():assert sha(z.read(n))==digest
  for helper in ('compare_tokenizer_model_parent.py','bf16_packed_state_gate.py'):
   assert sha(Path(__file__).with_name(helper).read_bytes())==sha(z.read(METHODS+helper))
  for parent,row in p['parents'].items():
   raw=z.read('results/'+parent+'-verified.json');q=json.loads(raw);assert sha(raw)==row['proof_sha256'] and q['status']==row['status'] and q['archive_sha256']==row['archive_sha256']==sha((args.proofs/(parent+'-evidence.zip')).read_bytes())
  for field in ('closed_reference','closed_native_admission'):
   entry=w[field];prior=args.proofs/(entry['job']+'-evidence.zip');assert sha(prior.read_bytes())==entry['archive_sha256'] and entry['cleanup_finished'] is True
   with zipfile.ZipFile(prior) as old:
    previous=json.loads(old.read('results/'+entry['job']+'/workflow.json'));assert previous==doc('results/'+entry['job']+'/workflow.json') and previous['exit_code']==entry['exit_code'] and previous['cleanup_finished'] is True
    assert previous['source_sha256']==previous['source_after_sha256'] and previous['runtime_dependencies']==previous['runtime_dependencies_after']
    if field=='closed_native_admission':assert previous['library_unchanged'] is True
    else:assert type(previous['server_exit_code']) is int
  passed=w['exit_code']==0
  if passed:
   assert w['status']=='mtp_prefill16_model_collected_needs_replay'
   for ledger in ('cpu-ledger','gpu-ledger'):
    events=[json.loads(line) for line in z.read(base+ledger+'/commands.jsonl').splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[-1]['event']=='finished' and events[-1]['exit_code']==0
   assert b'Ran 41 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
   study=doc(base+'native/study-report.json');r=doc(base+'native/model/report.json');parent='tokenizer-nfc-model-consistency-attempt1';prefix='results/'+parent+'/b1/';retained=doc(prefix+'report.json')
   assert study['status']=='mtp_prefill16_short_model_collected_needs_replay' and study['sessions_constructed']==1 and len(study['close_observations'])==1 and study['actual_4k_qualified'] is False and study['quality_qualified'] is False and study['performance_admitted'] is False
   assert study['parent_archive_sha256']==p['parents'][parent]['archive_sha256'] and study['parent_proof_sha256']==p['parents'][parent]['proof_sha256']
   assert study['original_probe_sha256']==sha(z.read(METHODS+'probe_bf16_low_batch_model_fixed.py')) and study['candidate_module_sha256']==sha(z.read('Garnet/xModel/qwen35/vl_0_8b/cuda_mtp_prefill16_study.py'))
   assert r['status']=='bf16_low_batch_model_consistency_finished' and r['physical_batch']==1 and r['capacity']==4096 and r['library_sha256']==w['library_sha256'] and r['quality_qualified'] is False and r['performance_admitted'] is False
   comparison=compare_reports(r,retained,lambda n:z.read(base+'native/model/'+n),lambda n:z.read(prefix+n));assert comparison==study['retained_b1_comparison'] and comparison['cases']==8 and comparison['full_canonical_and_raw_vectors_exact']
   snapshots=[]
   def visit(value):
    if isinstance(value,dict):
     if 'logical' in value and 'vectors' in value:snapshots.append(value)
     else:
      for child in value.values():visit(child)
    elif isinstance(value,list):
     for child in value:visit(child)
   visit(r);assert len(snapshots)==comparison['captured_states']
   for state in snapshots:
    assert same_packed_bf16_row(state['logical'],0,state['logical'],0) and set(state['vectors'])=={'hidden','normalized'}
    for entry in state['vectors'].values():
     assert Path(entry['file']).name==entry['file'];raw=z.read(base+'native/model/'+entry['file']);assert len(raw)==entry['bytes']==20480 and sha(raw)==entry['sha256'] and all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',raw))
   assert [(v['key'],v['repeat']) for v in r['cases']]==[(key,repeat) for key in ('s3-text-c1-r0-greedy','s3-image-c1-r0-greedy','red','green') for repeat in (0,1)]
   assert len(study['prefill_call_controls'])==len(retained['cases'])==8
   for case,control,old in zip(r['cases'],study['prefill_call_controls'],retained['cases']):
    assert case['ordinary_mtp_exact'] and case['callbacks_exact'] and case['ordinary_ids']==case['generated_ids'];n=case['prefill_state']['logical']['slots'][0]
    old_calls=(n+3)//4+1;new_calls=(n+15)//16+1
    assert old['speculative_metrics']['mtp_prefill_calls']==old_calls and case['speculative_metrics']['mtp_prefill_calls']==new_calls
    assert control['key']==case['key'] and control['repeat']==case['repeat'] and control['baseline_prefill_calls']==old_calls and control['candidate_prefill_calls']==new_calls
    summaries.append({'key':case['key'],'repeat':case['repeat'],'actual_prompt_tokens':n,'actual_baseline_prefill_calls':old_calls,'actual_candidate_prefill_calls':new_calls,'complete_target_state_and_raw_vectors_exact':True})
   assert [v['count'] for v in r['prefix_controls']]==[0,1,2,4] and all(v['exact'] and v['transaction_closed'] for v in r['prefix_controls']) and len(r['native_cross_dialect_rejections'])==2 and r['pending_guards_rejected']==['parent-reset','nested-verify','prefix-overrun','prefix-bool','prefix-wrong-length']
   info=study['close_observations'][0];assert info['draft_prefill_chunk']==16 and info['draft_decode_chunk']==info['draft_chunk']==4 and info['target_prefill_chunk']==32 and info['extra_draft_engine_weights_duplicated'] is True and info['study_gpu_qualified'] is False and info['study_performance_admitted'] is False and info['prefill16_buffers_bytes']==983360
   graph=info['prefill16_graph'];assert graph['graph_capture_failures']==0 and graph['graph_captures']>=1 and graph['graph_replays']>=1 and graph['device_output_bytes']==33882112 and graph['builder_workspace_limit_bytes']==128<<20
   resource=doc(base+'resources/resource-monitor.json');samples=[json.loads(line) for line in z.read(base+'resources/resource-vram.jsonl').splitlines()];peak=max(v['used_bytes'] for v in samples);assert resource['child_exit_code']==0 and resource['gpu']['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6' and resource['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
  result={'status':'mtp_prefill16_short_target_model_replayed' if passed else 'mtp_prefill16_model_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'library_sha256':w['library_sha256'],'cases':summaries,'correctness_sampled_peak_adapter_used_bytes':peak,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':sha(z.read(METHODS+'verify_mtp_prefill16_model.py')),'actual_4k_qualified':False,'broad_quality_qualified':False,'http_qualified':False,'performance_admitted':False,'two_times_target_achieved':False,'scope':'Actual short8B1 target IDs/callbacks and every captured state/raw vector, prefix/reset/rejection and extra draft T16 engine/call-count/memory replay. Draft engine changes are explicit, target/decode/verifier source and native runtime unchanged. Not full4K/805quality/serving/repetition/speed or2x admission.'}
 target=args.proofs/(name+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
 if not target.exists():target.write_bytes(raw)
 print(json.dumps(result,indent=2));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
