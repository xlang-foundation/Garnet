"""Replay raw B1 HTTP study evidence; scores/timings remain explicitly provisional."""
from pathlib import Path
import argparse,hashlib,io,json,statistics,sys,tempfile,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from compare_http_runs import verify_run
from benchmark_stream import TokenEvents
from benchmark_fixed_context_native import final_ids
from test2026.cuda.qwen38_5090.verify_mmmu_http_quality import replay
from test2026.cuda.qwen38_5090.mmmu_http_quality import load_corpus
sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);a.add_argument('--corpus',type=Path,required=True);a.add_argument('--scratch',type=Path,required=True);args=a.parse_args();assert args.scratch.is_dir()
 name='tokenizer-nfc-http-study-attempt1';archive=args.proofs/(name+'-evidence.zip');base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/';profiles=[]
 with zipfile.ZipFile(archive) as z:
  doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json');assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
  for n,h in manifest['files'].items():assert sha(z.read(n))==h
  w=doc(base+'workflow.json');raw=z.read(methods+'evidence/tokenizer-nfc-http-preparation.json');p=json.loads(raw);assert sha(raw)==w['preparation_sha256']
  assert w['source_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime_dependencies'] and w['library_unchanged'] and w['cleanup_finished'] and not w['performance_admitted']
  assert w['library_sha256']==p['runtime_dependencies']['garnet.dll'] and w['parent_proof_sha256']==p['parent_proof_sha256'] and w['parent_archive_sha256']==p['parent_archive_sha256']==manifest['parent_archive_sha256']
  for n,h in p['files'].items():assert sha(z.read(n))==h
  parent='tokenizer-nfc-model-consistency-attempt1';raw=z.read('results/'+parent+'-verified.json');parent_proof=json.loads(raw);assert sha(raw)==p['parent_proof_sha256'] and parent_proof['status']=='tokenizer_nfc_model_consistency_replayed' and parent_proof['archive_sha256']==p['parent_archive_sha256'] and parent_proof['library_sha256']==w['library_sha256']
  passed=w['exit_code']==0
  if passed:
   assert w['status']=='tokenizer_nfc_http_study_collected_needs_replay' and [v['mode'] for v in w['variants']]==['ordinary','mtp'] and all(v['finished'] and v['exit_code']==0 for v in w['variants'])
   assert b'Ran 29 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
   _,rows,_=load_corpus(args.corpus)
   with tempfile.TemporaryDirectory(prefix='b1-http-replay-',dir=args.scratch) as temporary:
    folder=Path(temporary).resolve();assert folder.is_relative_to(args.scratch.resolve())
    for mode in ('ordinary','mtp'):
     prefix=base+mode+'/';report=doc(prefix+'report.json');assert report['status']=='tokenizer_nfc_http_profile_collected' and report['mode']==mode and report['library_sha256']==w['library_sha256'] and report['physical_batch']==1 and report['queued_concurrency'] and not report['performance_admitted'] and not report['broad_quality_qualified']
     assert len(report['protocol_cases'])==12 and len(report['boundary_controls'])==2 and report['exploratory_short_pilot_collected'] and report['fixed_context_benchmark_collected']
     quality_prefix=prefix+'mmmu-quality/';destination=folder/mode;destination.mkdir()
     for n in manifest['files']:
      if not n.startswith(quality_prefix):continue
      relative=n.removeprefix(quality_prefix);target=destination/relative;assert target.resolve().is_relative_to(destination.resolve());target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(n))
     screening=replay(args.corpus,destination);assert screening==doc(prefix+'mmmu-quality-replayed.json')==report['adapted_screening'] and screening['status']=='adapted_mmmu_http_quality_responses_replayed'
     model_report=doc('results/'+parent+'/b1/report.json');expected={c['key']:c['generated_ids'][0] for c in model_report['cases']}
     native_tokens={};native_completed={};ready=[];used=set()
     for line in z.read(prefix+'worker/worker-stdout.jsonl').splitlines():
      try:event=json.loads(json.loads(line)['line'])
      except ValueError:continue
      if not isinstance(event,dict):continue
      if event.get('event')=='ready':ready.append(event)
      elif event.get('event')=='token':
       key=event['id'];assert key not in native_completed;native_tokens.setdefault(key,[]).append(event['token_id']);assert event['sampled_tokens']==len(native_tokens[key])
      elif event.get('event')=='completed':
       key=event['id'];assert key not in native_completed and native_tokens[key]==event['generated_ids'] and event['physical_batch']==event['real_rows']==1 and event['padded_rows']==0;native_completed[key]=event
      elif event.get('event')=='error':raise AssertionError(event)
     assert len(ready)==1 and ready[0]==report['ready'] and ready[0]['library_sha256']==w['library_sha256']
     def native_match(key,prompt,completion):
      assert key not in used and key in native_completed;record=native_completed[key];assert record['prompt_ids']==prompt and record['generated_ids']==completion;used.add(key)
     for row in rows:
      response=doc(quality_prefix+'cases/'+row['id']+'/response.bin');choice=response['choices'][0];native_match(response['id'],choice['prompt_token_ids'],choice['response_token_ids'])
     for item in report['protocol_cases']:
      key=item['key']
      if key.startswith('protocol-'):
       original=key.split('-',2)[2];response=doc(prefix+key+'-response.json');choice=response['choices'][0];request=doc(prefix+key+'-request.json');oracle=doc('results/native-mtp3-http/'+original+'-response.json')['choices'][0];assert choice['prompt_token_ids']==oracle['prompt_token_ids'] and choice['response_token_ids']==expected[original] and item['completion_ids_exact'];native_match(response['id'],choice['prompt_token_ids'],choice['response_token_ids'])
      else:
       original=key.removeprefix('stream-');events=[json.loads(line) for line in z.read(prefix+key+'-events.jsonl').splitlines()];data=[json.loads(v['data']) for v in events if v['data']!='[DONE]'];final=[v for v in data if 'response_token_ids' in v];assert len(final)==1 and final[0]['response_token_ids']==expected[original] and final[0]['prompt_token_ids']==doc('results/native-mtp3-http/'+original+'-response.json')['choices'][0]['prompt_token_ids'];native_match(final[0]['id'],final[0]['prompt_token_ids'],final[0]['response_token_ids'])
       state=TokenEvents()
       for event in events:state.feed(event['data'],event['monotonic'])
       result=doc(prefix+key+'-result.json');fresh=state.result(result['submitted'],result['completed'],doc(prefix+key+'-request.json')['max_tokens']);assert fresh['generated_tokens']==len(expected[original]) and result['status']=='completed'
     with zipfile.ZipFile(io.BytesIO(z.read(methods+'evidence/fixed-context-4096-corpus.zip'))) as context:
      manifest_raw=context.read('manifest.json');assert sha(manifest_raw)=='829d7770f2d749e30a66e027940224e8ab77a488d0b5f9c456e4809ae845987c';context_manifest=json.loads(manifest_raw);context_ids={r['key']:json.loads(context.read(r['expected_prompt_ids_file'])) for r in context_manifest['rows']}
      for case in ('text','image'):
       row=next(r for r in context_manifest['rows'] if r['key']=='ctx4096-s3-'+case+'-c1-warmup-0-0');expected_ids=json.loads(context.read(row['expected_prompt_ids_file']));body=json.loads(context.read(row['request_file']));body['stream']=False;assert doc(prefix+'boundary-'+case+'-request.json')==body
       response=doc(prefix+'boundary-'+case+'-response.json');choice=response['choices'][0];assert choice['prompt_token_ids']==expected_ids and len(expected_ids)==3968 and len(choice['response_token_ids'])==128 and response['usage']=={'prompt_tokens':3968,'completion_tokens':128,'total_tokens':4096};native_match(response['id'],choice['prompt_token_ids'],choice['response_token_ids'])
       if mode=='mtp':assert choice==doc(base+'ordinary/boundary-'+case+'-response.json')['choices'][0]
     if mode=='mtp':
      assert report['ordinary_mtp_screening_token_mismatches']==[]
      for row in rows:
       suffix='mmmu-quality/cases/'+row['id']+'/response.bin';a=doc(base+'ordinary/'+suffix)['choices'][0];b=doc(prefix+suffix)['choices'][0];assert (a['prompt_token_ids'],a['response_token_ids'])==(b['prompt_token_ids'],b['response_token_ids'])
     def benchmark_replay(directory,fixed):
      root=prefix+directory+'/';benchmarks=[n for n in manifest['files'] if n.startswith(root) and n.endswith('-manifest.json')];assert len(benchmarks)==1;benchmark=doc(benchmarks[0]);result=verify_run(benchmark,lambda n:z.read(root+n));assert len(result['cohorts'])==8 and benchmark['gpu']['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6'
      if fixed:assert benchmark['corpus_manifest_sha256']=='829d7770f2d749e30a66e027940224e8ab77a488d0b5f9c456e4809ae845987c' and benchmark['actual_input_tokens_required']==3968 and benchmark['actual_completion_tokens_required']==128
      samples=[json.loads(line) for line in z.read(root+benchmark['run_id']+'-vram.jsonl').splitlines()];memory={}
      for cohort in benchmark['cohorts']:
       phase=cohort['case']+'-c'+str(cohort['concurrency']);selected=[s for s in samples if s['phase']==phase];gaps=[b['monotonic']-a['monotonic'] for a,b in zip(selected,selected[1:])];peak=max(s['used_bytes'] for s in selected)
       assert peak==cohort['sampled_peak_adapter_used_bytes'] and len(selected)==cohort['samples'] and cohort['requested_interval_seconds']==.02 and cohort['observed_sample_gap_median_seconds']==(statistics.median(gaps) if gaps else None) and cohort['observed_sample_gap_max_seconds']==(max(gaps) if gaps else None)
       if fixed:assert cohort['prompt_tokens']==[3968]*8 and cohort['generated_tokens']==1024
       memory[phase]={'sampled_peak_adapter_used_bytes':peak,'samples':len(selected),'scope':cohort['scope']}
      actual_context_keys=set()
      for n in manifest['files']:
       if not(n.startswith(root) and n.endswith('-events.jsonl')):continue
       events=[json.loads(line) for line in z.read(n).splitlines()];values=[json.loads(v['data']) for v in events if v['data']!='[DONE]'];final=[v for v in values if 'response_token_ids' in v];assert len(final)==1
       if fixed:
        key=Path(n).name.removesuffix('-events.jsonl').removeprefix(benchmark['run_id']+'-');actual=final_ids(events,context_ids[key]);actual_context_keys.add(key)
        if mode=='mtp':
         ordinary_manifests=[v for v in manifest['files'] if v.startswith(base+'ordinary/benchmark4k/') and v.endswith('-manifest.json')];assert len(ordinary_manifests)==1;ordinary=doc(ordinary_manifests[0]);assert benchmark['ordinary_manifest_sha256']==sha(z.read(ordinary_manifests[0]));previous=[json.loads(line) for line in z.read(base+'ordinary/benchmark4k/'+ordinary['run_id']+'-'+key+'-events.jsonl').splitlines()];assert actual['response_token_ids']==final_ids(previous,context_ids[key])['response_token_ids']
       native_match(final[0]['id'],final[0]['prompt_token_ids'],final[0]['response_token_ids'])
      if fixed:assert actual_context_keys==set(context_ids) and len(actual_context_keys)==124
      return {'raw_replay':result,'sampled_memory':memory,'actual_input_tokens_each':3968 if fixed else None}
     short=benchmark_replay('benchmark',False);fixed=benchmark_replay('benchmark4k',True)
     assert used==set(native_completed) and len(used)==1067
     resource=doc(base+mode+'-resources/resource-monitor.json');all_samples=[json.loads(line) for line in z.read(base+mode+'-resources/resource-vram.jsonl').splitlines()];whole_peak=max(s['used_bytes'] for s in all_samples);assert resource['child_exit_code']==0 and resource['gpu']['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6' and resource['sampled_memory']['sampled_peak_adapter_used_bytes']==whole_peak
     profiles.append({'mode':mode,'screening':screening,'protocol_controls':12,'actual_4k_boundary_controls':2,'short_pilot':short,'fixed_4k_benchmark':fixed,'whole_study_sampled_peak_adapter_bytes':whole_peak,'scope':'Actual raw805screen,two4Kcontrols and complete124request4K/short cohorts. Single-slot queued native serving; whole-study memory includes startup/quality while benchmark memory is cohort-specific. Adapter-wide sampled maxima. Matched references,broad quality and2x remain open.'})
  archived=sha(z.read(methods+'verify_tokenizer_nfc_http.py'))
 report={'status':'tokenizer_nfc_http_study_replayed' if passed else 'tokenizer_nfc_http_study_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'library_sha256':w['library_sha256'],'profiles':profiles,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,'native_fixed_4k_metrics_replayed':passed,'matched_4k_reference_comparison_qualified':False,'broad_quality_qualified':False,'matched_tuned_reference_comparison_qualified':False,'performance_admitted':False,'two_times_target_achieved':False,'scope':p['scope']}
 (args.proofs/(name+'-verified.json')).write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
