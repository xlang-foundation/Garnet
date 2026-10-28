"""Replay the actual failed trial's complete screening and legitimate early-EOS stream."""
from pathlib import Path
import argparse,hashlib,io,json,statistics,sys,tempfile,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.verify_mmmu_http_quality import replay
from benchmark_stream import TokenEvents
from metrics import summarize
sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);a.add_argument('--corpus',type=Path,required=True);a.add_argument('--scratch',type=Path,required=True);a.add_argument('--tokenizer',type=Path,required=True);args=a.parse_args();name='tokenizer-nfc-http-study-attempt1';base='results/'+name+'/';prefix=base+'ordinary/';proof=json.loads((args.proofs/(name+'-verified.json')).read_bytes());assert proof==json.loads((args.proofs/(name+'-remote-verified.json')).read_bytes()) and proof['status']=='tokenizer_nfc_http_study_failure_preserved';archive=args.proofs/(name+'-evidence.zip');assert sha(archive.read_bytes())==proof['archive_sha256']
 assert sha(args.tokenizer.read_bytes())=='0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3'
 import tokenizers
 tokenizer=tokenizers.Tokenizer.from_file(str(args.tokenizer));stop_id=tokenizer.token_to_id('<|im_end|>')
 with zipfile.ZipFile(archive) as z:
  manifest=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None
  for path,digest in manifest['files'].items():assert sha(z.read(path))==digest
  doc=lambda path:json.loads(z.read(path));workflow=doc(base+'workflow.json');assert workflow['cleanup_finished'] and workflow['exit_code']==1 and workflow['source_sha256']==workflow['source_after_sha256'] and workflow['runtime_dependencies']==workflow['runtime_dependencies_after'] and workflow['library_unchanged']
  with tempfile.TemporaryDirectory(prefix='natural-eos-replay-',dir=args.scratch) as temporary:
   folder=Path(temporary).resolve();assert folder.is_relative_to(args.scratch.resolve())
   for n in manifest['files']:
    if not n.startswith(prefix+'mmmu-quality/'):continue
    target=folder/n.removeprefix(prefix+'mmmu-quality/');assert target.resolve().is_relative_to(folder);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(n))
   screening=replay(args.corpus,folder);assert screening==doc(prefix+'mmmu-quality-replayed.json') and screening['completed_requests']==805 and screening['status']=='adapted_mmmu_http_quality_responses_replayed'
  paths=[n for n in manifest['files'] if n.startswith(prefix+'benchmark4k/') and n.endswith('-manifest.json')];assert len(paths)==1;benchmark=doc(paths[0]);assert benchmark['status']=='failed';root=prefix+'benchmark4k/';short=[]
  def actual(key):
   body=doc(root+key+'-request.json');result=doc(root+key+'-result.json');state=TokenEvents();events=[json.loads(line) for line in z.read(root+key+'-events.jsonl').splitlines()]
   for event in events:state.feed(event['data'],event['monotonic'])
   fresh=state.result(result['submitted'],result['completed'],body['max_tokens'])
   for field,value in fresh.items():assert result[field]==value,(key,field)
   assert result['request_sha256']==sha(z.read(root+key+'-request.json'));return body,result,events
  for n in manifest['files']:
   if n.startswith(root) and n.endswith('-result.json') and doc(n).get('generated_tokens')!=128:short.append(n.removeprefix(root).removesuffix('-result.json'))
  assert len(short)==1;key=short[0];body,result,events=actual(key);values=[json.loads(e['data']) for e in events if e['data']!='[DONE]'];final=next(v for v in values if 'response_token_ids' in v);tokens=[v['token_id'] for v in values if 'token_id' in v]
  assert body.get('ignore_eos',False) is False and body['max_tokens']==128 and result['finish_reason']=='stop' and len(tokens)==74 and tokens==final['response_token_ids'] and tokens[-1]==stop_id and final['usage']=={'prompt_tokens':3968,'completion_tokens':74,'total_tokens':4042}
  native=[]
  for line in z.read(prefix+'worker/worker-stdout.jsonl').splitlines():
   try:event=json.loads(json.loads(line)['line'])
   except ValueError:continue
   if isinstance(event,dict) and event.get('event')=='completed' and event.get('id')==final['id']:native.append(event)
  assert len(native)==1 and native[0]['generated_ids']==tokens and native[0]['prompt_ids']==final['prompt_token_ids'] and native[0]['finish_reason']=='stop'
  with zipfile.ZipFile(io.BytesIO(z.read('Garnet/test2026/cuda/qwen38_5090/evidence/fixed-context-4096-corpus.zip'))) as context:
   corpus=json.loads(context.read('manifest.json'));row=next(r for r in corpus['rows'] if r['key']==key.removeprefix(benchmark['run_id']+'-'));assert context.read(row['request_file'])==z.read(root+key+'-request.json') and json.loads(context.read(row['expected_prompt_ids_file']))==final['prompt_token_ids']
  early_events_sha256=sha(z.read(root+key+'-events.jsonl'))
  partial=[];samples=[json.loads(line) for line in z.read(root+benchmark['run_id']+'-vram.jsonl').splitlines()]
  for cohort in benchmark['cohorts']:
   rows=[actual(k)[1] for k in cohort['request_keys']];metrics=summarize(rows)
   for field,value in metrics.items():assert cohort[field]==value
   phase=cohort['case']+'-c'+str(cohort['concurrency']);selected=[s for s in samples if s['phase']==phase];gaps=[b['monotonic']-a['monotonic'] for a,b in zip(selected,selected[1:])];assert max(s['used_bytes'] for s in selected)==cohort['sampled_peak_adapter_used_bytes'] and max(gaps)==cohort['observed_sample_gap_max_seconds']
   partial.append({'case':cohort['case'],'concurrency':cohort['concurrency'],'metrics':metrics,'sampled_peak_adapter_used_bytes':cohort['sampled_peak_adapter_used_bytes'],'observed_sample_gap_max_seconds':max(gaps),'performance_admitted':False})
 report={'status':'native_natural_eos_failure_and_complete_screen_replayed','archive_sha256':proof['archive_sha256'],'screening':screening,'failed_fixed_length_case':key,'actual_stream_and_final_and_native_ids_exact':True,'actual_prompt_tokens':3968,'actual_generated_tokens':74,'declared_max_tokens':128,'ignore_eos_requested':False,'terminal_token_id':stop_id,'terminal_token_name':'<|im_end|>','finish_reason':'stop','early_eos_events_file':root+key+'-events.jsonl','early_eos_events_sha256':early_events_sha256,'completed_partial_cohorts':partial,'verifier_sha256':sha(Path(__file__).read_bytes()),'performance_admitted':False,'two_times_target_achieved':False,'scope':'Original trial remains failed: valid74token natural-stop response violates its predeclared128output workload. All805raw screen replies independently replayed; partial cohort metrics are retained observations from a failed overall trial, excluded from admitted comparisons/charts. No MTP run or completed fixed4Kprofile. New explicit ignore_eos profile must keep all original failures and establish actual128tokens on every engine.'};target=args.proofs/'Garnet-Native-Natural-EOS-Failure-Reanalysis.json';assert not target.exists();target.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in report.items() if k not in ('screening','completed_partial_cohorts')},indent=2));print(json.dumps({'screening_count':screening['score']['count'],'screening_correct':screening['score']['correct'],'screening_accuracy':screening['score']['accuracy'],'partial_completed_cohorts':len(partial)}));return 0
if __name__=='__main__':raise SystemExit(main())
