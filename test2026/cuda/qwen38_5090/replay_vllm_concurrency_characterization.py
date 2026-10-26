"""Raw case/wave/variant replay, no model execution or performance admission."""
from pathlib import Path
import hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vllm_concurrency_characterization import control_request,validate_id_response,compare_outputs,overlap,replay_stream_record,sha

def replay(read,groups):
 manifest=json.loads(read('file-manifest.json'))
 for name,digest in manifest['files'].items():assert sha(read(name))==digest
 report=json.loads(read('report.json'));assert report['status']=='reference_concurrency_output_characterization_collected_needs_replay' and report['engine']=='vllm' and report['generation_mode']=='ordinary'
 assert report['benchmark_metrics_collected'] is False and report['physical_gpu_batch_observed'] is False and report['performance_admitted'] is False and report['quality_qualified'] is False
 records=report['records'];assert len(records)==288 and len({v['key'] for v in records})==288 and len(report['waves'])==135
 bykey={v['key']:v for v in records};expected_files={'report.json'};seen=[];wave_index=0;counts={'actual_nonstream_id_requests':0,'actual_full4k_streams':0,'actual_color_streams':0,'concurrent_color_nonstreams':0,'concurrent_color_streams':0,'changed_nonstream_outputs':0,'changed_stream_texts':0}
 for group,cases in groups.items():
  before={};current={};serial_comparisons={}
  phases=[(group+'-serial-before',1,False,cases)]
  phases.append((group+'-c1-stream',1,True,cases))
  for c in (2,4,8):
   phases.extend(((group+'-c'+str(c)+'-ids',c,False,cases),(group+'-c'+str(c)+'-rotated-ids',c,False,cases[3:]+cases[:3])))
   phases.append((group+'-c'+str(c)+'-stream',c,True,cases))
  phases.append((group+'-serial-after',1,False,cases))
  for label,c,stream,ordered in phases:
   for start in range(0,len(ordered),c):
    batch=ordered[start:start+c];rows=[];wave=report['waves'][wave_index];wave_index+=1
    keys=[label+'-'+case['key'] for case in batch]
    assert wave['label']==label and wave['declared_client_concurrency']==c and wave['stream'] is stream and wave['keys']==keys
    for case,key in zip(batch,keys):
     saved=bykey[key];seen.append(key)
     if stream:
      fresh=replay_stream_record(read,key,case);expected_files.update(key+suffix for suffix in ('-request.json','-events.jsonl','-result.json'));counts['actual_color_streams' if group=='mixed-color' else 'actual_full4k_streams']+=1
      if group=='mixed-color' and c>1:counts['concurrent_color_streams']+=1
      assert saved['serial_control_text_exact'] is (fresh['text']==before[case['key']]['text'])
      if c>1:assert saved['previous_concurrent_control_text_exact'] is (fresh['text']==current[case['key']]['text'])
      if not saved['serial_control_text_exact']:counts['changed_stream_texts']+=1
     else:
      wire=read(key+'-request.json');raw=read(key+'-response.json');assert wire==json.dumps(control_request(case['body'],'vllm'),separators=(',',':')).encode();parsed=validate_id_response(json.loads(raw),case);record=json.loads(read(key+'-record.json'))
      fresh={'key':key,'case_key':case['key'],'kind':case['kind'],'request_sha256':sha(wire),'response_sha256':sha(raw),'submitted':record['submitted'],'completed':record['completed'],'prompt_ids_sha256':sha(json.dumps(parsed['prompt_token_ids'],separators=(',',':')).encode()),'completion_token_ids':parsed['completion_token_ids'],'text':parsed['text'],'individual_completion_ids_observed':True};assert record==fresh
      expected_files.update(key+suffix for suffix in ('-request.json','-response.json','-record.json'));counts['actual_nonstream_id_requests']+=1
      if group=='mixed-color' and c>1:counts['concurrent_color_nonstreams']+=1
      if label.endswith('-serial-before'):before[case['key']]=fresh
      elif label.endswith('-serial-after'):serial_comparisons[case['key']]=compare_outputs(fresh,before[case['key']])
      else:
       comparison=compare_outputs(fresh,before[case['key']]);assert saved['serial_control_comparison']==comparison
       if not comparison['completion_ids_exact']:counts['changed_nonstream_outputs']+=1
       if '-rotated-ids' in label:assert saved['previous_concurrent_comparison']==compare_outputs(fresh,current[case['key']])
       else:current[case['key']]=fresh
     for field,value in fresh.items():assert saved[field]==value
     rows.append(fresh)
    assert wave['actual_overlap']==overlap(rows)==len(batch)
  assert report['serial_after_comparisons'][group]==serial_comparisons
 assert seen==[v['key'] for v in records] and wave_index==135 and counts['actual_nonstream_id_requests']==192 and counts['actual_full4k_streams']==64 and counts['actual_color_streams']==32 and counts['concurrent_color_nonstreams']==48 and counts['concurrent_color_streams']==24
 assert set(manifest['files'])==expected_files
 return {'status':'reference_concurrency_output_characterization_raw_replayed',**counts,'actual_client_overlap_waves':135,'serial_after_comparisons':report['serial_after_comparisons'],'physical_gpu_batch_observed':False,'cause_or_harmlessness_proven':False,'broad_quality_qualified':False,'performance_admitted':False,'two_times_target_achieved':False,'scope':'Raw288requests/135client-overlap waves with actual prompt/completion IDs for192nonstreams,64strict3968+128 streams,32color streams and48nonstream/24stream concurrent mixed-color/spatial controls. Short color streams may deliver all tokens together because no decode rate is measured. Every output variant retained. No GPU-batch, numerical-cause/equivalence, broad quality or speed/2x admission.'}
