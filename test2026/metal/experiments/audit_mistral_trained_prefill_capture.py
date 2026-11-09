from pathlib import Path
import hashlib,json,sys
s=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
fixture=json.loads((s/'fixture1.json').read_text());status=json.loads((s/'status1.json').read_text());assert status['status']=='collected_pending_audit' and [(x['label'],x['status'],x['returncode'],x['final_returncode']) for x in status['waves']]==[('control','completed',0,0),('capture','completed',0,0)]
all_events={};results={};boundaries={}
for name in ['control','capture']:
 records=[]
 for line in (s/(name+'-events1.jsonl')).read_text().splitlines():
  if line.startswith('{'):records.append(json.loads(line))
 assert not [r for r in records if r['event']=='error'];all_events[name]=records
 completed=[r for r in records if r['event']=='completed'];assert len(completed)==4
 results[name]={r['id']:r for r in completed};assert results[name]==json.loads((s/(name+'-results1.json')).read_text())
 for entry in fixture['requests']:
  r=results[name]['trained-'+str(entry['row'])];assert r['prompt_ids']==entry['token_ids'];assert r['cached_tokens']==0 and r['finish_reason']=='length' and len(r['generated_ids'])==8
 closes=[r for r in records if r['event']=='closed'];assert len(closes)==1 and records[-1]==closes[0];mem=closes[0]['memory'];assert all(mem[k]==0 for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']);assert mem['metal_allocated_bytes']<=4194304
 boundary=[r for r in records if r['event']=='diagnostic_prefill_capture_boundary'];assert len(boundary)==1;boundary=boundary[0];assert boundary['enabled']==(name=='capture') and boundary['invocation']==16
 expected_ids=[token for entry in fixture['requests'] for token in entry['token_ids'][1920:2048]];assert boundary['ids']==expected_ids and boundary['active']==[1.0]*512 and boundary['slots']==[float(i) for _ in range(4) for i in range(1920,2048)] and len(boundary['epochs'])==4 and all(x>0 for x in boundary['epochs']);boundaries[name]=boundary
 stop=[r for r in records if r['event']=='diagnostic_prefill_capture_stopped'];assert len(stop)==(1 if name=='capture' else 0)
 if name=='capture':assert stop[0]['path']==str(s/'trained-last-prefill1.gputrace')
 for record in records:
  if record['event']=='native_batch':assert len(record['active_rows'])==4 and sum(bool(v) for v in record['active_rows'])<=4
# The exercised call must cover the admitted full32-layer route without fallback.
for name,records in all_events.items():
 index=next(i for i,r in enumerate(records) if r['event']=='diagnostic_prefill_capture_boundary')
 after=records[index+1:]
 def next_event(kind):return next(r for r in after if r['event']==kind)
 affine=next_event('native_affine_profile');assert affine['stage']=='prefill' and affine['affine_dispatches']==224 and affine['misses']==0 and affine['active_rows']==affine['physical_rows']==512
 attention=next_event('native_chunk_attention_mps');assert attention['dispatches']==32 and attention['plan_count']==21 and attention['scratch_bytes']==1044162560 and attention['active_tokens']==512
 bucket=next_event('native_chunk_attention_bucket');assert bucket['dispatches']==32 and bucket['capacity']==2048 and bucket['unknown_controls']==0
 mpp=next_event('native_mpp');assert mpp['affine_dispatches']==224 and mpp['legacy_mpp_dispatches']==64 and mpp['packed_prefill_dispatches']==160
 assert next_event('native_prefill_pool')['bindings']==32 and next_event('native_chunk_kv_scatter')['dispatches']==64
for key in results['control']:
 assert results['control'][key]['generated_ids']==results['capture'][key]['generated_ids'];assert results['control'][key]['text']==results['capture'][key]['text']
assert {k:v for k,v in boundaries['control'].items() if k!='enabled'}=={k:v for k,v in boundaries['capture'].items() if k!='enabled'}
trace=s/'trained-last-prefill1.gputrace';files=[f for f in trace.rglob('*') if f.is_file()];size=sum(f.stat().st_size for f in files);assert size>4*1024**3 and len(files)>20
report={'passed':True,'scope':'One trained native32layer B4/T128 prefill call at offsets1920..2047, exact frozen prompts with8-output diagnostic budgets; separate uncaptured control followed by captured engine. No reference rerun or performance result.','prompt_lengths':[e['prompt_tokens'] for e in fixture['requests']],'capture_rows':512,'capture_call':16,'captured_route':{'affine_dispatches':224,'affine_misses':0,'packed_prefill_dispatches':160,'legacy_mpp_dispatches':64,'MPS_attention_dispatches':32,'attention_bucket_capacity':2048},'captured_slots':[1920,2047],'complete_requests_per_engine':4,'output_token_ids_per_engine':32,'all_captured_outputs_equal_control':True,'all_prompt_ids_equal_frozen_fixture':True,'native_owners_closed':True,'postclose_metal_bytes':{name:all_events[name][-1]['memory']['metal_allocated_bytes'] for name in all_events},'trace_files':len(files),'trace_bytes':size,'stderr_warning':'Capture stderr includes Context leak detected, CoreAnalytics returned false. Native owner counters reach zero; these counters do not explain or dismiss the framework warning.','gpu_replay_inspected':False,'full_cache_or_logits_parity_claim':False,'serving_speed_claim':False,'quality_qualification_claim':False,'goal_complete':False}
(s/'capture-audit1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
