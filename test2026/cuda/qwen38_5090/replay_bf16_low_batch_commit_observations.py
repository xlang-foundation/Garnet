"""Pure CPU replay of the B1 collector stream, independent of archive transport."""
if __package__:
 from .bf16_low_batch_commit_oracle import case_plan,compare,expected
else:
 from bf16_low_batch_commit_oracle import case_plan,compare,expected

GPU_UUID='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6'

def replay(records,data,frozen_raw,*,gpu_uuid=GPU_UUID):
 assert len(records)==24
 identity=records[0];assert identity['event']=='gpu_identity' and type(identity['sm']) is int and identity['sm']==120 and identity['uuid']==gpu_uuid
 plan=case_plan();assert len(plan)==20
 cases=[r for r in records if r.get('event')=='case_collected'];profiles=[r for r in records if r.get('event')=='profile_passed']
 assert len(cases)==20 and profiles==[{'event':'profile_passed','batch':1,'chunk':chunk,'full_state_byte_cases':13 if chunk==4 else 7,'graph_profiles':2 if chunk==4 else 1,'graph_replays_each':3,'rejected_calls':33,'readonly_inputs_exact':True,'canaries_exact':True} for chunk in (4,32)]
 for profile in profiles:
  for key in ('batch','chunk','full_state_byte_cases','graph_profiles','graph_replays_each','rejected_calls'):assert type(profile[key]) is int
  assert profile['readonly_inputs_exact'] is True and profile['canaries_exact'] is True
 assert [r.get('event') for r in records[1:-1]]==['case_collected']*13+['profile_passed']+['case_collected']*7+['profile_passed']
 assert records[-1]=={'status':'bf16_low_batch_commit_collected_needs_byte_replay','cases':20,'witness_bytes':104704,'performance_admitted':False,'model_qualified':False}
 assert records[-1]['performance_admitted'] is False and records[-1]['model_qualified'] is False
 assert len(data)==len(frozen_raw)==104704
 checked=[];cursor=0
 for case,record in zip(plan,cases):
  wanted=expected(case)
  assert record=={'event':'case_collected',**case,'witness_offset':cursor,'witness_bytes':len(wanted),'gpu_full_state_exact':True,'host_full_state_exact':True,'canaries_exact':True}
  for key in ('id','batch','chunk','source','trial','start','count','witness_offset','witness_bytes'):assert type(record[key]) is int
  for key in ('gpu_full_state_exact','host_full_state_exact','canaries_exact'):assert record[key] is True
  assert frozen_raw[cursor:cursor+len(wanted)]==wanted
  raw=data[cursor:cursor+len(wanted)];checked.append(compare(case,raw));cursor+=len(wanted)
 assert cursor==len(data)
 return checked
