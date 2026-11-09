"""Audit current ordinary parity, real B1 proposals and C4 native lifecycles."""
from pathlib import Path
import copy,hashlib,importlib.util,json,sys
R=Path(__file__).resolve().parent;M=R.parent;B=M/'mistral-bounded-draft-long1'
sys.path.insert(0,str(B));sys.path.insert(0,str(R))
from audit_owner1 import check as owner_check
spec=importlib.util.spec_from_file_location('bounded_qualification',B/'audit_qualification1.py');bounded=importlib.util.module_from_spec(spec);spec.loader.exec_module(bounded)
bounded.ROOT=B;bounded.M=M;bounded.SG=M/'mistral-fused-mlp-serving1';bounded.N=M/'mistral-fused-q4-decode-native1'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for buf in iter(lambda:f.read(8*1024*1024),b''):h.update(buf)
 return h.hexdigest()
def need(v,m):
 if not v:raise ValueError(m)
PLAN=read(R/'qualification-plan1.json');BUILD=read(R/'build-audit1.json');REFERENCE=read(B/'qualification1-result.json')['records'][0]
TAG='native_mistral_verification_gqa16_scores32_optional_v1'
def b1_check(data,mode):
 need(data['gqa_scores_enabled']==(mode=='on') and (TAG in data['backend'])==(mode=='on'),'compiled GQA16 capability')
 need(data['split_scores_enabled'] is True,'control split capability changed')
 need(data['library']==PLAN['libraries'][mode]+'/libgarnet.dylib','wrong native library')
 summary=bounded.check(data)
 for row in data['records']:
  for k in ['case_id','prompt_ids','generated_ids','slots','state_sha256','hidden_sha256']:need(row[k]==REFERENCE[k],'ordinary output/full state mismatch '+k)
 for q in data['records'][1]['rounds']:
  if q['metrics']:
   need(q['metrics']['target_attention_gqa16_score_dispatches']==(32 if mode=='on' else 0),'GQA16 route proof')
   need(q['metrics']['target_attention_split_score_dispatches']==(0 if mode=='on' else 32),'wrong control route')
 return summary

def c4_check(ordinary,candidate):
 need(TAG in candidate['backend'] and TAG in ordinary['backend'],'C4 capability missing')
 need(candidate['mode'] in ['verify','reverse'],'wrong C4 candidate arm')
 copy_candidate=copy.deepcopy(candidate);copy_candidate['mode']='verify';summary=owner_check(ordinary,copy_candidate)
 counts=set();depths=set();calls=0;restores=0
 jobs={j['native_request']:j for j in candidate['jobs']}
 for tick in candidate['ticks']:
  for p,e in zip(tick['plan']['batches'],tick['executions']):
   if p['stage']!='decode':continue
   calls+=1;wantTokens=[];wantFlags=[];wantSlots=[];byrow={q['row']:q for q in p['requests']}
   for row in range(4):
    q=byrow.get(row);budget=jobs[q['id']]['limit']-q['generated_tokens'] if q else 0;depth=min(4,budget)
    if depth:depths.add(depth)
    for t in range(4):
     wantTokens.append(q['token_id'] if q and t==0 else 0)
     wantFlags.append(float(t<depth));wantSlots.append(float(q['slot']+t) if t<depth else 0.)
   need(e['submitted_tokens']==wantTokens and e['submitted_active']==wantFlags and e['submitted_positions']==wantSlots,'C4 zero proposals/native budget/positions')
   counts.add(int(sum(wantFlags)));m=e['native']['verification']
   need(m['target_attention_gqa16_score_dispatches']==32 and m['target_attention_split_score_dispatches']==0,'C4 actual GQA16 route')
   need(m['target_indexed_decode_dispatches']==225 and m['target_affine_dispatches']==225 and m['target_affine_misses']==m['target_mpp_affine_dispatches']==m['target_indexed_prefill_dispatches']==0,'C4 prepared FP32 projection route')
   wantRestores=64 if any(q['submitted']>q['consumed'] for q in e['native']['requests']) else 0
   need(m['target_kv_backups']==64 and m['commit_kv_restores']==wantRestores and m['commit_replay_layers']==0,'C4 rollback spans');restores+=wantRestores
 need(counts==set(range(1,17)) and depths=={1,2,3,4} and calls==45,'C4 physical row/depth coverage')
 return dict(**summary,submitted_physical_counts=sorted(counts),submitted_depths=sorted(depths),kv_restore_layer_dispatches=restores,gqa16_layer_dispatches=calls*32)

def main():
 need(not (R/'qualification-audit1.json').exists(),'audit overwrite')
 pre=read(R/'qualification1-preflight.json');post=read(R/'qualification1-postflight.json');need(pre['passed'] and post['passed'] and not pre['hashes_during_collection'],'collection incomplete')
 for name,h in pre['files_sha256'].items():need(sha(Path(name))==h,'Bound source/model changed '+name)
 data={};last=0;pids=set()
 for label,worker,mode,fixture,experiment in PLAN['sequence']:
  state=read(R/(label+'-status1.json'));need(state['status']=='terminal' and state['returncode']==0 and state['started_ns']>=last and state['finished_ns']>state['started_ns'] and state['pid'] not in pids,'Nonserial/fresh/terminal process')
  last=state['finished_ns'];pids.add(state['pid']);d=read(R/(label+'-result1.json'));need(d['completed'],'Incomplete model run')
  need(d['library']==PLAN['libraries'][mode]+'/libgarnet.dylib' and sha(Path(d['library']))==BUILD['library_sha256'][mode],'Library binding');data[label]=d
 summaries=[dict(mode=mode,summary=b1_check(data['b1-'+mode],mode)) for mode in ['on','off']]
 on=data['b1-on']['records'][1]['rounds'];off=data['b1-off']['records'][1]['rounds'];need(len(on)==len(off),'B1 target call count changed')
 for a,b in zip(on,off):
  for k in ['committed_before','slot','proposal','submitted_ids','emitted_ids','consumed','accepted_draft']:need(a[k]==b[k],'ON/OFF target history '+k)
  for k in ['reason','prompt_ids','generated_ids','executed_draft_ids','prefill_chunks','cached_tokens']:need(a['draft'].get(k)==b['draft'].get(k),'ON/OFF causal helper history '+k)
 ordinary=data['c4-ordinary'];frozen=read(M/'mistral-wide-lifecycle1/ordinary-result2.json')
 need(ordinary['jobs']==frozen['jobs'] and ordinary['witnesses']==frozen['witnesses'],'Current ordinary C4 differs from qualified prior control')
 c4=[dict(mode=mode,summary=c4_check(ordinary,data['c4-'+mode])) for mode in ['verify','reverse']]
 negative=[]
 def reject(name,fn):
  try:fn()
  except (ValueError,KeyError,TypeError,IndexError):negative.append(name);return
  raise AssertionError('Corruption admitted: '+name)
 def badb1(fn):
  d=copy.deepcopy(data['b1-on']);fn(d);return b1_check(d,'on')
 reject('wrong_b1_output',lambda:badb1(lambda d:d['records'][1]['generated_ids'].__setitem__(0,-1)))
 reject('wrong_b1_kv',lambda:badb1(lambda d:d['records'][1]['state_sha256'][0].__setitem__(0,'0'*64)))
 reject('wrong_b1_hidden',lambda:badb1(lambda d:d['records'][1].update(hidden_sha256='0'*64)))
 reject('false_gqa16_dispatch',lambda:badb1(lambda d:next(q['metrics'] for q in d['records'][1]['rounds'] if q['metrics']).update(target_attention_gqa16_score_dispatches=0)))
 reject('false_compiled_feature',lambda:badb1(lambda d:d.update(gqa_scores_enabled=False)))
 reject('future_helper_prefix',lambda:badb1(lambda d:next(q['draft'] for q in d['records'][1]['rounds'] if q['draft'].get('prompt_ids'))['prompt_ids'].append(1)))
 reject('missing_b1_rollback',lambda:badb1(lambda d:next(q['metrics'] for q in d['records'][1]['rounds'] if q['metrics'] and q['consumed']<len(q['proposal'])+1).update(commit_kv_restores=0)))
 def badc4(fn):
  d=copy.deepcopy(data['c4-verify']);fn(d);return c4_check(ordinary,d)
 reject('wrong_c4_kv_owner',lambda:badc4(lambda d:d['witnesses'][0]['states'][0][0].__setitem__(0,'0'*64)))
 reject('wrong_c4_known',lambda:badc4(lambda d:next(e for t in d['ticks'] for e in t['executions'] if e['stage']=='decode')['native']['requests'][0].update(known=-1)))
 reject('wrong_c4_mask',lambda:badc4(lambda d:next(e for t in d['ticks'] for e in t['executions'] if e['stage']=='decode')['submitted_active'].__setitem__(0,0.)))
 reject('missing_c4_invalid_guard',lambda:badc4(lambda d:d['rejections'].pop()))
 frames=sum(q['metrics'] is not None for q in on)
 result=dict(passed=True,scope=__doc__,fresh_engines=5,b1_current_ordinary_full_state_digest_pairs=260,b1_output_tokens_per_record=128,b1_gqa16_frames=frames,b1_gqa16_layer_dispatches=frames*32,b1_proposals_commits_helper_history_exact=True,b1_summary=summaries,c4_summary=c4,full_state_digest_pairs=260+sum(c['summary']['full_four_row_state_digest_pairs'] for c in c4),rejected_call_state_digest_pairs=sum(c['summary']['rejected_call_state_digest_pairs'] for c in c4),negative_controls_rejected=negative,native_b1_causal_drafting_qualified=True,native_c4_zero_proposal_lifecycle_qualified=True,learned_helper_batch4_qualified=False,serving_speed_claim=False,coding_quality_qualified=False,no_comparator_execution=True,goal_complete=False,files_sha256={str(p):sha(p) for p in [R/'audit_qualification1.py',R/'audit_owner1.py',R/'qualification-plan1.json',R/'qualification1-preflight.json',R/'qualification1-postflight.json']+[R/(label+'-result1.json') for label,*rest in PLAN['sequence']]})
 (R/'qualification-audit1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
