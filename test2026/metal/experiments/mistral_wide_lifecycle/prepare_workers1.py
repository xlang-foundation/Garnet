"""Extend the frozen T1 ownership fixtures to real T4 speculative rollback."""
from pathlib import Path
import ast,json
R=Path(__file__).resolve().parent;B=R.parent/'mistral-compact-verification-coexist1'
for name in ['worker1.py','probe_worker1.py']:
 s=(B/name).read_text();s=s.replace('decoder.prepare_verification(1)','decoder.prepare_verification(4)')
 s=s.replace('scheduled-B4-C4096-T128-verify1','scheduled-B4-C4096-prefill-up512-verify4-fp32-v1')
 s=s.replace('"""Untimed C4 real-model ownership reference/verification; no proposal oracle."""','"""Untimed C4/T4 ownership and rollback using zero proposals; no learned draft or speed claim."""')
 s=s.replace(' try:policy.verify_decode_json(*args)',' try:(decoder.session.verify_greedy if session else policy.verify_decode_json)(*args)').replace('def reject(label,args):','def reject(label,args,session=False):')
 old="     args=[decoder.io.stack_tokens([decoder.io.embedding(ids)]),val([s for s in slots for axis in range(3)],[4,1,3]),val(slots,[4,1]),val([float(x) for x in plan['active_rows']],[4,1]),ids,inputs['stops']]"
 new="""     byrow={q['row']:q for q in plan['requests']};flat=[];positions=[];flags=[];budgets=[0]*4;depths=[0]*4
     for row in range(4):
      q=byrow.get(row)
      if q is not None:budgets[row]=jobs[q['id']]['limit']-len(jobs[q['id']]['generated']);depths[row]=min(4,budgets[row])
      for t in range(4):
       on=t<depths[row];flat.append(ids[row] if t==0 else 0);positions.append(slots[row]+float(t) if on else 0.);flags.append(float(on))
     args=[decoder.io.stack_tokens([decoder.io.embedding([flat[row*4+t] for row in range(4)]) for t in range(4)]),val([s for s in positions for axis in range(3)],[4,4,3]),val(positions,[4,4]),val(flags,[4,4]),flat,inputs['stops']]"""
 assert s.count(old)==1;s=s.replace(old,new)
 old="      witness('before_invalid_mixed');free=plan['active_rows'].index(0);bad=list(args);flags=[float(x) for x in plan['active_rows']];flags[free]=1.;bad[3]=val(flags,[4,1]);reject('mixed_unscheduled_row',bad)\n      bad=list(args);known=list(ids);active_row=plan['requests'][0]['row'];known[active_row]=(known[active_row]+1)%32768;bad[4]=known;reject('wrong_known_mixed',bad);witness('after_invalid_mixed')"
 new="""      witness('before_invalid_mixed');free=plan['active_rows'].index(0);bad=list(args);bad_flags=list(flags);bad_flags[free*4]=1.;bad[3]=val(bad_flags,[4,4]);reject('mixed_unscheduled_row',bad)
      bad=list(args);known=list(flat);active_row=plan['requests'][0]['row'];known[active_row*4]=(known[active_row*4]+1)%32768;bad[4]=known;reject('wrong_known_mixed',bad)
      bad=list(args);bad[2]=val(positions,[16]);reject('wrong_slots_shape',bad)
      epochs=list(plan['epochs']);epochs[active_row]+=1;reject('wrong_native_epoch',list(args)+[budgets,epochs],True)
      remaining=list(budgets);remaining[active_row]=-1;reject('negative_native_budget',list(args)+[remaining,plan['epochs']],True)
      witness('after_invalid_mixed')
     if 'submitted_over_budget' not in rejections and any(0<b<4 for b in budgets):
      witness('before_invalid_budget');row=next(r for r in range(4) if 0<budgets[r]<4);bad=list(args);bad_flags=list(flags);bad_positions=list(positions)
      for t in range(4):bad_flags[row*4+t]=1.;bad_positions[row*4+t]=slots[row]+float(t)
      bad[1]=val([s for s in bad_positions for axis in range(3)],[4,4,3]);bad[2]=val(bad_positions,[4,4]);bad[3]=val(bad_flags,[4,4]);reject('submitted_over_budget',bad);witness('after_invalid_budget')"""
 assert s.count(old)==1;s=s.replace(old,new)
 s=s.replace("and q['consumed']==q['submitted']==1","and q['consumed']==1 and q['submitted']==min(4,q['remaining_before'])")
 old="     native=json.loads(policy.verify_decode_json(*args));completed=native['requests'];assert len(completed)==len(plan['requests'])"
 new=old+"\n     assert native['verification']['target_affine_misses']==0 and native['verification']['target_indexed_decode_dispatches']==225 and native['verification']['target_mpp_affine_dispatches']==0 and native['verification']['target_indexed_prefill_dispatches']==0"
 assert s.count(old)==1;s=s.replace(old,new)
 old="record['executions'].append(dict(stage='decode',native=native,completed=completed,compact_rows=0))"
 new="record['executions'].append(dict(stage='decode',native=native,completed=completed,compact_rows=0,submitted_tokens=list(flat) if mode!='ordinary' else None,submitted_positions=list(positions) if mode!='ordinary' else None,submitted_active=list(flags) if mode!='ordinary' else None))"
 assert s.count(old)==1;s=s.replace(old,new)
 s=s.replace(" assert 'native_compact_verify_b4t1_optional_v1' in garnet.backend_info()"," assert 'native_wide_verify_preparation_lifetime_v1' in garnet.backend_info() and decoder.session.wide_verification_preparation_open")
 ast.parse(s);(R/name).write_text(s)
for name in ['input1.json','probe-input1.json']:
 d=json.loads((B/name).read_text());d['verification_tokens']=4;d['scope']='C4/T4 zero-proposal rollback and independent request ownership; no draft/speed claim.';(R/name).write_text(json.dumps(d,indent=2)+'\n')
(R/'plan1.json').write_text(json.dumps(dict(scope='Prepared native Mistral B4/T4 lifetime, real C4 cancel/refill/survivor, B2 interleave.',guards=['off','on-claim','on-run','on-adapter-late','on-compact','on-capacity','on-batch'],primary_arms=['ordinary','verify','reverse'],probe_arms=['ordinary','verify'],submitted_extra_tokens='zero IDs; require rejection after first known input and exact full-state rollback',native_rejections=['mixed_unscheduled_row','wrong_known_mixed','wrong_slots_shape','wrong_native_epoch','negative_native_budget','submitted_over_budget','stale_tick_replay'],active_requests_required=[1,2,3,4],primary_shared_witnesses=['initial_prefilled','finish_A','claim_E','mid_E','cancel_C','claim_F','finish_E','finish_B','before_late_G','all_done'],probe_shared_witnesses=['first_B2_prefill','after_B2_verification','all_done'],actual_profile_identity='scheduled-B4-C4096-prefill-up512-verify4-fp32-v1',prefix_retention_qualified=False,learned_helper_qualified=False,serving_speed_claim=False,goal_complete=False),indent=2)+'\n')
print('Prepared wide rollback ownership and B2 workers')
