"""Prepare native model qualification without changing any captured fixture or reference."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent;Q=R.parent/'mistral-causal-batch4-helper1';N=R.parent/'mistral-gqa16-value-native1'
def write(n,s):
 p=R/n;assert not p.exists();p.write_text(s)
def replace(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
source=(Q/'productive_worker1.py').read_text()
s=replace(source,"output=sys.argv[4];mode=sys.argv[5];assert mode in ('ordinary','productive') and not os.path.exists(output)","output=sys.argv[4];configuration=sys.argv[5];mode='ordinary';assert configuration in ('enabled','disabled') and not os.path.exists(output);route_trace=[]")
s=replace(s,'completed=done,mode=mode,inputs=inputs','completed=done,mode=mode,configuration=configuration,route_trace=route_trace,inputs=inputs')
s=replace(s," base=[l['graph'] for l in decoder.layers]"," assert decoder.session.prefill_set_tensor_down(configuration=='enabled')\n base=[l['graph'] for l in decoder.layers]")
s=replace(s,"    predicted=decoder.session.predict_rows(final) if any(final) else None;completed=[]","    route_trace.append(dict(width=width,active_tokens=int(sum(active)),rows=[q['row'] for q in plan['requests']],dispatches=decoder.session.packed_tensor_down_dispatches))\n    predicted=decoder.session.predict_rows(final) if any(final) else None;completed=[]")
write('ordinary_worker1.py',s);write('ordinary-input1.json',(Q/'productive-input1.json').read_text())
source2=(N/'c4_worker1.py').read_text()
s2=replace(source2,"output=sys.argv[4];mode=sys.argv[5];assert mode in ('ordinary','verify','reverse') and not os.path.exists(output)","output=sys.argv[4];configuration=sys.argv[5];mode='ordinary';assert configuration in ('enabled','disabled') and not os.path.exists(output);route_trace=[]")
s2=replace(s2,'completed=done,mode=mode,inputs=inputs','completed=done,mode=mode,configuration=configuration,route_trace=route_trace,inputs=inputs')
s2=replace(s2," policy=garnet.continuous_scheduler();assert policy.initialize(decoder.session,16);assert policy.set_prefill_chunk(512)"," assert decoder.session.prefill_set_tensor_down(configuration=='enabled')\n policy=garnet.continuous_scheduler();assert policy.initialize(decoder.session,16);assert policy.set_prefill_chunk(512)")
s2=replace(s2,'    predicted=decoder.session.predict_rows(final_epochs) if any(final_epochs) else None',"    route_trace.append(dict(width=width,active_tokens=int(sum(flags)),rows=[q['row'] for q in plan['requests']],dispatches=decoder.session.packed_tensor_down_dispatches))\n    predicted=decoder.session.predict_rows(final_epochs) if any(final_epochs) else None")
write('lifecycle_worker1.py',s2);write('lifecycle-input1.json',(N/'input2.json').read_text())
for s in [s,s2]:compile(s,'worker','exec')
write('qualification-plan1.json',json.dumps(dict(serial=True,untimed=True,no_comparator_execution=True,cases=[dict(name='on-enabled',worker='ordinary',build='on',configuration='enabled'),dict(name='on-disabled',worker='ordinary',build='on',configuration='disabled'),dict(name='off-disabled',worker='ordinary',build='default-off',configuration='disabled'),dict(name='lifecycle-on',worker='lifecycle',build='on',configuration='enabled')],base_workers_sha256={str(Q/'productive_worker1.py'):hashlib.sha256(source.encode()).hexdigest(),str(N/'c4_worker1.py'):hashlib.sha256(source2.encode()).hexdigest()},performance_claim=False),indent=2)+'\n')
