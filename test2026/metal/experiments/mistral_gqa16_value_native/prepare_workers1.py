"""Derive causal model workers and explicit shared-value CPU validators."""
from pathlib import Path
import hashlib,json,shutil
R=Path(__file__).resolve().parent;M=R.parent;N=M/'mistral-gqa16-score-native1';B=M/'mistral-bounded-draft-long1'
def derive(name,source,changes):
 s=source.read_text()
 for old,new in changes:
  assert s.count(old)==1,(name,old,s.count(old));s=s.replace(old,new)
 (R/name).write_text(s)
derive('b1_worker1.py',N/'b1_worker1.py',[
 ("gqa_enabled='native_mistral_verification_gqa16_scores32_optional_v1' in garnet.backend_info()","gqa_enabled='native_mistral_verification_gqa16_scores32_optional_v1' in garnet.backend_info()\nvalue_enabled='native_mistral_verification_gqa16_values64_32_optional_v1' in garnet.backend_info()"),
 ('gqa_scores_enabled=gqa_enabled)','gqa_scores_enabled=gqa_enabled,gqa_values_enabled=value_enabled)'),
 ("assert metrics['target_attention_vector4_dispatches']==32","assert metrics['target_attention_vector4_dispatches']==(0 if value_enabled else 32)\n      assert metrics['target_attention_gqa16_value_dispatches']==(32 if value_enabled else 0)")])
derive('c4_worker1.py',N/'c4_worker1.py',[("assert native['verification']['target_attention_gqa16_score_dispatches']==32","assert native['verification']['target_attention_gqa16_score_dispatches']==32\n     assert native['verification']['target_attention_gqa16_value_dispatches']==32\n     assert native['verification']['target_attention_vector4_dispatches']==0")])
derive('audit_owner1.py',N/'audit_owner1.py',[("metrics=native['verification'];need(metrics['target_attention_gqa16_score_dispatches']==32,'native GQA16 route missing');", "metrics=native['verification'];need(metrics['target_attention_gqa16_score_dispatches']==32 and metrics['target_attention_gqa16_value_dispatches']==32 and metrics['target_attention_vector4_dispatches']==0,'native GQA16 score/value routes missing');")])
shutil.copyfile(N/'input2.json',R/'input2.json')
derive('audit_bounded_value_common1.py',B/'audit_bounded_common1.py',[
 ("metrics=step['metrics']", "metrics=step['metrics']\n     shared='native_mistral_verification_gqa16_values64_32_optional_v1' in data['backend']\n     require(data['gqa_values_enabled']==shared,'wrong compiled shared-value capability')\n     require(metrics['target_attention_gqa16_value_dispatches']==(32 if shared else 0),'wrong actual shared-value route')"),
 ("('target_attention_vector4_dispatches',32)","('target_attention_vector4_dispatches',0 if shared else 32)")])
derive('audit_bounded_qualification1.py',B/'audit_qualification1.py',[("from audit_bounded_common1 import validate","from audit_bounded_value_common1 import validate")])
derive('audit_qualification1.py',N/'audit_qualification1.py',[
 ("spec=importlib.util.spec_from_file_location('bounded_qualification',B/'audit_qualification1.py')","spec=importlib.util.spec_from_file_location('bounded_value_qualification',R/'audit_bounded_qualification1.py')"),
 ("TAG='native_mistral_verification_gqa16_scores32_optional_v1'","TAG='native_mistral_verification_gqa16_values64_32_optional_v1'"),
 ("need(data['gqa_scores_enabled']==(mode=='on') and (TAG in data['backend'])==(mode=='on'),'compiled GQA16 capability')","need(data['gqa_values_enabled']==(mode=='on') and (TAG in data['backend'])==(mode=='on'),'compiled shared-value capability')\n need(data['gqa_scores_enabled'] is True and 'native_mistral_verification_gqa16_scores32_optional_v1' in data['backend'],'shared-key QK control changed')"),
 ("need(q['metrics']['target_attention_gqa16_score_dispatches']==(32 if mode=='on' else 0),'GQA16 route proof')","need(q['metrics']['target_attention_gqa16_score_dispatches']==32,'shared-key QK route proof')\n   need(q['metrics']['target_attention_gqa16_value_dispatches']==(32 if mode=='on' else 0),'shared-value route proof')\n   need(q['metrics']['target_attention_vector4_dispatches']==(0 if mode=='on' else 32),'legacy vector4 actual route proof')"),
 ("need(q['metrics']['target_attention_split_score_dispatches']==(0 if mode=='on' else 32),'wrong control route')","need(q['metrics']['target_attention_split_score_dispatches']==0,'wrong QK control route')"),
 ("need(m['target_attention_gqa16_score_dispatches']==32 and m['target_attention_split_score_dispatches']==0,'C4 actual GQA16 route')","need(m['target_attention_gqa16_score_dispatches']==32 and m['target_attention_gqa16_value_dispatches']==32 and m['target_attention_vector4_dispatches']==0 and m['target_attention_split_score_dispatches']==0,'C4 actual shared QK/value routes')"),
 ("reject('false_compiled_feature',lambda:badb1(lambda d:d.update(gqa_scores_enabled=False)))","reject('false_compiled_feature',lambda:badb1(lambda d:d.update(gqa_values_enabled=False)))\n reject('false_shared_value_dispatch',lambda:badb1(lambda d:next(q['metrics'] for q in d['records'][1]['rounds'] if q['metrics']).update(target_attention_gqa16_value_dispatches=0)))\n reject('false_legacy_vector_dispatch',lambda:badb1(lambda d:next(q['metrics'] for q in d['records'][1]['rounds'] if q['metrics']).update(target_attention_vector4_dispatches=32)))")])
derive('run_qualification1.py',N/'run_qualification1.py',[
 ("checks=read(B/'qualification1-preflight.json')['files_sha256'].copy();", "checks={n:h for n,h in read(B/'qualification1-preflight.json')['files_sha256'].items() if not n.startswith(str(W/'Garnet')+'/')};"),
 ("'audit_qualification1.py','input2.json'", "'audit_qualification1.py','audit_bounded_qualification1.py','audit_bounded_value_common1.py','input2.json'")])
derive('run_measurement1.py',N/'run_measurement1.py',[])
derive('audit_measurement1.py',N/'audit_measurement1.py',[("reject('false_gqa_route',lambda d:next(q['metrics'] for q in d['records'][0]['rounds'] if q['metrics']).update(target_attention_gqa16_score_dispatches=0))","reject('false_gqa_value_route',lambda d:next(q['metrics'] for q in d['records'][0]['rounds'] if q['metrics']).update(target_attention_gqa16_value_dispatches=0))")])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=['b1_worker1.py','c4_worker1.py','audit_owner1.py','input2.json','audit_bounded_value_common1.py','audit_bounded_qualification1.py','audit_qualification1.py','run_qualification1.py','run_measurement1.py','audit_measurement1.py']
(R/'workers-bindings1.json').write_text(json.dumps(dict(base_files_sha256={str(p):sha(p) for p in [N/'b1_worker1.py',N/'c4_worker1.py',N/'audit_owner1.py',N/'audit_qualification1.py',B/'audit_bounded_common1.py',B/'audit_qualification1.py']},derived_files_sha256={n:sha(R/n) for n in names},new_counter_validated_without_changing_old_counter=True),indent=2)+'\n')
print('Prepared five-engine qualification and serial ABBA measurement programs.')
