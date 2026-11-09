from pathlib import Path
import hashlib,json,shutil
R=Path(__file__).resolve().parent;M=R.parent;B=M/'mistral-wide-lifecycle1';N=M/'mistral-verification-split-score-native1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=(N/'worker1.py').read_text()
a="split_enabled='native_mistral_verification_split_scores128_default_v1' in garnet.backend_info()"
assert s.count(a)==1;s=s.replace(a,a+"\ngqa_enabled='native_mistral_verification_gqa16_scores32_optional_v1' in garnet.backend_info()")
a='experiment=experiment,split_scores_enabled=split_enabled)';assert s.count(a)==1;s=s.replace(a,'experiment=experiment,split_scores_enabled=split_enabled,gqa_scores_enabled=gqa_enabled)')
a="assert metrics['target_attention_split_score_dispatches']==(32 if split_enabled else 0)";assert s.count(a)==1;s=s.replace(a,"assert metrics['target_attention_gqa16_score_dispatches']==(32 if gqa_enabled else 0)\n      assert metrics['target_attention_split_score_dispatches']==(32 if split_enabled and not gqa_enabled else 0)")
(R/'b1_worker1.py').write_text(s)
s=(B/'worker1.py').read_text();a="native=json.loads(policy.verify_decode_json(*args));completed=native['requests'];assert len(completed)==len(plan['requests'])";assert s.count(a)==1
s=s.replace(a,a+"\n     assert native['verification']['target_attention_gqa16_score_dispatches']==32")
(R/'c4_worker1.py').write_text(s);shutil.copyfile(B/'input2.json',R/'input2.json')
s=(B/'audit_owner2.py').read_text();a="metrics=native['verification'];need(";assert s.count(a)==1;s=s.replace(a,"metrics=native['verification'];need(metrics['target_attention_gqa16_score_dispatches']==32,'native GQA16 route missing');need(")
(R/'audit_owner1.py').write_text(s)
(R/'workers-bindings1.json').write_text(json.dumps(dict(base_files_sha256={str(p):sha(p) for p in [N/'worker1.py',B/'worker1.py',B/'audit_owner2.py',B/'input2.json']},derived_files_sha256={n:sha(R/n) for n in ['b1_worker1.py','c4_worker1.py','audit_owner1.py','input2.json']}),indent=2)+'\n')
