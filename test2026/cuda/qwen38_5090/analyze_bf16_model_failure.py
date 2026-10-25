"""Describe the archived mismatch without promoting the failed candidate."""
from pathlib import Path
import hashlib,json,sys,zipfile
out=Path(sys.argv[1]);name='bf16-model-b4-t32-attempt1';archive=out/(name+'-evidence.zip');sha=lambda b:hashlib.sha256(b).hexdigest()
proof=json.loads((out/(name+'-verified.json')).read_bytes());assert proof['status']=='bf16_model_b4_t32_failure_preserved' and proof['archive_sha256']==sha(archive.read_bytes())
with zipfile.ZipFile(archive) as z:
 base='results/'+name+'/';r=json.loads(z.read(base+'native/report.json'));assert r['status']=='failed' and len(r['cases'])==1;c=r['cases'][0]
 records=[]
 def mismatch(x,y):return next((i for i,(u,v) in enumerate(zip(x,y)) if u!=v),None if len(x)==len(y) else min(len(x),len(y)))
 for row,(a,b,e) in enumerate(zip(c['ordinary_ids'],c['generated_ids'],c['expected_ids'])):
  records.append({'row':row,'key':c['keys'][row],'ordinary_mtp_first_mismatch':mismatch(a,b),'ordinary_fp32_first_mismatch':mismatch(a,e),'output_counts':[len(a),len(b),len(e)]})
 raw=[z.read(base+'native/group-0-repeat-0-'+mode+'-last-hidden.f32') for mode in ('ordinary','mtp')]
 assert raw[0]==raw[1] and raw[0][:40960]==raw[0][40960:]
 a={'status':'bf16_model_failure_analyzed','archive_sha256':sha(archive.read_bytes()),'report_sha256':sha(z.read(base+'native/report.json')),'producer_sha256':sha(Path(__file__).read_bytes()),'rows':records,
    'ordinary_mtp_full_state_hash_witnesses_exact':c['ordinary_state']==c['mtp_state'],'ordinary_mtp_raw_hidden_exact':raw[0]==raw[1],
    'duplicate_pair_raw_hidden_exact':raw[0][:40960]==raw[0][40960:],'runtime_metadata_gate':r['runtime_metadata_gate'],
    'negative_commit_calls':len(r['negative_commit_gate']['rejected_calls']),'negative_zero_prefix_state_unchanged':r['negative_commit_gate']['before']==r['negative_commit_gate']['after'],
    'precision_and_prefill_effect_not_separated':True,'quality_gate_passed':False,'performance_admitted':False,
    'scope':'One failed group/repeat only. BF16 KV/T32 prefill/register128 changes were combined. MTP agrees with ordinary but does not justify retained-FP32 divergence or broad quality. Full canonical arrays absent.'}
(out/(name+'-failure-analysis.json')).write_text(json.dumps(a,indent=2),encoding='utf8');print(json.dumps(a,indent=2))
