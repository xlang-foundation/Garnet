"""Compare closed joined/unjoined B4 execution counters; no timing inference."""
from pathlib import Path
import hashlib,json,sys,zipfile
out=Path(sys.argv[1]);sha=lambda b:hashlib.sha256(b).hexdigest();names=['shared-mtp3-full-b4-t4-host-embedding-workspace-attempt1','shared-mtp3-full-b4-t4-host-joined-attempt1'];reports=[];inputs={}
for name in names:
 proof=json.loads((out/(name+'-verified.json')).read_bytes());assert proof['model_gate_passed']
 assert sha((out/(name+'-evidence.zip')).read_bytes())==proof['archive_sha256']
 with zipfile.ZipFile(out/(name+'-evidence.zip')) as z:
  manifest=json.loads(z.read('evidence-manifest.json'))
  for n,h in manifest['files'].items():assert sha(z.read(n))==h
  reports.append(json.loads(z.read('results/'+name+'/native/report.json')))
 inputs[name]=proof['archive_sha256']
old,new=reports;original={(c['group'],c['repeat']):c for c in old['cases']};rows=[]
for c in new['cases']:
 key=(c['group'],c['repeat']);ref=original[key]
 assert c['speculative_metrics']==ref['speculative_metrics']
 assert c['generated_ids']==ref['generated_ids'] and c['ordinary_state']==ref['ordinary_state'] and c['mtp_state']==ref['mtp_state']
 def pool_bytes(case):
  m=case['decoder_memory'];return m['compressed_projection_weight_bytes']+m['compressed_projection_scale_bytes']
 rows.append({'group':key[0],'repeat':key[1],'all_speculative_counters_and_iterations_exact':True,'target_verification_calls':c['speculative_metrics']['target_verification_calls'],'partial_prefix_replays':c['speculative_metrics']['target_prefix_replays'],'proposed_draft_tokens':c['speculative_metrics']['proposed_draft_tokens'],'accepted_draft_tokens':c['speculative_metrics']['accepted_draft_tokens'],'old_pool_bytes':pool_bytes(ref),'joined_pool_bytes':pool_bytes(c),'old_projection_entries':ref['decoder_memory']['compressed_projection_pool_entries'],'joined_projection_entries':c['decoder_memory']['compressed_projection_pool_entries']})
assert len(rows)==4
result={'status':'joined_b4_closed_work_counters_exact','inputs':inputs,'cases':rows,'scope':'Closed four-case correctness workload only. Identical IDs/states and speculative counters/iterations. Projection grouping changes192to128NVFP4 plus208FP8. Not actual kernel launch counts, HTTP cohort work equivalence, causal speedup or performance admission.'}
(out/'Garnet-B4-Joined-Work-Audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
