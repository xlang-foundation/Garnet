"""Opt-in short-model admission; reuse unchanged actual B1 state/prefix probe."""
from pathlib import Path
import hashlib,json,os,runpy,sys
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 root,out=map(Path,sys.argv[1:3]);assert not out.exists();m=root/'Garnet/test2026/cuda/qwen38_5090';repo=root/'Garnet'
 parent='tokenizer-nfc-model-consistency-attempt1';proofpath=root/'results'/(parent+'-verified.json');proof=json.loads(proofpath.read_bytes());assert proof['status']=='tokenizer_nfc_model_consistency_replayed'
 assert sha(root/'results'/(parent+'-evidence.zip'))==proof['archive_sha256'] and sha(root/'build/bin/garnet.dll')==proof['library_sha256']
 closed=json.loads((root/'results'/parent/'workflow.json').read_bytes());assert closed['exit_code']==0 and closed['cleanup_finished'] and closed['library_unchanged']
 for name,digest in closed['source_sha256'].items():assert sha(root/name)==digest,('Qualified target/source drift',name)
 for name,digest in closed['runtime_dependencies'].items():assert sha(root/'build/bin'/name)==digest
 out.mkdir();report={'status':'running','parent_archive_sha256':proof['archive_sha256'],'parent_proof_sha256':sha(proofpath),'original_probe_sha256':sha(m/'probe_bf16_low_batch_model_fixed.py'),'candidate_module_sha256':sha(repo/'xModel/qwen35/vl_0_8b/cuda_mtp_prefill16_study.py'),'sessions_constructed':0,'close_observations':[],'actual_4k_qualified':False,'quality_qualified':False,'performance_admitted':False,'scope':'Existing unchanged B1 eight-case short text/image/color ordinary/MTP/reset/prefix probe with explicit separate T16 draft-prefill candidate. Exact target IDs/state/raw-vector comparison to actual retained1a1d B1 baseline. Shared explicit draft KV, unchanged T4 decode/verifier; additional engine weights and memory must be recorded. No4Kmodel,805quality, serving, memory-fit-for-full4K or speed admission.'}
 def save():(out/'study-report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
 save();sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,str(repo))
 from xModel.qwen35.vl_0_8b import cuda_speculative as original
 from xModel.qwen35.vl_0_8b.cuda_mtp_prefill16_study import CUDAMTPPrefill16StudySession
 old_factory=original.CUDASpeculativeSession;original_close=CUDAMTPPrefill16StudySession.close
 def factory(*args,**kwargs):
  value=CUDAMTPPrefill16StudySession(*args,**kwargs);report['sessions_constructed']+=1;save();return value
 def close_observed(value):
  try:
   if value.program is not None and value.prefill16_program is not None:
    report['close_observations'].append(value.memory_info());save()
  finally:original_close(value)
 original.CUDASpeculativeSession=factory;CUDAMTPPrefill16StudySession.close=close_observed;argv=sys.argv
 try:
  sys.argv=[str(m/'probe_bf16_low_batch_model_fixed.py'),str(root),'1',str(out/'model')]
  runpy.run_path(sys.argv[0],run_name='__main__')
  actual=json.loads((out/'model/report.json').read_bytes());assert actual['status']=='bf16_low_batch_model_consistency_finished' and len(actual['cases'])==8 and report['sessions_constructed']==1 and len(report['close_observations'])==1
  from test2026.cuda.qwen38_5090.compare_tokenizer_model_parent import compare_reports
  baseline=root/'results'/parent/'b1';retained=json.loads((baseline/'report.json').read_bytes())
  comparison=compare_reports(actual,retained,lambda n:(out/'model'/n).read_bytes(),lambda n:(baseline/n).read_bytes())
  report['retained_b1_comparison']=comparison;report['retained_baseline']=parent;report['prefill_call_controls']=[]
  for case in actual['cases']:
   old=next(row for row in retained['cases'] if (row['key'],row['repeat'])==(case['key'],case['repeat']))
   report['prefill_call_controls'].append({'key':case['key'],'repeat':case['repeat'],'baseline_prefill_calls':old['speculative_metrics']['mtp_prefill_calls'],'candidate_prefill_calls':case['speculative_metrics']['mtp_prefill_calls'],'scope':'Actual call counters only, not a timing or speedup.'})
  report['status']='mtp_prefill16_short_model_collected_needs_replay';save();print('MTP_PREFILL16_SHORT_TARGET_STATE_CONTROLS_PASS',flush=True)
 except BaseException as error:report.update(status='failed',error=repr(error));save();raise
 finally:original.CUDASpeculativeSession=old_factory;CUDAMTPPrefill16StudySession.close=original_close;sys.argv=argv;save()
 return 0
if __name__=='__main__':raise SystemExit(main())
