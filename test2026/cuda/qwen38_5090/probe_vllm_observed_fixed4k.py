"""Untimed reference controls and frozen4K cohorts; an outer owner starts/stops the server.

No reference server may coexist with the native HTTP job. This probe requires
its actual closed proof and declared, separately checked server metadata. It
does not attest that metadata or promote screening to broad model quality.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
from urllib.request import urlopen
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.mmmu_http_quality import load_corpus,collect,validate_response
from test2026.cuda.qwen38_5090.verify_mmmu_http_quality import replay
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
REVISION='482ca0f3832238542f8f5295dde86b5f22711d80'

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',type=Path,required=True);a.add_argument('--engine',choices=('vllm','sglang'),required=True);a.add_argument('--mode',choices=('ordinary',),required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--metadata',type=Path,required=True);a.add_argument('--launcher',type=Path,required=True);a.add_argument('--server-log',type=Path,required=True);a.add_argument('--ordinary',type=Path);args=a.parse_args()
 root=args.root;methods=Path(__file__).resolve().parent;out=args.output;parent='fixed-length-http-study-attempt1';proofpath=root/'results'/(parent+'-verified.json')
 if not proofpath.is_file():print(json.dumps({'status':'reference4k_native_http_proof_missing_no_requests_no_output'}));return 2
 proof=json.loads(proofpath.read_bytes());assert proof['status']=='fixed_length_http_study_replayed' and proof['native_fixed_4k_metrics_replayed'] and proof['archive_sha256']==sha(root/'results'/(parent+'-evidence.zip'))
 closed=json.loads((root/'results'/parent/'workflow.json').read_bytes());assert closed['cleanup_finished'] and closed['exit_code']==0 and closed['library_unchanged']
 metadata=json.loads(args.metadata.read_bytes());assert metadata['engine']==args.engine and metadata['generation_mode']==args.mode and metadata['context_capacity']==4096 and metadata['checkpoint_revision']==REVISION
 assert (args.ordinary is not None)==(args.mode=='mtp') and not out.exists()
 assert metadata['launcher_sha256']==sha(args.launcher) and metadata['source_after_startup_verified'] is True and metadata['owned_server_pid_verified'] is True
 assert metadata['image_cache_policy']==('dual_idle_reset_before_timing' if args.engine=='vllm' else 'preprocess0_prefixMMoff_globalMMoff')
 assert args.engine=='vllm' and args.mode=='ordinary' and args.ordinary is None
 from reference_vllm_observed_output import admit_observation
 observed_proof=root/'results/vllm-output-characterization-attempt1-verified.json';observed_archive=root/'results/vllm-output-characterization-attempt1-evidence.zip'
 observation=admit_observation(observed_proof.read_bytes(),observed_archive.read_bytes(),metadata)
 url='http://127.0.0.1:'+('18080' if args.engine=='vllm' else '18081');corpus=root/'tools/mmmu-quality-corpus';_,rows,gold=load_corpus(corpus)
 if args.ordinary is not None:
  ordinary=json.loads((args.ordinary/'report.json').read_bytes());assert ordinary['status']=='reference_fixed4k_profile_collected' and ordinary['engine']==args.engine and ordinary['mode']=='ordinary' and ordinary['checkpoint_revision']==REVISION
  assert ordinary['native_http_proof_sha256']==sha(proofpath)
 out.mkdir();report={'status':'running','engine':args.engine,'mode':args.mode,'checkpoint_revision':REVISION,'native_http_proof_sha256':sha(proofpath),'metadata_sha256':sha(args.metadata),'launcher_sha256':sha(args.launcher),'performance_admitted':False,'observed_output_policy':observation,'scope':'Ordinary vLLM with the replayed current-profile output/isolation diagnosis. Same two early boundaries/fourVLM/16short controls/805adapted screen/124actual serial ID controls and124frozen3968+128 streams. Record every stream-text variant after timing; individual stream IDs unobserved. No long-output semantic/numerical equivalence, strongest tuning/repetition or2x promotion.'}
 def save():(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
 save()
 try:
  with urlopen(url+'/health',timeout=15) as response:assert response.status==200;(out/'health.bin').write_bytes(response.read())
  with urlopen(url+'/v1/models',timeout=15) as response:models=response.read()
  (out/'models.json').write_bytes(models);assert any(row.get('id')=='qwen38-27b-nvfp4' for row in json.loads(models)['data'])
  assert metadata['configured_engine_context_capacity']==4096 and metadata['engine_context_reserved_slots']==0
  from reference_fixed4k_vllm_boundary import collect as collect_boundary
  boundary=collect_boundary(methods/'evidence/fixed-length-4096-corpus.zip',out/'boundary-controls',url,args.ordinary/'boundary-controls' if args.ordinary is not None else None)
  report['early_full4k_controls']=boundary;save()
  command=[sys.executable,str(methods/'run_reference_greedy_candidate.py'),'--study',str(root),'--engine',args.engine,'--url',url,'--output',str(out/'greedy-controls'),'--launcher',str(args.launcher),'--server-log',str(args.server_log)]
  if args.ordinary is not None:command+=['--ordinary-oracle',str(args.ordinary/'greedy-controls/greedy/equivalence.json')]
  subprocess.run(command,check=True);report['short_greedy_controls_passed']=True;save()
  identity={'engine':args.engine,'model_alias':'qwen38-27b-nvfp4','generation_mode':args.mode,'checkpoint_revision':REVISION,'server_metadata':metadata,'metadata_sha256':sha(args.metadata),'scope':'Outer controller declaration; independently checked launcher/config/source/PID evidence must accompany this client.'}
  (out/'server-identity.json').write_text(json.dumps(identity,indent=2),encoding='utf8')
  collect(corpus,rows,gold,out/'mmmu-quality',args.engine,'qwen38-27b-nvfp4',url,identity,timeout=300)
  screening=replay(corpus,out/'mmmu-quality');assert screening['status']=='adapted_mmmu_http_quality_responses_replayed';(out/'mmmu-quality-replayed.json').write_text(json.dumps(screening,indent=2),encoding='utf8');report['screening']=screening;save()
  if args.ordinary is not None:
   prior=replay(corpus,args.ordinary/'mmmu-quality');assert prior['status']=='adapted_mmmu_http_quality_responses_replayed' and prior['engine']==args.engine
   for row in rows:
    expected=json.loads((corpus/row['prompt_ids_file']).read_bytes());relative=Path('mmmu-quality/cases')/row['id']/'response.bin'
    previous=validate_response((args.ordinary/relative).read_bytes(),args.engine,'qwen38-27b-nvfp4',expected);current=validate_response((out/relative).read_bytes(),args.engine,'qwen38-27b-nvfp4',expected)
    assert previous['completion_token_ids']==current['completion_token_ids'],('MTP screening differs from ordinary',row['id'])
   report['all805_ordinary_mtp_completion_ids_exact']=True;save()
  command=[sys.executable,str(methods/'benchmark_vllm_observed_output.py'),'--engine',args.engine,'--url',url,'--corpus-archive',str(methods/'evidence/fixed-length-4096-corpus.zip'),'--output',str(out/'benchmark4k'),'--metadata',str(args.metadata)]
  if args.ordinary is not None:command+=['--ordinary-controls',str(args.ordinary/'benchmark4k/controls')]
  command+=['--observed-proof',str(observed_proof),'--observed-archive',str(observed_archive)]
  subprocess.run(command,check=True);report['fixed_4k_cohorts_collected']=True;report['status']='reference_fixed4k_profile_collected';save()
 except BaseException as error:report.update(status='failed',error=repr(error));save();raise
 return 0
if __name__=='__main__':raise SystemExit(main())
