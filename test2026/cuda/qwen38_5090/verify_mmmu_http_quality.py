"""Recompute adapted HTTP screening scores from exact raw replies and wire bytes."""
from pathlib import Path
import argparse,gzip,json
from .mmmu_http_quality import MANIFEST_SHA256,load_corpus,adapt_request,validate_response,sha,encoded,decoded,require,child,endpoint
from .mmmu_quality_protocol import PROTOCOL,score_complete

def replay(corpus,run):
 corpus=Path(corpus).resolve(strict=True);run=Path(run).resolve(strict=True);_,rows,gold=load_corpus(corpus)
 raw=(run/'file-manifest.json').read_bytes();manifest=decoded(raw);actual={p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file()}
 require(actual==set(manifest['files'])|{'file-manifest.json'},'Unexpected/missing run files')
 for name,item in manifest['files'].items():
  data=child(run,name).read_bytes();require(sha(data)==item['sha256'] and len(data)==item['bytes'],'Run bytes differ: '+name)
 workflow=decoded((run/'workflow.json').read_bytes());engine=workflow['engine'];alias=workflow['model_alias']
 require(workflow['corpus_manifest_sha256']==MANIFEST_SHA256 and workflow['protocol']==PROTOCOL and workflow['expected_requests']==805,'Run corpus differs')
 require(workflow['endpoint']==endpoint(workflow['endpoint'].removesuffix('/v1/chat/completions')),'Endpoint differs')
 require(workflow['retries']==0 and type(workflow['retries']) is int and workflow['concurrency']==1 and type(workflow['concurrency']) is int,'Quality request scheduling differs')
 require(workflow['performance_admitted'] is False and workflow['quality_promotion_admitted'] is False and workflow['runtime_image_preprocessing_verified'] is False and workflow['server_identity_attested_by_client'] is False,'Run overstates admission')
 method_hashes={n:sha(Path(__file__).with_name(n).read_bytes()) for n in ('mmmu_http_quality.py','mmmu_quality_protocol.py')}
 require(workflow['method_sha256']==workflow['method_after_sha256']==method_hashes,'Collector/protocol changed')
 identity=decoded((run/'server-identity.json').read_bytes());require(sha(encoded(identity))==workflow['server_identity_sha256'] and identity['engine']==engine and identity['model_alias']==alias,'Declared server identity differs')
 folders={p.name for p in (run/'cases').iterdir() if p.is_dir()};require(len(folders)<=805 and folders=={r['id'] for r in rows[:len(folders)]},'Attempt order/subset differs')
 complete=workflow['status']=='completed_needs_independent_replay';require(workflow['status'] in ('completed_needs_independent_replay','failed','collecting'),'Unknown collection status')
 records=[];failed=[];attempts=0
 canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
 for i,row in enumerate(rows[:len(folders)]):
  folder=run/'cases'/row['id'];record=decoded((folder/'record.json').read_bytes());response_file=folder/'response.bin';response=response_file.read_bytes() if response_file.exists() else b''
  require(record['id']==row['id'] and record['canonical_request_sha256']==row['canonical_request_sha256'],'Case/request identity differs')
  if 'response_sha256' in record:require(record['response_sha256']==sha(response) and record['response_bytes']==len(response),'Raw response differs')
  else:require(not complete and record['status']=='started','Completed/failed response identity missing')
  if not response_file.exists():require(not complete and record['status']=='started','Required response bytes missing')
  wire_file=folder/'wire-request.json.gz'
  if wire_file.exists():
   wire=gzip.decompress(wire_file.read_bytes());wanted=adapt_request(child(corpus,row['request_file']).read_bytes(),engine,alias)
   require(wire==wanted and record['wire_request_sha256']==sha(wire) and record['wire_request_bytes']==len(wire),'Transmitted request adapter differs');attempts+=1
  else:require(not complete and record['status']!='completed','Completed request wire missing')
  if record['status']=='completed':
   expected=decoded(child(corpus,row['prompt_ids_file']).read_bytes());parsed=validate_response(response,engine,alias,expected)
   wanted={'id':row['id'],'canonical_request_sha256':row['canonical_request_sha256'],'status':'completed','wire_request_sha256':sha(wire),'wire_request_bytes':len(wire),'http_status':200,'response_sha256':sha(response),'response_bytes':len(response),**parsed}
   require(canonical(record)==canonical(wanted),'Persisted case does not match fresh response replay');records.append(record)
  else:
   require(not complete and i==len(folders)-1 and record['status'] in ('failed','started'),'Nonterminal/error case ordering differs');failed.append({'id':row['id'],'status':record['status'],'error_type':record.get('error_type'),'error':record.get('error'),'response_sha256':sha(response) if response_file.exists() else None,'scope':'Historical transport/error cause is preserved, not re-executed or promoted.'})
 require(type(workflow['attempted_requests']) is int and workflow['attempted_requests']==attempts and type(workflow['completed_requests']) is int and workflow['completed_requests']==len(records),'Request counters differ')
 score=None
 if complete:
  require(len(records)==len(rows)==805 and not failed,'Complete cohort missing');score=score_complete(gold,records);require(canonical(score)==canonical(decoded((run/'score.json').read_bytes())),'Stored score differs from raw response replay')
 return {'status':'adapted_mmmu_http_quality_responses_replayed' if complete else 'adapted_mmmu_http_quality_incomplete_failure_preserved','corpus_manifest_sha256':MANIFEST_SHA256,'run_file_manifest_sha256':sha(raw),'files_verified':len(manifest['files']),'engine':engine,'model_alias':alias,'attempted_requests':attempts,'completed_requests':len(records),'failed_or_interrupted_cases':failed,'score':score,'unadmitted_score_file_preserved':not complete and (run/'score.json').exists(),'official_mmmu_score':False,'server_identity_attested_by_client':False,'runtime_image_grid_verified':False,'runtime_image_preprocessing_verified':False,'quality_promotion_admitted':False,'performance_admitted':False,'verifier_sha256':sha(Path(__file__).read_bytes()),'scope':'Raw response/wire/prompt-ID screening replay. This805single-image256pixel adaptation is not the full official900-question MMMU, an independent server/model identity attestation, pixel-preprocessing proof, broad quality promotion or timing result.'}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 require(not a.output.resolve().is_relative_to(a.run.resolve()),'Write replay report outside the immutable run')
 result=replay(a.corpus,a.run);raw=encoded(result)
 if a.output.exists():require(a.output.read_bytes()==raw,'Preserve previous differing replay report')
 else:a.output.write_bytes(raw)
 print(json.dumps({k:v for k,v in result.items() if k!='score'}));return 0 if result['status']=='adapted_mmmu_http_quality_responses_replayed' else 1
if __name__=='__main__':raise SystemExit(main())
