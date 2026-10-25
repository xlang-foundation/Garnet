"""Independently replay closed EOS-policy evidence; this checker is separately versioned."""
from pathlib import Path
import argparse,base64,hashlib,io,json,sys,zipfile
from benchmark_fixed_length_native import final_ids,ARCHIVE_SHA256,MANIFEST_SHA256
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();name='fixed-length-eos-consistency-attempt1';archive=args.proofs/(name+'-evidence.zip');base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/';profiles=[]
 with zipfile.ZipFile(archive) as z:
  doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json');assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
  for path,digest in manifest['files'].items():assert sha(z.read(path))==digest
  w=doc(base+'workflow.json');p=doc(methods+'evidence/fixed-length-eos-preparation.json');assert sha(z.read(methods+'evidence/fixed-length-eos-preparation.json'))==w['preparation_sha256']
  assert w['cleanup_finished'] and w['library_unchanged'] and w['source_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime_dependencies'] and w['parents']==p['parents'] and not w['performance_admitted']
  for path,digest in p['files'].items():assert sha(z.read(path))==digest
  for parent,row in p['parents'].items():
   raw=z.read('results/'+parent+'-verified.json');q=json.loads(raw);assert sha(raw)==row['proof_sha256'] and q['status']==row['status'] and q['archive_sha256']==row['archive_sha256']
  assert sha(z.read(methods+'benchmark_fixed_length_native.py'))==sha(Path(__file__).with_name('benchmark_fixed_length_native.py').read_bytes())
  passed=w['exit_code']==0
  if passed:
   assert w['status']=='fixed_length_eos_collected_needs_replay' and [v['mode'] for v in w['variants']]==['ordinary','mtp'] and all(v['finished'] and v['exit_code']==0 for v in w['variants'])
   assert b'Ran 43 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
   historical=doc(methods+'evidence/native-natural-eos-failure-reanalysis.json');events=[json.loads(line) for line in z.read(historical['early_eos_events_file']).splitlines()];old=next(json.loads(e['data']) for e in events if e['data']!='[DONE]' and 'response_token_ids' in json.loads(e['data']));old_ids=old['response_token_ids'];assert len(old_ids)==74 and old_ids[-1]==248046
   raw=z.read(methods+'evidence/fixed-length-4096-corpus.zip');assert sha(raw)==ARCHIVE_SHA256
   with zipfile.ZipFile(io.BytesIO(raw)) as context:
    assert sha(context.read('manifest.json'))==MANIFEST_SHA256;corpus=json.loads(context.read('manifest.json'));row=next(r for r in corpus['rows'] if r['key']=='ctx4096-s3-image-c2-r0');body=json.loads(context.read(row['request_file']));prompt=json.loads(context.read(row['expected_prompt_ids_file']));assert prompt==old['prompt_token_ids'] and len(prompt)==3968
   image_hash=sha(base64.b64decode(body['messages'][0]['content'][0]['image_url']['url'].split(',',1)[1],validate=True));ordinary={}
   for mode in ('ordinary','mtp'):
    prefix=base+mode+'/';report=doc(prefix+'report.json');assert report['status']=='fixed_length_eos_profile_collected' and report['mode']==mode and len(report['cases'])==4 and not report['performance_admitted'] and report['ready']['library_sha256']==w['library_sha256'] and report['ready']['explicit_ignore_eos_supported']
    tokens={};completed={};ready=[];used=set()
    for line in z.read(prefix+'worker/worker-stdout.jsonl').splitlines():
     try:event=json.loads(json.loads(line)['line'])
     except ValueError:continue
     if not isinstance(event,dict):continue
     if event.get('event')=='ready':ready.append(event)
     elif event.get('event')=='error':raise AssertionError(event)
     elif event.get('event')=='token':
      key=event['id'];tokens.setdefault(key,[]).append(event);assert event['sampled_tokens']==len(tokens[key])
     elif event.get('event')=='completed':
      key=event['id'];assert key not in completed and [v['token_id'] for v in tokens[key]]==event['generated_ids'];completed[key]=event
    assert ready==[report['ready']] and len(completed)==4
    def match(value,ignore):
     key=value['id'];choice=value['choices'][0];ids=choice.get('response_token_ids',value.get('response_token_ids'));actual_prompt=choice.get('prompt_token_ids',value.get('prompt_token_ids'));assert key in completed and key not in used;used.add(key);record=completed[key];assert type(record['ignore_eos']) is bool and record['ignore_eos']==ignore and record['generated_ids']==ids and record['prompt_ids']==actual_prompt==prompt and record['image_sha256']==image_hash and record['image_grid']==[1,16,16]
     assert len(ids)==(128 if ignore else 74) and ids[:74]==old_ids and all(type(v) is int and 0<=v<248320 for v in ids) and value['usage']=={'prompt_tokens':3968,'completion_tokens':len(ids),'total_tokens':3968+len(ids)} and choice['finish_reason']==('length' if ignore else 'stop')
     assert all(v['stop'] is False for v in tokens[key][:-1]) and tokens[key][-1]['stop'] is (not ignore);return ids
    for label,ignore in (('natural-before',False),('forced',True),('natural-after',False)):
     expected=json.loads(json.dumps(body));expected.update(stream=False,ignore_eos=ignore);assert doc(prefix+label+'-request.json')==expected;value=doc(prefix+label+'-response.json');ids=match(value,ignore)
     if mode=='ordinary':ordinary[label]=value['choices'][0]
     else:assert value['choices'][0]==ordinary[label]
    assert doc(prefix+'forced-stream-request.json')==body;events=[json.loads(line) for line in z.read(prefix+'forced-stream-events.jsonl').splitlines()];value=final_ids(events,prompt);ids=match(value,True);assert ids==doc(prefix+'forced-response.json')['choices'][0]['response_token_ids'] and used==set(completed)
    resource=doc(base+mode+'-resources/resource-monitor.json');assert resource['child_exit_code']==0 and resource['gpu']['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6';profiles.append({'mode':mode,'actual_requests':4,'natural_stop_tokens':74,'forced_actual_tokens':128,'prefix74_exact':True,'normal_stop_restored':True,'forced_stream_ids_exact':True,'ordinary_mtp_exact':mode=='mtp','performance_admitted':False})
 report={'status':'fixed_length_eos_consistency_replayed' if passed else 'fixed_length_eos_consistency_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'library_sha256':w['library_sha256'],'profiles':profiles,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'checker_in_runtime_archive':False,'performance_admitted':False,'two_times_target_achieved':False,'scope':'Actual closed8request EOS-policy/source/runtime/normal-reset/ordinary-MTP/SSE replay. Checker is separately versioned on study branch, not injected into immutable runtime archive. No full805quality/fixed-length performance/reference/tuning/2x admission.'};raw=json.dumps(report,indent=2).encode();target=args.proofs/(name+'-verified.json');assert not target.exists() or target.read_bytes()==raw
 if not target.exists():target.write_bytes(raw)
 print(json.dumps(report,indent=2));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
