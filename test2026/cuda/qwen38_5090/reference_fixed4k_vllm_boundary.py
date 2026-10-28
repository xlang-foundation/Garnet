"""Untimed early text/image boundary checks using unchanged frozen requests.

Preserve failed bytes; no timings, IDs inferred from text, or relaxed128 gate.
"""
from pathlib import Path
import hashlib,json,zipfile
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from benchmark_fixed_length_reference import control_request,control_response,paired_control
from benchmark_fixed_length_native import ARCHIVE_SHA256,MANIFEST_SHA256
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def frozen_cases(archive):
 assert sha(Path(archive).read_bytes())==ARCHIVE_SHA256
 with zipfile.ZipFile(archive) as z:
  raw=z.read('manifest.json');assert sha(raw)==MANIFEST_SHA256;manifest=json.loads(raw);cases=[]
  for case in ('text','image'):
   row=next(v for v in manifest['rows'] if v['case']==case)
   body=z.read(row['request_file']);ids=z.read(row['expected_prompt_ids_file'])
   assert sha(body)==row['request_sha256'] and sha(ids)==row['expected_prompt_ids_sha256']
   body=json.loads(body);ids=json.loads(ids);assert body['ignore_eos'] is True and body['max_tokens']==128 and len(ids)==3968
   cases.append((row['key'],case,json.dumps(control_request(body,'vllm'),separators=(',',':')).encode(),ids))
 return cases

def collect(archive,output,url,ordinary=None):
 output=Path(output);assert url=='http://127.0.0.1:18080' and not output.exists();cases=frozen_cases(archive)
 if ordinary is not None:replay(archive,ordinary)
 output.mkdir();report={'status':'running','cases':[],'timed':False,'actual_workload_context_capacity':4096,'scope':'Early actual3968prompt/128output-ID and4096usage checks before805screen; unchanged frozen requests, no timing or performance admission.'}
 def save():(output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
 save()
 try:
  for key,case,wire,ids in cases:
   (output/(key+'-request.json')).write_bytes(wire)
   try:
    with urlopen(Request(url+'/v1/chat/completions',data=wire,headers={'Content-Type':'application/json'}),timeout=300) as response:raw=response.read()
   except HTTPError as error:(output/(key+'-error.bin')).write_bytes(error.read());raise
   (output/(key+'-response.json')).write_bytes(raw);parsed=control_response(json.loads(raw),'vllm',ids)
   if ordinary is not None:
    ordinary=Path(ordinary);assert (ordinary/(key+'-request.json')).read_bytes()==wire
    paired_control(parsed,control_response(json.loads((ordinary/(key+'-response.json')).read_bytes()),'vllm',ids))
   report['cases'].append({'key':key,'case':case,'request_sha256':sha(wire),'response_sha256':sha(raw),'prompt_ids':3968,'completion_ids':128,'ordinary_ids_exact':ordinary is not None});save()
  report['status']='two_actual_full4k_boundaries_passed';save()
 except BaseException as error:report.update(status='failed',error=repr(error));save();raise
 assert replay(archive,output)==report
 return report

def replay(archive,output):
 output=Path(output);report=json.loads((output/'report.json').read_bytes())
 assert report['status']=='two_actual_full4k_boundaries_passed' and report['timed'] is False and report['actual_workload_context_capacity']==4096 and len(report['cases'])==2
 expected={'report.json'}
 for (key,case,wire,ids),row in zip(frozen_cases(archive),report['cases']):
  assert row['key']==key and row['case']==case and row['prompt_ids']==3968 and row['completion_ids']==128 and type(row['ordinary_ids_exact']) is bool
  request=output/(key+'-request.json');response=output/(key+'-response.json');raw=response.read_bytes()
  assert request.read_bytes()==wire and sha(wire)==row['request_sha256'] and sha(raw)==row['response_sha256']
  control_response(json.loads(raw),'vllm',ids);expected.update((request.name,response.name))
 assert {p.name for p in output.iterdir()}==expected
 return report
