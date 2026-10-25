"""Derive explicit ignore_eos fixed-length requests without changing any prompt or image."""
from pathlib import Path
import argparse,hashlib,json,zipfile
sha=lambda raw:hashlib.sha256(raw).hexdigest()
OLD_ARCHIVE='ca78cdc12239810851e9943ccf6b318e8b29074b0fb0908728dd58132ae6923f'
OLD_MANIFEST='829d7770f2d749e30a66e027940224e8ab77a488d0b5f9c456e4809ae845987c'

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--output',required=True,type=Path);args=a.parse_args();m=Path(__file__).resolve().parent;old=m/'evidence/fixed-context-4096-corpus.zip';assert sha(old.read_bytes())==OLD_ARCHIVE and not args.output.exists()
 with zipfile.ZipFile(old) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()));raw=z.read('manifest.json');assert sha(raw)==OLD_MANIFEST;manifest=json.loads(raw);data={n:z.read(n) for n in z.namelist() if n!='manifest.json'}
 assert len(manifest['rows'])==124 and len(manifest['images'])==62
 for row in manifest['rows']:
  name=row['request_file'];old_wire=data[name];assert sha(old_wire)==row['request_sha256'];body=json.loads(old_wire);assert 'ignore_eos' not in body and body['max_tokens']==128;before=json.loads(old_wire);body['ignore_eos']=True;wire=json.dumps(body,separators=(',',':')).encode();assert {k:v for k,v in body.items() if k!='ignore_eos'}==before
  ids=data[row['expected_prompt_ids_file']];assert sha(ids)==row['expected_prompt_ids_sha256'] and len(json.loads(ids))==3968
  data[name]=wire;row['request_sha256']=sha(wire);row['ignore_eos']=True
 manifest.update(status='fixed_length_corpus_prepared_unrun',fixed_length_policy={'ignore_eos':True,'input_tokens':3968,'required_output_tokens':128,'capacity':4096},derivation={'parent_archive_sha256':OLD_ARCHIVE,'parent_manifest_sha256':OLD_MANIFEST,'method_sha256':sha(Path(__file__).read_bytes()),'only_request_change':'Add ignore_eos:true to every canonical request. Prompt text, images/nonces, all prompt-ID bytes, row keys and cohorts remain identical.'},scope='Explicit fixed-length generation benchmark:3968actual input IDs plus128actual output tokens, ignoring EOS for this workload only. Natural-stop original trial remains failed/preserved. Same raw request bodies must be used for native/vLLM/SGLang.805adapted quality continues normal stopping and is separate. CPU corpus derivation is not actual model/HTTP/timing qualification.')
 data['manifest.json']=json.dumps(manifest,indent=2).encode()
 with zipfile.ZipFile(args.output,'x',zipfile.ZIP_DEFLATED) as z:
  for name,raw in sorted(data.items()):z.writestr(name,raw)
 result={'status':manifest['status'],'archive_sha256':sha(args.output.read_bytes()),'manifest_sha256':sha(data['manifest.json']),'requests':124,'unchanged_image_nonces':62,'unchanged_prompt_ids_each':3968,'fixed_length_policy':manifest['fixed_length_policy'],'scope':manifest['scope']};target=args.output.with_name('fixed-length-4096-preparation.json');assert not target.exists();target.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
