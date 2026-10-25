"""Compare every recorded raw hidden/normalized vector, including transient prefix controls."""
from pathlib import Path
import argparse,hashlib,json,zipfile
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();name='tokenizer-nfc-model-consistency-attempt1';old='bf16-low-batch-model-consistency-attempt2'
 needed=[args.proofs/(name+s) for s in ('-evidence.zip','-verified.json','-remote-verified.json')]
 if not all(f.is_file() for f in needed):print(json.dumps({'status':'model_vector_audit_prerequisite_missing_no_output'}));return 2
 proof=json.loads((args.proofs/(name+'-verified.json')).read_bytes());assert proof==json.loads((args.proofs/(name+'-remote-verified.json')).read_bytes()) and proof['status']=='tokenizer_nfc_model_consistency_replayed';archive=args.proofs/(name+'-evidence.zip');assert sha(archive.read_bytes())==proof['archive_sha256'];profiles=[]
 with zipfile.ZipFile(archive) as z:
  manifest=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
  for n,h in manifest['files'].items():assert sha(z.read(n))==h
  for label,batch in (('b4-smoke',4),('b1',1)):
   prefix='results/'+name+'/'+label+'/';parent='results/'+old+'/'+label+'/';actual={n.removeprefix(prefix):n for n in manifest['files'] if n.startswith(prefix) and n.endswith('.f32')};retained={n.removeprefix(parent):n for n in manifest['files'] if n.startswith(parent) and n.endswith('.f32')};assert actual and set(actual)==set(retained);rows=[]
   for key,n in sorted(actual.items()):
    raw=z.read(n);previous=z.read(retained[key]);assert len(raw)==batch*20480 and raw==previous;rows.append({'file':key,'bytes':len(raw),'sha256':sha(raw)})
   profiles.append({'label':label,'physical_batch':batch,'all_recorded_vectors':len(rows),'vectors':rows,'scope':'All persisted .f32 vectors, including prefix-before/restored/rollback snapshots not referenced as report state objects; full raw bytes exact against8ae.'})
 result={'status':'tokenizer_nfc_all_recorded_vectors_match_retained8ae','model_archive_sha256':proof['archive_sha256'],'library_sha256':proof['library_sha256'],'profiles':profiles,'total_raw_vectors':sum(p['all_recorded_vectors'] for p in profiles),'auditor_sha256':sha(Path(__file__).read_bytes()),'performance_admitted':False,'scope':'Supplement to both actual closed model replays: exhaustive full raw hidden/normalized vector file-set/byte comparison. State metadata, token/callback and ordinary/MTP/prefix checks remain those of the base model proof. No quality, actual4Kserving or timing admission.'};target=args.proofs/(name+'-all-vectors-verified.json');data=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==data
 if not target.exists():target.write_bytes(data)
 print(json.dumps({k:v for k,v in result.items() if k!='profiles'},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
