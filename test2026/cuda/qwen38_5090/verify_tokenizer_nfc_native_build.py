"""Independent actual native build/source/ABI evidence replay, never model admission."""
from pathlib import Path
import hashlib,io,json,sys,zipfile
out=Path(sys.argv[1]);name='tokenizer-nfc-native-build-attempt1';archive=out/(name+'-evidence.zip');base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/';sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(archive) as z:
 doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json');assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
 for n,h in manifest['files'].items():assert sha(z.read(n))==h
 p=doc(methods+'evidence/tokenizer-nfc-native-build-preparation.json');w=doc(base+'workflow.json');assert sha(z.read(methods+'evidence/tokenizer-nfc-native-build-preparation.json'))==w['preparation_sha256'] and w['cleanup_finished'] and not w['performance_admitted']
 assert w['parents']==p['parents'] and w['runtime_before']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['methods_after']==p['methods'] and w['retained_library_sha256']==p['runtime']['garnet.dll'] and w['cuda_objects_before']==p['cuda_objects_before']
 for n,h in {**p['methods'],**p['inputs']}.items():assert sha(z.read(n))==h
 for n,row in p['parents'].items():
  raw=z.read('results/'+n+'-verified.json');q=json.loads(raw);assert sha(raw)==row['proof_sha256'] and q['archive_sha256']==row['archive_sha256'] and q['status']==row['status']
 cpu=doc('results/tokenizer-nfc-cpu-qualified-verified.json');assert cpu==p['cpu_proof'] and cpu['status']=='tokenizer_nfc_cpu_evidence_replayed' and sha(z.read('results/tokenizer-nfc-cpu-qualified-evidence.zip'))==cpu['archive_sha256'] and cpu['corpus_cases_exact']==955 and cpu['normalization_scalar_cases_exact']==1112064
 with zipfile.ZipFile(io.BytesIO(z.read(base+'original-source.zip'))) as old:
  assert set(old.namelist())=={n for n,h in p['candidate_base'].items() if h is not None} and len(old.namelist())==2
  for n in old.namelist():assert sha(old.read(n))==p['candidate_base'][n]
 with zipfile.ZipFile(io.BytesIO(z.read(p['source_zip']))) as candidate:
  assert len(candidate.namelist())==len(set(candidate.namelist()))==len(p['candidate_files'])==5
  for n,h in p['candidate_files'].items():assert sha(candidate.read(n))==h==cpu['source_sha256'][n]
 passed=w['exit_code']==0
 if passed:
  assert w['status']=='tokenizer_native_built_cpu_qualified_model_unqualified' and w['installed_source']==w['source_after']=={**p['base_files'],**{'Garnet/'+n:h for n,h in p['candidate_files'].items()}}
  for n,h in w['source_after'].items():assert sha(z.read(n))==h
  assert w['candidate_library_sha256']==w['library_sha256']==w['runtime_after']['garnet.dll']!=p['runtime']['garnet.dll']
  for n,h in p['runtime'].items():
   if n!='garnet.dll':assert w['runtime_after'][n]==h
  assert b'Ran 49 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
  for ledger in ('cpu-ledger','build-ledger','native-cpu-ledger'):
   records=[json.loads(line) for line in z.read(base+ledger+'/commands.jsonl').splitlines()];assert len(records)==2 and records[0]['event']=='started' and records[1]['event']=='finished' and records[0]['id']==records[1]['id'] and records[1]['exit_code']==0
  r=doc(base+'native-tokenizer-cpu/report.json');actual=[json.loads(line) for line in z.read(base+'native-tokenizer-cpu/cases.jsonl').splitlines()];assert r['status']=='rebuilt_native_tokenizer_cpu_exact' and not r['mismatches'] and r['cases']==len(actual)==955 and r['library_sha256']==w['library_sha256'] and r['oracle_sha256']==cpu['oracle_archive_sha256'] and r['raw_cases_sha256']==sha(z.read(base+'native-tokenizer-cpu/cases.jsonl'))
  with zipfile.ZipFile(io.BytesIO(z.read('results/tokenizer-nfc-cpu-oracles.zip'))) as oracle:
   expected=[json.loads(line) for line in oracle.read('cases.jsonl').splitlines()];assert len(expected)==955
  for a,b in zip(actual,expected):assert a['key']==b['key'] and a['kind']==b['kind'] and a['exact'] and a['code']==0 and a['actual_ids']==b['expected_ids']
 else:
  assert w['status']=='tokenizer_native_build_failed_restored' and w['runtime_after']==p['runtime'] and not w['restore_skipped_source_drift']
  assert w['source_after']=={**p['base_files'],**{'Garnet/'+n:h for n,h in p['candidate_base'].items()}}
 archived=sha(z.read(methods+'verify_tokenizer_nfc_native_build.py'))
report={'status':'tokenizer_nfc_native_build_replayed' if passed else 'tokenizer_nfc_native_build_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'library_sha256':w.get('library_sha256'),'retained_library_sha256':w['retained_library_sha256'],'source_files_installed':5 if passed else 0,'cpu_regressions_passed':49 if passed else 0,'native_cpu_tokenizations_exact':955 if passed else 0,'cuda_objects_bitwise_unchanged':w['cuda_objects_before']==w['cuda_objects_after'],'cuda_objects_count':len(w['cuda_objects_before']),'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,'gpu_model_qualified':False,'streaming_http_qualified':False,'performance_admitted':False,'scope':p['scope']};(out/(name+'-verified.json')).write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2));raise SystemExit(0 if passed else 1)
