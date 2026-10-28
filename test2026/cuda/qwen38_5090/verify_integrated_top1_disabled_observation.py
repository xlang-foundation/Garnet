"""Replay a finished mode snapshot while the original two-mode controller lives.

This deliberately cannot emit the final model proof filename or admit serving.
"""
from pathlib import Path
import hashlib,json,sys,zipfile
from integrated_top1_model_replay import compare

def replay(proofs):
 proofs=Path(proofs);m=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
 parent='bf16-padded-lifecycle-scope-fixed-attempt2';job='integrated-batch-top1-model-lifecycle-attempt1'
 archive=proofs/'integrated-batch-top1-disabled-observation.zip';parent_archive=proofs/(parent+'-evidence.zip')
 assert sha(archive.read_bytes())=='f0269fb8e0a797465df20a8c5199f68edff99a17c496f430e08dc36d2b532692'
 proofraw=(proofs/(parent+'-verified.json')).read_bytes();proof=json.loads(proofraw);assert proof==json.loads((proofs/(parent+'-remote-verified.json')).read_bytes())
 assert proof['status']=='bf16_padded_lifecycle_replayed' and sha(parent_archive.read_bytes())==proof['archive_sha256']
 prepraw=(m/'evidence/integrated-top1-model-preparation.json').read_bytes();p=json.loads(prepraw)
 assert sha(proofraw)==p['parents'][parent]['proof_sha256'] and proof['archive_sha256']==p['parents'][parent]['archive_sha256']
 assert sha((m/'integrated_top1_model_replay.py').read_bytes())==p['files']['Garnet/test2026/cuda/qwen38_5090/integrated_top1_model_replay.py']
 with zipfile.ZipFile(parent_archive) as z:
  manifest=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
  for n,h in manifest['files'].items():assert sha(z.read(n))==h
  parent_count=len(manifest['files'])
  prefix='results/'+parent+'/native/';old=json.loads(z.read(prefix+'report.json'));old_vectors={n.removeprefix(prefix):z.read(n) for n in manifest['files'] if n.startswith(prefix) and n.endswith('.f32')}
 with zipfile.ZipFile(archive) as z:
  manifest=json.loads(z.read('observation-manifest.json'));assert manifest['status']=='closed_disabled_variant_observation_only'
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1==106
  for n,h in manifest['files'].items():assert sha(z.read(n))==h
  base='results/'+job+'/';w=json.loads(z.read(base+'workflow.json'));assert w['status']=='running' and w['cleanup_finished'] is False
  assert w['preparation_sha256']==sha(prepraw) and w['source_sha256']==p['files'] and w['runtime_dependencies']==p['runtime_dependencies']
  assert w['library_sha256']==p['runtime_dependencies']['garnet.dll']=='5ac61e3a15466be3cea151190eb52c0403732bb3ad2abdb39a6ba086a5c21473'
  disabled,enabled=w['variants'];assert disabled['label']=='disabled' and disabled['mode']=='0' and disabled['finished'] is True and disabled['exit_code']==0 and disabled['parent_parity_passed'] is True
  assert enabled=={'label':'enabled','mode':'1','finished':False}
  prefix=base+'disabled/';new=json.loads(z.read(prefix+'report.json'));new_vectors={n.removeprefix(prefix):z.read(n) for n in manifest['files'] if n.startswith(prefix) and n.endswith('.f32')}
  parity=compare(old,new,old_vectors,new_vectors,w['library_sha256'],'0');assert parity==json.loads(z.read(base+'disabled-parent-parity.json'))
 return {'status':'disabled_mode_parent_parity_observed_whole_run_pending','snapshot_archive_sha256':sha(archive.read_bytes()),'snapshot_files_verified':105,'parent_archive_sha256':sha(parent_archive.read_bytes()),'parent_files_verified':parent_count,
  'mode':'0','cases':parity['cases'],'raw_vectors_exact':parity['raw_vectors_exact'],'raw_vector_bytes':parity['raw_vector_bytes'],'dispatch_before':parity['dispatch_before'],'dispatch_after':parity['dispatch_after'],
  'library_sha256':w['library_sha256'],'preparation_sha256':sha(prepraw),'final_source_runtime_closure_available':False,'enabled_mode_finished':False,'model_qualified':False,'performance_admitted':False,
  'observer_verifier_sha256':sha(Path(__file__).read_bytes()),'scope':'Independent local CPU replay of the actual finished disabled-mode snapshot agrees with remote controller parity: all14case report/ID/state/callback/prefix/reset/retirement fields and102raw vectors. Enabled mode remains live; no final source/runtime-after proof or whole-job archive exists in this snapshot. This is not the final model proof and does not unblock B1 runtime factories, broad quality, actual4K, HTTP or speed.'}

if __name__=='__main__':
 out=Path(sys.argv[1]);result=replay(out);(out/'Garnet-Parallel-Top1-Disabled-Observation.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
