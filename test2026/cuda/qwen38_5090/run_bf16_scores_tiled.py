"""Exclusive isolated score-tiling numerical attempt; no model or timing run."""
from pathlib import Path
import argparse,hashlib,json,os,sys,zipfile
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--closed-http-archive-sha256',required=True);args=a.parse_args()
 root=Path('D:/CantorAI/5090-study-20261023');m=Path(__file__).resolve().parent;sys.path.insert(0,str(m))
 from run_logged import run
 from run_bf16_low_batch_native_build import idle
 assert m==(root/'Garnet/test2026/cuda/qwen38_5090').resolve()
 native='fixed-length-http-study-attempt1';native_archive=root/'results'/(native+'-evidence.zip')
 assert sha(native_archive)==args.closed_http_archive_sha256
 closed=json.loads((root/'results'/native/'workflow.json').read_bytes());assert closed['exit_code']==0 and closed['cleanup_finished'] and closed['library_unchanged']
 with zipfile.ZipFile(native_archive) as z:assert z.testzip() is None and json.loads(z.read('results/'+native+'/workflow.json'))==closed
 prep=m/'evidence/bf16-scores-tiled-run-preparation.json';p=json.loads(prep.read_bytes());assert p['status']=='bf16_scores_tiled_gpu_correctness_prepared_unrun'
 for n,digest in p['files'].items():assert sha(root/n)==digest,('Isolated input drift',n)
 parent=p['parent'];proof=json.loads((root/'results'/(parent+'-verified.json')).read_bytes());assert proof['status']=='tokenizer_nfc_model_consistency_replayed' and proof['archive_sha256']==p['parent_archive_sha256']==sha(root/'results'/(parent+'-evidence.zip'))
 library=root/'build/bin/garnet.dll';assert sha(library)==p['observed_unused_library_sha256']==proof['library_sha256']
 name='bf16-scores-tiled-correctness-attempt1';out=root/'results'/name;archive=root/'results'/(name+'-evidence.zip');assert not out.exists() and not archive.exists()
 idle();out.mkdir();os.chdir(root/'Garnet');os.environ['CUDA_VISIBLE_DEVICES']='0';code=1
 w={'status':'running','source_sha256':p['files'],'preparation_sha256':sha(prep),'parent_proof_sha256':p['parent_proof_sha256'],'parent_archive_sha256':p['parent_archive_sha256'],
    'closed_http_archive_sha256':args.closed_http_archive_sha256,'observed_unused_library_sha256':sha(library),'runtime_loaded':False,'cleanup_finished':False,'performance_admitted':False,'scope':p['scope']}
 def save():(out/'workflow.json').write_text(json.dumps(w,indent=2),encoding='utf8')
 common=[root/n for n in p['files']]+[prep,root/'results'/native/'workflow.json'];save()
 try:
  code=run([sys.executable,'-m','unittest','test2026.cuda.qwen38_5090.test_bf16_low_batch_candidate','test2026.cuda.qwen38_5090.test_bf16_kv_overlay_oracle'],out/'cpu-ledger','correctness','Existing B1 profile and FP64 oracle controls before separate tiled-score GPU probe',common)
  if code:raise RuntimeError('Isolated attention CPU controls failed')
  executable=root/'tools/bf16-kv-native/bf16-scores-tiled.exe'
  code=run([sys.executable,str(m/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--',str(executable),str(out/'witnesses.f32')],out/'gpu-ledger','correctness','Actual44case full output/workspace/graph/prefix/readonly/canary and both baseline/candidate malformed-call controls; FP64 replay mandatory',common)
  if code:raise RuntimeError('Tiled score numerical collection failed')
  w['status']='bf16_scores_tiled_collected_needs_fp64_replay'
 except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
 finally:
  w.update(exit_code=code,source_after_sha256={n:sha(root/n) for n in p['files']},library_unchanged=sha(library)==w['observed_unused_library_sha256']);save()
  idle();w['cleanup_finished']=True;save()
  paths=[f for f in out.rglob('*') if f.is_file()]+common;raw={f.relative_to(root).as_posix():f.read_bytes() for f in paths}
  manifest={'files':{n:hashlib.sha256(data).hexdigest() for n,data in raw.items()},'scope':p['scope']}
  with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
   for n,data in raw.items():z.writestr(n,data)
   z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
  print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'files':len(raw),'bytes':archive.stat().st_size}),flush=True)
 return code
if __name__=='__main__':raise SystemExit(main())
