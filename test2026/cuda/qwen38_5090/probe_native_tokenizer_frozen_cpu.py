"""Actual rebuilt native C API versus955frozen text-tokenization oracles; no model."""
from pathlib import Path
import argparse,ctypes,hashlib,json,os,zipfile
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--oracle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir();root=a.study;library=root/'build/bin/garnet.dll';assets=root/'models/Qwen3.8-27B-NVFP4'
 assert sha(assets/'tokenizer.json')=='0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3'
 handles=[os.add_dll_directory(str(f)) for f in (root/'build/bin',root/'tools/TensorRT/bin',root/'tools/cuda13/nvidia/cu13/bin/x86_64') if f.is_dir()];lib=ctypes.CDLL(str(library));fn=lib.GarnetQwenTokenizerEncode;fn.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_longlong),ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.c_char_p,ctypes.c_int];fn.restype=ctypes.c_int
 with zipfile.ZipFile(a.oracle) as z:
  assert z.testzip() is None and set(z.namelist())=={'manifest.json','cases.jsonl'};raw=z.read('cases.jsonl');manifest=json.loads(z.read('manifest.json'));assert hashlib.sha256(raw).hexdigest()==manifest['sha256'];rows=[json.loads(line) for line in raw.splitlines()];assert len(rows)==manifest['cases']==955
 mismatches=[];counts={}
 with (a.output/'cases.jsonl').open('x',encoding='utf8') as stream:
  for row in rows:
   assert '\0' not in row['text'];buf=(ctypes.c_longlong*8192)();count=ctypes.c_int();error=ctypes.create_string_buffer(512);code=fn(str(assets).encode('utf8'),row['text'].encode('utf8'),buf,len(buf),ctypes.byref(count),error,len(error));actual=list(buf[:count.value]);exact=code==0 and actual==row['expected_ids'];record={'key':row['key'],'kind':row['kind'],'actual_ids':actual,'code':code,'error':error.value.decode('utf8',errors='replace'),'exact':exact};stream.write(json.dumps(record)+'\n');counts[row['kind']]=counts.get(row['kind'],0)+1
   if not exact:mismatches.append(row['key'])
 assert counts=={'frozen805':805,'frozen4k':124,'edge':26}
 report={'status':'rebuilt_native_tokenizer_cpu_exact' if not mismatches else 'rebuilt_native_tokenizer_cpu_failed','library_sha256':sha(library),'oracle_sha256':sha(a.oracle),'cases':955,'counts':counts,'mismatches':mismatches,'raw_cases_sha256':sha(a.output/'cases.jsonl'),'native_library_loaded':True,'gpu_model_executed':False,'performance_admitted':False,'scope':'Actual full rebuilt native DLL C API tokenization only.955text templates include64image placeholders; image processing/model/HTTP gates remain separate.'};(a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2));return 1 if mismatches else 0
if __name__=='__main__':raise SystemExit(main())
