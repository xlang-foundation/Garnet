"""CPU-only native tokenizer versus pinned Rust tokenizer/frozen corpus IDs.

Load only an explicitly supplied, standalone tokenizer library; no Garnet model,
SDK or GPU runtime. Retain every actual mismatch; never change frozen inputs.
"""
from pathlib import Path
import argparse,ctypes,hashlib,importlib.metadata,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.fixed_context_workload import study_prompt_ids
from test2026.cuda.qwen38_5090.mmmu_http_quality import load_corpus
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();m=Path(__file__).resolve().parent
 observation=json.loads((m/'evidence/fixed-context-tokenizer-source-observation.json').read_bytes())
 for name in ('tokenizer.json','tokenizer_config.json'):
  raw=(a.assets/name).read_bytes();assert sha(raw)==observation['files'][name]['sha256'] and len(raw)==observation['files'][name]['bytes']
 from tokenizers import Tokenizer
 reference=Tokenizer.from_file(str(a.assets/'tokenizer.json'));rust=lambda s:reference.encode(s,add_special_tokens=False).ids
 library=ctypes.CDLL(str(a.library.resolve()));encode=library.GarnetQwenTokenizerEncode;encode.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_longlong),ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.c_char_p,ctypes.c_int];encode.restype=ctypes.c_int
 def native(text):
  assert '\0' not in text;buffer=(ctypes.c_longlong*8192)();count=ctypes.c_int();error=ctypes.create_string_buffer(512);code=encode(str(a.assets.resolve()).encode('utf8'),text.encode('utf8'),buffer,len(buffer),ctypes.byref(count),error,len(error));assert code==0,(code,error.value);return list(buffer[:count.value])
 a.output.mkdir();records=[]
 def compare(key,text,expected,actual,kind):
  index=next((i for i,(x,y) in enumerate(zip(expected,actual)) if x!=y),min(len(expected),len(actual)));record={'key':key,'kind':kind,'text':text,'expected_ids':expected,'actual_ids':actual,'exact':expected==actual,'first_difference':None if expected==actual else index};records.append(record)
  with (a.output/'cases.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(record,ensure_ascii=True)+'\n')
 _,rows,_=load_corpus(a.corpus)
 for row in rows:
  body=json.loads((a.corpus/row['request_file']).read_bytes());expected=json.loads((a.corpus/row['prompt_ids_file']).read_bytes());assert study_prompt_ids(body,rust,248056)==expected;compare(row['id'],body['messages'][0]['content'][1]['text'],expected,study_prompt_ids(body,native,248056),'frozen805')
 with zipfile.ZipFile(m/'evidence/fixed-context-4096-corpus.zip') as z:
  manifest=json.loads(z.read('manifest.json'))
  for row in manifest['rows']:
   body=json.loads(z.read(row['request_file']));expected=json.loads(z.read(row['expected_prompt_ids_file']));assert study_prompt_ids(body,rust,248056)==expected;compare(row['key'],None,expected,study_prompt_ids(body,native,248056),'frozen4k')
 samples=['Maximum  possible','x  word','x   word','x\tword','x\t word','x  123','x \nword','x \r\nword','x  !word','x  ','\n\tword',"DON'T", "WE'RE", "It'S",'café naïve résumé','αβ ΓΔ','中文 ABC','x² + y³ = 2','e\u0301cole','\u00a0word','one\u2003two','\t\n word','12\n\n','a,b,c',',word',"HELLO'S"]
 for i,text in enumerate(samples):compare('edge-'+str(i),text,rust(text),native(text),'edge')
 mismatches=[r for r in records if not r['exact']];report={'status':'cpu_tokenizer_corpus_exact' if not mismatches else 'cpu_tokenizer_corpus_mismatches_preserved','library_sha256':sha(a.library.read_bytes()),'source_sha256':sha(a.source.read_bytes()),'tokenizer_json_sha256':sha((a.assets/'tokenizer.json').read_bytes()),'tokenizers_version':importlib.metadata.version('tokenizers'),'counts':{kind:{'cases':sum(r['kind']==kind for r in records),'mismatches':sum(r['kind']==kind and not r['exact'] for r in records)} for kind in ('frozen805','frozen4k','edge')},'mismatch_keys':[r['key'] for r in mismatches],'raw_cases_sha256':sha((a.output/'cases.jsonl').read_bytes()),'gpu_executed':False,'scope':'Actual standalone native CPU encode IDs compared to pinned Rust/frozen IDs. All inputs unchanged; no model, GPU, HTTP quality or timing promotion.'};(a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2));return 1 if mismatches else 0
if __name__=='__main__':raise SystemExit(main())
