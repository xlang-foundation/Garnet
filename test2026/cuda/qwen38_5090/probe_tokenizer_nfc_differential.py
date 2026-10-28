"""Actual private C++ NFC/encode differential against pinned Rust; no GPU/model."""
from pathlib import Path
import argparse,ctypes,hashlib,json,random,struct,sys
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir();out=a.output
 import tokenizers
 from tokenizers import normalizers,pre_tokenizers,Regex,Tokenizer
 assert tokenizers.__version__=='0.22.2';nfc=normalizers.NFC().normalize_str;lib=ctypes.CDLL(str(a.library.resolve()));fn=lib.GarnetStudyNormalizeNfc;fn.argtypes=[ctypes.POINTER(ctypes.c_uint32),ctypes.c_int,ctypes.POINTER(ctypes.c_uint32),ctypes.c_int,ctypes.POINTER(ctypes.c_int)];fn.restype=ctypes.c_int
 input_buffer=(ctypes.c_uint32*256)();output_buffer=(ctypes.c_uint32*1024)();count=ctypes.c_int();counts={'scalar':0,'sequence':0,'encode':0,'configuration':0,'negative':0};failures=[];report={'status':'running','library_sha256':sha(a.library.read_bytes()),'counts':counts,'failures':failures,'gpu_executed':False}
 def save():(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
 def normalize(text):
  assert len(text)<=len(input_buffer)
  for i,c in enumerate(text):input_buffer[i]=ord(c)
  code=fn(input_buffer,len(text),output_buffer,len(output_buffer),ctypes.byref(count));assert code==0,code
  return list(output_buffer[:count.value])
 save()
 try:
  with (out/'scalar-witness.u32').open('xb') as raw:
   for cp in range(0x110000):
    if 0xD800<=cp<=0xDFFF:continue
    expected=list(map(ord,nfc(chr(cp))));actual=normalize(chr(cp));raw.write(struct.pack('<II',cp,len(expected)));raw.write(struct.pack('<'+'I'*len(expected),*expected));raw.write(struct.pack('<I',len(actual)));raw.write(struct.pack('<'+'I'*len(actual),*actual));counts['scalar']+=1
    if expected!=actual:failures.append({'kind':'scalar','cp':cp,'expected':expected,'actual':actual});raise AssertionError(failures[-1])
  save();print(json.dumps({'stage':'all_unicode_scalars_exact','cases':counts['scalar']}),flush=True)
  universe=''.join(chr(cp) for cp in range(0x110000) if not 0xD800<=cp<=0xDFFF);marks=[c for text,_ in pre_tokenizers.Split(Regex(r'[^\p{M}]+'),behavior='removed').pre_tokenize_str(universe) for c in text];rng=random.Random(20261025)
  cases=['e\u0301cole','A\u030a\u0301','\u1e9b\u0323','a\u0345\u0334','\u212b','\u0344','\u1100\u1161\u11a8','\uac00\u11a8','','\0','\u0345\u0301\u0334','x\u0301\u0300\u0323']
  for i in range(30000):
   if i%3==0:cases.append(rng.choice(['a','e','A','\u03b1','\u05d0','\u0b47','\u1100'])+''.join(rng.choices(marks,k=rng.randrange(1,12))))
   elif i%3==1:cases.append(''.join(rng.choices(marks+list('abcXYZ\u1100\u1161\u11a8\uac00\u212b'),k=rng.randrange(0,24))))
   else:cases.append(''.join(rng.choice(universe) for _ in range(rng.randrange(0,20))))
  with (out/'sequence-witness.jsonl').open('x',encoding='utf8') as raw:
   for i,text in enumerate(cases):
    expected=list(map(ord,nfc(text)));actual=normalize(text);row={'case':i,'text':text,'expected':expected,'actual':actual};raw.write(json.dumps(row,ensure_ascii=True)+'\n');counts['sequence']+=1
    if actual!=expected:failures.append({'kind':'sequence',**row});raise AssertionError(failures[-1])
  save();print(json.dumps({'stage':'seeded_normalization_sequences_exact','cases':counts['sequence']}),flush=True)
  encode=lib.GarnetQwenTokenizerEncode;encode.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_longlong),ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.c_char_p,ctypes.c_int];encode.restype=ctypes.c_int
  def encode_raw(folder,wire):
   buf=(ctypes.c_longlong*8192)();size=ctypes.c_int();error=ctypes.create_string_buffer(512);code=encode(str(folder.resolve()).encode('utf8'),wire,buf,len(buf),ctypes.byref(size),error,len(error));return code,list(buf[:size.value]),error.value.decode('utf8',errors='replace')
  reference=Tokenizer.from_file(str(a.assets/'tokenizer.json'));pieces=['word','Maximum','  ',' ','\t','\r\n','\n\n',"DON'T","WE'RE","'ſ","'S",',word','cafe\u0301','\u03b1\u03b2','中文','\u1100\u1161\u11a8','\u212b','\u0344','\u0301','\u00a0','\u2003','²','123','e\u0301cole','<|im_start|>','<|im_end|>','!\n']
  texts=list(pieces)+[''.join(rng.choices(pieces,k=rng.randrange(1,20))) for _ in range(5000)]
  with (out/'encode-witness.jsonl').open('x',encoding='utf8') as raw:
   for i,text in enumerate(texts):
    expected=reference.encode(text,add_special_tokens=False).ids;code,actual,error=encode_raw(a.assets,text.encode('utf8'));row={'case':i,'text':text,'expected':expected,'actual':actual,'code':code,'error':error};raw.write(json.dumps(row,ensure_ascii=True)+'\n');counts['encode']+=1
    if code or actual!=expected:failures.append({'kind':'encode',**row});raise AssertionError(failures[-1])
  # Small actual BPE vocab exercises configured NFC, none and unsupported modes
  # without making copies of the12MB production tokenizer or loading model code.
  vocab={value:i for i,value in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))};pre=json.loads((a.assets/'tokenizer.json').read_bytes())['pre_tokenizer'];configurations=[]
  for label,normalizer in (('none',None),('nfc',{'type':'NFC'}),('unsupported',{'type':'NFD'}),('malformed','NFC')):
   folder=out/('config-'+label);folder.mkdir();body={'model':{'type':'BPE','vocab':vocab,'merges':[]},'normalizer':normalizer,'pre_tokenizer':pre,'added_tokens':[]};(folder/'tokenizer.json').write_text(json.dumps(body),encoding='utf8');code,actual,error=encode_raw(folder,'e\u0301cole'.encode('utf8'))
   if label in ('none','nfc'):
    expected=Tokenizer.from_str(json.dumps(body)).encode('e\u0301cole',add_special_tokens=False).ids;assert code==0 and actual==expected
   else:expected=None;assert code!=0 and 'supports only NFC or no normalizer' in error
   configurations.append({'case':label,'actual_ids':actual,'expected_ids':expected,'code':code,'error':error});counts['configuration']+=1
  (out/'configuration-controls.json').write_text(json.dumps(configurations,indent=2),encoding='utf8')
  negative=[]
  for wire in (b'\x80',b'\xc0\xaf',b'\xe2\x82',b'\xed\xa0\x80',b'\xf4\x90\x80\x80',b'\xf0\x28\x8c\xbc'):
   code,actual,error=encode_raw(a.assets,wire);assert code!=0;negative.append({'wire_hex':wire.hex(),'code':code,'error':error});counts['negative']+=1
  for cp in (0xD800,0xDFFF,0x110000):
   input_buffer[0]=cp;code=fn(input_buffer,1,output_buffer,len(output_buffer),ctypes.byref(count));assert code!=0;negative.append({'invalid_scalar':cp,'code':code});counts['negative']+=1
  (out/'negative-controls.json').write_text(json.dumps(negative,indent=2),encoding='utf8');report['status']='native_nfc_and_encode_differential_exact'
 except BaseException as error:report.update(status='failed',error=repr(error));raise
 finally:
  report['artifact_sha256']={f.name:sha(f.read_bytes()) for f in out.iterdir() if f.is_file() and f.name!='report.json'};report['scope']='Actual standalone native CPU normalization for every Unicode scalar and seeded sequences, real encode IDs and config/invalid input controls. No GPU/model/HTTP or performance claim.';save()
 print(json.dumps(report,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
