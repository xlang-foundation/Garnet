#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Derive research-only compact Qwen proposal heads from static tokenizer rules.

No benchmark outputs select rows. Input embeddings and every decoder tensor are
unchanged. Output indices must be explicitly mapped back to original token IDs;
these assets are not generic Qwen checkpoints or target-model replacements.
"""
import argparse,hashlib,json,mmap,shutil,struct,time
from pathlib import Path

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def byte_inverse():
 bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs.copy();n=0
 for b in range(256):
  if b not in bs:bs.append(b);cs.append(256+n);n+=1
 return {chr(c):b for b,c in zip(bs,cs)}

def select(tokenizer,limit):
 inverse=byte_inverse();chosen=set();one_byte=set()
 for token,index in tokenizer['model']['vocab'].items():
  raw=bytes(inverse[c] for c in token)
  if len(raw)==1:one_byte.add(index)
  if len(raw)==1 or (index<limit and all(c in (9,10,13) or 32<=c<=126 for c in raw)):chosen.add(index)
 assert len(one_byte)==256
 chosen.update(t['id'] for t in tokenizer['added_tokens'])
 return sorted(chosen)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 model=a.model.resolve();root=a.output.resolve();root.mkdir(exist_ok=False)
 config=json.loads((model/'config.json').read_text());tokenizer=json.loads((model/'tokenizer.json').read_text())
 assert config['model_type']=='qwen3' and config['hidden_size']==1024 and config['tie_word_embeddings'] is True
 assert tokenizer['model']['type']=='BPE' and tokenizer['decoder']['type']=='ByteLevel'
 manifest={'created_ns':time.time_ns(),'source':str(model),'source_files':{p.name:digest(p) for p in model.iterdir() if p.is_file()},'selection':'Static printable ASCII plus tab/LF/CR, all256 single-byte tokens, all added tokens. Second profile applies original ID<65536 to multibyte ASCII tokens. No measured output is consulted.','scope':__doc__,'profiles':[]}
 with (model/'model.safetensors').open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
  size=struct.unpack_from('<Q',mm)[0];header=json.loads(mm[8:8+size]);base=8+size;original={k:v for k,v in header.items() if k!='__metadata__'}
  for label,limit in [('ascii',config['vocab_size']),('ascii64k',65536)]:
   target=root/label;target.mkdir();ids=select(tokenizer,limit);assert ids==sorted(set(ids)) and all(0<=i<config['vocab_size'] for i in ids)
   mapping={'token_ids':ids,'source_vocabulary_size':config['vocab_size'],'rule_limit':limit,'research_only':True}
   (target/'output-token-map.json').write_text(json.dumps(mapping,separators=(',',':'))+'\n')
   head={}
   for suffix in ['weight','scales','biases']:
    meta=original['model.embed_tokens.'+suffix];assert meta['shape'][0]==config['vocab_size'];stride=(meta['data_offsets'][1]-meta['data_offsets'][0])//meta['shape'][0]
    data=bytearray()
    for i in ids:
     offset=base+meta['data_offsets'][0]+i*stride;data.extend(mm[offset:offset+stride])
    head['lm_head.'+suffix]=(dict(dtype=meta['dtype'],shape=[len(ids),*meta['shape'][1:]]),data)
   out_header={};offset=0
   for name,meta in original.items():
    n=meta['data_offsets'][1]-meta['data_offsets'][0];out_header[name]=dict(dtype=meta['dtype'],shape=meta['shape'],data_offsets=[offset,offset+n]);offset+=n
   for name,(meta,data) in head.items():out_header[name]=dict(meta,data_offsets=[offset,offset+len(data)]);offset+=len(data)
   encoded=json.dumps(out_header,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%8)
   with (target/'model.safetensors').open('xb') as out:
    out.write(struct.pack('<Q',len(encoded)));out.write(encoded)
    for meta in original.values():out.write(mm[base+meta['data_offsets'][0]:base+meta['data_offsets'][1]])
    for meta,data in head.values():out.write(data)
   derived=dict(config,tie_word_embeddings=False,garnet_research_output_token_map='output-token-map.json',garnet_research_only=True)
   (target/'config.json').write_text(json.dumps(derived,indent=2)+'\n')
   for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json']:
    if (model/name).exists():shutil.copy2(model/name,target/name)
   # Independently re-read the derived file and verify every unchanged tensor and selected row byte.
   with (target/'model.safetensors').open('rb') as cf,mmap.mmap(cf.fileno(),0,access=mmap.ACCESS_READ) as cm:
    cn=struct.unpack_from('<Q',cm)[0];ch=json.loads(cm[8:8+cn]);cb=8+cn
    for name,meta in original.items():
     other=ch[name];assert other['shape']==meta['shape'] and other['dtype']==meta['dtype']
     assert cm[cb+other['data_offsets'][0]:cb+other['data_offsets'][1]]==mm[base+meta['data_offsets'][0]:base+meta['data_offsets'][1]]
    selected_bytes=0
    for suffix in ['weight','scales','biases']:
     meta=original['model.embed_tokens.'+suffix];other=ch['lm_head.'+suffix];stride=(meta['data_offsets'][1]-meta['data_offsets'][0])//config['vocab_size']
     for row,i in enumerate(ids):
      assert cm[cb+other['data_offsets'][0]+row*stride:cb+other['data_offsets'][0]+(row+1)*stride]==mm[base+meta['data_offsets'][0]+i*stride:base+meta['data_offsets'][0]+(i+1)*stride]
      selected_bytes+=stride
   record={'label':label,'root':str(target),'head_rows':len(ids),'original_rows':config['vocab_size'],'selected_head_bytes':selected_bytes,'all_original_tensors_bitwise_unchanged':len(original),'every_selected_row_bitwise_unchanged':True,'token_ids_sorted_unique':True,'files':{p.name:digest(p) for p in target.iterdir() if p.is_file()}}
   manifest['profiles'].append(record);print(label,len(ids),'selected head bytes',selected_bytes,flush=True)
 (root/'assets-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
