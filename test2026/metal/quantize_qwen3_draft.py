#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only deterministic min/max affine INT4/g64 conversion of pinned Qwen3.

Draft artifact only: quantization may change draft predictions. Mistral target
weights/IDs stay unchanged and native target verification remains mandatory.
No PyTorch, MLX, Rust runtime or full dequantized model is used for inference.
"""
import argparse,hashlib,json,mmap,shutil,struct
from pathlib import Path
import numpy as np

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def convert(source,output):
 source=source.resolve();output=output.resolve();output.mkdir()
 manifest=json.loads((source/'download-manifest.json').read_text())
 if manifest['repo']!='Qwen/Qwen3-0.6B' or manifest['revision']!='c1899de289a04d12100db370d81485cdf75e47ca':raise ValueError('Unpinned source')
 for row in manifest['files']:
  if (source/row['file']).stat().st_size!=row['bytes'] or sha(source/row['file'])!=row['sha256']:raise ValueError('Source asset mismatch')
 config=json.loads((source/'config.json').read_text())
 if config['model_type']!='qwen3' or config['hidden_size']!=1024 or not config['tie_word_embeddings']:raise ValueError('Unqualified draft architecture')
 with (source/'model.safetensors').open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
  length=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+length]);base=8+length
  emb=header['model.embed_tokens.weight'];head=header['lm_head.weight']
  if emb['dtype']!='BF16' or emb['shape']!=head['shape'] or head['dtype']!='BF16':raise ValueError('Invalid tied head')
  ea,eb=emb['data_offsets'];ha,hb=head['data_offsets']
  if eb-ea!=hb-ha:raise ValueError('Unequal tied payload lengths')
  for offset in range(0,eb-ea,8*1024*1024):
   n=min(8*1024*1024,eb-ea-offset)
   if mapping[base+ea+offset:base+ea+offset+n]!=mapping[base+ha+offset:base+ha+offset+n]:raise ValueError('Tied head differs from embedding')
  planned={};offset=0
  for name,meta in header.items():
   if name in ('__metadata__','lm_head.weight'):continue
   shape=meta['shape'];dtype=meta['dtype']
   if dtype!='BF16' or len(shape) not in (1,2):raise ValueError('Unqualified source tensor '+name)
   entries=[(name,dtype,shape)]
   if len(shape)==2:
    n,k=shape
    if k%64 or not name.endswith('.weight'):raise ValueError('Unpackable tensor '+name)
    root=name[:-7];entries=[(name,'U32',[n,k//8]),(root+'.scales','F16',[n,k//64]),(root+'.biases','F16',[n,k//64])]
   for target,kind,dims in entries:
    count=int(np.prod(dims));size=count*(4 if kind=='U32' else 2);planned[target]=dict(dtype=kind,shape=dims,data_offsets=[offset,offset+size]);offset+=size
  wire=json.dumps(planned,separators=(',',':')).encode();wire+=b' '*((-len(wire))%8);start=8+len(wire);stats=[]
  with (output/'model.safetensors').open('xb') as target:
   target.write(struct.pack('<Q',len(wire)));target.write(wire);target.truncate(start+offset)
   def write(name,row,values):
    info=planned[name];stride=int(np.prod(info['shape'][1:]));target.seek(start+info['data_offsets'][0]+row*stride*values.dtype.itemsize);target.write(values.tobytes())
   for name,meta in header.items():
    if name in ('__metadata__','lm_head.weight'):continue
    a,b=meta['data_offsets'];shape=meta['shape']
    if len(shape)==1:target.seek(start+planned[name]['data_offsets'][0]);target.write(mapping[base+a:base+b]);continue
    n,k=shape;root=name[:-7];max_error=0.;squared=0.;source_squared=0.;elements=0
    for row in range(0,n,1024):
     count=min(1024,n-row);raw=np.frombuffer(mapping,dtype='<u2',count=count*k,offset=base+a+row*k*2)
     values=(raw.astype(np.uint32)<<16).view(np.float32).reshape(count,k//64,64);del raw
     if not np.isfinite(values).all():raise ValueError('Nonfinite source '+name)
     low=values.min(axis=-1);high=values.max(axis=-1);bias=low.astype('<f2');scale=((high-low)/np.float32(15)).astype('<f2')
     constant=high==low;scale[constant]=np.float16(1.)
     if not np.isfinite(bias).all() or not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('FP16 coefficient overflow/underflow '+name)
     codes=np.clip(np.rint((values-bias.astype(np.float32)[...,None])/scale.astype(np.float32)[...,None]),0,15).astype(np.uint32)
     packed=np.zeros((count,k//8),dtype='<u4');flat=codes.reshape(count,k//8,8)
     for nibble in range(8):packed|=flat[:,:,nibble]<<(4*nibble)
     restored=(codes.astype(np.float64)*scale.astype(np.float64)[...,None]+bias.astype(np.float64)[...,None]).astype(np.float32)
     error=restored.astype(np.float64)-values.astype(np.float64)
     max_error=max(max_error,float(np.max(np.abs(error))));squared+=float(np.sum(error*error));source_squared+=float(np.sum(values.astype(np.float64)**2));elements+=values.size
     write(name,row,packed);write(root+'.scales',row,scale);write(root+'.biases',row,bias)
    stats.append(dict(name=name,shape=shape,elements=elements,max_abs_quantization_error=max_error,l2_relative_error=(squared/source_squared)**.5 if source_squared else 0.))
  if (output/'model.safetensors').stat().st_size!=start+offset:raise ValueError('Incomplete converted checkpoint')
 config['quantization']=dict(bits=4,group_size=64,mode='affine');(output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
 for row in manifest['files']:
  if row['file'] not in ('model.safetensors','config.json'):shutil.copyfile(source/row['file'],output/row['file'])
 report=dict(source=str(source),source_repo=manifest['repo'],source_revision=manifest['revision'],source_files=manifest['files'],converter_sha256=sha(Path(__file__)),policy='min/max per64, FP16 scale/bias, ties-to-even nearest stored-coefficient code, clamp0..15; low-nibble first; constant groups scale1; identical duplicated tied head omitted',matrices=stats,files=[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(output.iterdir()) if p.is_file()],scope=__doc__)
 (output/'conversion-manifest.json').write_text(json.dumps(report,indent=2)+'\n');print('converted',len(stats),'matrices','bytes',(output/'model.safetensors').stat().st_size,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();convert(a.source,a.output)
