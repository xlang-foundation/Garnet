#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Actual packed Qwen2/Qwen3 projection samples with an independent FP64 dot oracle."""
import argparse,json,mmap,struct
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir();cells=[]
config=json.loads((a.model/'config.json').read_text());family=config['model_type'];assert family in ('qwen2','qwen3');vocab=config['vocab_size']
with (a.model/'model.safetensors').open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
 n=struct.unpack_from('<Q',mapping)[0];h=json.loads(mapping[8:8+n]);base=8+n
 def selected(name,rows):
  def read(label,dtype):
   meta=h[label];return np.frombuffer(mapping,dtype=dtype,count=int(np.prod(meta['shape'])),offset=base+meta['data_offsets'][0]).reshape(meta['shape'])[rows].copy()
  packed=read(name,'<u4');scale=read(name[:-7]+'.scales','<f2').astype(np.float64);bias=read(name[:-7]+'.biases','<f2').astype(np.float64)
  codes=((packed[...,None]>>(4*np.arange(8,dtype=np.uint32)))&15).reshape(len(rows),-1)
  return (codes*scale.repeat(64,axis=-1)+bias.repeat(64,axis=-1)).astype(np.float32)
 names=['model.embed_tokens.weight']
 for layer in (0,config['num_hidden_layers']-1):
  names+=['model.layers.'+str(layer)+'.'+suffix+'.weight' for suffix in ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']]
 for name in names:
  n,packed_k=h[name]['shape'];k=packed_k*8;columns=sorted(set([0,1,n//7,n//4,n//2,3*n//4,n-2,n-1]));weight=selected(name,columns)
  for rows in (1,2,4,16):
   x=(np.sin(np.arange(rows*k,dtype=np.float64)*.173)*.05).astype('<f4').reshape(rows,k);path=(a.output/(str(len(cells))+'-input.f32')).resolve();x.tofile(path)
   left=x.astype(np.float64);right=weight.astype(np.float64)
   if rows>4:left=x.astype('<f2').astype(np.float64);right=weight.astype('<f2').astype(np.float64)
   expected=left@right.T;absolute=np.abs(left)@np.abs(right).T
   cells.append(dict(weight=name,rows=rows,inner=k,outputs=n,columns=columns,input=str(path),expected=expected.tolist(),absolute=absolute.tolist(),operand_dtype='fp16' if rows>4 else 'fp32'))
 ids=[0,1,100,1000,vocab//2,vocab-2,vocab-1];embedding=selected('model.embed_tokens.weight',ids).astype('<f2').astype('<f4')
 (a.output/'fixture.json').write_text(json.dumps(dict(checkpoint=str((a.model/'model.safetensors').resolve()),cells=cells,family=family,vocabulary_size=vocab,checkpoint_format='mlx_'+family+'_affine_int4_g64_fp16',embedding_ids=ids,embedding_expected=embedding.reshape(-1).tolist(),scope=__doc__),indent=2)+'\n');print('prepared',len(cells),'actual-weight projection cells')
