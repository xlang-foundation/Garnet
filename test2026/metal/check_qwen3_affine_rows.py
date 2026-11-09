#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent FP64 dequantization/FP16 embedding oracle for native codec rows."""
import argparse,hashlib,json,mmap,struct,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir()
checkpoint=a.model/'model.safetensors'
with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
 n=struct.unpack_from('<Q',mapping)[0];h=json.loads(mapping[8:8+n]);base=8+n;requests=[]
 for name,m in h.items():
  if m['dtype']=='U32':requests.append(dict(tensor='model.language_model.embed_tokens.weight' if name=='model.embed_tokens.weight' else name,rows=sorted(set([0,m['shape'][0]//2,m['shape'][0]-1]))))
 request=a.output/'request.json';request.write_text(json.dumps(requests,indent=2)+'\n')
 subprocess.run([str(a.binary.resolve()),str(checkpoint.resolve()),str(request.resolve()),str((a.output/'native').resolve()),'qwen3'],check=True)
 native=json.loads((a.output/'native/native.json').read_text());words=0;records=[]
 def tensor(name,kind):
  meta=h[name];return np.frombuffer(mapping,dtype=kind,count=int(np.prod(meta['shape'])),offset=base+meta['data_offsets'][0]).reshape(meta['shape'])
 for row in native:
  name=row['tensor'].replace('model.language_model.','model.');stem=name[:-7];indices=row['rows']
  q=tensor(name,'<u4')[indices];s=tensor(stem+'.scales','<f2')[indices].astype(np.float64);b=tensor(stem+'.biases','<f2')[indices].astype(np.float64)
  codes=((q[...,None]>>(4*np.arange(8,dtype=np.uint32)))&15).reshape(len(indices),-1).astype(np.float64)
  oracle=(codes*s.repeat(64,axis=-1)+b.repeat(64,axis=-1)).astype('<f4');rounded=oracle.astype('<f2').astype('<f4')
  actual=np.fromfile(row['raw'],dtype='<f4').reshape(oracle.shape);embedding=np.fromfile(row['rounded'],dtype='<f4').reshape(oracle.shape)
  if not np.array_equal(actual,oracle) or not np.array_equal(embedding,rounded):raise ValueError('Native codec differs: '+name)
  words+=oracle.size;records.append(dict(tensor=name,rows=indices,words=oracle.size,raw_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),rounded_sha256=hashlib.sha256(rounded.tobytes()).hexdigest()))
 report=dict(completed=True,matrices=len(records),words=words,records=records,binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope=__doc__)
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('native codec PASS',len(records),'matrices',words,'FP32 values and FP16-rounded embeddings')
