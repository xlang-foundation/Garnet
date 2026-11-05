#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare native typed coefficient/embedding decoding to real MLX dequantize.

Map the pinned checkpoint on CPU and submit only the selected rows to Metal.
Never expand or upload the entire checkpoint as part of this component check.
"""
import argparse,hashlib,importlib.metadata,json,pathlib,struct
import numpy as np
import mlx.core as mx

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=pathlib.Path,required=True)
 p.add_argument('--native',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True)
 a=p.parse_args();a.output.mkdir(exist_ok=False)
 with a.checkpoint.open('rb') as stream:
  length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
 records=json.loads((a.native/'native.json').read_text());results=[]
 for index,record in enumerate(records):
  base=record['tensor'][:-len('.weight')];selected=[]
  for name in [record['tensor'],base+'.scales',base+'.biases']:
   info=header[name];dtype={'U32':'<u4','F16':'<f2'}[info['dtype']]
   mapping=np.memmap(a.checkpoint,mode='r',dtype=dtype,offset=8+length+info['data_offsets'][0],shape=tuple(info['shape']))
   selected.append(mx.array(np.array(mapping[record['rows']],copy=True)));del mapping
  weights,scales,biases=selected
  # An output dtype override alone retains FP16 coefficient arithmetic in
  # MLX0.32.4. Cast the exact coefficient values first for the FP32-math check.
  raw=mx.dequantize(weights,scales.astype(mx.float32),biases.astype(mx.float32),group_size=64,bits=4,dtype=mx.float32)
  output_override=mx.dequantize(weights,scales,biases,group_size=64,bits=4,dtype=mx.float32)
  rounded=mx.dequantize(weights,scales,biases,group_size=64,bits=4)
  if rounded.dtype!=mx.float16:raise AssertionError('MLX did not infer FP16 embedding output')
  rounded=rounded.astype(mx.float32);mx.eval(raw,rounded,output_override)
  override_words=np.array(output_override,dtype='<f4').reshape(-1).view(np.uint32)
  rounded_words=np.array(rounded,dtype='<f4').reshape(-1).view(np.uint32)
  row=dict(tensor=record['tensor'],rows=record['rows'],columns=record['columns'],output_dtype_override_equals_fp16_arithmetic=bool(np.array_equal(override_words,rounded_words)),comparisons=[])
  for label,reference in [('raw',raw),('rounded',rounded)]:
   expected=np.array(reference,dtype='<f4').reshape(-1);got=np.fromfile(record[label],dtype='<f4')
   if len(expected)!=len(got):raise AssertionError('wrong native output size')
   mismatches=int(np.count_nonzero(expected.view(np.uint32)!=got.view(np.uint32)))
   path=a.output/(str(index)+'-'+label+'-mlx.f32');expected.tofile(path)
   row['comparisons'].append(dict(kind=label,words=len(got),mismatches=mismatches,native_sha256=sha(pathlib.Path(record[label])),reference_sha256=sha(path)))
  results.append(row)
 report=dict(passed=all(c['mismatches']==0 for r in results for c in r['comparisons']),mlx_version=importlib.metadata.version('mlx'),numpy_version=np.__version__,checkpoint=str(a.checkpoint),native_report_sha256=sha(a.native/'native.json'),verifier_sha256=sha(pathlib.Path(__file__)),results=results,scope='Selected real rows only; no whole-model generation or speed claim')
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
 raise SystemExit(0 if report['passed'] else 1)

if __name__=='__main__':main()
