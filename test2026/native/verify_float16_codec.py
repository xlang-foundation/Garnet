#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent NumPy oracle for the native IEEE binary16 codec."""
import argparse,hashlib,json,pathlib,subprocess
import numpy as np

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--binary',type=pathlib.Path,required=True)
 p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 rng=np.random.default_rng(30917)
 half=np.arange(65536,dtype=np.uint16).view(np.float16);finite=np.isfinite(half)
 with np.errstate(invalid='ignore'):expanded=half.astype(np.float32)
 values=np.sort(expanded[finite]);mid=((values[:-1].astype(np.float64)+values[1:].astype(np.float64))/2).astype(np.float32)
 random=rng.integers(0,2**32,size=1000000,dtype=np.uint32).view(np.float32)
 values=np.concatenate([expanded[finite],mid,np.nextafter(mid,np.float32(-np.inf)),np.nextafter(mid,np.float32(np.inf)),random]).astype('<f4')
 values.tofile(a.output/'input-f32.bin')
 result=subprocess.run([str(a.binary.resolve()),str(a.output/'input-f32.bin'),str(a.output/'output-f16.bin'),str(a.output/'decoded-f32.bin')],capture_output=True,text=True,check=True)
 (a.output/'native.log').write_text(result.stdout+result.stderr)
 actual=np.fromfile(a.output/'decoded-f32.bin',dtype='<f4');valid=~np.isnan(expanded)
 if len(actual)!=65536:raise AssertionError('wrong native decode count')
 decode_bad=int(np.count_nonzero(actual[valid].view(np.uint32)!=expanded[valid].view(np.uint32)))
 decode_nan_bad=int(np.count_nonzero(np.isnan(actual)!=np.isnan(expanded)))
 with np.errstate(over='ignore',invalid='ignore'):expected=values.astype(np.float16)
 got=np.fromfile(a.output/'output-f16.bin',dtype='<u2').view(np.float16);valid=~np.isnan(expected)
 if len(got)!=len(values):raise AssertionError('wrong native encode count')
 encode_bad=int(np.count_nonzero(got[valid].view(np.uint16)!=expected[valid].view(np.uint16)))
 encode_nan_bad=int(np.count_nonzero(np.isnan(got)!=np.isnan(expected)))
 passed=decode_bad==decode_nan_bad==encode_bad==encode_nan_bad==0
 source=pathlib.Path(__file__).resolve().parents[2]/'src/model/float16_codec.h'
 report=dict(passed=passed,decode_cases=65536,finite_roundtrips=int(finite.sum()),encode_cases=len(values),decode_mismatches=decode_bad,decode_nan_classification_mismatches=decode_nan_bad,encode_mismatches=encode_bad,encode_nan_classification_mismatches=encode_nan_bad,oracle='numpy '+np.__version__,seed=30917,scope='CPU codec only; FP16 quantized Metal kernels and checkpoint admission not qualified',files={f.name:sha(f) for f in a.output.iterdir() if f.is_file()},source_sha256=sha(source),binary_sha256=sha(a.binary),verifier_sha256=sha(pathlib.Path(__file__)))
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
 raise SystemExit(0 if passed else 1)

if __name__=='__main__':main()
