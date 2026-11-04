"""Isolated MLX operator diagnostic; never a replacement vLLM serving baseline.

Uses the native projection probe's deterministic INT4/group64/BF16 operands.
MLX stores BF16 output whereas Garnet stores FP32; report that difference.
Run only when all serving engines and native GPU tests/builds have stopped.
"""
import argparse,hashlib,importlib.metadata,json,os,statistics,time
from pathlib import Path
import mlx.core as mx
import numpy as np
from run_local_coder_baseline import engine_processes

def bf16_float(values):
    values=np.asarray(values,dtype=np.float32)
    bits=values.view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cohorts',type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.cohorts<=10:parser.error('Invalid cohort count')
    live=[p for p in engine_processes() if int(p.split(None,1)[0])!=os.getpid()]
    if live:raise RuntimeError('Another inference engine is live: '+str(live))
    mx.set_default_device(mx.gpu)
    dimensions=((48,5120),(1024,5120),(5120,6144),(5120,17408),(6144,5120),(10240,5120),(12288,5120),(17408,5120))
    # This is a large-prefill probe. MLX's small-row quantized matvec uses
    # grouped scale/bias arithmetic instead of the rounded-BF16 matrix tile.
    # Do not apply this matrix oracle or infer decode performance there.
    rows=(64,128,256,512)
    report=dict(status='running',scope=__doc__,mlx_version=importlib.metadata.version('mlx'),device=mx.device_info(),
        operand_dtype='bfloat16',output_dtype='bfloat16',native_output_dtype='float32',
        group_size=64,bits=4,warmups=3,repetitions=9,cohorts=args.cohorts,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cells=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        def save():
            stream.seek(0);json.dump(report,stream,indent=2);stream.write('\n');stream.truncate();stream.flush()
        save()
        for cohort in range(1,args.cohorts+1):
            for m in rows:
                for n,k in dimensions:
                    weight=(np.arange(n*(k//8),dtype=np.uint32)*np.uint32(2654435761))^np.uint32(0x76543210)
                    indices=np.arange(n*(k//64),dtype=np.int64)
                    scales=bf16_float((indices%13-6).astype(np.float32)*np.float32(.017)).reshape(n,-1)
                    biases=bf16_float((indices%17-8).astype(np.float32)*np.float32(.031)).reshape(n,-1)
                    x=bf16_float((np.sin(np.arange(m*k,dtype=np.float64)*.173)*.05).astype(np.float32)).reshape(m,k)
                    a=mx.array(x,dtype=mx.bfloat16);w=mx.array(weight.reshape(n,k//8));s=mx.array(scales,dtype=mx.bfloat16);b=mx.array(biases,dtype=mx.bfloat16)
                    mx.eval(a,w,s,b);mx.synchronize()
                    samples=[]
                    for repeat in range(-3,9):
                        began=time.monotonic_ns()
                        y=mx.quantized_matmul(a,w,scales=s,biases=b,transpose=True,group_size=64,bits=4)
                        mx.eval(y);mx.synchronize()
                        elapsed=(time.monotonic_ns()-began)/1e6
                        if repeat>=0:samples.append(elapsed)
                    actual=np.array(y.astype(mx.float32));maximum=0.0
                    if not np.isfinite(actual).all():raise RuntimeError('Nonfinite MLX projection')
                    for r,c in ((0,0),(m//2,n//2),(m-1,n-1)):
                        words=weight.reshape(n,k//8)[c]
                        q=((words[:,None]>>(4*np.arange(8,dtype=np.uint32)))&15).reshape(-1).astype(np.float64)
                        decoded=bf16_float((q*scales[c].repeat(64).astype(np.float64)+biases[c].repeat(64)).astype(np.float32))
                        expected=float(np.dot(x[r].astype(np.float64),decoded.astype(np.float64)))
                        error=abs(float(actual[r,c])-expected);maximum=max(maximum,error)
                        if error>max(.001,abs(expected)*.008):raise RuntimeError('MLX BF16 output differs from FP64 sample')
                    cell=dict(cohort=cohort,rows=m,outputs=n,inner=k,wall_ms=samples,median_wall_ms=statistics.median(samples),sample_max_abs=maximum)
                    report['cells'].append(cell)
                    save()
                    del a,w,s,b,y,actual,weight,indices,scales,biases,x
                    mx.clear_cache()
            print(json.dumps(dict(cohort=cohort,cells=len(report['cells']))),flush=True)
        report['status']='passed'
        save()
if __name__=='__main__':main()
