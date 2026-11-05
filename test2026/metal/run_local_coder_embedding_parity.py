"""Native requested-row parity. Args: oracle.json native-build source out.json.

Run with XLang3, separately from the MLX oracle and all measured campaigns.
"""
import sys,json,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet
with open(sys.argv[1]) as stream:oracle=json.load(stream)
io=garnet.tensor_io()
try:
    io.open(oracle['checkpoint'])
    assert io.checkpoint_format=='mlx_qwen35_affine_int4_g64_bf16'
    actual=io.embedding(oracle['tokens']).tolist()
    expected=[value for row in oracle['expected'] for value in row]
    assert len(actual)==len(expected)==len(oracle['tokens'])*oracle['width']
    differences=0;maximum=0
    for x,y in zip(actual,expected):
        assert math.isfinite(x) and math.isfinite(y)
        if x!=y:differences+=1
        maximum=max(maximum,abs(x-y))
    result=dict(status='passed' if differences==0 else 'failed',tokens=oracle['tokens'],
                compared_values=len(actual),differing_values=differences,max_abs=maximum,
                backend=garnet.backend_info(),scope=oracle['scope'])
    with open(sys.argv[4],'w') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    assert differences==0,result
    print('native_mlx_embedding_parity PASS',len(actual),flush=True)
finally:
    io.close()
    garnet.shutdown()
