"""Fixed-order FP32 norm: independent FP64 bounds and exact row geometry."""
import hashlib,json,math,os,struct,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T
request_path,output_path=map(Path,sys.argv[1:3]);assert not output_path.exists()
request=json.loads(request_path.read_text());fixed=os.environ.get('GARNET_QWEN35_FIXED_RMS')=='1'
io=garnet.cuda_tensor_io();graph=None
out={'fixed_rms_requested':fixed,'cases':[],'all_math_bounds_passed':False,'all_row_geometry_exact':False,'full_goal_met':False}
def save():output_path.write_text(json.dumps(out,indent=2)+'\n')
try:
    all_math=True;all_exact=True
    for case in request['cases']:
        h=case['hidden'];known={};gamma=case['gamma'];eps=case['epsilon'];vectors=case['vectors']
        unit=2.**-24;steps=math.ceil(h/256)+8;gam=steps*unit/(1-steps*unit)
        relative_bound=gam+16*unit
        # The bound is frozen before dispatch: positive-square FP32 sum with
        # ceil(H/256) local FMAs +8 tree additions, then inverse norm/scaling.
        refs=[]
        for vector in vectors:
            denom=math.sqrt(math.fsum(float(x)*float(x) for x in vector)/h+eps)
            refs.append([float(x)/denom*float(w) for x,w in zip(vector,gamma)])
        shapes=[[1,h],[1,4,h],[1,8,h],[1,16,h],[1,32,h],[1,64,h],[1,128,h],[2,4,2,h],[8,8,2,h]]
        for shape in shapes:
            rows=math.prod(shape[:-1]);values=[]
            for row in range(rows):values.extend(vectors[row%len(vectors)])
            @T.fusion(name='qwen35_fixed_rms_primitive',boundary='required')
            def normalize(value):
                result=value*T.unary_op('qwen35_rms_norm',weight_name=case['weight_name'],eps=eps)
                return result*T.unary_op('reshape',shape=[1,rows*h])
            symbol=tensor.input('fixed_rms_input',shape=shape,dtype=tensor.float32)
            graph=garnet.compiled_graph();graph.build(normalize(symbol),[symbol],case['checkpoint']);graph.enable_cuda_graph(True)
            graph.enable_serial_graph_reuse(True)
            input_tensor=io.zeros(shape);io.update(input_tensor,values)
            before=io.finite_f32_bytes(input_tensor);actual=io.finite_f32_bytes(graph.run_device([input_tensor]))
            assert io.finite_f32_bytes(input_tensor)==before
            assert io.finite_f32_bytes(graph.run_device([input_tensor]))==actual
            words=struct.unpack('<'+str(rows*h)+'f',actual);maximum=0.;fraction=0.;different_rows=0
            for row in range(rows):
                index=row%len(vectors);raw=actual[row*h*4:(row+1)*h*4]
                if index in known:
                    if known[index]!=raw:different_rows+=1
                else:known[index]=raw
                for col in range(h):
                    value=words[row*h+col];reference=refs[index][col]
                    assert math.isfinite(value)
                    error=abs(value-reference);bound=2e-7+relative_bound*abs(reference)
                    maximum=max(maximum,error);fraction=max(fraction,error/bound)
            all_math=all_math and fraction<=1;all_exact=all_exact and different_rows==0
            out['cases'].append({'model':case['model'],'hidden':h,'shape':shape,'rows':rows,'words':rows*h,
                'max_abs_fp64_error':maximum,'max_fraction_of_predeclared_bound':fraction,'different_repeated_vector_rows':different_rows,
                'row_geometry_exact':different_rows==0,'input_unchanged':True,'exact_replay':True,'output_sha256':hashlib.sha256(actual).hexdigest(),
                'graph_stats':json.loads(graph.stats_json())})
            save();graph.close();graph=None
            print('FIXED_RMS_PRIMITIVE',h,shape,fraction,different_rows,flush=True)
    out['all_math_bounds_passed']=all_math;out['all_row_geometry_exact']=all_exact;save()
    assert all_math,'Independent predeclared FP64 norm bound failed'
    if fixed:assert all_exact,'Fixed norm changed a row with its batch/token geometry'
finally:
    if graph is not None:graph.close()
    io.close();out['closed']=True;save()
