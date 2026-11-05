"""Real TensorRT TTS operators versus independent scalar CPU SiLU projections.

Run using native XLang3. No checkpoint, Python package or Cantor node is needed.
"""
import argparse,json,math,struct,sys,tempfile
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--native-root');args=parser.parse_args()
if args.native_root:sys.path.insert(0,str(Path(args.native_root).resolve()))
import garnet
assert sys.implementation.name=='xlang3'
if args.native_root:assert Path(str(garnet.native_library_path())).resolve()==Path(args.native_root).resolve()/'garnet.dll'
def flatten(value):
    if not isinstance(value,list):return [value]
    result=[]
    for item in value:result.extend(flatten(item))
    return result
text=[[0.,0,0,0],[-2.,-1,.5,2],[1.,-.5,3,-2],[.2,-1.7,.3,1.4],[1.,2,-3,.5]]
codec=[[i/13-.5 for i in range(row*4,row*4+4)] for row in range(5)]
predictor=[[x*.3 for x in row] for row in reversed(codec)]
w1=[[float(i==j) for j in range(4)] for i in range(4)];b1=[.2,-.3,.4,-.5]
w2=[[3.,.6,0,0],[0,1.5,-.3,0],[0,0,3.,.9],[.6,0,0,2.4]];b2=[.1,-.2,.3,-.4]
def linear(value,weight,bias):return [sum(a*b for a,b in zip(row,value))+offset for row,offset in zip(weight,bias)]
def project(value):
    first=linear(value,w1,b1)
    return linear([x/(1+math.exp(-x)) for x in first],w2,b2)
weights={'talker.model.text_embedding.weight':text,'talker.model.codec_embedding.weight':codec,
    'talker.code_predictor.model.codec_embedding.0.weight':predictor,
    'talker.text_projection.linear_fc1.weight':w1,'talker.text_projection.linear_fc1.bias':b1,
    'talker.text_projection.linear_fc2.weight':w2,'talker.text_projection.linear_fc2.bias':b2}
ids=[[1,-1],[2,1],[3,2],[4,3]]
prompt=[a+b for t,c in ids for a,b in zip(project(text[t]),codec[c] if c>=0 else [0]*4)]
decode=[a+b+c for a,b,c in zip(codec[1],predictor[2],project(text[3]))]
with tempfile.TemporaryDirectory(prefix='garnet-tts-projection-') as directory:
    root=Path(directory);header={};payload=bytearray()
    for key,value in weights.items():
        shape=[len(value),len(value[0])] if isinstance(value[0],list) else [len(value)]
        values=flatten(value);start=len(payload);payload.extend(struct.pack('<'+'f'*len(values),*values))
        header[key]={'dtype':'F32','shape':shape,'data_offsets':[start,len(payload)]}
    raw=json.dumps(header,separators=(',',':')).encode();raw+=b' '*((-len(raw))%8)
    (root/'model.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw+payload)
    (root/'config.json').write_text('{}')
    source=Path(__file__).resolve().parent/'models/tts_projection_fixture.py'
    for name,shape,values,expected in [('prompt',[1,4,2],flatten(ids),prompt),('decode',[1,2],[1,2],decode)]:
        model=garnet.load_model(str(source),weights=str(root),runtime_mode='compiled_xmodel',
            entry_function=name,input_shapes=[shape],input_dtypes=['int64'],cache_dir=str(root/name),
            backend='tensorrt',precision='bf16',compile={'builder_workspace_mb':64,'builder_optimization_level':0})
        assert model.runtime_status()['ready'],model.runtime_status()
        try:
            value=garnet.tensor_from_host(values,dtype='int64',shape=shape,device='cuda')
            result=model.forward({'inputs':[value]});assert result['status']=='ok'
            actual=flatten(garnet.tensor_to_cpu(result['output']).tolist());assert len(actual)==len(expected)
            error=max(abs(a-b) for a,b in zip(actual,expected))
            assert error<.035,(name,error,actual,expected)
            print(json.dumps({'operator':name,'max_absolute_error':error,'passed':True}),flush=True)
        finally:assert model.release_runtime()
