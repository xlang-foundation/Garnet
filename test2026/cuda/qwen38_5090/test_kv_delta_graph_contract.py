"""CPU graph-construction contracts, not a TensorRT/CUDA execution test."""
import ast,copy,hashlib,json,types,unittest
from pathlib import Path
from xModel.qwen35.vl_0_8b.kv_delta_layout import build_kv_delta_layout

ROOT=Path(__file__).resolve().parents[3]
MODEL=ROOT/'xModel/qwen35/vl_0_8b'
CONFIG=json.loads((Path(__file__).parent/'evidence/nvfp4-config.json').read_bytes())['text_config']
class Trace:
    def __init__(self):self.rows=[]
    def add(self,kind,name,args,kw=None):
        row={'kind':kind,'name':name,'args':args,'kw':{} if kw is None else kw}
        result=Value(self,len(self.rows));self.rows.append(row);return result
    def input(self,name):return self.add('input',name,[])
    def unary_op(self,name,**kw):return ('unary',name,kw)
    def binary_op(self,name,**kw):return ('binary',name,kw)
    def fusion(self,**kw):return lambda function:function
    def clean(self,value):
        if isinstance(value,Value):return {'node':value.index}
        if isinstance(value,dict):return {k:self.clean(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [self.clean(v) for v in value]
        return value
    def digest(self):return hashlib.sha256(json.dumps(self.clean(self.rows),sort_keys=True,separators=(',',':')).encode()).hexdigest()
class Value:
    def __init__(self,trace,index):self.trace=trace;self.index=index
    def __mul__(self,op):
        kind,name,kw=op
        if kind=='binary':return Bound(self,op)
        return self.trace.add(kind,name,[self],kw)
class Bound:
    def __init__(self,left,op):self.left=left;self.op=op
    def __mul__(self,right):return self.left.trace.add(self.op[0],self.op[1],[self.left,right],self.op[2])
def functions(source,names,namespace):
    nodes=[copy.deepcopy(n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
    for n in nodes:n.decorator_list=[]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual-model-functions','exec'),namespace)
def member(source,cls,name,namespace):
    node=copy.deepcopy(next(n for c in ast.parse(source).body if isinstance(c,ast.ClassDef) and c.name==cls for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-model-method','exec'),namespace);return namespace[name]
def target_trace(enabled=False,strategy=1,program_source=None,chunk_source=None):
    trace=Trace();ns={'T':trace}
    program_source=program_source or (MODEL/'cuda_packed_program.py').read_bytes()
    chunk_source=chunk_source or (MODEL/'cuda_chunk.py').read_bytes()
    functions(program_source,('decoder_state_layout','packed_view','concat','flat_concat'),ns)
    functions(chunk_source,('Qwen35CUDAChunk',),ns)
    canonical,width=ns['decoder_state_layout'](CONFIG,4,4096);compact=build_kv_delta_layout(canonical)
    p=types.SimpleNamespace(config=CONFIG,batch=4,size=4,hidden_width=20480,layout=canonical,flat_concat=True,delta_layout=compact if enabled else None,
        external_compressed_weights=True,workspace=0,retain=True,register_delta_128=False,join_input_projections=False,joined_mlp_projections=True,kv_overlay_strategy=strategy,
        weight_names=[],weight_slots=[],weight_layers=[])
    symbols=[trace.input(n) for n in ('hidden','canonical','active','positions','slots')];weights=[]
    for layer,small in zip(canonical,compact['layers']):
        layer['payload_width']=small.get('payload_width',0);indices=[];prefix='model.language_model.layers.'+str(layer['index'])+'.'
        suffixes=['mlp.gate_up_proj.weight','mlp.down_proj.weight']+(['linear_attn.'+n+'.weight' for n in ('in_proj_qkv','in_proj_z','out_proj')] if layer['kind']=='linear_attention' else ['self_attn.'+n+'.weight' for n in ('q_proj','k_proj','v_proj','o_proj')])
        for suffix in suffixes:
            name=prefix+suffix;indices.append(len(p.weight_names));p.weight_names.append(name)
            slots=[]
            for role in (('packed','scales') if suffix.startswith('mlp.') else ('packed',)):
                slots.append(len(weights));weights.append(trace.input(name+'/'+role))
            p.weight_slots.append(slots)
        p.weight_layers.append(indices)
    forward=member(program_source,'CUDAPackedDecoderProgram','_forward',ns)(p)
    forward(*symbols,*weights)
    return trace,p,ns

class KVDeltaGraphContractTests(unittest.TestCase):
    def test_disabled_graph_matches_retained_construction(self):
        golden=json.loads((Path(__file__).parent/'evidence/kv-delta-disabled-graph-golden.json').read_bytes())
        trace,_,_=target_trace();self.assertEqual(trace.digest(),golden['trace_sha256']);self.assertEqual(len(trace.rows),golden['trace_nodes'])

    def test_compact_graph_binds_canonical_without_full_kv_slices(self):
        for strategy in (0,1):
            trace,p,_=target_trace(True,strategy);full=[r for r in trace.rows if r['name']=='qwen35_full_decoder_chunk']
            self.assertEqual(len(full),16)
            for i,row in enumerate(full):
                self.assertEqual(row['kw']['canonical_kv_overlay'],1);self.assertEqual(row['kw']['kv_overlay_strategy'],strategy)
                expected=p.delta_layout['layers'][i*4+3]
                self.assertEqual(row['kw']['canonical_state_width'],173932544);self.assertEqual(row['kw']['canonical_key_offset'],expected['key_offset']);self.assertEqual(row['kw']['canonical_value_offset'],expected['value_offset'])
            self.assertEqual(sum(r['name']=='qwen35_attention_bind_canonical' for r in trace.rows),16)
            self.assertFalse(any(r['name'] in ('qwen35_attention_bind_key','qwen35_attention_bind_value') for r in trace.rows))
            canonical_slices=[r for r in trace.rows if r['name']=='qwen35_packed_view' and r['args'][0].index==1]
            self.assertEqual(len(canonical_slices),96);self.assertTrue(all(r['kw']['width'] in (40960,786432) for r in canonical_slices))
            result=trace.rows[-1];self.assertEqual(result['name'],'qwen35_concat_parts')
            parts=result['kw']['parts'];self.assertEqual(len(parts),112)
            widths=[trace.rows[v.index]['kw']['shape'][1] for v in parts]
            self.assertEqual(20480+sum(widths),43030528)

    def test_partial_replay_returns_only_linear_state(self):
        for enabled in (False,True):
            trace,p,ns=target_trace(enabled)
            for layer,small in zip(p.layout,build_kv_delta_layout(p.layout)['layers']):
                if layer['payload_width']:layer['payload_offset']=small['payload_offset'] if enabled else 20480+173932544+sum(x['payload_width'] for x in p.layout[:layer['index']])
            p.register_delta_128=False;owner=types.SimpleNamespace(decoder=types.SimpleNamespace(config=CONFIG,batch=4,decode_chunk_size=4),kv_delta=enabled)
            start=len(trace.rows);forward=member((MODEL/'cuda_packed_replay.py').read_bytes(),'CUDAPackedDeltaReplayVerifier','_forward',ns)(owner,p)
            args=[trace.input(n) for n in ('packed','backup','active')+(() if enabled else ('spans',))]
            forward(*args);rows=trace.rows[start:]
            self.assertEqual(sum(r['name']=='qwen35_linear_replay_chunk' for r in rows),48)
            self.assertEqual(sum(r['name']=='qwen35_restore_kv_prefix' for r in rows),0 if enabled else 32)
            result=rows[-1];self.assertEqual(len(result['kw']['parts']),47 if enabled else 79)

if __name__=='__main__':unittest.main()
