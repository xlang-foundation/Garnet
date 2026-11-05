"""Independent NumPy FP32 oracle for the explicit raw-BF16 MTP graph.

Args: runtime, native build, source, NEW output directory; optional raw MTP
checkpoint and full model config. Auxiliary-only, never a generation benchmark.
The fixed draft-operator criterion is abs(error) <= 5e-4 + 5e-4*abs(reference).
Untouched cache spans must match raw FP32 bytes exactly.
"""
import hashlib, json, os
from pathlib import Path
import struct, subprocess, sys
import numpy as np

runtime, build, source, destination = sys.argv[1:5]
out = Path(destination); out.mkdir(exist_ok=False)
rng = np.random.default_rng(380028)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
wide = 'profiles' in sys.argv[5:]
trained = len(sys.argv) > 5 and sys.argv[5] != 'profiles'
if trained:
    checkpoint = Path(sys.argv[5]); config_path = Path(sys.argv[6])
    assert sha(checkpoint) == 'dec6630788d1ba4557230ca02b5df2db239f2b2bbcfff1c6742776e231e7540b'
    assert sha(config_path) == '14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85'
    c = json.loads(config_path.read_text())['text_config']
else:
    c = dict(hidden_size=64, intermediate_size=128, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, rms_norm_eps=1e-6,
        rope_parameters=dict(partial_rotary_factor=1, rope_theta=10000,
                             mrope_section=[6,5,5], mrope_interleaved=True))
    h=c['hidden_size']; d=c['head_dim']; q=c['num_attention_heads']*d; k=c['num_key_value_heads']*d; m=c['intermediate_size']
    shapes={'mtp.fc.weight':[h,2*h]}
    shapes.update({'mtp.'+n+'.weight':[h] for n in ('pre_fc_norm_embedding','pre_fc_norm_hidden','norm')})
    shapes.update({'mtp.layers.0.'+n+'.weight':[h] for n in ('input_layernorm','post_attention_layernorm')})
    shapes.update({'mtp.layers.0.self_attn.'+n+'.weight':[d] for n in ('q_norm','k_norm')})
    shapes.update({'mtp.layers.0.self_attn.'+n+'.weight':s for n,s in dict(q_proj=[2*q,h],k_proj=[k,h],v_proj=[k,h],o_proj=[h,q]).items()})
    shapes.update({'mtp.layers.0.mlp.'+n+'.weight':s for n,s in dict(gate_proj=[m,h],up_proj=[m,h],down_proj=[h,m]).items()})
    header={}; payload=bytearray()
    for name, shape in shapes.items():
        a=rng.normal(0,.04,shape).astype('<f4'); bits=a.view(np.uint32)
        raw=((bits+0x7fff+((bits>>16)&1))>>16).astype('<u2').tobytes()
        header[name]=dict(dtype='BF16',shape=shape,data_offsets=[len(payload),len(payload)+len(raw)])
        payload.extend(raw)
    encoded=json.dumps(header,separators=(',',':')).encode(); encoded+=b' '*(-len(encoded)%8)
    checkpoint=out/'mtp.safetensors';checkpoint.write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)

weights={}
with checkpoint.open('rb') as stream:
    size=struct.unpack('<Q',stream.read(8))[0]; header=json.loads(stream.read(size))
for name, info in header.items():
    if name=='__metadata__':continue
    assert info['dtype']=='BF16'
    a=np.memmap(checkpoint,dtype='<u2',mode='r',offset=8+size+info['data_offsets'][0],shape=tuple(info['shape']))
    weights[name]=(np.asarray(a,dtype=np.uint32)<<16).view(np.float32)

f32=np.float32;h=c['hidden_size'];d=c['head_dim'];heads=c['num_attention_heads'];kv=c['num_key_value_heads'];rope=c['rope_parameters'];rotary=int(d*rope['partial_rotary_factor']);norm_offset=f32(1)
def norm(x,name):
    return (x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True,dtype=np.float32)+f32(c['rms_norm_eps'])))*(weights[name+'.weight']+norm_offset)
def project(x,name):return x@weights[name+'.weight'].T
def rotate(x,name,positions):
    x=norm(x,name); y=x.copy(); half=rotary//2
    freq=np.asarray([float(rope['rope_theta'])**(-2.*i/rotary) for i in range(half)],dtype=np.float32)
    axes=np.asarray([1 if i%3==1 and i<rope['mrope_section'][1]*3 else 2 if i%3==2 and i<rope['mrope_section'][2]*3 else 0 for i in range(half)])
    angles=positions[:,axes]*freq;co=np.cos(angles)[:,None,:];si=np.sin(angles)[:,None,:]
    y[...,:half]=x[...,:half]*co-x[...,half:rotary]*si
    y[...,half:rotary]=x[...,half:rotary]*co+x[...,:half]*si
    return y
def oracle(inputs,b,t,cap,swap=False):
    target,embedding,key,value,positions,slots,active=inputs
    target=np.where(active[...,None]!=0,target,0).reshape(b*t,h)
    embedding=np.where(active[...,None]!=0,embedding,0).reshape(b*t,h)
    parts=[norm(embedding,'mtp.pre_fc_norm_embedding'),norm(target,'mtp.pre_fc_norm_hidden')]
    hidden=project(np.concatenate(parts[::-1] if swap else parts,axis=-1),'mtp.fc')
    n=norm(hidden,'mtp.layers.0.input_layernorm');p='mtp.layers.0.self_attn.'
    qg=project(n,p+'q_proj').reshape(b*t,heads,2*d)
    q=rotate(qg[...,:d],p+'q_norm',positions.reshape(b*t,3)).reshape(b,t,heads,d)
    k=rotate(project(n,p+'k_proj').reshape(b*t,kv,d),p+'k_norm',positions.reshape(b*t,3)).reshape(b,t,kv,d)
    v=project(n,p+'v_proj').reshape(b,t,kv,d);gate=qg[...,d:].reshape(b,t,heads,d)
    key=key.copy();value=value.copy();context=np.zeros((b,t,heads,d),dtype=np.float32)
    for row in range(b):
        for token in range(t):
            if not active[row,token]:continue
            slot=int(slots[row,token]);key[row,:,slot]=k[row,token];value[row,:,slot]=v[row,token]
            for head in range(heads):
                kh=head//(heads//kv);score=key[row,kh,:slot+1]@q[row,token,head]/f32(np.sqrt(d))
                probability=np.exp(score-np.max(score));probability/=np.sum(probability,dtype=np.float32)
                context[row,token,head]=(probability@value[row,kh,:slot+1])/(f32(1)+np.exp(-gate[row,token,head]))
    residual=hidden+project(context.reshape(b*t,heads*d),p+'o_proj')
    n=norm(residual,'mtp.layers.0.post_attention_layernorm');gate=project(n,'mtp.layers.0.mlp.gate_proj');up=project(n,'mtp.layers.0.mlp.up_proj')
    result=norm(residual+project((gate/(f32(1)+np.exp(-gate)))*up,'mtp.layers.0.mlp.down_proj'),'mtp.norm')
    return np.concatenate([result.reshape(b,t*h),key.reshape(b,-1),value.reshape(b,-1)],axis=1).astype('<f4')

cases=[];sensitivity=[]
profiles=[(1,4,64),(1,4,4096)] if trained else [(1,1,16),(1,2,16),(1,3,16),(1,4,16),(2,2,16),(4,1,16),(1,4,4096)]
if wide:
    profiles += [(1,8,64),(1,128,256),(1,128,4096)] if trained else [(1,8,16),(1,32,64),(1,128,256),(2,64,256),(4,32,256),(1,128,4096)]
for b,t,cap in profiles:
    previous=None
    trials=sorted(set([0,1,min(73,t),t-1,t])) if b*t>4 else ([0,1,2,3,4] if trained else list(range(t+1)))
    # Every profile also has two carried calls; the T4 boundary cases reach cap-1.
    for trial in trials+['carry-first','carry-second']:
        carry=trial=='carry-second'
        lengths=[min(2,t) if carry or trial=='carry-first' else (int(trial)+row)%(t+1) for row in range(b)]
        if trial==t:lengths=[t]*b
        starts=[cap-t if cap==4096 else 1]*b
        if carry:starts=[starts[row]+min(2,t) for row in range(b)]
        active=np.asarray([[int(j<lengths[row]) for j in range(t)] for row in range(b)],dtype='<f4')
        slots=np.asarray([[start+j for j in range(t)] for start in starts],dtype='<f4')
        positions=np.stack([slots,slots+f32(.25),slots+f32(.5)],axis=-1)
        target=rng.normal(0,1,(b,t,h)).astype('<f4');embedding=rng.normal(0,.1,(b,t,h)).astype('<f4')
        target[active==0]=np.nan;embedding[active==0]=np.nan
        if carry:
            key=previous[:,t*h:t*h+kv*cap*d].reshape(b,kv,cap,d).copy();value=previous[:,t*h+kv*cap*d:].reshape(b,kv,cap,d).copy()
        else:
            key=rng.normal(0,.015,(b,kv,cap,d)).astype('<f4');value=rng.normal(0,.015,key.shape).astype('<f4')
        inputs=[target,embedding,key,value,positions,slots,active];expected=oracle(inputs,b,t,cap);previous=expected
        if not sensitivity and np.any(active):
            wrong_order=oracle(inputs,b,t,cap,swap=True)
            norm_offset=f32(0);wrong_offset=oracle(inputs,b,t,cap);norm_offset=f32(1)
            for name,wrong in [('swapped_concatenation',wrong_order),('missing_raw_norm_offset',wrong_offset)]:
                ratio=float(np.max(np.abs(wrong[:,:t*h]-expected[:,:t*h])/(f32(5e-4)+f32(5e-4)*np.abs(expected[:,:t*h]))))
                assert ratio>1,(name,'oracle is insensitive',ratio)
                sensitivity.append(dict(mutation=name,worst_element_ratio=ratio))
        label=f'B{b}T{t}C{cap}-{trial}';folder=out/label;folder.mkdir()
        paths=[]
        for i,x in enumerate(inputs):
            path=folder/f'input{i}.f32';x.tofile(path);paths.append(str(path))
        path=folder/'expected.f32';expected.tofile(path)
        cases.append(dict(label=label,batch=b,chunk=t,capacity=cap,inputs=paths,expected=str(path),carry=carry,lengths=lengths,slots=slots.tolist()))
fixture=out/'fixture.json';fixture.write_text(json.dumps(dict(checkpoint=str(checkpoint),config=c,cases=cases,output=str(out/'result.json'),trained=trained,atol=5e-4,rtol=5e-4,sensitivity=sensitivity)))
runner=Path(__file__).with_name('run_metal_mtp.py');argv=[runtime,'-B',str(runner),str(fixture),build,source]
paths=[checkpoint,Path(build)/'libgarnet.dylib',Path(runtime),Path(__file__),runner,Path(source)/'xModel/qwen38/vl_27b/metal_mtp.py',Path(source)/'src/backends/metal/compiled_graph.cpp',Path(source)/'src/backends/metal/compiled_graph.h']
(out/'provenance.json').write_text(json.dumps(dict(sha256={str(p):sha(p) for p in paths},argv=argv,benchmark=False,oracle='Independent NumPy FP32, raw BF16 matrices and +1 norms, embedding-first concatenation'),indent=2)+'\n')
assert os.environ.get('GARNET_METAL_PROFILE_OPS','0')!='1'
print('mtp_oracle_prepared',len(cases),'trained',trained,flush=True)
subprocess.run(argv,check=True)
