"""Replay preserved synthetic fixtures on CPU and retain projection boundaries."""
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

root = Path(sys.argv[1]).resolve()
stem = root/'qwen35-mtp-stage-v1-reference'
assert not Path(str(stem)+'.json').exists()
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
cases = json.loads((root/'mtp-component-fixtures.json').read_text())
records = []
sources = {}
def data(spec):
    path = Path(spec['path']);raw = path.read_bytes()
    sources[str(path)] = hashlib.sha256(raw).hexdigest()
    return torch.from_numpy(np.frombuffer(raw,dtype='<f4').copy().reshape(spec['shape']))
def raw(t):
    return t.contiguous().view(torch.uint8).numpy().tobytes()
for case_index,c in enumerate(cases):
    checkpoint = Path(c['checkpoint'])
    sources[str(checkpoint)] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    w = {k.removeprefix('mtp.'):v for k,v in load_file(str(checkpoint),device='cpu').items() if k.startswith('mtp.')}
    b,t = c['batch'],c['chunk'];h = c['config']['hidden_size']
    heads,kv,dim = [c['config'][k] for k in ('num_attention_heads','num_key_value_heads','head_dim')]
    eps = c['config']['rms_norm_eps'];capacity=c['capacity']
    assert h==8 and dim==256
    keys = torch.zeros(b,kv,capacity,dim);values = keys.clone()
    path = Path(str(stem)+f'-c{case_index}.bin')
    record = {'case':case_index,'file':str(path),'steps':[]}
    with path.open('xb') as stream:
        def save_tensor(value):
            payload=raw(value);offset=stream.tell();stream.write(payload)
            return {'offset':offset,'bytes':len(payload),'shape':list(value.shape),
                    'bf16':value.dtype==torch.bfloat16,'sha256':hashlib.sha256(payload).hexdigest()}
        for step_index,step in enumerate(c['steps']):
            target,embedding,positions,slots,active = [data(s) for s in step['inputs']]
            stages = []
            def norm(x,k):
                return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*(1+w[k].float())
            def project(x,k):
                bf = x.to(torch.bfloat16)
                y = F.linear(bf,w[k])
                stages.append({'weight':'mtp.'+k,'parts':[save_tensor(x),save_tensor(bf),save_tensor(y)]})
                return y.float()
            x=project(torch.cat([norm(embedding,'pre_fc_norm_embedding.weight'),norm(target,'pre_fc_norm_hidden.weight')],-1),'fc.weight')
            x=x*active[...,None];n=norm(x,'layers.0.input_layernorm.weight')
            qg=project(n,'layers.0.self_attn.q_proj.weight').reshape(b,t,heads,2*dim)
            q,gate=qg.chunk(2,-1);q=norm(q,'layers.0.self_attn.q_norm.weight')
            k=norm(project(n,'layers.0.self_attn.k_proj.weight').reshape(b,t,kv,dim),'layers.0.self_attn.k_norm.weight')
            v=project(n,'layers.0.self_attn.v_proj.weight').reshape(b,t,kv,dim)
            axes=[1 if i%3==1 and i<33 else 2 if i%3==2 and i<30 else 0 for i in range(32)]
            freq=torch.tensor([10000000.**(-2*i/64) for i in range(32)])
            angle=positions[...,axes]*freq;angle=torch.cat([angle,angle],-1).unsqueeze(2)
            def rotate(a):
                part=a[...,:64];left,right=part.chunk(2,-1)
                return torch.cat([part*angle.cos()+torch.cat([-right,left],-1)*angle.sin(),a[...,64:]],-1)
            q,k=rotate(q),rotate(k);cores=[]
            for token in range(t):
                for row in range(b):
                    if active[row,token]:
                        keys[row,:,int(slots[row,token])]=k[row,token]
                        values[row,:,int(slots[row,token])]=v[row,token]
                rk=keys.repeat_interleave(heads//kv,1);rv=values.repeat_interleave(heads//kv,1)
                score=torch.einsum('bhd,bhsd->bhs',q[:,token],rk)/math.sqrt(dim)
                score=score.masked_fill(torch.arange(capacity)[None,None,:]>slots[:,token,None,None],-1e30)
                core=torch.einsum('bhs,bhsd->bhd',score.softmax(-1),rv)*gate[:,token].sigmoid()
                cores.append(core.reshape(b,1,heads*dim)*active[:,token,None,None])
            x=x+project(torch.cat(cores,1),'layers.0.self_attn.o_proj.weight')*active[...,None]
            n=norm(x,'layers.0.post_attention_layernorm.weight')
            x=x+project(F.silu(project(n,'layers.0.mlp.gate_proj.weight'))*project(n,'layers.0.mlp.up_proj.weight'),'layers.0.mlp.down_proj.weight')
            hidden=norm(x,'norm.weight')
            for name,actual,spec in zip(('hidden','key','value'),(hidden,keys,values),step['expected']):
                assert raw(actual)==raw(data(spec)),('CPU regeneration differs from original complete fixture',case_index,step_index,name)
            assert len(stages)==8
            record['steps'].append({'step':step_index,'stages':stages,'original_complete_outputs_exact':True})
    record['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    records.append(record)
assert not torch.cuda.is_initialized()
result={'cases':records,'source_sha256':sources,'torch':torch.__version__,
        'cuda_initialized':False,'all_original_complete_outputs_bit_exact':True,
        'scope':'CPU projection-boundary reference reproduces all14 original synthetic fixture steps bit-exact; not native parity or quality acceptance.'}
Path(str(stem)+'.json').write_text(json.dumps(result,indent=2))
print('MTP_STAGE_CPU_REFERENCE_COMPLETE',len(records),'ORIGINAL_FULL_OUTPUTS_BIT_EXACT',flush=True)
