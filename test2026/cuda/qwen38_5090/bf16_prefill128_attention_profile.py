"""Separate unwired B1/T128 attention geometry; old T4/T32 profiles stay closed."""
from .bf16_prefill128_layout import build_bf16_prefill128_layout

def profile(batch=1,tokens=128,heads=24,kv=4,capacity=4096,dim=256,strategy=1):
    values=(batch,tokens,heads,kv,capacity,dim,strategy)
    if any(type(v) is not int for v in values):raise ValueError('Explicit integer attention study dimensions required')
    if values[:6]!=(1,128,24,4,4096,256) or strategy not in (0,1):
        raise ValueError('Only separate B1/T128 baseline0 or tiled16 strategy1 is admitted for component study')
    layout=build_bf16_prefill128_layout()
    return {'contract':'garnet_qwen38_bf16_prefill128_attention_component_v1','batch':batch,'tokens':tokens,'heads':heads,'kv':kv,'capacity':capacity,'dim':dim,
            'strategy':strategy,'workspace_bytes':batch*tokens*heads*capacity*4,'query_bytes':batch*tokens*heads*dim*4,
            'chunk_kv_bytes':batch*tokens*kv*dim*4,'control_bytes':batch*tokens*4,'canonical_bytes':layout['canonical_state_width']*4,
            'full_attention_offsets':[(r['offset'],r['offset']+r['state1_width']) for r in layout['layers'] if r['kind']=='full_attention'],
            'runtime_qualified':False,'performance_admitted':False,
            'scope':'Standalone forward attention candidate only. Caller validates active-prefix, contiguous integer slots and bounds before enqueue. No runtime/parser/default/profile widening or model qualification.'}
