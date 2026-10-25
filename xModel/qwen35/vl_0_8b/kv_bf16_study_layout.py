"""Unwired Qwen3.8 study layout: FP32 GDN state and packed BF16 full-attention KV.

This metadata does not enable a runtime profile. It deliberately has a new
contract tag; the existing B4/FP32 native commit helper must reject it. Logical
tensor sizes are not allocator usage or proof that B8 fits the RTX 5090.
"""

CONTRACT='garnet_qwen38_bf16_kv_delta_study_v1'
CANONICAL_WORDS=106823680
LINEAR_WORDS=39714816

def build_bf16_kv_study_layout(batch=8,capacity=4096,chunk=4,*,retain_linear_inputs=True):
    if any(type(v) is not int for v in (batch,capacity,chunk)):
        raise ValueError('Study dimensions require integers')
    if batch not in (4,8) or capacity!=4096 or chunk not in (4,32):
        raise ValueError('BF16 KV study requires B4/B8, cap4096, T4/T32')
    if type(retain_linear_inputs) is not bool or retain_linear_inputs!=(chunk==4):
        raise ValueError('T4 retains replay inputs; T32 is prefill-only without replay payload')
    layers=[];canonical=0;target=chunk*5120;replay=0
    for index in range(64):
        full=index%4==3
        if full:
            # Each physical FP32 word carries two BF16 cache entries. Shape
            # below describes storage; logical K/V each have head_dim256.
            first=second=[batch,4,4096,128]
            w1=w2=4*4096*128
            out1=out2=chunk*4*256
            precision='packed_bf16';logical=[batch,4,4096,256]
        else:
            first=[batch,10240,4];second=[batch,48,128,128]
            w1=40960;w2=786432;out1=w1;out2=w2;precision='fp32';logical=None
        entry={'index':index,'kind':'full_attention' if full else 'linear_attention',
            'offset':canonical,'state1_shape':first,'state2_shape':second,
            'state1_width':w1,'state2_width':w2,'state_precision':precision,
            'target_offset':target,'output_state1_width':out1,'output_state2_width':out2,
            'replay_offset':None if full else replay,'payload_width':0,'payload_offset':None}
        if full:entry['logical_kv_shape']=logical
        else:replay+=w1+w2
        canonical+=w1+w2;target+=out1+out2;layers.append(entry)
    assert canonical==CANONICAL_WORDS and replay==LINEAR_WORDS
    payload=0
    if retain_linear_inputs:
        for layer in layers:
            if layer['kind']=='linear_attention':
                layer['payload_width']=chunk*(10240+6144+96)
                layer['payload_offset']=target+payload;payload+=layer['payload_width']
    return {'contract':CONTRACT,'batch':batch,'capacity':capacity,'chunk':chunk,
        'hidden':5120,'canonical_dtype':'fp32_carrier_with_bf16_kv','kv_dtype':'bf16',
        'linear_dtype':'fp32','canonical_state_width':canonical,'hidden_width':chunk*5120,
        'target_state_end':target,'payload_width':payload,'output_width':target+payload,
        'linear_replay_width':replay,'retain_linear_inputs':retain_linear_inputs,
        'layers':layers,'runtime_qualified':False}

def bf16_kv_byte_offset(layout,layer_index,which,row,head,slot,dimension):
    """Reference offset for future native operator tests; never executes CUDA."""
    if layout.get('contract')!=CONTRACT or layout.get('canonical_state_width')!=CANONICAL_WORDS:
        raise ValueError('Wrong BF16 KV study contract')
    values=(layer_index,row,head,slot,dimension)
    if any(type(v) is not int for v in values):raise ValueError('KV indices require integers')
    if not(0<=layer_index<64 and layer_index%4==3 and 0<=row<layout['batch'] and
        0<=head<4 and 0<=slot<4096 and 0<=dimension<256) or which not in ('key','value'):
        raise ValueError('KV index outside explicit study profile')
    layer=layout['layers'][layer_index]
    if layer['index']!=layer_index or layer['state_precision']!='packed_bf16':raise ValueError('Wrong KV layer')
    offset=layer['offset']+(layer['state1_width'] if which=='value' else 0)
    return 4*(row*CANONICAL_WORDS+offset)+2*((head*4096+slot)*256+dimension)
