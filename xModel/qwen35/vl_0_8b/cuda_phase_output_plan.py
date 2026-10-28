# SPDX-License-Identifier: Apache-2.0
"""Unwired, pure CPU contract for partitioned prefill and replay-only decode.

Prefill parts expose only their own layer outputs, then commit those outputs
before the shared storage is reused. Decode retains authoritative replay
payload and compact K/V deltas, never a second full recurrent-state image.
No tensor allocation, compiler call, alias permission or serving default here.
"""

PROTOCOL='qwen35-phase-outputs-v1'
LIMIT=(1<<31)-1


def integer(value,low,high,name):
    if type(value) is not int or not low<=value<=high:raise ValueError('Invalid '+name)
    return value


def canonical_layout(config,batch,capacity):
    integer(batch,1,8,'batch')
    if batch not in (1,2,4,8):raise ValueError('Unsupported batch')
    integer(capacity,32,8192,'capacity')
    h=integer(config['hidden_size'],1,16384,'hidden size')
    kh=integer(config['linear_num_key_heads'],1,64,'linear key heads')
    vh=integer(config['linear_num_value_heads'],1,64,'linear value heads')
    if vh%kh:raise ValueError('Value heads must group key heads')
    if type(config['linear_key_head_dim']) is not int or config['linear_key_head_dim']!=128 or type(config['linear_value_head_dim']) is not int or config['linear_value_head_dim']!=128:
        raise ValueError('Replay requires128-wide linear K/V state')
    kernel=integer(config['linear_conv_kernel_dim'],1,4,'convolution kernel')
    kv=integer(config['num_key_value_heads'],1,32,'KV heads')
    dim=integer(config['head_dim'],1,256,'head dimension')
    kinds=config['layer_types']
    if not isinstance(kinds,list) or not 1<=len(kinds)<=128 or any(x not in ('linear_attention','full_attention') for x in kinds):
        raise ValueError('Invalid layer types')
    if integer(config['num_hidden_layers'],1,128,'layer count')!=len(kinds):raise ValueError('Layer count mismatch')
    channels=(2*kh+vh)*128;result=[];cursor=0
    for index,kind in enumerate(kinds):
        if kind=='linear_attention':
            first=[batch,channels,kernel];second=[batch,vh,128,128]
            w1=channels*kernel;w2=vh*128*128
        else:
            first=second=[batch,kv,capacity,dim];w1=w2=kv*capacity*dim
        result.append({'index':index,'kind':kind,'offset':cursor,'state1_shape':first,'state2_shape':second,'state1_width':w1,'state2_width':w2})
        cursor+=w1+w2
    integer(cursor,1,LIMIT,'canonical width')
    return result,cursor


def commit_spec(config,batch,capacity,tokens,start,count,replay_only):
    """Return a fully checked source/destination layout; no device side effects."""
    if type(replay_only) is not bool:raise ValueError('Replay mode must be boolean')
    layers,state_width=canonical_layout(config,batch,capacity)
    integer(tokens,1,16 if replay_only else 128,'tokens')
    if tokens>capacity:raise ValueError('Phase token count exceeds cache capacity')
    integer(start,0,len(layers)-1,'first layer');integer(count,1,len(layers)-start,'partition length')
    if replay_only and (start!=0 or count!=len(layers)):raise ValueError('Decode replay must cover every layer')
    selected=[dict(x) for x in layers[start:start+count]]
    if not any(x['kind']=='linear_attention' for x in selected):raise ValueError('Phase requires a linear layer')
    if replay_only and not any(x['kind']=='full_attention' for x in selected):raise ValueError('Replay requires compact KV output')
    cursor=0;full=0
    for layer in selected:
        layer['output_offset']=cursor
        if layer['kind']=='full_attention':
            w1=w2=config['num_key_value_heads']*tokens*config['head_dim']
            layer['control_index']=full;full+=1
        elif replay_only:w1=w2=0
        else:w1,w2=layer['state1_width'],layer['state2_width']
        layer['output_state1_width']=w1;layer['output_state2_width']=w2
        cursor+=w1+w2
    hidden=tokens*config['hidden_size'];payload=hidden+cursor
    if not 0<cursor<=state_width:raise ValueError('Phase output-state width exceeds canonical contract')
    channels=(2*config['linear_num_key_heads']+config['linear_num_value_heads'])*128
    for layer in selected:
        width=tokens*(channels+config['linear_num_value_heads']*128+2*config['linear_num_value_heads']) if replay_only and layer['kind']=='linear_attention' else 0
        layer['payload_width']=width
        if width:layer['payload_offset']=payload;payload+=width
    spec={'phase_output_layout':PROTOCOL,'config':config,'layout':selected,'batch':batch,'tokens':tokens,'capacity':capacity,
        'state_width':state_width,'hidden_width':hidden,'output_state_width':cursor,'status_offset':payload,
        'packed_width':payload+full,'compact_kv_outputs':True,'copy_only':not replay_only,
        'replay_only_linear_state':replay_only}
    if not replay_only:spec['copy_layer_range']=[start,count]
    integer(spec['packed_width'],1,LIMIT,'packed width')
    return spec


def phase_output_plan(config,batch=1,capacity=512,prefill_tokens=64,decode_tokens=4,partition_layers=4):
    """Prospective two-phase layout; backend/kernel/model validation still required."""
    layers,state_width=canonical_layout(config,batch,capacity)
    if partition_layers not in (4,8):raise ValueError('Only four/eight-layer partitions are admitted')
    integer(partition_layers,4,8,'partition layers')
    if len(layers)%partition_layers:raise ValueError('Partitions must cover every layer exactly')
    if prefill_tokens not in (32,64,128) or decode_tokens!=4:raise ValueError('Phase prototype requires P32/64/128 and D4')
    parts=[commit_spec(config,batch,capacity,prefill_tokens,start,partition_layers,False) for start in range(0,len(layers),partition_layers)]
    decode=commit_spec(config,batch,capacity,decode_tokens,0,len(layers),True)
    width=max([decode['packed_width']]+[x['packed_width'] for x in parts])
    return {'protocol':PROTOCOL,'batch':batch,'capacity':capacity,'state_width':state_width,
        'partition_layers':partition_layers,'prefill_parts':parts,'decode_replay':decode,
        'shared_output_pool_bytes':batch*width*4,'implemented_gpu_execution':False,
        'scope':'Layout contract only. Canonical FP32 state, output/input disjointness, deferred decode ownership and rollback remain required; no measured saving.'}
