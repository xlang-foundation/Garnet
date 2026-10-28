"""Bounded complete-state byte witnesses; no shared native compact offset code."""
import hashlib
from test2026.cuda.qwen38_5090.bf16_float_reference_state import expand_bf16_carrier,require_bf16_roundtrip


def logical_state_witness(io,state,layout,batch,width,slots,*,packed_bf16,block_words=1048576):
    if type(packed_bf16) is not bool or any(type(v) is not int or v<1 for v in (batch,width,block_words)) or block_words>1048576:
        raise ValueError('Invalid bounded state witness profile')
    if len(slots)!=batch or any(type(v) is not int or v<0 for v in slots):raise ValueError('Invalid state slots')
    cursor=0
    for i,layer in enumerate(layout):
        if layer.get('index')!=i or layer.get('offset')!=cursor or layer.get('kind') not in ('linear_attention','full_attention'):
            raise ValueError('State witness requires contiguous complete layer metadata')
        for role in (1,2):
            words=layer.get('state'+str(role)+'_width')
            if type(words) is not int or words<1:raise ValueError('Invalid state component width')
            cursor+=words
    if cursor!=width:raise ValueError('State witness metadata does not cover canonical row')
    physical=[hashlib.sha256() for _ in range(batch)];logical=[hashlib.sha256() for _ in range(batch)];buffers={};components=[];logical_words=0
    for layer in layout:
        cache=layer['kind']=='full_attention'
        for role in (1,2):
            words=layer['state'+str(role)+'_width'];base=layer['offset']+(layer['state1_width'] if role==2 else 0)
            component=[hashlib.sha256() for _ in range(batch)]
            for offset in range(0,words,block_words):
                n=min(block_words,words-offset)
                if n not in buffers:buffers[n]=io.zeros([batch,n])
                part=buffers[n];io.slice_into(state,base+offset,part)
                for row in range(batch):
                    raw=bytes(io.finite_f32_bytes(io.row(part,row)))
                    if len(raw)!=n*4:raise ValueError('State witness reader returned truncated component')
                    physical[row].update(raw)
                    expanded=(expand_bf16_carrier(raw) if packed_bf16 else require_bf16_roundtrip(raw)) if cache else raw
                    logical[row].update(expanded);component[row].update(expanded)
            logical_words+=words*(2 if cache and packed_bf16 else 1)
            components.append({'layer':layer['index'],'kind':layer['kind'],'role':role,'logical_words':words*(2 if cache and packed_bf16 else 1),'row_sha256':[h.hexdigest() for h in component]})
    return {'physical_row_sha256':[h.hexdigest() for h in physical],'logical_row_sha256':[h.hexdigest() for h in logical],
        'physical_words_per_row':width,'logical_words_per_row':logical_words,'slots':list(slots),'components':components,
        'reference_bf16_roundtrip_checked':not packed_bf16,'scope':'Complete bounded host byte/hash observation. Physical words include all canonical slots; logical hashes expand BF16 carriers. Raw caches are not archived by this function. Not a timing or quality measurement.'}
