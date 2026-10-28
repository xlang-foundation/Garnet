"""Separate diagnostic layout; original B1 T32/T128 observer contract is unchanged."""
LAYERS=(0,1,2,3,4,10,11,12)
STEPS=7
COMMON_STEPS=5
HIDDEN=5120
TOKENS=4
ATTENTION_LAYERS=(3,11)
def validate_profile(batch,capacity,tokens):
    if type(batch) is not int or batch not in (1,4) or type(capacity) is not int or capacity!=4096 or type(tokens) is not int or tokens!=TOKENS:
        raise ValueError('Decode boundary study requires explicit B1/B4, capacity4096, T4')
def offsets(primary_width,hidden_width):
    if type(primary_width) is not int or primary_width<=0 or type(hidden_width) is not int or hidden_width!=TOKENS*HIDDEN:raise ValueError('Invalid bounded decoder diagnostic widths')
    return [(layer,primary_width+i*hidden_width,hidden_width) for i,layer in enumerate(LAYERS)]
def value_cache_word_offset(layout,layer,head,slot):
    if len(layout)!=64 or type(layer) is not int or layer not in ATTENTION_LAYERS or type(head) is not int or not 0<=head<4 or type(slot) is not int or not 0<=slot<4096:raise ValueError('Invalid bounded decode V cache address')
    entry=layout[layer];base=sum(2*2097152 if i%4==3 else 40960+786432 for i in range(layer))
    if entry['index']!=layer or entry['kind']!='full_attention' or entry['offset']!=base or entry['state1_width']!=2097152 or entry['state2_width']!=2097152:raise ValueError('Expected complete packed BF16 canonical V cache')
    return base+2097152+(head*4096+slot)*128
