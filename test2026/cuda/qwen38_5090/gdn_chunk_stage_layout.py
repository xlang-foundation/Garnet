"""Pure metadata, shared by embedded collector and independent CPU replay."""
READER_WORDS=1048576
STAGES=(('masked_input',5120),('input_norm',5120),('qkv',10240),('z',6144),('beta',48),('a',48),('gdn_gated_output',6144),('attention_output',5120),('residual',5120),('mlp_input_norm',5120),('output_hidden',5120))

def layout(tokens,diagnostic):
    assert type(tokens) is int and tokens in (32,128) and type(diagnostic) is bool
    primary=tokens*5120+40960+786432;offset=primary;rows=[{'name':'primary','offset':0,'words':primary}]
    if diagnostic:
        for name,width in STAGES:
            words=tokens*width;rows.append({'name':name,'offset':offset,'words':words});offset+=words
    return {'tokens':tokens,'primary_words':primary,'total_words':offset,'rows':rows}

def pieces(words):
    assert type(words) is int and words>0
    return [(offset,min(READER_WORDS,words-offset)) for offset in range(0,words,READER_WORDS)]

def check_graph(stats,tokens,diagnostic):
    assert stats['checkpoint_weight_precision']=='modelopt_mixed'
    for key,value in (('native_nvfp4_projection_weights',2),('native_fp8_projection_weights',3),('external_compressed_weight_inputs',5),('native_int4_g128_projection_weights',0),('native_fp8_m128_canonical_projections',1 if tokens==128 else 0)):
        assert type(stats[key]) is int and stats[key]==value,(key,stats[key])
    for key in ('device_output_bytes','graph_capture_failures','graph_captures','graph_replays'):assert type(stats[key]) is int
    assert stats['device_output_bytes']==layout(tokens,diagnostic)['total_words']*4 and stats['graph_capture_failures']==0
    assert stats['graph_captures']==1 and stats['graph_replays']>=1
    return True
