"""Small weights exercise the real TTS projection lowerings, without a model."""
import garnet
T=garnet.tensor()
GARNET_MODEL_SPEC={'arguments':[{'name':'ids','kind':'tensor'},
    {'name':'weights','kind':'weights'},{'name':'config','kind':'config'}]}

@T.fusion(name='tts_projection_prompt_check',boundary='required')
def prompt(ids,weights,config):
    return ids*T.unary_op('qwen3_tts_aligned_prompt_embedding',
        text_embedding_name='talker.model.text_embedding.weight',
        text_fc1_weight_name='talker.text_projection.linear_fc1.weight',
        text_fc1_bias_name='talker.text_projection.linear_fc1.bias',
        text_fc2_weight_name='talker.text_projection.linear_fc2.weight',
        text_fc2_bias_name='talker.text_projection.linear_fc2.bias',
        codec_embedding_name='talker.model.codec_embedding.weight')

@T.fusion(name='tts_projection_decode_check',boundary='required')
def decode(ids,weights,config):
    return ids*T.unary_op('qwen3_tts_decode_embedding',
        text_embedding_name='talker.model.text_embedding.weight',
        text_fc1_weight_name='talker.text_projection.linear_fc1.weight',
        text_fc1_bias_name='talker.text_projection.linear_fc1.bias',
        text_fc2_weight_name='talker.text_projection.linear_fc2.weight',
        text_fc2_bias_name='talker.text_projection.linear_fc2.bias',
        codec_embedding_name='talker.model.codec_embedding.weight',
        predictor_embedding_prefix='talker.code_predictor.model.codec_embedding.',
        tts_pad_token_id=3,num_code_groups=2)
