"""Weightless numerical qualification of the native speech sampling operator."""
import garnet
T=garnet.tensor()
GARNET_MODEL_SPEC={'arguments':[{'name':'logits','kind':'tensor'},
                              {'name':'gumbels','kind':'tensor'}]}
TOP_K=50
@T.fusion(name='tts_sampling_numeric',boundary='required')
def forward(logits,gumbels):
    return logits*T.binary_op('qwen3_tts_sample_top_k',group=1,
                              top_k=TOP_K,temperature=0.9)*gumbels
