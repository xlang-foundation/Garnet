# Mistral 7B native Metal text adapter

Research adapter for the pinned MLX affine INT4/group64
Mistral-7B-Instruct-v0.3 checkpoint documented in
`docs/research/mistral7b-metal-onboarding.md`.

The tested Apple M5 / 24 GiB profile uses native FP16 scale/bias decoding,
FP16 tensor operands with FP32 accumulation, FP32 KV state, and C++
tokenization. It accepts text embeddings directly and has no dummy-image input.
The coefficient type is fixed by the compiled engine; BF16 weights require
the BF16 engine profile. Payloads with the wrong coefficient type are rejected.

```python
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder

decoder = MistralMetalDecoder(checkpoint, config, batch=1, capacity=256)
decoder.prepare_prefill(16)  # Compile this graph family during engine setup.
```

Context capacity is an upper bound, not a required prompt length. The current
functional gate uses a 25-token prompt, a capacity of 256 and chunk limit 16.
Only the final chunk's unused token positions are masked. Native affine
pipelines separately specialize dimensions/strides and allow actual row counts
up to their prepared maximum; execution does not compile new Metal pipelines.

Small functional checks pass at batch 1 and batch 2, including one row stopping
after 8 tokens while the other reaches 32. Both ordinary and chunked prefill
match the frozen MLX 32-token reference. This is one reference case, not general
coding-quality, long-context, continuous-admission, prefix-reuse, multimodal,
ANE, xMind production integration or comparative speed qualification.

An expanded batch 1 check enables `GARNET_METAL_COMPACT_ATTENTION_INDEX` and
prepares capacity 4096/chunk 16. Actual 256- and 2056-token context probes each
match 64 MLX output IDs; a complete 4096-token request is not tested. The eight
unchanged coding tasks match five complete MLX sequences, but both engines
score 0/8 under the strict code-only contract. The earliest divergence occurs
at an exact MLX logit tie; ordinary and chunked native paths choose the same
alternative. These results do not establish coding-quality or speed admission.

The initial engine enables `GARNET_METAL_AFFINE_FP16_COEFFICIENTS`,
`GARNET_METAL_MPP_AFFINE_FP16`, strict register MPP, indexed projections/decode,
bounded affine profiles and 32-bit proven profile indexes. The underlying
Metal4 tensor path requires the actual compatible Apple GPU. Only the current
M5 profile has this functional evidence; other chips and feature combinations
remain unqualified. Larger prefill chunks, batch 4 and fuller context/batching
coverage require their own numerical, memory and workload admissions.
