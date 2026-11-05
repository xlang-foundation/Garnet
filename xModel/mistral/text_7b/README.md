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

The later B4/context4096/chunk128/global-budget512 admission now completes 22
frozen HTTP requests and 4,844 output tokens with native scheduling, exact frozen
native coding IDs, and no retained native owners. It remains one cohort with no
warmups, uses no prefix cache, and does not establish repeated performance or
the 2x targets. See the research document for all timings, regressions and scope.

`GARNET_METAL_MISTRAL_DECODE_REUSE=ON` is an optional compiled profile for exact
FP32 four-row projection reuse. It requires FP16 affine coefficients, bounded
affine profiles and indexed decode. The serving profile must also declare
`"mistral_decode_reuse": true`. Only measured Mistral Q/O, MLP and head shapes at
four active rows use the new tile; K/V and smaller occupancies keep the control.
It defaults off. The observed short B4 end-to-end rate is 40.45 tok/s, versus
36.46 in the prior native admission and frozen 45.24 /42.19 reference medians;
different system conditions limit this comparison. Long-prompt performance
remains dominated by other work.

With `GARNET_METAL_VERIFY_ATTENTION=ON` and the existing prepared FP32 projection
extensions, engine setup can call `decoder.prepare_verification(tokens=4)`.
This prepares the native target verifier, including its KV backups and accepted
state commit. Dense Mistral does not require the recurrent Qwen extensions.
The current bound is four physical token rows total: B1/T4, B2/T2 or B4/T1.
This API alone does not supply draft tokens or add speculative serving.
`medusa_draft.py` provides the separately qualified raw-BF16-weight/FP32-math
single-path head experiment. It preserves tested greedy outputs but is slower
than ordinary decode, is not a tree implementation, and is not selected by
default. The auxiliary artifact's training-base identity remains unverified.

The optional `GARNET_METAL_CHUNK_KV_SCATTER=ON` profile updates only a validated
contiguous token span in each identical private KV cache view. It requires the
resident/chunk in-place extensions and a cache capacity within the compiled
attention limit. A serving profile declares
`"chunk_kv_scatter": true` and
`"expected_chunk_kv_scatter_dispatches": 64`; the native session exposes
`chunk_kv_scatter_dispatches` for route checks. Gaps, overwrites, unknown controls
and separate output caches use the existing path. The interface to xModel
layers stays the same. It defaults off.

The corresponding single-cohort admission preserves all 22 frozen requests and
4,844 output tokens. Short B4 aggregate throughput is 46.20 tok/s and long B4
is 10.43 tok/s; short/long C1 TTFT is 1.802s /11.620s. Native output/state checks
pass, but all 2x and half-TTFT targets remain unmet. These observations have not
established repeated superiority over the frozen reference engines.

`GARNET_METAL_LAYOUT_VECTOR4=ON` optionally compiles aligned float4 chunk
transfers with bounded 32-bit addressing. It preserves exact FP32 words and
retains scalar layout fallbacks. A serving profile declares
`"chunk_layout_vector4": true` and
`"expected_chunk_layout_vector_dispatches": 256`; the native session exposes
`layout_vector_dispatches`. Compilation occurs before readiness, without new
per-run scratch. It defaults off and preserves the xModel layer API.

Its one-cohort 22-request admission preserves frozen outputs. Short/long C1
TTFT is 1.389s /10.449s; short/long B4 aggregate rates are 43.75 /10.47 tok/s.
Short B4 regresses from the prior 46.20 tok/s observation. Separate runs have
different system conditions and do not establish repeated performance or
isolated causation. All 2x and half-TTFT targets remain unmet; see the research
document for the complete table, numerical gates and limitations.

The native-only `GARNET_METAL_PREPARED_KV_ROW_SPANS` option prepares private cache
row views with shared backing readiness and offset-correct access. It requires
resident/chunk in-place updates and defaults off. This is the private cache-view
foundation; it does not change the xModel layer API. The earlier foundation-only
HTTP regression was slower and remains recorded in the research document.

`GARNET_METAL_COMPACT_OWNER_PREFILL=ON` additionally prepares native B1/B2 graph
families over those same B4-owned caches. Call `prepare_compact_prefill(128)` on
the decoder before any execution or request claim, or explicitly pass
`compact=True` to the batch-study factory. Native leased execution selects a
prepared contiguous one/two-owner span. Gapped, three/four-owner, small FP32,
decode and unleased calls keep the B4 path. Output hidden tensors and predictions
retain physical-row ownership. The extension defaults off and excludes dirty
or deferred reset, MTP and speculative target workspaces in the same session.

The qualified family has 14 shapes (B1/B2, chunks2..128), 21 MPS attention plans,
1,044,162,560 scratch bytes and 126 context buckets. Families, cache views,
physical hidden routing and scratch are prepared before readiness. This does
not yet select B1/T512 or B2/T256 or pack gapped owners. Fused attention requires
the separate experimental extension below.

Actual model checks preserve complete cache/hidden/prediction identity, prefix
restore and running survivors. Two opposite-order same-session late-chunk
studies show 1.18–1.30x body ratios for one owner and 1.11–1.16x for two. These
are repeated identical chunk measurements, excluding embedding, head and HTTP.
The separate single-wave HTTP result remains below the long-context references;
all 2x/half-TTFT targets are unmet. See the research document for the full table.

`GARNET_METAL_CHUNK_FLASH_ATTENTION=ON` experimentally fuses chunk attention
using FP32 SIMD-group matrix tiles and online softmax. It defaults off and
requires the prepared MPS/index32/context-bucket paths for fallbacks. Completed
contiguous active-prefix controls, head dimension128, up to4 owners and bounded
chunks8..512 select the kernel; other inputs retain MPS or scalar execution.
The current compact serving family still stops at chunk128. The backend reports
`chunk_flash_attention_fp32_operands`, and serving profiles must explicitly set
`"chunk_flash_attention": true`. Native counters report actual fused dispatches.

FP32 inputs are retained, but reduction order changes. Full KV/hidden hashes
therefore differ from the original MPS policy. Independent FP64 checks, unchanged
complete output sequences for22 requests and same-policy cache/prefix parity
pass. Operator gains do not translate into a qualified serving win: paired
late-chunk body work improves about1% for one owner and regresses3–9% for four.
The separate HTTP observation is slower in every cell. This implementation is
an experimental comparison path, not an enabled performance improvement. See
the research document for exact scope, evidence and remaining targets.

`GARNET_METAL_WIDE_OWNER_PREFILL=ON` adds optional prepared B1/T512 and B2/T256
families under the same global512-token bound. It requires compact owner
families and defaults off. Prepare with `decoder.prepare_compact_prefill(512)`
before execution or row claims. `prefill_profile_graphs(chunk)` only retrieves
prepared fallback handles; it never builds a padded B4/T256 or B4/T512 GPU graph.
The layer operators and physical KV owners stay unchanged. Native scheduling
selects a prepared width from remaining lengths, contiguous live owners and
the tick budget after decode tokens. Gapped or three/four-owner spans retain
up-to128 fallback profiles. Width is a maximum; short tails use smaller shapes.

The explicit serving profile uses `wide_owner_prefill: true`,
`maximum_owner_prefill_chunk: 512` and17 compact families. Native tick/route
traces are independently audited. All22 complete output ID sequences match
the earlier compact128 collection. Full KV/hidden/prediction comparisons,
dynamic A/E/A-return refill and exact/appended/changed prefix checks pass.
The separate serving observation does not establish a performance improvement;
long C1/C2/C4 aggregate rates are7.44/9.46/10.92 tok/s, below the frozen
reference engines. Flash remains off in this admission. All2x/half-TTFT and
coding-quality targets remain unmet; wider profiles are an experimental option.

`GARNET_METAL_CHUNK_ATTENTION_SCRATCH=ON` optionally includes completely written
attention results in the existing serialized shared scratch pool. It requires
shared scratch, MPS attention, index32 and context buckets, and defaults off.
The backend reports `chunk_attention_full_write_scratch`; serving profiles
declare `chunk_attention_scratch: true`. Existing shape, first-write, protected
owner and completion-lease checks remain in force. Layer APIs and attention
arithmetic are unchanged. KV caches retain their private native ownership.

On the tested M5, preparation saves1,586,528,256 Metal bytes with17 compact
families and915,439,616 bytes in the compact128 serving profile. All nine
old/new full-state cases are bit-exact. Poisoned-output operator checks,
dynamic refill, prefix reuse and all22 serving output sequences pass. Native
tests pass26 with this extension,27 with the optional Flash combination and25
with it disabled. Flash remains off in the serving measurement. The separate
serving observation does not establish a speed improvement; all2x/half-TTFT
and coding-quality targets remain open. See the research document for paired
timing results, exact scope and remaining projection work.
