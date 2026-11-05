# Mistral 7B native onboarding

The user moved Mistral 7B onboarding, native tokenization and a Core ML/ANE
plugin ahead of further Qwen27B performance work. The Mistral target is at least
2x single-request decode speed and 2x aggregate throughput against both
vLLM-Metal and SGLang in matched benchmarks, while retaining correctness and
coding-quality gates. The existing Qwen27B targets
and comparator evidence remain incomplete and preserved. This document records
onboarding and a small functional gate, not production or performance qualification.

## Follow-on model scope

The user added both **Gemma 4 E4B and Gemma 4 E2B** after Mistral, in that order,
before returning to Qwen27B, then Qwen9B and Qwen4B. Start Gemma with text coding;
multimodal model membership does not require dummy image inputs for text requests.
Each Gemma size needs its own pinned checkpoint, quantization, native C++
tokenizer parity, architecture implementation, KV ownership, batch/continuous
scheduling checks and matched serving baselines. Earlier Gemma tokenizer checks
do not establish Gemma 4 tokenizer or inference support.

Gemma profiles remain upper bounds chosen for the machine's memory budget;
initial planning uses context up to 4096 and batch up to four. Those are intended
profiles, not qualified configurations. Compare each size against the same size
and weight representation in supported vLLM-Metal and SGLang backends, one engine
at a time. Preserve valid comparator results across Garnet iterations. Unsupported
backend/model combinations must be reported rather than substituted silently.

Gemma 4 E4B has 4.5B effective parameters and 8B including embeddings; E2B has
2.3B effective and 5.1B including embeddings. Effective compute size does not
determine weight residency or the KV-cache budget. See the
[official model architecture](https://ai.google.dev/gemma/docs/core/model_card_4)
and [memory guidance](https://ai.google.dev/gemma/docs/core).

## Candidate checkpoint

`mlx-community/Mistral-7B-Instruct-v0.3-4bit`, revision
`a4b8f870474b0eb527f466a03fbc187830d271f5`, is downloaded under
`/Users/shawn/CantorAI/out/models/Mistral-7B-Instruct-v0.3-MLX-4bit`.
The model file has 4,077,480,215 bytes and SHA256
`d7182d886f93a9efa73d6d556ee33ffe839a4e09e15965eeef46f295ab5b503d`.
The pinned repository API supplies its LFS hash; the complete download was
verified against that size and hash before replacing the final filename.
Configuration/tokenizer files have separate recorded hashes. This is a public
community quantization of Mistral-7B-Instruct-v0.3, not a newly converted model.

| Property | Checkpoint metadata |
|---|---|
| Layers / hidden / MLP | 32 / 4096 / 14336 |
| Attention | 32 query heads, 8 KV heads, dimension 128 |
| Positional encoding | RoPE theta 1,000,000; no scaling or sliding window |
| Normalization / MLP | RMSNorm epsilon 1e-5; SwiGLU; no Q/K norm |
| Vocabulary | 32,768; untied embedding/head |
| Quantization | MLX affine INT4, group size 64 |
| Stored tensor types | 226 U32 matrices; 517 F16 coefficient/norm tensors |

The checkpoint does **not** fit the existing BF16 coefficient contract merely
because it uses INT4/group64. Its scales, additive biases and norms are FP16.
Rounding those values to BF16 would change the checkpoint and is not admitted
as an optimization. Native support must interpret them exactly.

## Completed prerequisites

The native C++ tokenizer implementation and API were committed in `d679d0e16`.
Mistral's audited pipeline uses metaspace, byte fallback and configured BOS;
tokenization requires no image input. See `native-tokenizers.md` for the separate
five-family tokenizer checks and limits.

An IEEE binary16 codec now supports exact FP16-to-FP32 expansion and FP32-to-FP16
round-to-nearest-even, including subnormals and signed zero. An independent
NumPy 2.5.4 comparison passed all 65,536 binary16 decode patterns, 63,488 finite
round trips and 1,253,949 FP32 encoding cases, including halfway neighbors.
NaNs are compared by classification rather than unspecified payload identity.

The native Metal raw-weight loader now accepts F16 weights and expands each
value exactly to FP32. Twelve graph checks, spanning batches 1/2/4 and widths
3/17/128/4096, compare F16-loaded norm weights to their exact FP32 expansion:
all 29,708 output words match. All four instrumented native owner counts are
zero after close. The residual Metal device allocation snapshot includes other
device/driver resources and is not claimed to be zero or a peak measurement.

The checkpoint adapter now recognizes canonical Mistral packed roots and
validates typed coefficient layouts. Selected-row CPU decoding supports FP16
coefficients and rounds embedding values to FP16 before exposing host FP32.
Four actual checkpoint matrices (embedding, head, layer0 Q projection and MLP
down projection), with eight selected rows each, match MLX 0.32.4 in two separate
checks: 212,992 FP32-arithmetic words and 212,992 FP16-rounded words. Native
runtime embedding calls at batch sizes 1/2/4/8 match 61,440 reference words;
invalid token IDs and attempts to use FP16 coefficients in the BF16 Metal
engine are rejected. Twelve metadata fixtures cover valid FP16/BF16 payloads,
mixed/unsupported coefficient types, layout errors and architecture guards.
The existing BF16 affine numerical/tail/ownership/rejection suite also passes.

An important precision distinction is retained in the failed first MLX row
comparison: with FP16 scale/bias inputs, asking `dequantize(dtype=float32)`
still preserved FP16 intermediate rounding in this probe. Explicitly casting
the original coefficient values to FP32 before dequantization verifies the
FP32-arithmetic path. The default FP16 dequantizer separately verifies actual
embedding rounding. Neither comparison alone admits quantized matmul or full
model precision. Earlier mismatch records remain preserved.

Native Metal FP16 affine profiles and the text adapter now pass the small
functional gate below. General coding quality and comparative speed remain
unqualified.

## FP16 native profile and small functional gate

The explicit `GARNET_METAL_AFFINE_FP16_COEFFICIENTS` engine profile decodes
original FP16 coefficient bits exactly. `GARNET_METAL_MPP_AFFINE_FP16` selects
FP16 tensor operands and FP32 accumulation. Register, indexed and ordinary
shader sources share typed coefficient decoding. The default BF16 profile is
preserved; each compiled engine rejects incompatible coefficient payloads.
Configuration rejects silently combining FP16 coefficients with BF16 tensor
operands. These are engine representation/precision choices, not model names.

An isolated source snapshot, excluding the pending Qwen blocked-projection
experiment, builds successfully and passes 13/13 native tests. Its FP16 affine
gate completes 749,002 checks over 478,835 FP32 output words. This includes every
finite binary16 coefficient value (scale and bias), subnormals, FP16 tensor
arithmetic, decode/prefill tails, sparse row masks, exact dense/indexed decode
agreement, wrong-format/nonfinite rejection and zero native owners at close.
Preparing an up-to-128-row profile from one actual row admits 1/2/4/5/63/127/128,
uses noncompiling fallback at 129 and permits explicit widening to 512. Runtime
encoding does not add pipelines. The existing BF16 affine regression also
passes after the shader changes. Only tested feature combinations are admitted;
the optional cooperative64 FP16 variant is not part of this gate.

`xModel/mistral/text_7b/metal_decoder.py` validates the current Mistral7B shape
and checkpoint representation, then reuses shared dense native GQA/RoPE/SwiGLU
operators. `prepare_prefill(16)` builds that graph family before requests. The
functional driver asserts no new prefill graph family or compile time appears
during request execution. Native tokenization exactly matches the reference's
rendered prompt IDs; no image input is used.

An independently generated MLX 0.32.4 / mlx-lm 0.32.2 greedy reference supplies
one 25-token coding prompt and 32 output tokens. The isolated native build
matches all 32 output IDs with both ordinary and 16-token chunked prefill at
batch 1. Batch 2 also matches when one row stops at 8 and the other reaches 32,
for both prefill paths. The independent audit checks 144 emitted tokens across
these repeated/prefix cases and rejects 10 corruptions; this is one unique
32-token reference sequence, not 144 independent quality samples. All four
native owner counters return to zero for each run.

The initial tested context capacity is 256. KV and intermediate native state remain
FP32; tensor operands use the explicitly selected FP16 path. This is not a
claim that every operation has the same precision as MLX. At this initial gate,
no sandboxed coding
suite, 4K context, batch 4, continuous admission, retained prefix reuse, ANE,
production xMind binding or serving-performance comparison is qualified yet.
The earlier greedy attempt that passed a whole chunk to the diagnostic head,
the missing eager-setup counter, and an incomplete source snapshot remain as
failed attempts; their results are not promoted to this gate.

## Expanded coding and bounded 4K-capacity checks

The dense adapter now uses parallel chunk attention up to capacity 4096 when
the compiled engine advertises `compact_attention_index`. Other engines retain
the previous 2048 bound and serial fallback. This reuses existing native compact
indexes without changing numeric policy. An isolated FP16 engine enables only
that additional extension and passes 15/15 native tests. Added ungated Mistral
GQA cases use 32 query heads, 8 KV heads and dimension 128, covering empty and
2048-token prefixes, high cache slots, overwritten positions and inactive rows.

The frozen fixture retains all eight original coding prompts/contracts and
their greedy 1024-token budget. Two separate controlled-context probes have 256
and 2056 actual prompt tokens and 64-token output budgets. The nominal 2048 target
becomes 2056 after deterministic filler; manifests record actual IDs/counts.
Offline preparation uses Transformers 4.57.6. MLX 0.32.4 / mlx-lm 0.32.2 /
Transformers 5.19.0 and the native C++ tokenizer agree on every prompt ID.

| Check | MLX reference | Garnet native Metal |
|---|---|---|
| Unchanged strict code-only contracts | 0/8 | 0/8 |
| Complete coding output-ID sequences equal to MLX | Reference | 5/8 |
| 256-token context, 64 outputs | Reference | All 64 IDs equal |
| 2056-token context, 64 outputs | Reference | All 64 IDs equal |
| Native owner counters at close | Not instrumented here | All four zero |

All coding answers finish naturally at EOS. Their explanations/examples
violate the strict complete-code output contract, so the unchanged evaluator
reports zero passes. A secondary single-Python-fence diagnosis identifies one
unambiguous contract pass per engine; multiple-block outputs are not guessed
or stitched together, and other extracted implementations fail contracts.
It does not replace the strict score or establish general quality equivalence.

Three coding answers diverge after 3, 61 and 140 matching tokens. An eight-step
teacher-forced diagnosis of the earliest case captures complete logits from
ordinary and chunk16 native prefill. Both native modes choose identical tokens
at all inspected steps; their largest logit difference is 0.005864. At step 3 MLX
has an exact top-logit tie between its chosen token 1032 and native token 1040.
Native FP32 logits favor 1040 by 0.012589 / 0.014336 for ordinary/chunk16. Their
maximum absolute error against MLX at that step is 0.014937 / 0.014661. This
supports a precision/tie-breaking explanation for this early divergence; it
does not prove all discrepancies harmless. No output is altered to force parity.

Native requests use batch 1, capacity 4096 and eagerly prepared chunk 16 graphs.
The driver rejects request-time graph creation and checks final KV positions.
Two independent collection audits validate all IDs, finite first logits,
finish reasons/counts and owner cleanup, each rejecting nine corruptions. The
initial auditor assumed native `Tensor.tolist()` retained a batch dimension;
the corrected auditor explicitly checks flat native storage against MLX's
`[1,32768]`. Inference outputs were not rerun or rewritten.

Prepared Metal allocation is 13,422,149,632 bytes, versus a 1 GiB analytical FP32
KV payload. Sampled maximum RSS is about 5.49 GiB for native coding and 6.85 GiB for
native context; these are samples, not unique GPU memory or measured peaks.
System samples contain compression, page-ins/page-outs and nonzero swap-ins,
despite unchanged total swap usage. Owner counters reach zero while residual
Metal device allocations remain nonzero before process exit. These runs are
not paging-free and do not establish complete device-memory release.

Only the observed bounded context cases are admitted; no complete 4096-token
request, batch 2 at this capacity, batch 4, continuous admission or retained
prefix is tested here. The strict coding gate fails, and full MLX output parity
fails on three coding tasks. These functional/diagnostic runs have no repeated
serving measurements and are not vLLM/SGLang baselines or performance wins.

## Prepared dense profile and memory control

The opt-in `GARNET_METAL_DENSE_PREFILL_MAX_TOKENS=512` profile decouples bounded
dense prefill from the Qwen27B-specific flags. It prepares native kernel families,
private in-place KV updates, retired resident defaults/packs and shared prefill
scratch. The 512 physical token rows are global: batch1/chunk512,
batch2/chunk256, or batch4/chunk128. Width/context remain bounded at 4096.
Only this compiled 512 feature combination is qualified by the initial gate;
the other selectable maxima are not thereby tested or admitted.

An isolated source excludes the preserved Qwen blocked-projection experiment.
The build passes 15/15 native CTests and rejects four invalid configurations:
token bound 513, missing compact indexes, missing shared scratch and mixed legacy
wide flags. The expanded GQA operator oracle includes full 512-token chunks,
high slots and ragged batch4/chunk128, including a completely inactive row.
Direct K/V commits compare every stored word with independent latest-active-write
results, separately from the FP64 attention-read oracle.

At batch1/capacity4096/chunk16, the new memory profile matches all ten frozen
native control sequences exactly: 3,180 output tokens across eight coding tasks
and two controlled-context probes. Every prefill/decode frame reports 64 private
KV updates across 32 layers; request execution adds no graph family. A separate
auditor checks trace/ID integrity, exact control parity, sampled working-set
bounds and the unchanged 4 MiB close-allocation gate; it rejects 14 corruptions.
Native driver traces do not substitute for independent GPU state checks.

Prepared Metal allocation is 5,166,120,960 bytes, compared with the earlier
13,422,149,632-byte snapshot at the same batch/context/chunk. The new engine has
four eagerly prepared chunk families rather than one. These are allocation
snapshots, not measured peaks or proof of a serving speedup. All four native
owner counters reach zero; post-close Metal allocation is 720,896 bytes. Process
footprint/compression remains nonzero, so this is not a claim that all process
memory is released or that the run is paging-free.

The resident adapter also avoids creating host K/V arrays that native session
construction never consumes. This applies only during construction of the
prepared native resident profile; ordinary reset still clears native state,
and the legacy/nonresident path retains its host arrays. With the same immutable
native library, a separate batch1/chunk16 run again matches all ten controls
(3,180 tokens), passes working-set/close gates and records the actual loaded
library path. This proves observed control parity; no constructor peak or
startup speedup is claimed from these single runs.

The existing three MLX coding divergences and strict 0/8 coding results are
preserved. Exact agreement with the native controls does not remove them.
The subsequent actual-model wide-profile campaign also passes all five serial
checks below with the same native library. Every listed row matches its frozen
native control (or the declared shorter output prefix), every recorded frame
has the 64 required KV updates, no request creates a graph family, all sampled
working-set bounds pass, and every process closes with zero native owners and
720,896 Metal bytes. Each independent audit rejects 14 corruptions.

| Batch / maximum chunk | Collection | Observed rows | Prepared Metal GiB |
|---|---|---:|---:|
| 1 / 16 | Host-construction control, all ten cases | 10 | 4.811 |
| 1 / 512 | Eight coding + two context cases | 10 | 6.647 |
| 2 / 256 | Eight coding + two context cases | 10 | 7.670 |
| 4 / 128 | Same cases; final context group repeats its last row | 12 | 9.715 |
| 4 / 128 | Coding; first row stops at 8 in each group | 8 | 9.715 |

These are repetitions/prefixes of the same ten unique control sequences, not 50
independent quality cases. The batch4 long-context group keeps one row inactive
after its 256-token prefill while three 2056-token rows continue; the staggered
coding run also stops rows during decode. Final hidden vectors are gathered on
the host for this diagnostic driver, so its timings are not serving results.
The context tests still do not exercise a complete 4096-token request.

This establishes the observed fixed/ragged batch behavior, not continuous
arrival/admission or retained prefix reuse. The Mistral serving adapter, native
scheduler global-budget extension and actual HTTP measurements remain separate
qualifications; the 2x speed/throughput targets remain unmet.

Evidence under the local root: `dense-prefill-clean-source3`,
`dense-prefill-qualification1`, `mistral-dense-memory-control1`, and the immutable
`mistral-dense-prefill-library1` (SHA256
`a7354b4fb00e7fa13c35799b5e8d46a52a5f94df89db4046beae4c5efa2efabe`).
The construction and wide-profile checks use `mistral-resident-host-source1`
and the completed `mistral-dense-wide-qualification1` campaign.

## Remaining implementation and qualification

Fresh Mistral vLLM-Metal and SGLang profiles now have matched admission,
full serving prompt-ID checks and preserved strict coding failures. See
[comparator admission and fixed reference plan](mistral7b-metal-baselines.md).
Both repeated reference campaigns and their accounting/postflight audits have
now completed; preserve them across native optimization iterations.

1. Expand the initial FP16 profile admission to production workload shapes,
   chunk-size alternatives, memory limits and exact checkpoint/profile identity.
   Preserve the BF16 path and explicit coefficient/operand type matching.
2. Extend embedding and full-model precision/quality qualification;
   matching coefficient expansion alone does not prove embedding semantics.
3. Admit the Mistral text adapter with full GQA/RoPE/SwiGLU parity, explicit KV
   ownership, bounded prefill, batch/continuous scheduling and prefix reuse.
   The small native adapter gate passes; long context, batch 4, continuous
   scheduling and retained prefix reuse remain separate admissions.
4. Establish new pinned Mistral comparator baselines and correctness gates,
   running vLLM-Metal, SGLang and Garnet sequentially. Qwen27B baseline numbers
   cannot substitute for Mistral results. Freeze artifacts, tokenizer/chat
   rendering, prompt IDs, quantization, context, cache policy, concurrency,
   generation budgets, warmups and repetitions. Record precision differences
   and unsupported settings explicitly. Keep quality and prefix/batching tests
   alongside TTFT, prefill, decode and aggregate throughput measurements.
5. Add and measure the ANE/Core ML plugin separately. Core ML usage alone does
   not prove ANE placement or a speed improvement; preserve native Metal as a
   measured comparator and inspect actual placement and state-transfer cost.

Profiles specify maxima. For example, a context capacity of 4096 does not require
4096 input tokens; prefill chunk limits must permit smaller actual lengths and
tails. Layouts, kernels and admitted schedules are prepared for the profile.
For planning only, full FP32 KV storage at B1/context4096 is
`32 layers * 2 caches * 8 heads * 4096 tokens * 128 dimensions * 4 bytes`
= 1 GiB, before allocator padding, scratch, snapshots and additional batches.
This is an analytical lower-bound payload, not measured allocation or admission.

## Native HTTP admission and remaining performance gap

`mistral-native-http-admission1` collected one fresh native process, zero warmup
waves and one measured wave for each of six throughput cells and eight unchanged
coding tasks. All 22 requests completed. The independent
`audit_mistral_http_admission.py` checks frozen prompt IDs, streamed/native output
IDs and text, row leases/epochs, prefill offsets, the global 512-token budget,
projection route coverage, generation budgets and graceful cleanup. It rejected
11 deliberate trace/control corruptions. All eight coding outputs exactly match
the frozen native B1 controls; the unchanged sandboxed coding contracts still
score **0/8**, with all eight outputs classified as invalid code.

| Nominal input / concurrency | Garnet aggregate tok/s, one wave | vLLM-Metal median tok/s | SGLang median tok/s | 2x faster-reference target |
|---|---:|---:|---:|---:|
| 256 /1 | 16.110 | 22.021 | 22.042 | 44.084 |
| 256 /2 | 25.796 | 31.464 | 32.127 | 64.254 |
| 256 /4 | 29.320 | 45.235 | 42.194 | 90.470 |
| 2048 /1 | 2.586 | 12.345 | 12.518 | 25.036 |
| 2048 /2 | 2.916 | 16.536 | 14.482 | 33.071 |
| 2048 /4 | 3.054 | 21.179 | 16.337 | 42.358 |

Aggregate throughput includes prefill. Reference columns are the previously
frozen 15-wave pooled medians across three cohorts per engine, not new runs.
Native single-request decode was 22.988 tok/s for the short cell and 13.826 tok/s
for the long cell; corresponding client TTFT was 2.408s and 40.269s. Native results
are single-wave admission observations, not repeated speed qualification.
Paging/compression remains in the evidence. All 2x targets remain unmet.

The native trace reached four active rows; no wave submitted a replacement while
another request was still running. Thus this collection does not establish
dynamic row refill. Prefix reuse was disabled. All four native owner counters
were zero after shutdown, with a 720,896-byte final Metal allocation snapshot.
Neither that snapshot nor the sampled working-set checks establish peak memory.
See `native-audit.json`, `quality.json`, `comparison.json` and `comparison.md`
inside the admission directory for complete evidence and hashes.

Separate native stage probes (`mistral-native-stage-probe1`) showed GPU execution
dominating decode and prefill on the selected 84-token prompt. An isolated,
default-off `GARNET_METAL_INDEXED_AFFINE_RELAXED` candidate now permits explicit
FP16 relaxed tensor arithmetic in both regular and indexed prefill. The earlier
indexed shader hardcoded strict arithmetic, and CMake rejected combining indexed
projections with the regular relaxed flag. The candidate requires both explicit
precision policies and exact FP16 coefficients; it preserves FP32 accumulation,
decode and cache representation. Its 16 native tests passed without relaxing
their numerical bounds; four inconsistent compile profiles were rejected.
Three repetitions at B1 and B4 preserved all 32 selected control tokens per row
(480 repeated token observations in total). Those short-prompt stage results
differ in compiled serving features as well as precision policy, so they cannot
isolate the effect of relaxed math. Full-output parity and serving speed remain
pending. This is an experiment, not a qualified replacement.

`mistral-prefill-op-diagnosis1` serialized operations in the strict serving
engine's chunk at offset 1920, after populating real KV from the preceding
tokens of the frozen 2056-token context prompt. Physical batch was four, with
only row zero active and chunk capacity 128. Of 832 recorded operations,
attention accounted for 3360.323ms (80.13%) of summed GPU time, versus
545.073ms (13.00%) for matrix projections. Submission splitting changes timing;
these shares identify a candidate bottleneck rather than establishing serving
latency. This evidence prioritizes a separate test of the existing prepared
FP32 attention matrix/context-bucket path, with strict projection precision.
Native admission evidence is bound by
`mistral-native-http-admission-provenance1.json` (796 artifacts), SHA256
`f5383edc97f0e69772bcd846f00d8e097af0494a4557ff6c95419cba0a897336`.

The first attention-matrix build passed 17/18 native tests; its full 512-token test
exposed the old 256-token limit in the index proof. The corrected candidate
passes the compiled chunk maximum explicitly to both the index proof and the
context-bucket selector, retaining 256 as the helpers' legacy default. New CPU
checks cover explicit 512-token admission, token-stride overflow, sparse tails and
invalid/unrequested bounds. Encoder checks still enforce the global physical
row budget. The corrected isolated source/build is
`mistral-mps-attention-source2` / `native-mistral-mps-attention-build2`.

All 18 native tests passed. The subsequent B4/chunk128 actual-weight collection
passed all four existing profile-audit gates, without changing their limits:
collection integrity, exact frozen native controls, sampled working set and
post-close allocation. It generated 3308 token observations across 12 physical
rows representing ten unique cases (the last context case repeats to fill B4).
Prepared Metal allocation was 11,029,495,808 bytes; final allocation was 1,507,328
bytes, below the existing 4MiB close limit, with all four native owner counts
zero. This proves the recorded model/control and row-position checks, not
continuous arrivals or HTTP performance. See
`mistral-mps-attention-model-qualification1/audit.json`.
The immutable candidate library SHA256 is
`4aaf62f9df14043d67a78b2755232f14f5f878d068be873114fdbbc924764516`.

The subsequent `mistral-mps-native-http-admission1` run replayed the same
22 requests, with one native engine running at a time. All requests completed;
accounting reported zero issues. The native/SSE auditor also checked every
attention matrix/scalar-tail route, selected context extent, seven prepared
plans, 42 buckets and 596,664,320 bytes of attention scratch. All eight coding
outputs matched the frozen controls, and all 15 trace corruptions were rejected.
Strict coding quality remains 0/8. Final native owner counts were zero and Metal
allocation was 1,507,328 bytes after graceful shutdown.

| Nominal input / concurrency | Prior Garnet tok/s | Prepared-attention Garnet tok/s | vLLM-Metal median tok/s | SGLang median tok/s |
|---|---:|---:|---:|---:|
| 256 /1 | 16.110 | 16.305 | 22.021 | 22.042 |
| 256 /2 | 25.796 | 27.077 | 31.464 | 32.127 |
| 256 /4 | 29.320 | 32.042 | 45.235 | 42.194 |
| 2048 /1 | 2.586 | 4.962 | 12.345 | 12.518 |
| 2048 /2 | 2.916 | 6.435 | 16.536 | 14.482 |
| 2048 /4 | 3.054 | 7.639 | 21.179 | 16.337 |

These are aggregate output tokens/s including prefill. Each native column is
one wave; comparator columns retain the existing 15-wave medians. Native long
single-request TTFT fell from 40.269s to 16.489s; decode was 13.668 tok/s, versus
13.826 in the prior native wave. The improvement does not establish a decode
gain or repeated speed qualification. All 2x comparator targets remain unmet.
The later studies below address dynamic arrival/refill, direct native prefix
reuse and cached chunked HTTP scheduling. Full 4096-token requests and ANE remain
pending.

## Actual continuous arrivals and native prefix reuse

`mistral-continuous-http1` exercised the prepared-attention engine through real
HTTP requests: A/B/C/D started together, A finished after eight output tokens,
then E arrived for 16 tokens, followed by A returning for 32 tokens. B/C/D had
budgets of 256/320/384 and continued decoding through both replacements. The
independent auditor requires a native batch containing all four initial jobs,
both replacements before any survivor finishes, survivor decode frames across
both lifetimes, fresh row epochs and exact frozen token-prefix parity.

All six requests passed, with 1016 generated tokens. A's physical row was 1,
reused at epochs 2 -> 5 -> 6; surviving rows retained their leases. All four
native owner counters closed, with 1,507,328 Metal bytes after graceful shutdown.
Four overlap/epoch/row/trigger corruptions were rejected. The generic native/SSE
auditor was refactored to share parsing and accept this explicitly bounded
prefix-control study; its default 22-request/full-control gate still passed on
both earlier native HTTP collections. This test used prefix caching OFF and
separate native decode/prefill forwards (`mixed_forward=false`). It establishes
actual refill/interleaving, not a fused mixed kernel or a performance result.

Enabling the existing C++ token-prefix API on the same strict MPS source passed
all 18 native tests. `mistral-native-prefix-probe1` then tested actual Mistral
weights through that runtime API, with B/C/D kept alive while row 0 was reused.
The host cache was bounded to 96MiB and three owning checkpoints; observed
accounting reached 69,001,479 bytes. Stored boundaries were 64, 84 and 115 tokens.

| Direct native query | Returned checkpoint | Fresh input after restore | Observed output gate |
|---|---:|---:|---|
| E replacement | 0 | 120 | Frozen native control prefix |
| Exact A return | 84 | 0 | Frozen native control prefix |
| A plus 32 generated tokens | 115 | 1 | Same 32 output tokens as fresh execution |
| A with token 70 changed | 64 | 20 | Same 16 output tokens as fresh execution |

The 115-token checkpoint contains A's 84 prompt tokens plus 31 consumed decode
inputs; the last emitted token remains unconsumed and is the next fresh input
for the 116-token appended prompt. The changed prompt uses a stored checkpoint
before the edit. This does not establish arbitrary common-prefix truncation.

The probe produced 704 token observations: 608 checked against existing frozen
native controls, and 96 across independent fresh/restored runs of the two new
prompts. B/C/D each produced 176 unchanged control tokens. Stale restore and
capture epochs were rejected. The independent auditor checks identities,
token arrays, cold/warm parity, hit/miss/accounting progression, row positions,
leases and cleanup, and rejects 12 deliberately corrupted records. Final native
owners were zero and Metal allocation was 1,507,328 bytes. Restored checkpoint
counts are assertions executed by the frozen native API driver; trace positions
are driver observations, not independent GPU KV snapshots.

This study is direct native runtime qualification; the subsequent integration
below adds cached chunked HTTP scheduling. Longer prefixes still face the existing
512MiB maximum host-cache budget: a 2048-token Mistral FP32 KV payload alone is
512MiB before hidden/history/accounting overhead. No full 4K reuse, prefix speed,
general coding-quality or xMind production claim follows from these studies.
The immutable prefix-enabled library SHA256 is
`15a50423252ccb7c106ac04592813610fa21f285cd30041ec3b64383961550db`.

## Native cached-progress scheduling and HTTP integration

The prefix-enabled native scheduler now exposes
`submit_tokens(token_ids, output_budget, priority)`. It copies the token IDs
before queue admission and performs content lookup/restoration when a physical
row is allocated. The allocator reports the actual restored count and whether
retained hidden state can resume the output head. The portable policy uses that
count for its prefill offset and KV cursor. Its existing length-only submit API
and allocator defaults retain uncached behavior.

An exact prompt hit is an explicit zero-token prefill plan, admitted only with a
full restored prompt and ready hidden state. Packed metadata preserves that
row/epoch without claiming GPU token work. The Mistral bridge skips embeddings
and body execution for an entirely cached plan, then calls the native row head.
Before callbacks commit progress, the native bridge checks the actual tracked
token count. Native capture retains power-of-two intermediate checkpoints from
64 tokens onward, complete prompts, and consumed generation history before row
release. Budget rejection leaves a miss rather than failing generation. Python
transports token IDs and plan metadata; it does not match prefixes or choose
the restored position. Cached mixed-forward fusion remains disabled.

`mistral-cached-scheduler-draft1` passed all 18 native tests, including cached
heads, partial hits, duplicate callbacks, packed zero-token ranges, invalid
restored counts, missing hidden state, cancellation and row refill. The actual
Mistral session additionally rejected integer/token-contract violations and
callbacks acknowledging prefill before execution; cancelling queued/in-flight
requests left no tracked policy state. Native source and artifacts are isolated
from the pending blocked-decode and relaxed tensor experiments.

The HTTP profile is B4/context4096/chunk64 with a compiled global 512-token upper
bound, six prepared MPS plans (290,480,128 scratch bytes), 36 context buckets,
16 prefix entries and a 512MiB accounted host-cache limit. The two phases in
`mistral-cached-http1` ran sequentially on the same pinned library: two fresh
controls for new single-user prompt edits, then eight cached requests with
survivor overlap. No vLLM-Metal or SGLang baseline was rerun.

| Cached HTTP request | Reused tokens | Observed behavior |
|---|---:|---|
| Initial A/B/C/D and replacement E | 0 | Frozen native control prefixes |
| Exact A return | 84 | Head only; no prefill-body dispatch |
| User message with an appended instruction | 64 | Remaining suffix executed; outputs equal fresh control |
| User message with edited final wording | 64 | Remaining suffix executed; outputs equal fresh control |

All 1080 token observations passed: 1016 against existing frozen controls, plus
64 across the two new fresh/cached pairs. B/C/D remained active through every
return. Row 0 was reused at epochs 1 -> 5 -> 6 -> 7 -> 8. The independent auditor
checks native cursor/lease/route coverage, full native/SSE token identity,
constant cached-token usage, actual survivor overlap and cache accounting. It
rejects eight lifecycle/head/body/usage corruptions. Both earlier 22-request
native static collections still pass its unchanged default full-control gate.

Maximum accounted host-cache storage was 531,847,789 bytes (507.21MiB), below
536,870,912 bytes. Native sampled Metal allocation respected the existing
working-set bound. Both phases closed all four native owner counters with
1,507,328 Metal bytes after shutdown. Host accounting and task/device snapshots
are not peak or unique physical-memory measurements; fresh-phase task
compression was nonzero. Exact-hitting or partially reused work must not be
compared as an uncached speed gain against the frozen prefix-OFF comparators.

This establishes the checked cached HTTP lifecycle, not arbitrary common-prefix
truncation, full 4K snapshots, multimodal prefix reuse, production xMind binding,
new coding quality or a 2x speed qualification. The immutable library SHA256 is
`97b0bb9b1eca46da69e78caaee436f34d52a9f87bfe509332c6ef28ab9cc7f61`.

## Evidence

Local root: `/Users/shawn/CantorAI/out/research/mistral7b-local-coder/`.

- `mistral7b-mlx-checkpoint-manifest1.json`, `mistral7b-checkpoint-download1.log`:
  pinned download and complete checksum validation.
- `mlx-mistral7b-metadata1/`: pinned configuration, complete tensor header and
  metadata/tokenizer hashes.
- `float16-codec-check2/report.json`: CPU codec inputs, outputs, binary/source
  hashes and independent reference comparison using the saved verifier.
  `float16-codec-check1` retains the initial equivalent comparison.
- `fp16-weight-loading1/fixture.json`, `native3.log`: complete fixture data and
  successful instrumented native FP16 loading checks. `native.log` and
  `native2.log` preserve earlier driver failures (wrong tensor factory and a
  build lacking owner-counter capability). Configure failures are retained.
- `mistral-fp16-onboarding-provenance1.json`: frozen source/build/result hashes.
- `mistral-affine-rows-native1`, `mistral-affine-rows-mlx2`: complete native and
  MLX selected-row outputs. `mistral-affine-rows-mlx1` preserves the precision
  assumption failure described above.
- `mistral-embedding-runtime1.log`, `mistral-affine-metadata-tests1/report.json`,
  `mistral-affine-bf16-regression1.log`: runtime embedding checks, representation
  guards and the unchanged BF16 affine regression suite.
- `mistral-affine-embedding-provenance1.json`: source/binary/result hashes for
  this additional prerequisite qualification.
- `mistral-fp16-clean-source2`, `native-mistral-fp16-clean-build2`,
  `mistral-fp16-clean-ctest2.log`: isolated staged implementation and 13 native
  test results, without the pending Qwen experiment.
- `mistral-mlx-greedy1/reference.json`, `mistral-native-clean-greedy1.json`,
  `mistral-native-clean-greedy-batch2-1.json`: frozen small functional reference
  and native results, including complete first logits and prepared/cleanup
  snapshots. They are not performance baselines.
- `mistral-native-clean-functional-audit1.json`: independent exact-token audit
  and 10 corruption checks.
- `mistral-fp16-functional-provenance1.json`: immutable library/source/result
  hashes for this functional gate.
- `mistral-fp16-4k-source1`, `native-mistral-fp16-4k-build1`,
  `mistral-fp16-4k-ctest1.log`: isolated compact-index build and 15 native tests.
- `mistral-coding-context-fixture1`, `mistral-mlx-coding-context1`,
  `mistral-native-4k-coding1`, `mistral-native-4k-context1`: frozen prompts,
  complete IDs/text/logits, raw logs and sampled memory.
- `mistral-native-4k-coding-audit2.json`,
  `mistral-native-4k-context-audit1.json`: observed parity/corruption checks.
- `mistral-mlx-coding-quality1.json`, `mistral-native-4k-coding-quality1.json`:
  strict failures; separate `coding-block-diagnosis1.json` files are secondary.
- `mistral-mlx-teacher1`, `mistral-native-teacher1.json`,
  `mistral-teacher-analysis1.json`: fixed-continuation logits and diagnosis.
- `mistral-fp16-4k-provenance1.json`, `mistral-fp16-4k-library1`:
  expanded source/binary/result bindings and immutable built artifacts.

The first tokenizer runtime build directory was reused for FP16 loader work.
Its original provenance retains the original hashes, but that build output path
is mutable. A copied immutable library under `tokenizer-runtime-final-library1`
and a fresh 120-check `native-tokenizer-runtime-api2.log` supersede that runtime
artifact. The original five-family reference probe and per-case records remain
unchanged. The protected Qwen27B qualified library still has its original SHA256
`69491581eb6ff7a659dcfc065231823b3827e8a15c603fec4756a1a2dfdbcb34`.

## Strict cooperative64 projection candidate

The late-prefill trace in `mistral-prepared-attention-op-diagnosis1` attributed
48.96% of serialized GPU time to projections, 22.13% to attention and 13.11% to
layout work. These per-operation submissions are diagnostic, not serving time.
The existing cooperative64 kernel now has a compiled output threshold:
`GARNET_METAL_MPP_REUSE64_MIN_OUTPUTS=4096` admits Mistral Q/O, gate/up and down
projections. The default remains5120 and the candidate remains OFF by default.
Preparation, dense dispatch, indexed dispatch and backend/profile metadata use
the same constant. The only allowed thresholds are4096/5120; a nondefault value
requires explicitly enabling the candidate. FP16 coefficients/operands, strict
tensor arithmetic and FP32 accumulation remain unchanged.

`mistral-reuse64-draft1` records isolated control/candidate builds, each passing
all18 native tests. The candidate reference test checked8,123,261 output words,
including independent full FP64 dots at reduced dimensions, thresholds +/-1,
63/64/65 active rows, ragged output tiles, reverse indexed masks and inactive
zeros. Four invalid compile policies were rejected. The preserved source and
library are `mistral-reuse64-source1` and `mistral-reuse64-library1`; library SHA256
is `bb25ddbf4d71ecc7815871051980eef293fd6390911d66409e5c0ec4781a325d`.

`mistral-reuse64-operator-campaign1` ran eight processes sequentially: two
cohorts, dense/indexed routes and strict control/candidate, with reversed engine
order in the second cohort. Each of256 cells used three warmups/nine samples,
real Mistral projection dimensions and synthetic inputs/weights. All128 paired
output files matched bitwise (143,478,784 observed output words across runs).
The FP64 samples passed14,400 checks and49,459,200 inactive words were zero.
At128 active indexed rows, Q/O were1.16–1.18x, gate/up1.38–1.41x and down1.68–1.69x
faster. These operator results are not model quality or serving throughput.
The auditor rejects overlapping processes, changed binaries, missing cells,
wrong routes, runtime compilation, nonfinite timings and differing word hashes.

`mistral-reuse64-model-qualification2` passed all four profile gates:12 rows,
10 unique cases and3,308 output tokens matched frozen native controls. Owner
counts returned to zero; close Metal allocation was1,769,472 bytes. Attempt1
failed prompt tokenization before inference because a runner-added
`DYLD_LIBRARY_PATH` selected the older installed library. A direct probe proved
the loaded-path difference; attempt2 removed all DYLD overrides and recorded the
actual intended library. This is native-control parity, not independent MLX
parity or an improved coding-quality result.

`mistral-reuse64-native-http-admission1` completed22 requests with no accounting
issues. All eight coding outputs matched the frozen native controls; the strict
coding suite still passed0/8. SSE/native identities, row epochs, positions,
cleanup and all70 prefill projection traces passed. Eligible chunks executed
exactly160 cooperative projections across32 layers; chunks below64 active tokens
executed zero. The HTTP auditor rejected19 corruptions, including missing or
incorrect cooperative routing. The earlier two HTTP collections still pass the
updated auditor. The new operator benchmark also participates in the serial
engine guard.

| Nominal input / concurrency | Prior Garnet tok/s | New Garnet tok/s | vLLM-Metal tok/s | SGLang tok/s | 2x target |
|---|---:|---:|---:|---:|---:|
| 256 /1 |16.305|15.072|22.021|22.042|44.084|
| 256 /2 |27.077|26.350|31.464|32.127|64.254|
| 256 /4 |32.042|33.248|45.235|42.194|90.470|
| 2048 /1 |4.962|4.935|12.345|12.518|25.036|
| 2048 /2 |6.435|7.052|16.536|14.482|33.071|
| 2048 /4 |7.639|8.510|21.179|16.337|42.358|

Both native columns are one cohort, no warmups and one measured wave per cell;
comparator columns retain the frozen15-wave medians and were not rerun. These
descriptive values do not establish a causal or uniform serving gain. Arithmetic
differs and paging/compression remains. Single-request performance did not
improve; all2x/half-TTFT targets remain unmet. At that checkpoint Core ML/ANE was unimplemented;
none of this evidence uses ANE execution.

## Core ML plugin bring-up after the Metal checkpoint

The optional native plugin now has a functional state/API prototype described in
[coreml-ane-plugin.md](coreml-ane-plugin.md). It loads precompiled ML Programs,
owns persistent state, supports native typed/FP32 transport and checks anticipated
device plans. The19-test native build and3,079-check API fixture pass. This tiny
fixture prefers CPU even under CPU+ANE policy; it is not evidence of ANE
acceleration. Whole-model Mistral conversion, actual ANE placement, compiled profile families
and continuous-batch/prefix integration for this backend remain pending. The
Metal benchmark table above is unchanged by this plugin work.

## Prepared physical request spans for MPS attention

`GARNET_METAL_CHUNK_ATTENTION_ACTIVE_ROWS=ON` is an optional candidate alongside
the existing context buckets. The serving profile must explicitly set
`chunk_attention_active_rows=true`. The default is OFF. Preparation constructs
every contiguous physical request span for profiles with at most four request
owners, at every existing context bucket. A missing span cannot compile or
allocate GPU workspace during a request. Larger generic batches and pending
GPU-produced controls retain the full schedule.

Each span has bounded MPS matrix descriptors and fixed buffer offsets for the
existing query, live key/value, score, probability and result workspace. The
cache descriptor also contains only that span's physical KV owners. Runtime
binds those prepared layouts to the current private cache buffers. It never
compacts or copies KV state, changes a row lease, pads a short request to its
maximum length, or changes attention arithmetic. Nonadjacent owners use separate
spans. Full-batch execution reuses the original products. The original pack,
softmax and unpack kernels retain zero-output behavior for inactive tokens and
owners. At most eight actual tokens still use the existing scalar-tail route.

This distinction matters on this Mac: the first implementation changed only
MPS `batchStart` / `batchSize` while retaining full matrix descriptors. It
preserved outputs but gave effectively no speedup; separated spans were slower.
The retained gapped/overwrite study is `mistral-active-attention-operator1`, and
the contiguous-prefix repetition is `mistral-active-attention-operator2`.
The bounded-descriptor implementation is `mistral-active-attention-source3` /
`native-mistral-active-attention-build3`. No unsupported claim about MPS internals
is needed to explain the next experiment: it changes the actual operand-view
bounds as well as the requested batch range.

`mistral-active-attention-operator3` runs two fresh native processes in serial,
with alternating full/selected schedules, three warmups and nine measurements.
The fixed profile is B4/T128, Q32/KV8/D128, FP32 operands, context 4096. Inputs are
deterministic synthetic text-attention values with contiguous live prefixes.
The following GPU medians use the second cohort; both cohorts retain all raw
samples and outputs:

| Active physical owners | Prefix length | Full schedule (ms) | Bounded views (ms) | Operator speedup |
|---|---:|---:|---:|---:|
| Row0 |0 |1.1755 |0.6997 |1.680x |
| Row0 |3968 |13.9774 |6.5558 |2.132x |
| Row3 |3968 |13.9726 |6.5526 |2.132x |
| Rows0,2 |3968 |15.8244 |10.9587 |1.444x |
| Rows0,1 |3968 |15.8179 |10.8699 |1.455x |
| All four |3968 |19.5405 |19.5373 |1.000x |

These are attention operator latencies, not generated token rates or full-model
speedups. Every one of 41,943,040 compared output words is bit-identical to the
full schedule, with 480 independent FP64 dot/softmax/value samples across the two
cohorts. All inactive outputs are zero, cohort outputs repeat exactly, and native
owners return to zero. The counters report logical MPS matrix ranges encoded;
they do not establish observed hardware FLOPs. Prepared GPU scratch stays at
596,664,320 bytes across the seven graph families, as in the previous profile.
All 19 native tests pass, including every four-owner subset, token gaps and
overwrites, gated/ungated attention, nonzero starting owner, GPU-produced control
fallback, stale populated scratch, read-only KV and shared-plan lifetime checks.

The actual-weight model check in `mistral-active-attention-model-qualification1`
matches the frozen native outputs on all 12 rows (ten unique cases), with 3,308
generated tokens. It checks actual product-range counters and completed controls
at every prefill step; the existing profile auditor passes collection integrity,
exact control parity, sampled working-set and close-allocation gates. Native
owners close to zero, with 1,769,472 Metal bytes remaining. This preserves the
known native control behavior; it does not establish new coding quality or exact
agreement with the independent MLX path.

Reproduction tools are `run_chunk_attention_active_rows.py`,
`audit_chunk_attention_active_rows.py`, and the native
`garnet_chunk_attention_active_rows_test` executable (`--benchmark NEW_DIRECTORY`
for the large serial operator study). `local_coder_batch_worker.py` rejects a
profile/backend policy mismatch and verifies the encoded matrix counts against
actual owners. `audit_mistral_http_admission.py` also validates those ranges
against native request leases and rejects missing or changed schedule events.

The serving admission in `mistral-active-attention-native-http-admission1`
completes the same 22 frozen requests, including 4,844 generated output tokens,
with no accounting issues and exact frozen native output IDs on all eight
coding cases. Its audit rejects 22 corrupted evidence cases, including changed
matrix counts, wrong active owners and missing request-schedule events. All
native owners close to zero; remaining Metal allocation is 1,769,472 bytes.
Strict coding quality is still 0/8. The earlier native, MPS and cooperative64
collections also pass the extended auditor.

| Nominal input / concurrency | Prior Garnet tok/s | Bounded-view Garnet tok/s | vLLM-Metal tok/s | SGLang-MLX tok/s | 2x target |
|---|---:|---:|---:|---:|---:|
| 256 /1 |15.072 |16.969 |22.021 |22.042 |44.084 |
| 256 /2 |26.350 |29.733 |31.464 |32.127 |64.254 |
| 256 /4 |33.248 |36.464 |45.235 |42.194 |90.470 |
| 2048 /1 |4.935 |5.811 |12.345 |12.518 |25.036 |
| 2048 /2 |7.052 |8.068 |16.536 |14.482 |33.071 |
| 2048 /4 |8.510 |9.646 |21.179 |16.337 |42.358 |

Both Garnet columns are one cohort with no warmups and one wave per cell;
reference columns remain the frozen 15-wave medians. They are descriptive
comparisons under different system conditions, with paging/compression retained,
and do not establish the isolated cause of whole-model differences. No reference
engine was rerun. Native FP32 decode/KV/intermediates still differ from the
comparators' FP16 path. At concurrency 1, native TTFT is 1.871s /12.767s and decode
is 22.472 /13.737 tok/s for short /long prompts. All whole-model 2x decode,
2x throughput and half-TTFT targets remain unmet. The operator gain must not be
presented as achieving those targets.
