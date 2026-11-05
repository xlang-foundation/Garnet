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

## Exact native verification and trained-head experiment

Dense Mistral can now prepare native greedy verification without enabling the
Qwen recurrent convolution/delta extensions. `prepare_verification(tokens=4)`
prepares target graphs and shared FP32 projections before execution. The profile
currently allows at most four physical token rows: B1/T4, B2/T2 or B4/T1. The
native runtime owns KV span backups, target decisions and accepted-prefix
commit/restore. This does not yet accelerate B4 serving or implement a tree.
Recurrent verification still requires its recurrent extensions.

`mistral-verification-qualification1` checks 24 cases at capacities 256 and 4096:
accepted depths 1–4, partial active spans, first/last EOS, a carried second round,
near-capacity positions, and two actual 256-token prompt states. Complete FP32
KV hashes for all 32 layers and accepted hidden hashes match ordinary decode.
The near-capacity tests seed cache positions; they are not full 4k prompt runs.
Every round checks 225 prepared affine dispatches, no misses or FP16 prefill
route, 64 KV backups, 32 attention dispatches, and the required restore count.
All native owners close to zero, with 2,031,616 Metal bytes remaining. The clean
verification and head-enabled builds each pass all 21 native tests.

The optional explicit `medusa_residual_head` native IR operator supports one
raw BF16 residual layer with FP32 inputs/arithmetic, hidden width 4096 and
vocabulary 32768. It leaves target weights and state unchanged. A diagnostic
feature output exposes the actual residual used by the vocabulary projection.
`MistralMedusaDraft` loads up to three heads for the B1/T4 prototype. This uses
the pinned `leptonai/Mistral-7B-Instruct-v0.3-4heads` artifact at revision
`6e4b8a1a7b10167c7d52aba0c9badcdbd5edd944`, safetensors SHA256
`af35f952136cdb12626cbc272d6ff3334f6e807d49cb53bf121a8fcac7cbacc1`.
Its configuration names the training base as `N/A`; compatible dimensions and
the observed acceptance below do not prove its training-base identity.
The original BF16 activation path is not reproduced. Native target verification
protects the accepted greedy token stream.

`mistral-medusa-operator-probe2` independently checks 12,288 normalization words,
36,864 actual residual features and 54 vocabulary dot samples against FP64
calculations from the pinned weights. The original dot limits remain unchanged;
the residual bound propagates that limit through bias rounding and SiLU.
Maximum fractions of the allowed error are 0.03167, 0.00469 and 0.00304,
respectively. Seven invalid head/profile combinations are rejected. This is
arithmetic evidence, not model compatibility, coding quality or a speed claim.

`mistral-medusa-generation1` then generates the full original budgets on all ten
frozen coding/context cases using causal head predictions. Ordinary and
single-path top-1 head modes alternate order by case, sequentially in one
process. Both produce all 3,180 frozen native output IDs and identical final
complete KV/hidden hashes, for 6,360 output tokens total. The profile is
B1/context4096/prefill128/verifier4/prefix-off, which differs from the frozen B4
HTTP serving profile. These are one-run diagnostics with no warmups; timings
include host orchestration and trace capture. No reference engine was rerun.

| Case | Ordinary decode tok/s | Single-path heads tok/s | Speed ratio | Committed inputs/round |
|---|---:|---:|---:|---:|
| Merge intervals |24.620 |18.133 |0.736 |1.783 |
| Stable unique |24.778 |17.910 |0.723 |1.752 |
| Toposort |24.023 |15.910 |0.662 |1.679 |
| Half-open subtraction |24.073 |15.037 |0.625 |1.587 |
| Incremental frames |23.900 |15.709 |0.657 |1.712 |
| Bounded LRU |23.418 |16.129 |0.689 |1.844 |
| JSON pointer |24.057 |15.208 |0.632 |1.684 |
| Bisect repair |24.919 |15.683 |0.629 |1.667 |
| Context 256 |24.415 |17.458 |0.715 |1.853 |
| Context 2048 |14.256 |11.152 |0.782 |2.032 |

Across 1,838 rounds, 1,332 of 5,508 proposed inputs commit (24.18%); including
the known first input, 1.7247 inputs commit per round. Target plus commit GPU
duration averages about 89–105ms per round for coding cases and 173ms for the
long context case, versus about 40ms per ordinary short-context token. The
remaining roughly 9.7ms per round includes draft work, host work and other
unattributed costs; it is not an isolated draft GPU measurement. The four-row
verification cost outweighs the accepted extra tokens. This prototype is slower
and is not selected as a default optimization. It is not a Medusa tree
implementation, so the result does not evaluate tree decoding.

Reproduction tools are `run_mistral_greedy_verification.py`,
`audit_mistral_greedy_verification.py`, `run_mistral_medusa_probe.py`,
`audit_mistral_medusa_probe.py`, `run_mistral_medusa_generation.py` and
`audit_mistral_medusa_generation.py`. The generation CLI audit binds the terminal
collector, input paths and library SHA256, then checks the complete token/state
contract and per-round GPU duration accounting. CPU-only
`test_mistral_speculative_audits.py` rejects 39 corruptions of the retained real
target/generation evidence. Raw inputs, outputs, failures, libraries and clean
source snapshots remain under the external Mistral research directory. None of
this establishes 2x serving speed, half TTFT or improved coding quality; the
native frozen coding outputs retain the previously reported 0/8 strict result.

## Exact FP32 four-row projection reuse

The next opt-in engine profile, `GARNET_METAL_MISTRAL_DECODE_REUSE=ON`, shares
each FP32 input vector across four adjacent output channels. Each output keeps
the existing packed-word order, dequantization FMA, per-lane accumulation order
and SIMD sum. Prepared pipeline entries retain the output tile width and encode
the matching dispatch grid. All occupancy variants prepare before readiness;
actual row counts and strides remain bounded runtime data. This does not change
KV ownership, attention, weights, precision or scratch layout.

The admitted policy uses four-column reuse only for four active rows and these
Mistral projection shapes: K4096/N4096, K4096/N14336, K14336/N4096 and
K4096/N32768. K/V projections (K4096/N1024), smaller occupancies and other shapes
retain the control kernel. The option defaults off and requires native affine
FP16 coefficients, prepared profiles and indexed FP32 decode. Serving profiles
must explicitly set `mistral_decode_reuse: true`; the worker checks backend
agreement before loading the model and validates dispatch counts per stage.

`mistral-decode-reuse-operator2` explores two- and four-column tiles across all
five model projection shapes and four small/padded shapes, dense rows 1–4 and
every four-owner subset. Six processes run serially in forward then reverse
order, with three warmups and nine measured samples per cell. All 20,859,860
compared FP32 words match the control byte for byte; 6,804 independent FP64
samples pass the unchanged original dot bound. Owners close to zero in all six
processes. Some early small-occupancy cases in the first cohort are much slower
than their second-cohort counterparts; the cause was not isolated. All samples
remain in the collection. The policy admits only large four-row shapes with a
gain in both cohorts, and leaves the first-cohort K/V regression on the control.

| Four active rows, dense projection | First-cohort operator ratio | Second-cohort operator ratio |
|---|---:|---:|
| K4096/N4096 |1.476x |1.419x |
| K4096/N14336 |1.474x |1.540x |
| K14336/N4096 |1.578x |1.593x |
| K4096/N32768 |1.546x |1.550x |

These are projection GPU ratios, not generated token rates. The actual admitted
allowlist then matches all 4,171,972 output words in the same 180-case suite
(`mistral-decode-reuse-runtime-operator3`). The clean source snapshot is
`mistral-decode-reuse-source3`; its native library SHA256 is
`3ae950bbf3944f19d5e64a961d4799ed5135b56f18c733d01afca9de42f2a16f`.
All 21 native tests pass with the option on and again with it off. A configuration
without FP16 coefficients is rejected. All 24 actual-weight target-verification
cases preserve complete KV/hidden state against ordinary decode. The dispatcher
reports 161 reuse operations for a full four-token round. Clipped tails retain
control body projections; the four-row target head still reports one reuse
operation. This is prepared four-row head work, not an extra accepted token.

`mistral-decode-reuse-generation3` repeats every original full-budget causal
case. All 6,360 output tokens and final KV/hidden witnesses again match frozen
ordinary-native controls, and all reuse counters match the actual active body
rows. The extended structural audits reject 41 corruptions. Native owners close
to zero with 2,031,616 Metal bytes remaining. The raw BF16 head artifact and its
unverified training-base identity are unchanged.

| Case | Prior single-path heads tok/s | Reuse single-path heads tok/s | Same-run ordinary tok/s | Reuse / ordinary |
|---|---:|---:|---:|---:|
| Merge intervals |18.133 |22.329 |24.735 |0.903 |
| Stable unique |17.910 |22.036 |25.033 |0.880 |
| Toposort |15.910 |19.334 |24.144 |0.801 |
| Half-open subtraction |15.037 |18.371 |24.123 |0.762 |
| Incremental frames |15.709 |19.155 |24.092 |0.795 |
| Bounded LRU |16.129 |20.281 |23.541 |0.862 |
| JSON pointer |15.208 |18.894 |24.186 |0.781 |
| Bisect repair |15.683 |19.603 |25.063 |0.782 |
| Context 256 |17.458 |21.596 |24.454 |0.883 |
| Context 2048 |11.152 |12.433 |14.283 |0.870 |

Target plus commit GPU duration falls to about 70–81ms per coding round and
154ms for the long-context case. The single-path prototype still trails
ordinary decode in every case and remains unselected. Prior/new columns are
separate one-run diagnostics under different system conditions, not an isolated
whole-model causal experiment. No reference baseline was rerun or replaced.

The ordinary continuous-batch serving admission is
`mistral-decode-reuse-native-http-admission1`. It completes all 22 unchanged
requests and 4,844 output tokens, with no accounting issues and exact frozen
native coding IDs on all eight cases. Native owners close to zero with
1,769,472 Metal bytes remaining. The audit validates 40,734 reuse dispatches
across 254 stage events against actual row leases, including fallback for
smaller active batches, and rejects 25 evidence corruptions. Scratch remains
596,664,320 bytes across seven prefill graph families. This checks the ordinary
continuous-batch path; it does not add B4 speculative decoding.

| Nominal input / concurrency | Prior Garnet tok/s | Reuse Garnet tok/s | vLLM-Metal tok/s | SGLang-MLX tok/s | 2x target |
|---|---:|---:|---:|---:|---:|
| 256 /1 |16.969 |16.655 |22.021 |22.042 |44.084 |
| 256 /2 |29.733 |28.161 |31.464 |32.127 |64.254 |
| 256 /4 |36.464 |40.447 |45.235 |42.194 |90.470 |
| 2048 /1 |5.811 |5.551 |12.345 |12.518 |25.036 |
| 2048 /2 |8.068 |7.658 |16.536 |14.482 |33.071 |
| 2048 /4 |9.646 |9.615 |21.179 |16.337 |42.358 |

These are aggregate end-to-end output token rates. Garnet columns are separate
single-cohort, no-warmup admissions; reference columns remain the frozen
15-wave medians. System conditions, paging/compression and the FP32-versus-FP16
decode difference remain material limitations. The short four-request cell is
10.9% higher, while several other cells are lower; these runs do not isolate
whole-model causation. Single-request short/long TTFT is 2.011s /13.634s and
decode is 22.421 /13.503 tok/s. All 2x decode/throughput and half-TTFT targets
remain unmet. The long-prompt result still calls for prefill work. The option
stays off by default, and the trained-head path remains slower than ordinary
decode. Strict coding quality is unchanged at 0/8 by exact output identity.

## Prefill diagnosis and retained negative experiments

The next operator studies preserve the frozen reference engines and original
FP64 bounds. `mistral-prefill-tile-operator1` varies only cooperative reduction
staging width (K32/64/128) while retaining the K16 multiplication sequence.
Twelve serial processes cover two cohorts, dense/indexed layouts and 384 cells.
All 215,218,176 output words match the historical native control, including
143,478,784 candidate/control compared words; 21,600 FP64 samples pass. K32 is
generally slower. K128 offers only about 1% at 128 rows for large up projections,
and regresses several 64-row Q/O/down cases to roughly 0.64–0.70x control speed.
K64 remains selected.

`mistral-prefill-pack-operator2` converts selected activations once to compact
FP16 storage before repeated output-tile loads. Eight serial processes cover
256 cells. All 71,739,392 candidate/control words are bit-identical, with 14,400
FP64 samples passing. At 128 logical rows, Q/O and down projections show roughly
1.09–1.16x operator ratios, while up projections and 64-row cases regress. The
prototype is retained as an operator-only patch: its scratch belongs to the
preparing MetalContext, while resident sessions execute through another
context. Shared graph/session scratch ownership and binding are unimplemented;
no model or serving qualification follows. It is not part of the selected
runtime.

The locally installed MLX0.32.4 NAX implementation sets the tensor descriptor's
relaxed-precision flag. The installed Apple SDK documents that flag as allowing
accuracy/performance tradeoffs. `mistral-prefill-relaxed-operator3` tests Garnet's
existing explicit cooperative-only setting with strict fallback. In eight
serial processes /256 cells, all 71,739,392 candidate/control words remain
bit-identical and 14,400 original FP64 samples pass. The large model shapes show
essentially unchanged GPU medians in both cohorts. It is not selected as a
performance optimization. The inspected SDK/MLX file paths and hashes are
retained in `mistral-prefill-tile-draft1/primary-source-inspection.json`.

The two experimental source patches and reproduction notes are preserved under
`test2026/metal/experiments/`. Applying them to clean commit `85667ff7f` reproduces
all seven changed experimental files byte for byte. Their audit tools are
`audit_mistral_prefill_tiles.py` and
`audit_mistral_prefill_operand_study.py`. Keeping these experiments outside the
selected runtime avoids presenting small or negative operator findings as
model gains.

`mistral-prefill-cost-probe1` then uses the qualified native library on a real
late chunk of the frozen 2,056-token context prompt. Only row0 is active in the
B4/context4096/chunk128 engine. Earlier chunks populate KV normally; the chunk
at offset1920 splits operations into individual submissions. This changes the
execution schedule and cannot establish serving latency, quality or production
time fractions. The audit checks all 832 operations (26 per layer), the complete
16-chunk sequence, terminal collector/library bindings and zero native owners.

| Operation family | Calls | Sum of isolated GPU durations (ms) |
|---|---:|---:|
| MatMul |224 |378.033 |
| Layout |288 |131.153 |
| AttentionChunk |32 |98.041 |
| KVChunk |64 |63.822 |
| Silu |32 |53.629 |
| Add |64 |20.949 |
| NormRope |64 |11.999 |
| Norm |64 |7.322 |

The total is 764.946ms of isolated GPU durations. Unprofiled chunks in that same
process take about 0.697–0.768s, excluding embedding setup. Inspection confirms
that in-place chunk KV updates still dispatch over the full cache and copy
unwritten cache positions. Layout and elementwise operations also retain the
physical B4 token footprint for a lone active request. This motivates bounded
KV writes and, next, adapting prepared owner/chunk profiles to actual active
requests. The cost probe and `audit_mistral_prefill_cost_probe.py` are diagnostic
evidence, not a new baseline.

## Contiguous private KV span updates

`GARNET_METAL_CHUNK_KV_SCATTER=ON` adds an optional native update for identical
private cache views with a validated contiguous active prefix in each owner.
Each token/vector then owns a unique cache destination. The encoder dispatches
over the prepared chunk's token vectors rather than every cache position, and
leaves all other cache bytes untouched. Host-completed controls establish the
contiguous-prefix property; unknown controls, gaps, overwrites and separate
output caches retain the existing scan/copy path. The kernel also checks slot
bounds before forming a destination. It does not change dtype or KV ownership.

The option defaults off and requires native resident/chunk in-place updates.
The serving profile explicitly declares `chunk_kv_scatter: true` and
`expected_chunk_kv_scatter_dispatches: 64`. A native counter checks actual
per-stage routing; ordinary one-token decode uses its existing update path.
At B4/H8/D128/C4096/T128, the eligible grid contains 131,072 vectors, versus
4,194,304 vectors in the original cache-wide grid. These are logical dispatch
counts, not measured hardware memory traffic.

`mistral-kv-scatter-operator3` runs control/candidate, then candidate/control,
in four serial processes. Each case has three warmups and nine measurements.
Nine cases cover all four owners, row0, row3, rows0/2, clipped tails, all paused,
gaps/overwrites, separate output caches and untrusted-control fallback. All
603,979,776 full-cache words match an independent CPU reconstruction; the
candidate/control comparisons cover 301,989,888 words. All native owners close
to zero. The second-cohort medians are:

| KV update case | Original GPU ms | Span-update GPU ms | Operator ratio |
|---|---:|---:|---:|
| Four active owners |1.8034 |0.0529 |34.11x |
| Row0 |1.8331 |0.0384 |47.77x |
| Row3 |1.8142 |0.0409 |44.39x |
| Rows0/2 |2.3255 |0.0483 |48.20x |
| Clipped tail |1.3596 |0.0790 |17.20x |
| All paused |1.5657 |0.0281 |55.75x |
| Gaps/overwrites fallback |4.7066 |4.7067 |1.00x |
| Separate output fallback |1.0175 |1.0929 |0.93x |
| Untrusted controls fallback |4.8827 |4.8825 |1.00x |

These large ratios apply only to the KV update operator. Both complete cohorts,
including fallback variation, remain in the raw collection. All 22 native tests
pass with the option on and off; the reduced native reference includes 27 cases
and scalar fallback. Two isolated unsupported CMake configurations are rejected.
`mistral-kv-scatter-verification3` also preserves complete KV and accepted hidden
state on all 24 trained-model target-verification cases, including rollback,
partial spans, EOS and carried state. Every target reports 64 span updates.
Native owners close to zero with 2,031,616 Metal bytes remaining. The clean
library SHA256 is
`8c4074c18e7895041307889a433e95429a927e1d155a67009122d7883819b12e`.

The serving admission `mistral-kv-scatter-native-http-admission1` completes all
22 frozen requests and 4,844 output tokens, with exact frozen native coding IDs
on all eight tasks and no accounting issues. It validates 4,480 span updates in
70 prefill forwards and zero span updates in ordinary decode, across 3,878
stage events. The audit rejects 28 corruptions, including changed KV policy,
wrong dispatch count and missing stage coverage. The extended target structural
audit rejects 42 corruptions while rechecking the prior unchanged causal
generation evidence. Serving owners close to zero with 1,507,328 Metal bytes
remaining; the seven prefill families retain 596,664,320 scratch bytes.

| Nominal input / concurrency | Prior Garnet tok/s | Span-update Garnet tok/s | vLLM-Metal tok/s | SGLang-MLX tok/s | 2x target |
|---|---:|---:|---:|---:|---:|
| 256 /1 |16.655 |17.304 |22.021 |22.042 |44.084 |
| 256 /2 |28.161 |30.712 |31.464 |32.127 |64.254 |
| 256 /4 |40.447 |46.553 |45.235 |42.194 |90.470 |
| 2048 /1 |5.551 |6.185 |12.345 |12.518 |25.036 |
| 2048 /2 |7.658 |8.478 |16.536 |14.482 |33.071 |
| 2048 /4 |9.615 |10.048 |21.179 |16.337 |42.358 |

These are aggregate end-to-end output token rates. Both native columns are
separate one-cohort, no-warmup, single-wave admissions; references remain the
frozen 15-wave medians. System-condition variation, paging/compression and the
native FP32-versus-reference FP16 decode difference limit the comparison. The
short C4 value exceeds the reference medians in this run but overlaps their
retained variability; it does not establish repeated superiority or isolated
causation. Other cells remain below the references. Single-request short/long
TTFT is 1.780s /11.567s and decode is 22.650 /13.937 tok/s. All 2x throughput,
2x decode and half-TTFT targets remain unmet. Strict coding quality is unchanged
at 0/8 by complete output identity. The option remains off by default. Further
prefill work must address the active-owner layout and prepared chunk schedule.

The final selected source, `mistral-kv-scatter-source4`, additionally requires
cache capacity to stay within `kMetalAttentionChunkCapacityLimit` before taking
the span path. Larger low-level caches retain the original implementation.
The reduced reference now covers 28 cases, including an above-profile cache.
All 22 tests pass again with the option on and off, and
`mistral-kv-scatter-verification4` repeats all 24 complete state-parity cases.
The selected library SHA256 is
`850c5d4cb7c7b2b67e20d0085d79beecfa98297cb0e77ac1c431cb57ebd8cec6`.
This guard preserves the qualified <=4k kernel and dispatch behavior; it does
not enable larger contexts.

`mistral-kv-scatter-native-http-admission2` rechecks the selected build on all
22 requests /4,844 tokens, exact frozen coding outputs, 4,480 span updates and
28 corruption rejections. Owners again close to zero with 1,507,328 Metal bytes
remaining. Its single-wave rates are retained separately:

| Nominal input / concurrency | Prior Garnet tok/s | Final span-update Garnet tok/s | vLLM-Metal tok/s | SGLang-MLX tok/s | 2x target |
|---|---:|---:|---:|---:|---:|
| 256 /1 |16.655 |17.182 |22.021 |22.042 |44.084 |
| 256 /2 |28.161 |30.691 |31.464 |32.127 |64.254 |
| 256 /4 |40.447 |46.196 |45.235 |42.194 |90.470 |
| 2048 /1 |5.551 |6.157 |12.345 |12.518 |25.036 |
| 2048 /2 |7.658 |8.418 |16.536 |14.482 |33.071 |
| 2048 /4 |9.615 |10.431 |21.179 |16.337 |42.358 |

Both span-update collections are single-cohort admissions. They do not replace
the frozen 15-wave reference distributions or establish repeated superiority.
The selected build's short/long C1 TTFT is 1.802s /11.620s and decode is
22.557 /13.875 tok/s. The earlier source3 observation and final guard check both
remain available; the small timing differences do not measure the guard's
effect. All speed/TTFT goals remain unmet, with unchanged strict coding quality.

### Bounded vector layout candidate

The next opt-in extension is `GARNET_METAL_LAYOUT_VECTOR4` (default off).
The original chunk layout transfers one FP32 word per thread with 64-bit
division. The candidate transfers four adjacent words with 32-bit indexing.
Eligibility proves source/output column and stride alignment, one mask token
per vector, zero buffer offsets, and bounded address products. Incompatible
columns, token widths or strides retain the scalar path. It preserves reshape
order, padding semantics, and zeroing of inactive tokens. It changes neither
the cache representation nor the xModel layer interface.

The pipeline is compiled during Metal context construction, before engine
readiness. Dispatch adds no buffer, scratch or pipeline allocation. The native
session exposes `layout_vector_dispatches`; a serving profile selects
`"chunk_layout_vector4": true` and
`"expected_chunk_layout_vector_dispatches": 256`. Decode retains zero layout
dispatches in the qualified ordinary Mistral graph. This extension accelerates
transfers; it does not introduce zero-copy reshape aliases or active-owner
prefill graphs.

`mistral-layout-vector-source1` builds pass all 23 native tests with the option
on and off. The new test covers 29 reduced cases per build: reshape directions,
padded rows, all/first/last/gapped/paused masks, GPU-produced masks in the same
command, non-vector fallbacks, alias rejection, uninitialized inputs and abort
recovery. Complete CPU comparisons include signed zeros, subnormals, infinities
and NaN payloads. `mistral-layout-vector-operator1` runs two serial paired cohorts
in opposite variant orders, with three warmups and nine measurements per case.
All 76 cases /111,164,352 words match the independent CPU oracle; candidate and
control compare 55,582,176 words. Owners close to zero.

Example second-cohort isolated GPU medians follow. The retained complete table
includes all scenarios and both cohorts. The synthetic operator results vary
substantially between cases and do not establish model throughput or repeated
serving performance.

| Layout / live mask | Scalar ms | Vector ms | Operator ratio |
|---|---:|---:|---:|
| B4/T128/H4096 unpack /all |1.819 |0.336 |5.42 |
| B4/T128/H4096 unpack /row0 |0.731 |0.310 |2.36 |
| 512x4096 pack /unmasked |0.557 |0.097 |5.72 |
| B4/T128/H1024 unpack /row0 |0.187 |0.015 |12.20 |
| 512x3 scalar fallback |0.007 |0.007 |1.02 |
| B4/T128/H4096 /GPU-produced gapped mask |0.733 |0.079 |9.23 |

`mistral-layout-vector-verification1` passes all 24 native target cases, including
complete KV and accepted-hidden parity with ordinary FP32 decode. A separate
ordinary prefill route probe confirms 256 vector layout dispatches for the
B4/T128 graph with only physical row0 active. This probe is route/cleanup
evidence; the reused diagnostic driver's chosen-offset marker does not mean
that per-operation profiling was enabled. The selected library SHA256 is
`f2d3069b119eb135232edb8a8c7ef879feb763e97efcb3db4ad7a02ec5e4f2a5`.

The corresponding `mistral-layout-vector-native-http-admission1` completes all
22 requests /4,844 tokens with exact frozen native coding IDs on all eight
tasks. It checks 17,920 vector layout dispatches in 70 prefill forwards and zero
in ordinary decode across 3,878 stage events. All 31 injected corruptions are
rejected, including policy, dispatch-count and missing-stage changes for this
extension. Native owners close to zero with 1,769,472 Metal bytes remaining;
the seven prefill scratch families still use 596,664,320 bytes.

| Nominal input / concurrency | Prior KV-span Garnet tok/s | Vector-layout Garnet tok/s | vLLM-Metal tok/s | SGLang-MLX tok/s | 2x target |
|---|---:|---:|---:|---:|---:|
| 256 /1 |17.182 |18.093 |22.021 |22.042 |44.084 |
| 256 /2 |30.691 |30.696 |31.464 |32.127 |64.254 |
| 256 /4 |46.196 |43.746 |45.235 |42.194 |90.470 |
| 2048 /1 |6.157 |6.462 |12.345 |12.518 |25.036 |
| 2048 /2 |8.418 |8.515 |16.536 |14.482 |33.071 |
| 2048 /4 |10.431 |10.469 |21.179 |16.337 |42.358 |

These remain separate single-cohort, no-warmup, one-wave admissions. Short B4
throughput regresses relative to the prior observation, while C1 TTFT changes
from 1.802s /11.620s to 1.389s /10.449s for short/long prompts. C1 decode is
22.423 /13.592 tok/s. System conditions differ; these observations alone do not
isolate layout's effect. All 2x throughput/decode and half-TTFT targets remain
unmet. Strict coding quality remains 0/8 by complete output identity. The
reference engines were not rerun, and the new extension remains off by default.

To examine layout's effect without HTTP/decode dilution,
`mistral-layout-vector-actual-chunk1` runs eight sequential native processes in
two opposite-order cohorts. Each prepares the same B4/C4096/T128 graph, seeds
15 real prompt chunks, then replays the identical late chunk at offset1920
with three warmups and nine measurements. One or four physical owners are live.
Timing includes the native session call and GPU completion, but excludes
embedding, head, token emission, full-state hashing and request scheduling.
This is a steady body-work experiment, not a prompt-latency distribution or
serving throughput benchmark. The first 2,048 tokens of the frozen 2,056-token
context case supply the input; this does not claim complete prompt generation.

| Active owners | Cohort | Control wall s | Vector wall s | Body ratio |
|---|---:|---:|---:|---:|
|1 |0 |0.749 |0.666 |1.125 |
|1 |1 |0.844 |0.729 |1.158 |
|4 |0 |2.231 |2.122 |1.052 |
|4 |1 |2.304 |2.160 |1.066 |

All 72 measured forwards complete. The full 32-layer KV hashes and final hidden
hash match the original native control across both variants and cohorts for
each occupancy. Candidate forwards record exactly 256 vector transfers; the
control records zero. Metal allocations and native owner counts remain stable
during replay, and every process closes with zero owners and less than4MiB
Metal allocation. This supports a bounded prefill improvement under the tested
replay conditions. It does not remove the single-wave HTTP B4 regression or
establish the broader performance targets. `run_mistral_layout_chunk_probe.py`
and `audit_mistral_layout_chunk.py` retain the reproduction and structural
checks; the raw collection preserves its frozen driver and hashes.

### Private cache row-span foundation

`GARNET_METAL_PREPARED_KV_ROW_SPANS` is an optional native preparation primitive,
off by default. It requires resident and chunk in-place cache updates. A private
session-facing helper prepares contiguous spans of an initialized cache's
physical rows using MPS descriptors over the same buffer. It retains the backing
owner, checks geometry/device/storage bounds, and creates no GPU buffer. It is
not exposed as an XLang view API and does not yet activate smaller-owner
prefill graphs or scheduler decisions.

Views share the backing owner's readiness and pending-command identity. A write
through one view makes the parent and sibling views unreadable until completion.
The same command may consume its pending aliases; another context must wait.
Aborting invalidates the complete backing owner and every view. Partial host
updates, zeros or GPU writes cannot recover an aborted owner; complete backing
storage must be restored. Nested views retain the original owner and their
correct relative byte offset. Deferred clears of individual views are rejected.
Custom matrix bindings and host update/read/zero/prefix operations now account
for descriptor offsets, and in-place single-slot KV aliasing requires identical
offsets as well as compatible dimensions/strides.

`mistral-owner-spans-source4` passes 24 native tests with the option on, 23 off,
and 24 with both the option and cached matrix metadata on. The row-span test has
61 cases: 41 complete-cache CPU comparisons across every contiguous subset,
scalar/vector and gap/overwrite paths, and 20 MPS attention cases. Attention is
bit-identical to independently allocated compact cache copies, with 480 sampled
FP64 oracle values. The tests cover the final physical owner, a full
8-KV-head/D128/context4096/chunk128 cache, unchanged rows, host offsets, nested
lifetimes, shared pending state, abort/recovery, and invalid preparation.
Native owners close to zero. A separate positive control confirms completed
cross-context reads/writes succeed and pending input/output rejection follows
the intended readiness checks, rather than an unrelated device/layout error.

The selected native library SHA256 is
`c310501db049737638a5d72e14c95a23678a57a739f6d443b4d540575a543017`.
`mistral-owner-spans-verification4` preserves all 24 complete KV/accepted-hidden
cases. `mistral-owner-spans-native-http-regression4` completes the unchanged
22-request /4,844-token serving workload with exact frozen native coding IDs on
all eight tasks, 31 corruption rejections and zero retained native owners.
Cleanup allocation is 1,769,472 bytes. This remains the existing B4/T128 engine;
the new private views are exercised by native tests, not by compact-owner
serving. No speed gain follows from this regression admission.

| Nominal input / concurrency | Prior vector-layout tok/s | Row-readiness regression tok/s | Frozen vLLM-Metal tok/s | Frozen SGLang-MLX tok/s |
|---|---:|---:|---:|---:|
|256 /1 |18.093 |17.192 |22.021 |22.042 |
|256 /2 |30.696 |28.390 |31.464 |32.127 |
|256 /4 |43.746 |39.907 |45.235 |42.194 |
|2048 /1 |6.462 |6.101 |12.345 |12.518 |
|2048 /2 |8.515 |8.227 |16.536 |14.482 |
|2048 /4 |10.469 |10.131 |21.179 |16.337 |

Both native observations are separate single-wave runs under different system
conditions. This regression run is slower in every cell; it does not isolate
the effect of readiness bookkeeping. C1 short/long TTFT is 1.554s /11.384s and
decode is 21.603 /13.257 tok/s. Strict coding quality remains unchanged at0/8.
All throughput/decode/TTFT targets remain unmet and comparator processes were
not rerun. The new option stays off by default.

The final `mistral-owner-spans-source5` differs only by a CMake dependency guard.
Both missing-prerequisite configurations are rejected; the valid profile's
native compile flags match the tested source4 profile after source-include-path
normalization. Runtime C++/headers/tests are byte-identical. The first guard-log
checker mishandled CMake's wrapped whitespace; the corrected checker and its
rejections are retained. An initial cross-context diagnostic expected the input
error before the existing output-pending guard; the corrected positive control
requires both specific errors and successful completed transfers.

At that foundation checkpoint, remaining integration needed shared-weight B1/B2 graph families and every
permitted cache span before readiness, select them under native owner-epoch
validation, and route final hidden values/predictions to physical rows. First
prove exact owner compaction at the existing chunk size, then qualify wider
chunks separately. Gapped owners, stale leases, release/reclaim, prefix reuse,
running decode peers and allocation bounds still require full runtime tests.
No xMind compact-owner serving or new ANE/model offload is claimed here.

### Prepared compact-owner prefill families

The optional `GARNET_METAL_COMPACT_OWNER_PREFILL` extension now integrates
contiguous B1/B2 prefill families into native leased execution. It defaults off
and requires prepared KV spans, continuous row leases, per-row final hidden,
shared scratch and the bounded dense profile. Dirty/deferred reset and MTP
configurations are rejected. Compact families and speculative target workspaces
cannot coexist in one session. Eight invalid configuration checks preserve
these restrictions, including earlier dependency guards.

`MistralMetalDecoder.prepare_compact_prefill(128)` builds fourteen shared-weight
families, B1/B2 at chunk2/4/8/16/32/64/128, before any execution or row claim.
Native preparation retains graph generations and exact original weight-buffer
identities, prepares every contiguous private KV span and physical last-hidden
view, and reserves shared scratch/output storage. It does not copy weights or
allocate another KV cache. The inventory is21 MPS plans, 1,044,162,560 scratch
bytes and126 context buckets. No shape compilation occurs after readiness.

Native `run_leased` validates physical controls and owner epochs, then selects
the registered compact family for one/two contiguous active owners. Gapped,
three/four-owner, decode, unleased and small FP32 cases keep the B4 path. The
native context now permits last-hidden writes through private prepared spans
with validated offsets. The complete output hidden retains B4 physical rows,
with inactive rows zeroed. A compact prefill has257 vector-layout dispatches,
including physical output routing; the fallback has256. Layer ops remain the
same. Per-request token histories, prefix ownership and predictions use their
original physical rows.

This family stops at chunk128. B1/T512 and B2/T256 selection, packed gapped
owners and fused FlashAttention remain unimplemented. The global512-token upper
bound is therefore still underused when fewer than four requests prefill.
The current MPS attention path separately computes QK, masked softmax and PV;
it retains score/probability intermediates. The pinned vLLM-Metal source packs
scheduled token counts and includes tiled online-softmax attention and an M5
NAX path. Source support alone does not prove the selected kernel in every
saved baseline step or attribute the complete measured gap to that kernel.

`mistral-compact-prefill-source2` passes24 native tests with the extension on and
24 with it off (private row spans remain on). The row-span test additionally
checks last-hidden writes into every contiguous span with live/paused owners.
The first actual-model trial failed a legacy zero-offset last-hidden guard;
that failure is retained. The corrected private-span guard does not relax
numerical checks. Library2 SHA256 is
`f77445b21714e9663974a78b82f7652c41fe251e11f96de044f3d462be18ba96`.
Source3 changes only the post-build Python batch-factory entry point.
The same library also preserves all24 actual-model target-verification cases
at capacities256/4096 without compact families prepared, including full cache,
accepted hidden and rollback parity. This does not qualify simultaneous
compact-family and speculative-target workspaces.

The final twelve actual-model cases compare all32 layers of physical KV,
complete hidden output and row predictions against a separate B4 control.
They include row0/row3, adjacent pairs12/23, gaps, all owners, small fallback,
tail, chunk128, release/reclaim and reset. Missing/stale epochs and late
preparation are rejected. Seven tracked-prefix cases retain exact initial,
returned, appended and changed-prompt continuations while B/C/D each produce
176 unchanged survivor tokens. A row3 capture/restore compares302 tracked
tokens, complete cache hashes and next prediction. All native owners close to
zero. These tests do not establish a new coding-quality score.

The unchanged22-request/4,844-output-token HTTP collection passes accounting and
native route/identity checks, with all eight frozen coding outputs exact.
Its first audit replay failed because the corruption injector changed the first
log event instead of the scheduler-bound event after a new preparation event
was introduced. The corrected auditor selects the named event and rejects all
36 corruptions against the original raw collection; no engine rerun was needed.
The initial failed audit and corrected frozen auditor remain preserved.

| Nominal input / concurrency | Previous Garnet tok/s | Compact-owner Garnet tok/s | Frozen vLLM-Metal tok/s | Frozen SGLang-MLX tok/s | 2x target |
|---|---:|---:|---:|---:|---:|
|256 /1 |17.192 |19.390 |22.021 |22.042 |44.084 |
|256 /2 |28.390 |33.133 |31.464 |32.127 |64.254 |
|256 /4 |39.907 |47.698 |45.235 |42.194 |90.470 |
|2048 /1 |6.101 |7.729 |12.345 |12.518 |25.036 |
|2048 /2 |8.227 |9.725 |16.536 |14.482 |33.071 |
|2048 /4 |10.131 |10.288 |21.179 |16.337 |42.358 |

These native observations are separate single-cohort/no-warmup/one-wave runs;
references remain the frozen15-wave medians. C1 TTFT is1.017s /7.496s and decode
22.829 /14.032 tok/s. The unchanged B4 path also improves in this observation,
so these separate runs cannot isolate compaction's causal effect. Native FP32
decode/KV versus reference FP16 and system paging/compression remain material
differences. All2x/half-TTFT targets are unmet. Strict coding remains0/8 by full
output identity. No comparator was rerun, and no ANE acceleration is measured.

`mistral-compact-prefill-paired-stage1` provides a separate paired body study:
one session, same weights/cache/scratch, control via unleased B4 and candidate
via leased compact selection. Each owner group seeds15 real chunks, then
replays the same chunk at offset1920 three warmup and nine measured times per
variant. Two fresh processes reverse variant order. All108 measured forwards
complete with exact full-cache/hidden hashes across variants and cohorts;
steady Metal allocations/owner counts remain fixed, and cleanup is below4MiB
with zero owners. The independent auditor rejects seven evidence corruptions.

| Active physical owners | Cohort | B4 control wall s | Compact wall s | Body ratio |
|---|---:|---:|---:|---:|
|0 |1 |0.695 |0.587 |1.184 |
|0 |2 |0.778 |0.599 |1.298 |
|3 |1 |0.763 |0.587 |1.299 |
|3 |2 |0.774 |0.594 |1.304 |
|1,2 |1 |1.319 |1.140 |1.157 |
|1,2 |2 |1.313 |1.180 |1.113 |

This timing includes the session call and GPU completion, excludes embedding,
head, hashes, HTTP and scheduling, and replays identical inputs/slots. It proves
a bounded body-work improvement under those conditions, not repeated serving
superiority or the2x end-to-end target. The new driver and auditor are
`run_mistral_compact_prefill_stage.py` and
`audit_mistral_compact_prefill_stage.py`. Raw failed and passing trials,
immutable sources, route traces, hashes and comparison reports are retained
under the local Mistral evidence root.

### Experimental fused FP32 attention

`GARNET_METAL_CHUNK_FLASH_ATTENTION` now supplies an optional fused tiled
online-softmax implementation for the Metal path. It defaults off. The pinned
vLLM-Metal tiled kernel informed register query/output tiling, online softmax
and the8x8 fragment lane mapping. This is a FP32 SIMD-group implementation,
not vLLM's M5 NAX implementation. No FP16 conversion is introduced in attention.
The backend tag is `chunk_flash_attention_fp32_operands` and the native session
exposes `chunk_flash_attention_dispatches`. A serving profile must explicitly
declare `"chunk_flash_attention": true` to use the compiled extension.

The32-query/16-key/D128 tile keeps queries, running output, maxima and
denominators in registers. It reuses the initial shared query storage for
key/value tiles, using16,896 bytes of threadgroup storage. It reads existing
private KV before the chunk's first slot and the appropriate live-token values
after that boundary. Per-query causal masking excludes future live values.
Inactive physical owners and padded queries write zero. Cache/view bindings
honor descriptor offsets and retain existing native readiness/abort rules.

Only completed host controls proving a contiguous active prefix per owner
select the new path. Its native envelope is up to4 owners, D128, heads<=32,
KV heads<=8, chunk8..512 in multiples of8, and the existing global token-row,
context and index32 bounds. Tiny tails, gaps, overwrites, other dimensions and
GPU-pending controls retain MPS/scalar execution. The current compact serving
family still stops at chunk128. Shader pipelines are created during context
construction, and the registered fallback profile must exist before encoding.
There are no per-dispatch GPU allocations. MPS fallback score buffers remain
allocated: the existing21-plan/1,044,162,560-byte scratch inventory is retained.
This experiment removes their traffic on eligible calls, not their reservation.

FP32 reduction order changes, so bit identity against the old MPS policy is
not claimed. The existing independent finite-input FP64 bound remains
`1e-5 + 2e-4*abs(expected)`. Source2 passes all25 native tests. Source3 changes
only the native test, adding unprepared-profile, output-alias, uninitialized
query, pending-read and abort/recovery coverage. Its final15 operator cases
check228,376 FP64 samples and preserve read-only KV; full-profile tests also
check every output word finite and every inactive word zero. The largest
observed error uses0.027 of the fixed bound. The off build passes all24 original
tests. Invalid dependency combinations are rejected by configuration guards.

The selected source2 library SHA256 is
`19bd5f3346bb3e32b5d3a6e004da1bfa0fdc6a11b84d478947cdb7db55ec3d38`.
The expanded source3 test uses byte-identical runtime C++/shader sources. The
initial build lacked the native test's third-party include directory; the
corrected target builds without changing numerical policy.

The actual small greedy control remains exact, and12 compact-owner cases
compare complete physical KV, hidden output and predictions against a B4
control using the same new policy. Seven tracked-prefix cases preserve the
returned/appended/changed continuation checks,528 survivor tokens and a302-token
nonzero-owner restore. Full state and predictions remain exact within that
policy. These checks do not assert equality to old MPS floating-point state.

The unchanged HTTP workload completes22 requests/4,844 tokens and passes39
corruption rejections. All22 complete generated ID sequences, including the
performance requests, exactly match the previous pinned compact128 native run.
The66 non-tail prefill forwards dispatch2,112 fused layer calls; four tails
retain128 scalar calls, and all3,808 ordinary decode forwards have zero fused
attention calls. Strict coding remains0/8 by full output identity; this is not
a new quality score or high-quality-coder admission. No ANE work is involved.

The isolated synthetic operator alternates variants with three warmups and
nine measurements per variant. Examples of median GPU durations follow:

| Owners / cached prefix | MPS ms | Fused FP32 ms | Operator ratio |
|---|---:|---:|---:|
|1 /0 |0.567 |0.137 |4.139 |
|1 /1920 |2.211 |1.948 |1.135 |
|1 /3968 |4.924 |3.954 |1.245 |
|4 /0 |1.972 |0.470 |4.194 |
|4 /1920 |8.837 |6.885 |1.283 |
|4 /3968 |19.973 |14.384 |1.389 |

These operator gains do not establish a whole-model improvement. The separate
paired body study runs eight sequential processes in opposite variant orders,
with the same compact128 families/weights/inputs,15 real seed chunks and
three warmup/nine measured replays of the chunk at offset1920. Across all72
measured forwards, allocations/owners remain fixed; all processes close cleanly.
Complete state hashes are deterministic across cohorts within each policy,
but differ between MPS and fused policies. Post-prefill token predictions match.

| Active owners | Cohort | MPS body wall s | Fused body wall s | Body ratio |
|---|---:|---:|---:|---:|
|1 |0 |0.617 |0.610 |1.011 |
|1 |1 |0.576 |0.568 |1.015 |
|4 |0 |2.326 |2.394 |0.972 |
|4 |1 |2.388 |2.611 |0.915 |

This is repeated identical chunk work, excluding embedding, head, hashes,
HTTP and scheduling. The one-owner benefit is about1%; four-owner body work
regresses. The result does not justify enabling this implementation by default.
Actual model execution includes projection/layout/cache traffic and interactions
not represented by an isolated attention microbenchmark. Their individual
contributions to this regression have not been causally isolated.

| Nominal input / concurrency | Previous compact Garnet tok/s | Fused FP32 Garnet tok/s | Frozen vLLM-Metal tok/s | Frozen SGLang-MLX tok/s | 2x target |
|---|---:|---:|---:|---:|---:|
|256 /1 |19.390 |17.579 |22.021 |22.042 |44.084 |
|256 /2 |33.133 |30.057 |31.464 |32.127 |64.254 |
|256 /4 |47.698 |41.843 |45.235 |42.194 |90.470 |
|2048 /1 |7.729 |6.969 |12.345 |12.518 |25.036 |
|2048 /2 |9.725 |8.864 |16.536 |14.482 |33.071 |
|2048 /4 |10.288 |9.780 |21.179 |16.337 |42.358 |

The fused HTTP observation is slower in every cell. C1 TTFT is1.212s /8.275s;
decode is20.969 /12.603 tok/s. These are separate one-cohort/no-warmup/one-wave
observations under different system conditions; unchanged ordinary decode also
slows, so the entire difference cannot be assigned to attention from these
data alone. Reference columns remain frozen15-wave medians. All2x and half-TTFT
targets remain unmet. The selected extension stays experimental/default-off.

Evidence roots are `mistral-flash-attention-draft1`, immutable source2/3 and
library2, `mistral-flash-attention-admission2`,
`mistral-flash-attention-native-http-admission2`,
`mistral-flash-attention-native-parity3`, `mistral-flash-attention-prefix3` and
`mistral-flash-attention-paired-stage1`. Their independent auditors bind raw
results, native source/library identities, full output sequences and process
ordering. Comparator binaries/results were not rerun or replaced.

Remaining work includes prepared B1/T512 and B2/T256 selection under the same
four-owner/global512 upper bound, projection/layout traffic and M5-specific
attention execution. A wider compact input must not require a padded B4/T512
GPU graph, which would exceed that global bound. Native scheduling must select
an admitted family from current owner spans and remaining token budget, while
retaining the existing narrow fallback for gapped or mixed work. No speed gain
or correctness admission for those future profiles is established here.
