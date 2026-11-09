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

### Prepared wider owner profiles and native adaptive chunk selection

The subsequent optional `GARNET_METAL_WIDE_OWNER_PREFILL` extension now prepares
B1/T512 and B2/T256 without constructing a B4/T256 or B4/T512 GPU model graph.
It requires compact owner families and a compiled dense512-token upper bound;
it defaults off. The unchanged four physical KV owners remain native runtime
state. Contiguous one/two-owner spans select their prepared compact programs;
gapped and three/four-owner work retain up-to128 fallback programs. A wide
physical host input is copied to the selected compact row span before dispatch.
The fallback graph still has B4/min(chunk,128) GPU geometry, so unsupported
wide spans are rejected before execution rather than silently overrun it.

Native `set_prefill_chunk(512)` supplies an upper bound. After reserving stable
row leases, the scheduler selects a registered width using remaining prompt
lengths, physical contiguity and the global512-token tick budget minus actual
decode tokens. For example, one long prefill with three decodes uses256 rather
than512. Long remainders choose a power-of-two width no larger than the
remainder/maximum; tails up to128 choose the smallest covering prepared width.
Head-only prefix hits contribute no live prefill rows. Deferral retains leases.
The portable scheduler's selector is optional; existing callers keep their
previous policy. Layer operations are unchanged and execution performs no new
graph compilation. `prefill_profile_graphs(chunk)` retrieves prepared handles.

The wider inventory is17 compact families,24 MPS attention plans,
1,885,120,512 scratch bytes and144 context buckets, compared with14/21/
1,044,162,560/126 for compact128. Preparation reports15,918,710,784 tracked
Metal bytes on the M5/24GiB test host. These are device-resource snapshots,
not peak memory or system free RAM. The selected immutable source2 library
SHA256 is `e49ad5d051d0f828c96f653d826e39cc53602dbb40aa15530bf65199e784dfb4`.
Flash attention is off throughout this admission. INT4g64 coefficients/prefill
projection operands remain FP16; attention/KV/decode remain FP32. Reference
engines use FP16 state. Different chunk boundaries can reassociate arithmetic.

The enabled build passes25 native tests; disabling only the wider extension
passes all24 legacy tests. The new CPU policy test covers30 planning/lifecycle
cases. The checked C++ host `tensor_io.rows(source,first,count)` helper preserves
contiguous FP32 bytes without Python list expansion:31 positive cases and18
invalid bounds checks pass. This helper copies transport data and does not
change GPU ownership or expose an alias into private KV.

Nine actual-model cases compare every physical KV word through hashes, hidden
output and routed predictions against independent B1/B2 native caches using
the same immutable programs/weights. They cover rows0/3 at512, appended256,
a short tail, reclamation/reset, adjacent rows1/2 and2/3 at256, and unequal
valid lengths. All match exactly within each selected program. Five negative
cases reject stale reclaimed epochs, gapped wide work, two512-token owners,
unleased wide execution and preparation after execution, with unchanged caches.
The first parity attempt failed in its host slice harness before inference;
the corrected row-copy probe and its failed predecessor are both retained.

The real-weight native scheduler test completes nine jobs across75 batch
steps. A/B/C/D start together, E waits for A's row, and returned A reuses that
row with another fresh epoch while B/C/D still decode. It also exercises
solo512 and adjacent-pair256 scheduling, checks the global budget including
decode, and hashes inactive physical KV rows before/after each prefill. All
162 emitted IDs match frozen native controls. The seven-case prefix probe
also preserves exact return, appended/changed continuation and528 survivor
tokens. Neither test establishes serving speed or coding quality.

The unchanged HTTP workload completes22 requests/4,844 output tokens, matches
all22 prior compact128 generated-ID sequences and passes43 corruption
rejections. The auditor independently checks prepared width choice and the
combined decode/prefill tick budget. Strict coding remains0/8 by full output
identity; matching IDs is not a new quality score. No ANE work is involved.

| Nominal input / concurrency | Compact128 Garnet tok/s | Adaptive Garnet tok/s | Frozen vLLM-Metal tok/s | Frozen SGLang-MLX tok/s | 2x target |
|---|---:|---:|---:|---:|---:|
|256 /1 |19.390 |19.230 |22.021 |22.042 |44.084 |
|256 /2 |33.133 |32.710 |31.464 |32.127 |64.254 |
|256 /4 |47.698 |46.594 |45.235 |42.194 |90.470 |
|2048 /1 |7.729 |7.443 |12.345 |12.518 |25.036 |
|2048 /2 |9.725 |9.459 |16.536 |14.482 |33.071 |
|2048 /4 |10.288 |10.916 |21.179 |16.337 |42.358 |

C1 TTFT is0.919s /7.924s and decode22.188 /13.724 tok/s. The native columns
are separate one-cohort/no-warmup/one-wave observations; reference columns are
the unchanged15-wave medians. Five cells regress from the earlier native run
and one improves under different system conditions. These data do not establish
an isolated causal benefit or repeated superiority. All2x and half-TTFT targets
remain unmet. Wider shapes are a tested scheduling option, not a demonstrated
performance improvement.

The paired full-prefill study uses one prepared engine per process, the same
first2048 prompt tokens, the same weights/physical owners and all17 families.
Two serial processes reverse variant order. Each group/variant has one warmup
and three complete cold-cache prefills, for24 measured full prefills. Reset,
row claims, input construction, vocabulary head and hashing are outside timing.
Sampled Metal allocation remains fixed. Full KV hashes are deterministic across
both cohorts within each width; final predictions agree across widths. Chunk
boundaries can change FP32 state, so cross-width bit identity is not claimed.

| Active owners | Cohort | Compact128 body s | Wider body s |128/wider ratio |
|---|---:|---:|---:|---:|
|1, wider512 |1 |7.880 |8.052 |0.979 |
|1, wider512 |2 |8.181 |8.301 |0.986 |
|2, wider256 |1 |15.522 |16.143 |0.962 |
|2, wider256 |2 |15.950 |16.197 |0.985 |

Wider profiles regress1.5–4.0% in this bounded body study. In the first cohort,
GPU time accounts for7.805 of7.880 seconds at one owner and15.432 of15.522
seconds at two owners for compact128. The measured gap is primarily inside GPU
execution, rather than scheduler/HTTP overhead. This does not identify the
individual cause or establish performance on other profiles/devices.

A separate `GARNET_METAL_PROFILE_OPS=1` diagnostic instruments only the final
chunk in the same2048-token prefill, at widths128 and512. Their cached-prefix
lengths are1920 and1536 respectively. Both retain the paired
control's post-prefill predictions. This existing mode serializes operations,
adds waits and disables several fusions. Its times describe that diagnostic
policy; they are not an ordinary production trace or serving measurement.

| Diagnostic width | Projection GPU ms | Attention GPU ms | Total diagnostic GPU ms | Projection fraction |
|---|---:|---:|---:|---:|
|128 |371.69 |71.92 |470.91 |78.9% |
|512 |1467.73 |442.51 |2047.08 |71.7% |

Each diagnostic observes224 matrix projections and32 chunk-attention calls.
These results make quantized projections a priority alongside attention;
they do not prove an attention-only change can meet the2x whole-model target.
The deduplicated prepared resource inventory also exposes13,830,839,840 bytes
through declared graph/session owners:4,002,430,976 weight bytes,
4,294,967,296 private KV bytes and5,533,441,568 other declared bytes. Of the
latter,4,922,081,280 bytes are graph-only FP32 writer slots without a shared
scratch-pool reference. Context scratch/driver resources may be absent from
this inventory. Conservative resident references are not proof that these
buffers can safely be freed or aliased; their producer/consumer lifetimes need
a native proof before further memory reuse.

Evidence roots are `mistral-wide-owner-draft1`, immutable source2/library2,
`mistral-host-tensor-rows1`, `mistral-wide-owner-native-parity2`,
`mistral-wide-owner-native-http-admission1`, `mistral-wide-owner-scheduler1`,
`mistral-wide-owner-prefix1`, `mistral-wide-owner-paired-stage1` and
`mistral-wide-owner-op-diagnostic1`. The enabled/disabled builds, failed earlier
harness, raw serving streams, opposite-order body runs and diagnostic logs
remain separate. Comparator results are frozen and were not rerun. Remaining
work includes projection efficiency, proved scratch/hidden reuse and M5-specific
attention, with all performance/quality targets still open.

### Reusing completely written attention output buffers

The optional `GARNET_METAL_CHUNK_ATTENTION_SCRATCH` extension adds chunk
attention to the native scratch planner's complete-write initialization proof.
It defaults off and requires the existing shared scratch, MPS attention,
index32 and context-bucket extensions. The backend tag is
`chunk_attention_full_write_scratch`; a serving profile explicitly declares
`"chunk_attention_scratch": true`. Layer operations, attention equations,
precision, physical KV owners and cache capacity are unchanged.

The planner previously omitted this producer even though its scalar kernel,
MPS merge and optional fused kernel write every logical query/head/dimension,
including zero for inactive queries. Other eligibility checks are retained:
default physical layout, protected inputs/weights/returned parts, first full
write before any read, unique storage in the non-alias path, and the scratch
owner's serialization lease through GPU completion. Pool bindings are prepared
before readiness. No new per-dispatch scratch allocation or graph compilation
is introduced. This change reuses attention result storage across prepared
layers; private KV never enters the scratch pool.

The new native test poisons outputs with NaNs and a nonzero finite value before
two executions of each of nine cases. It checks16,789,504 output words for full
finite writes,3,037,824 inactive zero words and1,514 independent FP64 samples
under the original `1e-5 + 2e-4*abs(expected)` bound. The largest observed error
uses0.00414 of that bound. Cases include B1/T512 initial/cached attention,
B2/T256 unequal lengths, B4/T128 gaps, scalar tails, all-inactive work,
overwrites, pending controls and gating. KV remains read-only and native
allocation owners/plans remain fixed. The first build caught a mixed-type
`auto` declaration in this test; immutable source2 fixes only that test line
relative to source1. Both the failure and corrected build remain available.

The selected source2 library SHA256 is
`c415b745410a5034a69bfef6b473df971278911b42061a9d8a76aff97b6eb4e8`.
It passes all26 native tests. Enabling the optional Flash extension passes27
native tests, including the poison checks and existing fused numerical gates;
that combination has operator evidence only, not a new full-model admission.
Disabling attention scratch passes all25 prior native tests. The actual model
and serving gates below keep Flash off.

At17 compact families, preparation falls from15,918,710,784 to14,332,182,528
tracked Metal bytes, saving1,586,528,256 bytes. At the14-family compact128
serving profile, ready-state allocation falls from12,824,494,080 to
11,909,054,464 bytes, saving915,439,616 bytes. The latter still has21 MPS plans,
1,044,162,560 MPS scratch bytes and126 context buckets. These are sampled
resource allocations on this M5/24GiB host, not system free RAM or peak memory.
The separate MPS attention scratch remains reserved.

All nine real-weight full-state cases exactly match the previous immutable
library's complete physical KV, hidden output and predictions, and also match
their independent B1/B2 cache controls. Five invalid/stale dispatch cases retain
unchanged KV. Dynamic scheduling again completes nine jobs/75 batch steps with
162 frozen IDs, including E taking A's row and returned A overlapping three
live survivors. Seven token-prefix cases preserve exact return, appended and
changed continuations, including528 survivor IDs. These tests do not establish
coding quality or serving speed.

The unchanged uncached compact128 HTTP workload completes22 requests/4,844
output tokens and passes37 corruption rejections. All22 complete generated-ID
sequences match the prior compact128 collection. The auditor also checks the
explicit compiled scratch policy. Strict coding remains0/8 by full output
identity, without a new quality evaluation. No ANE work is involved.

| Nominal input / concurrency | Prior compact128 Garnet tok/s | Attention scratch Garnet tok/s | Frozen vLLM-Metal tok/s | Frozen SGLang-MLX tok/s |2x target |
|---|---:|---:|---:|---:|---:|
|256 /1 |19.390 |16.724 |22.021 |22.042 |44.084 |
|256 /2 |33.133 |31.694 |31.464 |32.127 |64.254 |
|256 /4 |47.698 |42.889 |45.235 |42.194 |90.470 |
|2048 /1 |7.729 |7.229 |12.345 |12.518 |25.036 |
|2048 /2 |9.725 |9.162 |16.536 |14.482 |33.071 |
|2048 /4 |10.288 |10.314 |21.179 |16.337 |42.358 |

C1 TTFT is1.094s /8.259s and decode19.398 /13.466 tok/s. Native entries are
separate one-cohort/no-warmup/one-wave observations under different system
conditions; references remain frozen15-wave medians. Five native cells regress
and one is essentially flat. These data do not establish a speed gain or
isolated causal regression. The resource saving is independently measured;
performance still needs a same-work comparison. All2x and half-TTFT targets
remain unmet, and attention scratch remains optional/default off.

A paired body study holds compact128 geometry, weights, inputs, context and
owner identities fixed while changing the native library. Four serial processes
run control/candidate/candidate/control. Each has one warmup and three measured
complete2048-token prefills at one and four owners, for24 measured full
prefills. Reset/claims, input construction, vocabulary head and hashes are
outside timing. The independent audit requires complete KV/hidden/prediction
identity across all four processes and sampled allocation stability within
each group; both pass. Each candidate preparation saves915,439,616 bytes.

| Active owners | Control / candidate cohorts | Control body s | Candidate body s |Control/candidate ratio |
|---|---|---:|---:|---:|
|1 |1 /2 |7.466 |9.158 |0.815 |
|4 |1 /2 |35.495 |37.461 |0.948 |
|1 |4 /3 |9.468 |9.351 |1.012 |
|4 |4 /3 |33.963 |39.879 |0.852 |

One-owner measurements vary with run order. Four-owner work regresses in both
orders, by5.5% and17.4% in wall time. These data reject a consistent speed-win
claim for the selected implementation. They do not independently isolate GPU
resource hazards, clock behavior or system pressure. A single `pmset` snapshot
reported no recorded thermal/performance warning and low-power mode disabled;
it is not a frequency/temperature trace or evidence of constant conditions.
The implementation remains an optional memory-capacity experiment, off by
default. Reducing temporary allocation does not solve the projection bottleneck
identified by the previous diagnostic.

Evidence roots are `mistral-attention-scratch-draft1`, immutable source2/library2,
`mistral-attention-scratch-native-parity1`,
`mistral-attention-scratch-scheduler1`, `mistral-attention-scratch-prefix1`,
`mistral-attention-scratch-native-http-admission1` and
`mistral-attention-scratch-paired-stage1`. Enabled, Flash-combination and disabled
native builds remain separate. Frozen comparators were neither rerun nor
replaced. Further work must improve the quantized projection kernels and
validate whole-model speed; no2x, half-TTFT, quality, ANE or renewed27B admission
is established by this checkpoint.

At that checkpoint, a separate next-step source prototype changed the quantized projection tile to
reuse an unpacked weight block across128 rows with eight SIMD groups/256
threads, retaining two strict FP32 accumulators per SIMD group and FP16
operands. Its evidence at that checkpoint was offline Metal4 syntax/type compilation only.
The subsequent native integration and measurements are recorded below. Increasing the projection tile is distinct
from the already-tested larger scheduler chunks; no gain is claimed for this
prototype. Its local source/compile evidence is `mistral-reuse128-design2`.


### Preparing a128-row cooperative projection tile

`GARNET_METAL_MPP_AFFINE_REUSE128` adds an optional native128x64 projection
tile with256 threads/eight SIMD groups. It defaults off. The existing64x64
kernel uses128 threads/four groups. Both stage the same9,216-byte FP16 weight
tile and keep two strict FP32 tensor accumulators per SIMD group. The new
kernel shares unpacking across twice as many input rows while retaining the
ordered reduction, FP16 coefficient/operand conversion and FP32 outputs.
Layer operations and xModel APIs are unchanged.

The candidate requires prepared affine profiles, bounded32-bit indices, strict
FP16 coefficients/operands and the cooperative64 fallback. Relaxed arithmetic
and the old128-thread compiler limit are rejected. Runtime device checks require
SIMD width32,256 supported threads and sufficient threadgroup memory. Libraries
and bounded pipelines are prepared before readiness; a missing or undersized
candidate profile throws before any output zeroing or projection is encoded.
The profile remains an up-to bound. Selection uses actual active rows128..512,
the compiled output threshold and an even number of64-row tiles, avoiding an
extra padded64-row tile. Small counts, odd tile counts and narrow outputs keep
the previous route. No dispatch-time compilation or new KV layout is added.

Backend tags report `mpp_affine_reuse128_optional`, minimum rows128,256 threads
and the even64-tile policy. Serving profiles explicitly require these values.
Separate native counters retain actual64-row versus128-row dispatch identity;
the existing64-row counter is not relabeled to include both. The new
`garnet_reuse128_affine_test --benchmark NEW_DIRECTORY` executable compares the
old control entry point with normal candidate selection in one native context.
The older `garnet_affine_fp16_projection_benchmark` assumes the64-only route and
is not qualified for a reuse128-enabled build; use the dedicated executable.

Immutable source1/library1 passes26 native tests with the extension enabled
and25 with it disabled. Library SHA256 is
`fc31978830d31cb17a1fc5460988ccf178e16224fa8e7f48623796c48f25ea5e`.
All full-model gates keep attention scratch and Flash off. The17 primitive cases
cover zero/small counts,128/129/160/192/193/255/256/384/512, shuffled physical
rows, a511-row ragged layout,8193 outputs and the narrow-output fallback.
Complete old/new words, finite/inactive zeros and1,450 independent FP64 samples
pass. Five guards cover preparation before writes, duplicate/out-of-range rows,
uninitialized input and abort/recovery. Native owners and prepared pipeline
counts remain fixed.

The separate alternating operator study uses real Mistral projection dimensions
with synthetic data, three warmups and nine measured executions per variant.
All32 cells pass96,468,992 complete output-word comparisons and3,040 FP64
samples. The selected128/256/512-row cells show the following old/new GPU
latency ratios. A ratio above1 means the candidate was faster.

| Outputs / inner |128 rows |256 rows |512 rows |
|---|---:|---:|---:|
|4096 /4096 |1.038 |1.088 |1.052 |
|14336 /4096 |1.028 |1.032 |1.051 |
|4096 /14336 |1.035 |1.083 |1.057 |

These are isolated operator results, not model throughput or a2x admission.
Nine actual-weight full-state cases exactly match the previous immutable
64-row library and their independent controls, including complete physical KV,
hidden output and predictions. Five invalid/stale cases retain state. Dynamic
scheduling preserves nine jobs/75 steps/162 IDs, and seven prefix cases preserve
528 survivor IDs, including returned, appended and changed requests.

The uncached compact128 serving campaign completes22 requests/4,844 output
IDs, all exactly matching the previous compact128 control. The auditor validates
actual projection routing for70 prefill batches and rejects37 corruptions,
including a swapped64/128 route with unchanged total counts, wrong thread count,
missing route event and changed policy. Strict coding remains0/8 by unchanged
output identity; these checks do not establish improved coding quality.

| Nominal input / concurrency | Prior compact128 Garnet tok/s | Reuse128 Garnet tok/s | Frozen vLLM-Metal tok/s | Frozen SGLang-MLX tok/s |2x target |
|---|---:|---:|---:|---:|---:|
|256 /1 |19.390 |15.988 |22.021 |22.042 |44.084 |
|256 /2 |33.133 |27.859 |31.464 |32.127 |64.254 |
|256 /4 |47.698 |40.143 |45.235 |42.194 |90.470 |
|2048 /1 |7.729 |6.773 |12.345 |12.518 |25.036 |
|2048 /2 |9.725 |8.654 |16.536 |14.482 |33.071 |
|2048 /4 |10.288 |9.745 |21.179 |16.337 |42.358 |

Single-owner TTFT is1.297s /8.461s and decode18.991 /12.186 tok/s. This is
one native cohort/no warmups/one measured wave per cell, collected separately
from the prior native run and frozen15-wave reference medians. All six native
rates are lower than the earlier collection. Those different-condition samples
do not isolate the kernel's causal effect. Native KV/decode remains FP32 while
the reference state is FP16. All2x/half-TTFT targets remain unmet; the candidate
remains optional/default off pending full-model performance qualification.


A separate full-model body study holds compact128 geometry, actual weights,
2,048-token inputs, context and owner identities fixed. Four serial processes
run control/candidate/candidate/control. Each has one warmup and three complete
measured prefills for one and four owners, for24 measured full prefills. Input
construction, reset/claims, vocabulary head and full-state hashes are outside
timing. Every chunk asserts160 eligible old64 or new128 projection dispatches;
complete KV/hidden/prediction identities agree across all four processes.
Sampled allocation is stable and both variants prepare12,824,494,080 Metal
bytes. No resource-capacity gain is claimed.

| Active owners | Control / candidate cohorts | Control body s | Candidate body s |Control/candidate ratio |
|---|---|---:|---:|---:|
|1 |1 /2 |8.620 |9.140 |0.943 |
|4 |1 /2 |36.268 |32.223 |1.126 |
|1 |4 /3 |9.382 |8.246 |1.138 |
|4 |4 /3 |42.407 |32.143 |1.319 |

Four-owner body work is faster in both process orders, with observed ratios
1.126 and1.319. One-owner results change direction. The two control processes
also differ substantially, so these ratios do not independently isolate kernel
speed from time-varying system conditions. This is body-only evidence, not a
repeated serving improvement or a2x admission. The default remains off; stronger
within-context comparison and a qualified serving gain are still required.
The independent stage auditor additionally rejects six corrupted collections:
changed compiled policy, swapped native route counts, changed complete state,
growing allocation, missing sample and unfinished process.

Four invalid CMake configurations reject the candidate when cooperative64 or
bounded index32 is disabled, relaxed cooperative arithmetic is requested, or
the128-thread compiler cap is enabled. The prior wide/scratch HTTP collections
also retain their43/37 corruption rejections under the updated auditor. No
comparator inference was rerun. Evidence roots are `mistral-reuse128-draft1`,
immutable source1/library1, `mistral-reuse128-operator1`,
`mistral-reuse128-native-parity1`, `mistral-reuse128-scheduler1`,
`mistral-reuse128-prefix1`, `mistral-reuse128-native-http-admission1` and
`mistral-reuse128-paired-stage1`. All2x decode/throughput, half-TTFT and quality
targets remain open. No ANE, Gemma, renewed27B or xMind production admission is
established by this checkpoint.


### Alternating prepared projection routes in one native session

The earlier separate-process comparison has substantial timing variation. A
verification-only native control now permits the full model to alternate the
already-prepared64-row and128-row projection routes without changing its engine,
weights, buffers, owner layout or graph compilation. This is a diagnostic method,
not a new production layer operation or a serving performance admission.

The control is compiled only when both `GARNET_METAL_MPP_AFFINE_REUSE128` and
`GARNET_METAL_VERIFY_ATTENTION` are enabled. Normal selection remains unchanged.
`session.verification_set_affine_reuse128(bool)` locks the session and changes
only its MetalContext's route policy. The native context rejects selection
while a command batch is open or partial submissions remain undrained. It does
not recompile kernels, allocate tensors, change precision or transfer cache
ownership. Both bounded pipelines must already be prepared before execution.
The backend capability tag is `verification_affine_route_control`.

Immutable alternating source1/library1 passes all26 native tests. Its library
SHA256 is `600fd761015a6a508eb4b5d9f3936abfafd5a95060e1ab85374a4214398e39f0`.
The expanded primitive guard checks unchanged complete outputs under toggling,
rejection during open and partially submitted work, unchanged policy after
rejection, independent context policy and recovery after abort. All prior
native numerical tests retain their original bounds. A separate build with
verification disabled compiles and passes the dedicated projection test; that
build does not expose the new diagnostic control.

`run_mistral_reuse128_alternating.py` prepares compact128/B4/context4096 once,
then measures one, two and four active owners with actual2,048-token inputs.
Each owner group warms both routes once, followed by four paired runs in
control/candidate, candidate/control, control/candidate, candidate/control order.
Every run starts with reset private KV and uses all16 chunks. Route selection,
reset/claims, embedding/input construction, vocabulary head and complete state
hashes are outside timing. Actual native counters, no profile misses, fixed
prepared pipeline count and stable sampled allocation are checked for every
run. The independent audit requires all30 runs, including six warmups and24
measured full prefills, balanced pair order and complete state/prediction parity.

The reference compiler-setting hypothesis was also checked against MLX0.32.4's
installed headers and pinned upstream JIT source. Its qmm NAX builder uses the
default compile options, whose math mode is Safe; the device maps that to Metal
Safe mode. This does not support a claim that an unmeasured general Fast-mode
setting explains the gap. The NAX tensor descriptor's relaxed flag was already
measured separately in the earlier negative experiment. This source inspection
is not a historical dispatch trace or an AOT build qualification. Sources and
hashes are retained in `mistral-math-mode-source-inspection1`.

The completed independent audit passes all24 measured full prefills and six
warmups, with exact full KV/hidden/prediction agreement. One- and four-owner
states also match the earlier immutable control library. Preparation remains
12,824,494,080 tracked Metal bytes and55 specialized affine pipelines. All
per-chunk native route counts match selection and no runtime profile misses or
pipeline-count growth occur. Eight corrupted studies are rejected for changed
state, wrong route, allocation growth, pair order, missing run, pipeline growth,
warmup classification and owner coverage.

| Active owners | Control median body s | Candidate median body s |Median paired ratio | Paired ratio range |
|---|---:|---:|---:|---:|
|1 |8.757 |8.428 |1.031 |1.009–1.045 |
|2 |18.167 |17.183 |1.050 |1.003–1.070 |
|4 |35.200 |36.232 |1.021 |0.917–1.033 |

The paired-ratio median differs from the ratio of independent duration
medians. For four owners the candidate duration median is higher, despite a
paired-ratio median above1, because one pair regresses materially. Across all
four pairs, total control/candidate body time ratios are 1.029 at1 owners, 1.043 at2 owners, 0.996 at4 owners.
These summaries retain the slower pair and do not establish a dependable
four-owner gain.

Ratios are control/candidate body wall time within each adjacent pair. They
are not serving token rates. One- and two-owner observations show modest gains;
four-owner pairs change direction. This closer comparison does not confirm the
earlier separate-process1.32x observation as a dependable kernel advantage.
Four pairs per owner in one process remain limited evidence across system
conditions. The extension stays optional/default off. Frozen reference results
are preserved and were not rerun; no new2x or half-TTFT admission follows.
Coding-quality targets remain open.

Evidence roots are `mistral-reuse128-alternating-draft1`, immutable alternating
source1/library1 and `mistral-reuse128-alternating-stage1`. The next layout
hypothesis is recorded separately in `mistral-tiled-affine-layout-design1`:
repack the native INT4 payload and coefficient bits into64-output/64-inner
tiles before readiness, with no duplicate full GPU weight copy. A24-case host
address bijection/roundtrip probe passes. No C++ runtime integration, GPU
dispatch, numerical qualification, allocation measurement or speed gain is
established for that prototype; decode locality and all fallback kernels must
be checked before selection.

### Rejected native tiled weight layout and decode staging

A separate native experiment repacks the GPU representation of affine INT4
weights and FP16 coefficient bits into64-output/64-inner tiles during
preparation. Checkpoint files and xModel layer operations are unchanged. Only
one GPU weight representation is retained. Ragged output rows are zero-padded;
the host packer validates byte lengths, overflows and overlaps before writes.
Specialized index32 proofs include padded extents, and uploads require a
closed/drained context plus the existing256MiB working-set margin. Every
affine shader, including ordinary/indexed decode and fallback paths, uses the
same representation-specific address helpers.

The host test passes24 cases/179,712 checked bytes and17 rejection cases.
Source1 passes27 native tests, including the previous numerical bounds. Its
decode kernels use the new addresses with their original dispatch structure.
Although some128-row Q/O/down operator cases improve, decode becomes much
slower. Source2 adds a matching FP32 decode kernel:128 threads cooperatively
stage32 output channels and256 inner elements into4,608 bytes of threadgroup
memory. Each output retains the original word/lane sequence, dequantization
FMA, FP32 accumulation and SIMD sum. The candidate reports its actual32-output
decode geometry instead of the older four-row reuse capability.

Source2 also passes27 native tests; the patched source with the extension
disabled passes all26 original tests. Source2 library SHA256 is
`14a683b28c642d895ac2bab16aef602d67922c029e97278da70c6d014223f51d`.
Neither numerical bounds nor precision were relaxed. No source/build artifact
replaces the protected Qwen library.

Each operator study runs four serial control/candidate/candidate/control
processes for prefill and another four for decode. Each process uses three
warmups/nine measurements per cell. Forty indexed prefill cells cover real
Mistral dimensions at4..512 actual rows;180 decode cells cover dense1..4 rows,
every four-row active mask and ragged/small/real model dimensions. The prefill
driver explicitly selects the64-row cooperative policy in both libraries so
weight layout is the variable being tested. Per study,374,274,636 complete
output-word comparisons pass against the immutable row-major control, along
with the original independent FP64 checks. Eight corrupted collections are
rejected, including changed process provenance, oracle bounds, route counts,
timed sample coverage and an altered output byte.

The matching decoder does not fix the performance regression. Hot synthetic
operator timings do not establish whole-model behavior, so a separate actual-
weight stage study checks both process orders at batches1 and4. It uses one
fixed84-token coding prompt and32 generated IDs, with three repetitions per
process. The first repetition is excluded from summarized timings. All24
sequences/1,920 token observations match the frozen control. Nine additional
real-model cases match the previous library's complete physical KV, hidden
state and predictions; five invalid/stale dispatch cases remain rejected.

| Batch | Control/candidate cohorts | Control direct decode tok/s | Candidate direct decode tok/s |Speed ratio |
|---|---|---:|---:|---:|
|1 |1 /2 |27.011 |23.583 |0.873 |
|1 |4 /3 |26.659 |23.581 |0.885 |
|4 |1 /2 |87.323 |63.295 |0.725 |
|4 |4 /3 |82.533 |62.514 |0.757 |

These are direct native stage rates, not HTTP throughput or the frozen
vLLM/SGLang benchmark cells. Rows repeat one prompt; this is not a new coding-
quality evaluation or a long-context qualification. Candidate decode speed
falls12–13% at batch1 and24–28% at batch4. The small short-prefill gain does
not justify that loss. No serving, continuous-refill or prefix-cache admission
is claimed for the candidate.

The experiment is rejected for selection. The active runtime is restored
byte-for-byte, preserving the pre-existing unrelated work. Its implementation
is retained as `test2026/metal/experiments/mistral-tiled-affine-layout.patch`,
against clean commit `ce3fa91816fdc2948a7a7e1af717a0d443a99f44`. Apply it only
in an isolated checkout, with `GARNET_METAL_AFFINE_TILED_WEIGHTS=ON` plus the
FP16/index32 affine dependencies. Patch replay reconstructs all15 candidate
files and344 compiled native source files exactly. The immutable source/build
snapshots, failed-to-improve source1 variant and complete raw outputs remain
under `mistral-tiled-affine-*`. No vLLM/SGLang inference was rerun.

### Weight-traffic estimate and history-only speculative feasibility

The pinned checkpoint contains4,001,366,016 bytes across225 linear payloads
and their scale/bias fields. Apple lists153GB/s memory bandwidth for the base
[M5 MacBook Pro](https://support.apple.com/en-us/125405). Assuming one complete
linear-weight read per ordinary autoregressive output gives an optimistic
streaming estimate of38.24 tok/s, before KV, activation, compute or launch
costs. The short C1 target of50.134 tok/s would require200.60GB/s for those
weights alone under that assumption. This is an analytical estimate, not a
measured traffic trace or a proof of a universal hardware ceiling: cache
residency and sustained bandwidth are unmeasured. It motivates amortizing
target-weight reads through multi-token verification rather than expecting
kernel scheduling alone to deliver the full single-request target.

The frozen vLLM-Metal proposer wraps history-only n-gram drafting with miss
throttling and target verification; the frozen SGLang source also exposes a
native n-gram corpus path. Source inspection does not mean either was enabled
in the frozen comparator runs. No comparator configuration or baseline is
changed.

A fixed native-runtime-independent replay now tests a simple local policy:
match the longest3..5-token suffix in the current request's committed history,
copy up to three following known tokens, and score acceptance against saved
future target IDs only after proposal construction. No future IDs enter the
proposal function, and no cross-request or external corpus is used. Maximum
draft length3 matches the existing four-token verification bound. The count
model assumes free drafting and a verification call as cheap as one ordinary
decode call; it is an optimistic call-compression bound, not measured speed.

| Frozen output case | Optimistic target-call compression |
|---|---:|
|merge-closed-intervals |1.166 |
|stable-unique-generator |1.104 |
|deterministic-toposort |1.068 |
|half-open-subtraction |1.070 |
|incremental-length-frames |1.121 |
|bounded-lru |1.129 |
|patch-json-pointer |1.068 |
|repair-bisect-left |1.574 |
|p256-c1-o128 |1.058 |
|p2048-c1-o128 |1.041 |

The throughput prompts obtain only1.058x /1.041x even under those optimistic
assumptions. This rules out this fixed prompt-lookup policy alone as a route
to2x on those saved sequences. It does not establish actual KV rollback,
runtime correctness, coding quality or latency. A stronger learned draft or
proposal mechanism remains necessary to pursue; the existing authoritative
target verifier and tokenizer alignment still require end-to-end qualification.
Evidence is `mistral-speculative-feasibility1`. All2x decode/aggregate, half-
TTFT and coding-quality targets remain open; ANE, Gemma and renewed27B work
are not admitted by this checkpoint.


### Native cross-tokenizer learned drafting and compact resynchronization

A new causal experiment uses the already-pinned `Qwen/Qwen3-0.6B` BF16
checkpoint, revision `c1899de289a04d12100db370d81485cdf75e47ca`, as a
proposal model for the same affine INT4 Mistral target. All six draft assets
match their retained download-manifest hashes. This is an inference experiment,
not a claim that the draft is the best coding model or an independently
qualified coding engine. No comparator inference is rerun.

The target remains authoritative. The draft receives the original user
messages in its pinned, thinking-disabled Qwen chat template plus only the
already-emitted Mistral continuation. Native C++ tokenizers decode that known
prefix and translate draft text back into target IDs. The prefix must roundtrip
exactly and remain an exact prefix of the combined tokenization. Otherwise the
proposer declines and ordinary target decode continues. It never splices token
IDs across different vocabularies. Up to eight draft tokens produce at most
three target proposals. A native Mistral four-input verifier accepts matching
prefix tokens, emits the target correction/bonus and rolls back rejected KV.
Frozen future target IDs are consulted only after sequence generation, to check
results; they never enter the proposer or verifier.

Native request-row leases and token-prefix snapshots also own draft state.
Before drafting, capture the current committed draft input; speculative draft
KV is never captured as committed input. The next round releases/reclaims the
row, restores the longest matching token prefix, then ingests only missing
input. The cache is bounded to two entries and256MiB of accounted host memory.
The selected native runtime is unchanged: orchestration and measurement live
in an experimental XLang runner, using the existing C++/Metal owners and
kernels, with no public xModel operator or serving API change.

A fixed16 resynchronization profile was needlessly executing16 physical token
rows after a cache hit even for one or two new tokens. A second policy chooses
the smallest prepared1/2/4/8/16 family: one new token uses ordinary native
step; other tails select the smallest prepared chunk and mask only its tail.
Both policies share the same maximum profile, weights, precision, cache budget
and target verifier. All graph families are prepared before requests. The
experiment asserts unchanged family counts throughout and zero target affine
verification misses; it does not establish that every possible backend helper
has no lazy work.

The final collection runs all three modes sequentially in one process for each
of eight coding prompts, reversing mode order on alternate cases. It uses
B1/C512, target prefill up-to128, draft prefill up-to16 and exactly64 generated
target IDs per case/mode. There is one sequence per mode/case and no warmup;
this is a clipped native stage diagnosis, not the frozen B4/C4096 HTTP study,
a repeated statistical speed qualification, a long-context test or a coding-
quality admission. Decode rates count63 post-prefill target tokens. Setup,
prefill, final-state hashing and trace-file writes are outside decode timing;
decode includes proposal generation, prefix capture/restore, host translation,
target verification/rollback and ordinary fallback. Prefill/setup times and
raw traces are retained separately.

| Coding case | Ordinary target tok/s | Fixed16 draft tok/s | Compact draft tok/s | Compact / ordinary |
|---|---:|---:|---:|---:|
|merge-closed-intervals |26.588 |24.065 |25.683 |0.966 |
|stable-unique-generator |26.660 |24.045 |25.714 |0.965 |
|deterministic-toposort |26.195 |21.266 |22.920 |0.875 |
|half-open-subtraction |25.999 |19.155 |21.204 |0.816 |
|incremental-length-frames |25.970 |22.038 |24.171 |0.931 |
|bounded-lru |25.770 |18.425 |20.015 |0.777 |
|patch-json-pointer |25.647 |16.823 |18.150 |0.708 |
|repair-bisect-left |23.845 |23.974 |25.155 |1.055 |

Across the eight sequential cases, summing504 post-prefill outputs and their
decode durations gives25.807 tok/s ordinary,20.878 fixed16 and22.548 compact.
These are pooled serial rates, not concurrent aggregate serving throughput.
Compact resynchronization is1.080x fixed16 in this collection (per-case ratios
1.049–1.107), but only0.874x ordinary overall. Its best descriptive case ratio
is1.055x ordinary; the study is not designed to establish that small difference
as a statistically reliable win. Neither path is selected as the default.

Both draft policies make214 target calls for504 post-prefill outputs, accept
290 of621 proposed target tokens, and obtain201 native prefix hits among210
draft attempts. The compact policy changes272 physical resync chunks from all
16-row chunks to65x16,6x8,95x4,39x2 and67 ordinary single-token steps. Acceptance
and target-call count are unchanged. Draft-side time falls13.955s to12.156s;
target-side time remains10.173s /10.184s. Improving shape selection alone does
not remove the learned-draft cost or raise acceptance enough to reach2x.

All24 sequences /1,536 target-token observations match the corresponding frozen
native controls. For every prompt, all32 layers' complete physical K/V hashes
and final hidden hash match ordinary decode in both speculative modes. The
prepared allocation is6,030,737,408 bytes, within the device recommendation
with256MiB margin. Accounted host prefix-cache use stays below256MiB; these are
snapshots, not measured peak/system-free memory. All native owner counts are
zero after close and residual Metal allocation is1,769,472 bytes, below the
unchanged4MiB guard.

An independent CPU audit re-tokenizes the recorded native text/IDs using the
pinned tokenizer JSONs, re-renders chat prompts against the original fixture,
recomputes causal acceptance and checks native routes, rollback, final state,
profile policy and timing arithmetic. Sixteen altered collections are rejected,
including changed draft conditioning, proposal, target output, cursor, KV,
hidden state, rollback, shape policy, memory ownership and reported rate. The
first integration smoke failed because an internal C++ tracking-count method
has no XLang binding; its failed artifact and cleanup are retained. The runner
uses the existing native cache-capture validation instead. A second smoke and
the two complete eight-case collections succeed.

Reproduction tools are `test2026/metal/prepare_mistral_cross_tokenizer_draft.py`,
`run_mistral_cross_tokenizer_draft.py` and
`audit_mistral_cross_tokenizer_draft.py`; each documents its arguments. The
preparer verifies all draft files, creates immutable input/manifest/command
files in a new directory and never runs inference. Re-rendering reproduces the
original frozen input byte-for-byte (SHA256
`4fac9f11fbb45951f0cc7a79b3ae666667865aabe43320e0993ff87b6d4b7888`).
Raw results, both original source versions, failed/successful smoke evidence,
pre-run manifests, audits and summary remain under
`mistral-small-draft-native1`. This is a correctness-qualified clipped research
path only. Full coding quality, B4 speculative batching, C4096/long-context
behavior, HTTP admission and all original2x / half-TTFT targets remain open.


### Explicit native packed-Qwen3 draft format and complete-output evidence

The C++ checkpoint representation adapter now admits dense `qwen3` affine
INT4/group64 artifacts in addition to its existing Mistral and hybrid Qwen3.5
formats. Dense architecture is an explicit enum, so canonical root mapping
and convolution/vision exclusions apply to both dense families. Existing
`DenseMistral()` semantics remain unchanged. `tensor_io.checkpoint_format`
reports `mlx_qwen3_affine_int4_g64_fp16` or its BF16-coefficient counterpart;
raw BF16 checkpoints still report `bf16`. Only the FP16-coefficient Metal
engine and the pinned Qwen3-0.6B draft are GPU-qualified here. Other Qwen3
sizes, quantized Qwen2, hybrid/MoE families and BF16-coefficient GPU execution
are not admitted by these measurements. xModel operators, graph programs and
public inference entry points are unchanged; native tensor loading selects
its existing typed affine kernels. There is no Rust runtime integration.

The offline CPU converter starts from the six verified files of
`Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`.
It performs per64 min/max affine quantization, rounds scale/bias to FP16, then
chooses nearest ties-to-even unsigned4-bit codes against those stored
coefficients and packs low nibble first. Constants use scale1. Norm vectors
retain their original BF16 bits. The source's duplicated tied head is checked
byte-for-byte against the embedding before omission; native inference already
uses the tied embedding. This is a deterministic local conversion in the
existing affine wire format, not a claim of reproducing another converter's
quantization algorithm or preserving draft predictions exactly.

The new checkpoint is335,450,656 bytes, SHA256
`2e2c01e65f3a69d222a2512621448f1c1207ca887932cbb3620bd84b7d143188`.
The original1,503,300,328-byte source file contains a duplicated311,164,928-byte
tied head that native inference does not load as a second matrix. File-size
ratios therefore are not runtime weight-bandwidth ratios. The Mistral target
artifact and FP32 target verification policy remain unchanged.

Validation is independent of the speculative speed test. Both dense families
pass12 native metadata cases each, including wrong coefficient types/shapes,
missing bias, unsupported family, incompatible quantization and ambiguous
roots. All26 existing native CTest cases pass in the isolated build. Native
selected-row decoding from every one of197 packed matrices matches an
independent FP64 expansion for863,232 raw FP32 values and the same number of
FP16-rounded embedding values. Real GPU projections from the first/last
layers and tied head cover60 shape cells,1/2/4/16 rows,2,760 independent FP64
dot samples with the original `2e-6 +4e-6*sum(abs(products))` bound, and7,168
native IO embedding values. Existing BF16 draft proposal/generated traces on
all eight clipped cases remain identical to the previous runtime.

The first same-process comparison retains the ordinary target, BF16 compact
draft, INT4 fixed16 draft and INT4 compact draft before readiness. Eight cases
reverse the four-mode order on alternate cases, with one64-token sequence per
mode and no warmup. All32 sequences /2,048 target-token observations match the
frozen controls; every paired complete physical KV and hidden hash matches.
Seventeen corrupted result sets are rejected by the independent audit.

| Clipped B1/C512 mode | Pooled serial decode tok/s |
|---|---:|
|Ordinary Mistral |25.971 |
|BF16 compact draft |22.163 |
|INT4 fixed16 draft |26.184 |
|INT4 compact draft |28.250 |

The INT4 compact path is1.088x ordinary in this clipped collection. It accepts
286/629 proposals versus290/621 for BF16; fewer bytes and lower draft cost
outweigh the slight acceptance loss here. These are native stage measurements,
not HTTP/comparator or repeated statistical qualification.

Complete-output tests then use target B1/C4096, target prefill up-to128, draft
prefill up-to16, native four-input verification and the original up-to1024
output/EOS budget. Every actual output contains171–484 IDs and stops on the
same EOS as its frozen control. The first study gives both engines C4096.
The next keeps the target C4096 and bounds only the draft to C512; a known
input plus the fixed eight-token proposal allowance exceeding C512 declines
drafting and performs ordinary target decode. The host prefix cache remains
bounded to two entries/512MiB in both studies. Runs and modes are sequential;
mode order alternates by case. Each study has one sequence per mode/case and
no warmup. Separate-process results are descriptive, not an isolated causal
estimate or significance test.

| Complete coding case | C4096 draft: ordinary tok/s | C4096 draft: speculative tok/s | C512 draft: ordinary tok/s | C512 draft: speculative tok/s |
|---|---:|---:|---:|---:|
|merge-closed-intervals |24.520 |22.669 |24.255 |26.474 |
|stable-unique-generator |24.757 |23.183 |24.646 |26.809 |
|deterministic-toposort |23.821 |20.621 |23.689 |23.495 |
|half-open-subtraction |23.810 |19.821 |23.651 |22.634 |
|incremental-length-frames |23.770 |19.812 |23.328 |22.502 |
|bounded-lru |23.164 |18.795 |23.142 |21.453 |
|patch-json-pointer |23.459 |18.514 |23.578 |21.015 |
|repair-bisect-left |24.691 |25.263 |20.095 |25.827 |

Summing3,044 post-prefill outputs and their decode durations gives:

| Target/draft capacity | Ordinary tok/s | Speculative tok/s | Speculative/ordinary |
|---|---:|---:|---:|
|4096/4096 |23.917 |20.574 |0.860 |
|4096/512 |23.488 |23.424 |0.997 |

Each complete study matches all16 outputs /6,104 target-token observations,
plus all32 layers' full physical K/V and final hidden hashes for every
ordinary/speculative pair. Each independent audit rejects16 corrupted result
sets. The smaller draft bound produces29 capacity fallbacks, validated against
known input lengths, and still reproduces every complete target output. No
model token is accepted from the proposer without native target verification.
The smaller bound reduces prepared allocation7,081,967,616 to6,246,580,224
bytes, with stable prepared family counts and no target affine misses. All
native owners are zero after close; the final smaller-bound collection leaves
1,769,472 Metal bytes, within the unchanged4MiB guard. These are allocation
snapshots, not measured peak memory.

The full-length results override the encouraging clipped result: there is no
pooled end-to-end native decode speedup in the smaller-bound complete study,
and ordinary decode remains the selected default. The bounded study spends
53.227s on drafting/translation/cache work and76.654s on target execution,
accepting1,796/3,645 proposals across1,248 target calls. Existing native prefix
snapshots read/write the growing KV prefix through host memory every draft
round; the measurement does not isolate that copy cost from other drafting
work. A next implementation should reconcile the known token prefix within
native GPU-resident draft state and clear only rejected suffix state, while
retaining lease/owner checks and target parity. Target verification cost and
acceptance also need improvement; eliminating draft time alone does not prove
the original2x requirement.

Full target outputs are unchanged, so this is no new coding-quality score or
improvement. Strict coding qualification, frozen B4/C4096 HTTP admission,
speculative continuous batching, long-input draft behavior and all original
2x decode/aggregate and half-TTFT requirements remain open. No vLLM/SGLang
baseline was rerun. ANE/Gemma/27B qualification is not inferred from this work.

Reproduction/validation entry points live in `test2026/metal/`:
`quantize_qwen3_draft.py`, `check_qwen3_affine_rows.py`,
`prepare_qwen3_affine_reference.py`, `run_qwen3_affine_reference.py`,
`prepare_mistral_packed_draft.py`, `run_mistral_packed_draft_comparison.py`,
`audit_mistral_packed_draft_comparison.py`, `run_mistral_packed_draft_full.py`
and `audit_mistral_packed_draft_full.py`. The existing cross-tokenizer input
preparer supplies base inputs. The packed preparer verifies both source and
converted assets; clipped, full and bounded-full inputs reproduce the frozen
JSON byte-for-byte. Full runner input defaults the draft to target capacity,
with optional `draft_capacity=512` and native bounds/fallback checks. Raw
source variants, numerical oracles, CMake/source manifests, logs, audits,
failed setup evidence, full output/KV witnesses and all timings are retained
under `mistral-packed-draft1`, with the immutable source/build/library under
`mistral-packed-draft-*`. The native code adds a representation capability;
the Python/XLang programs are experimental orchestration, not a new serving
runtime API or an admitted default speculative scheduler.


### Native dense token reconciliation retains working KV on Metal

The runtime now provides `session.reconcile_dense_tokens(row, epoch, tokens)`.
It compares the requested text token IDs with the current tracked history of
that live leased row, retains the exact longest common prefix in the existing
Metal KV buffers, and returns the input count already present. It clears only
the discarded KV suffix plus the invalid per-row final-hidden bank in one
synchronous Metal blit batch. The prefix is not copied to a host snapshot or
reinstalled. These are shared Metal buffers on unified memory; this is buffer
residency/reuse, not a claim about a separate VRAM transfer bus.

All source owners/generations, row bounds, epochs, integer/vocabulary tokens,
context lengths and fully dense KV layouts are validated before mutation.
An empty query is rejected. Querying a shorter prefix requires replay of its
last token when the historical final hidden is unavailable; the returned
count includes that replay requirement. Truncation invalidates row prediction
and prefix capture until new input completes. `hidden()` also rejects the old
global frame until a successful inference or full reset. Other rows' final
hidden banks and predictions stay usable. A failed GPU clear poisons the
session through the existing full-reset recovery path. Statistics are exposed
through `dense_reconcile_stats_json()`; the zero-byte count includes the
per-row hidden clear and `host_snapshot_bytes` describes this operation only.

The extension explicitly rejects convolution/recurrent/hybrid state, including
a fresh hybrid row, and MTP auxiliary state. It does not pretend KV truncation
recovers a recurrent checkpoint. Existing stored hybrid prefix snapshots and
restore/prediction still pass. Cross-request stored-prefix reuse remains the
existing native cache API; this new operation reconciles a currently leased
working row. No xModel operator, tokenizer, target math or default serving
scheduler changes. Implementation and ownership are native C++/Metal; the
experimental XLang runner controls only which tested path is measured.

Validation covers seven actual packed-Qwen3 B2/C64 cases: fresh, exact, append,
changed, shortened, unrelated and near-capacity queries. Every layer/part
retains the expected prefix bits, has zero in each discarded physical slot,
and preserves the other row. Continued ordinary FP32 decoding matches a fresh
control in complete physical KV, prediction and final hidden; a second
session remains unchanged. Thirteen invalid/released/untracked/stale/closed
owner cases reject without changing state. Two hybrid guards preserve state
and the original hybrid snapshot restore. The GPU primitive checks43 witnesses
and12 invalid ranges, including a bad second descriptor after a valid first,
padded rows, prepared row-span offsets, open commands and uninitialized
storage. The new primitive is registered as `metal_kv_row_clear_ranges` in
CTest; all27 tests pass. An independent audit rejects11 corrupted component
result sets. The earlier broad hybrid-prefill harness fails its pre-existing
dense-profile admission before reaching reconciliation; that failed artifact
is retained, and the focused hybrid test uses supported base steps.

The complete-output comparison uses one prepared process with three sequential
modes per case: ordinary Mistral, the previous host-snapshot INT4 draft, and
the resident INT4 draft. Mode order reverses on alternate cases. Target B1,
C4096/up-toT128; draft B1,C512/up-toT16; original up-to1024 output/EOS budget;
no warmup or repetitions. All graph families are prepared before requests.
The resident mode holds one auxiliary row lease through the request and
includes its final release in decode timing. It uses no host prefix captures.
Both draft paths fall back to ordinary target decode when the known draft
input plus the eight-token proposal allowance exceeds512.

| Complete coding case | Ordinary tok/s | Snapshot draft tok/s | Resident draft tok/s | Resident / ordinary |
|---|---:|---:|---:|---:|
|merge-closed-intervals |24.436 |27.394 |32.317 |1.323 |
|stable-unique-generator |24.854 |27.579 |32.539 |1.309 |
|deterministic-toposort |24.032 |24.510 |29.369 |1.222 |
|half-open-subtraction |23.720 |23.443 |27.572 |1.162 |
|incremental-length-frames |23.788 |23.258 |27.328 |1.149 |
|bounded-lru |23.189 |22.092 |25.661 |1.107 |
|patch-json-pointer |23.860 |20.629 |24.477 |1.026 |
|repair-bisect-left |24.670 |29.818 |34.036 |1.380 |

Across3,044 post-prefill outputs per mode, pooled serial rates are23.985,
24.201 and28.488 tok/s respectively. Resident/ordinary is1.188x and
resident/snapshot1.177x in this collection. These are native stage rates on
eight coding prompts, not the fixed HTTP throughput cells, a repeated
statistical qualification or a2x result. Per-case resident/ordinary ratios
range1.026–1.380. All24 complete sequences /9,156 target-token observations
match the frozen native outputs. Every ordinary/speculative pair matches all
32 layers' complete physical KV and final hidden hash. The full audit also
reconstructs resident histories from actually executed draft IDs, checks each
native LCP/clear/replay count and rejects20 corrupted result sets.

Both draft paths have identical proposals and generated draft IDs at all1,248
aligned target rounds, accept1,796/3,645 proposals and take the same29 capacity
fallbacks. Resident reconciliation is called1,219 times. It retains previously
computed matching draft inputs as well as the prior committed prefix, reducing
inputs processed from4,382 to2,501. Thus its benefit includes avoided
recomputation as well as removed snapshot copies/whole-row replacements; the
experiment does not isolate these contributions. The operation reports
261,795,840 logical cleared bytes total,24.298ms of summed GPU clear spans and
zero host KV snapshot bytes. No shortened-query replay happens in this
collection; that branch is covered in the real-model component test.

Measured drafting/translation work falls52.302s to32.213s, while target work
is73.428s /74.597s. Removing all remaining draft time, holding these target
calls/costs fixed, would still not establish the original2x target. Target
verification cost and/or accepted depth must improve too. Prepared allocation
is6,246,580,224 bytes; native family counts stay fixed and target affine
verification misses stay zero. Owner counts are zero after close and remaining
Metal allocation is within the unchanged4MiB guard. These are snapshots,
not measured peak/system-free memory.

The measured source/library1 and selected source/library2 have identical five
runtime implementation files. Source2 additionally registers the range test
and makes its row-span checks conditional on that existing feature;27 CTest
cases and the actual-model7-case/13-rejection checker pass there too. Both
immutable versions and their library hashes are retained. The measured run
continues to bind its exact original library, not the later rebuilt filename.

Reproduction tools are `test2026/metal/check_dense_reconcile.py`,
`check_dense_reconcile_hybrid.py`, `audit_dense_reconcile.py`,
`run_mistral_resident_draft.py`, `audit_mistral_resident_draft.py` and
`kv_row_clear_ranges_test.cpp`. The previous packed-input preparer generates
the unchanged B1/C4096-target/C512-draft input; this runner measures all three
modes. Native clear-only descriptors and session/hidden guards are checked
separately from timing. Full raw traces, frozen pre-run manifest, source/CMake
commands, failed harness, component/control states and audits are retained
under `mistral-dense-reconcile1`, with immutable source/build/library siblings.
Pre-existing unrelated changes are preserved and excluded from the checkpoint.

This is a positive native optimization, still experimental. Ordinary serving
remains the default. Full HTTP/B4 speculative batching, long-input policy,
strict coding qualification and all original2x decode/aggregate and half-TTFT
gates remain open. No comparator engine was rerun; this does not admit ANE,
Gemma or renewed27B/9B/4B qualification.

### Encoder timestamps identify the remaining verifier cost (2026-11-05)

The next diagnostic replays the **same saved resident-mode target inputs**
from the previous eight complete coding cases. It runs only Mistral target
inference; it does not run the draft or remeasure either frozen comparator.
The instrumented run and then an unmodified-library control are strictly
sequential. Both reproduce all3,052 target IDs, every layer's complete physical
KV hash and final hidden hash. There are1,218 multi-token verifier calls and30
ordinary fallback calls. This preserves the original1,024/EOS output budget,
B1/C4096/T128 prefill and physical four-token verification profile. Saved
future proposals are appropriate for this cost replay; this is **not a causal
draft-generation or end-to-end speculative-throughput measurement**.

A research-only native patch attaches timestamp samples at the beginning and
end of existing custom compute encoders. It adds no dispatches, barriers or
kernel arithmetic. M5 supports stage-boundary sampling, but not dispatch or
blit sampling. A32KiB buffer holds4,096 timestamp samples /2,048 encoder pairs.
The first model attempt exceeded Metal's live-counter-buffer limit by allocating
one per graph context; the selected patch shares one bank per device and locks
it from Begin through resolution/abort. This diagnostic consequently supports
one active command batch at a time and is not admitted for concurrent serving.
It requires macOS26 Apple unified memory; unset
`GARNET_METAL_ENCODER_TIMESTAMPS_PATH` leaves sampling disabled. The patch stays
outside the production runtime and must be applied to a separate source copy.

Timestamp calibration matters: `queryTimestampFrequency` reports24MHz on this
M5, while resolved counter timestamps and `sampleTimestamps` use matching
nanosecond values. Dividing the resolved differences by24MHz produced a wrong
41.667x duration and was rejected by a standalone two-encoder calibration.
The selected patch instead samples CPU/GPU clocks before and after each batch
and applies their measured span ratio, following Apple's
[CPU-time conversion procedure](https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time).
The corrected standalone encoder span11.0907ms agrees with independent
command-buffer GPU time11.0908ms. These values are calibration evidence, not
model speed. Failed attempts and logs remain in the research directory.

All1,218 verifier commands have complete valid sampling with no overflow or
dropped encoders. Their independent command-buffer spans total72,107.339ms.
The union of sampled compute intervals covers68,030.141ms (94.346%); summed
encoder durations are68,575.849ms. Their545.708ms overlap means the category
percentages below are duration ratios, **not an additive partition**. Blits,
MPS-internal encoders and gaps are outside the sampling scope.

| Native verifier work | Summed encoder time, ms | Duration / command GPU time |
|---|---:|---:|
| MLP gate/up/down projections |35,912.806 |49.805% |
| Other projections, including head |11,596.230 |16.082% |
| Attention scores and softmax |9,495.445 |13.168% |
| Ordered attention value accumulation |7,461.686 |10.348% |
| Other custom compute |4,109.683 |5.699% |

KV-backup kernels are392.800ms /0.545%, already included in other compute.
They are not the leading target cost. The full four-row gate/up shape is
4x4096x14336; its77,568 encoders average0.298535ms. Down projections are
4x14336x4096 and average0.324503ms across38,784 encoders. Tail calls using
indexed active-row projections are also included in the aggregate MLP row.
The next experiment should target these exact projection shapes before changing
KV ownership or weakening the existing FP32 attention rounding contract.

The unmodified-library replay totals71,293.952ms target GPU time and74.693s
replay wall time, versus72,107.339ms and79.401s instrumented. Observed overhead
is1.141% GPU and6.303% wall in this single ordered comparison; order/thermal
and time drift are not isolated. Therefore neither the breakdown nor this
control is a new speed qualification. The previously measured resident native
stage rate remains28.488tok/s, versus23.985tok/s ordinary in its own campaign.
No new HTTP, coding-quality, B4 speculative,2x or half-TTFT gate has passed.

Reproduction tools are `replay_mistral_verifier_profile.py`,
`audit_mistral_verifier_profile.py`, `test_mistral_verifier_profile_audit.py`,
`experiments/metal_encoder_timestamps.patch` and
`experiments/metal_counter_calibration.mm`, under `test2026/metal/`.
The auditor binds each target command to its replay GPU span, checks calibration
units and timestamp bounds, accounts for overlaps and verifies dispatch counts,
complete IDs/KV/hidden and cleanup. Its synthetic suite rejects28 corruptions.
All27 CTest cases pass with the diagnostic library and sampling disabled.
Counter-buffer ownership is outside Garnet's matrix/weight-owner counters;
both actual replay runs close to zero tracked owners and below4MiB Metal
allocation, rather than claiming those counters prove arbitrary driver cleanup.

Artifacts are in
`out/research/mistral7b-local-coder/mistral-verifier-profile1/`: raw samples2,
instrumented-result2, control-result1, selected-analysis2, overhead-comparison1,
auditor-corruptions2, ctest1, calibrated probe, failures, source/CMake commands
and protected-file hashes. The immutable source/build/library2 siblings use
base commit `81f920f1c2d6a70438adb3445f0fedc14620c449` and context base SHA256
`bd9142b1148a57162473e85cedad6825a7b9f031b71189052e5dc7f0b5737325`.
Instrumented context SHA256 is
`1efdf929446a5dec24f7d15df10c1a81e158cd3e84f4c822e3e3789630166560`;
library SHA256 is
`5a47669c8b58490a8d625b6b4dbe44ce8e4c2dacecb5c356ef8b078f49a919dc`.
The unmodified control binds selected dense-reconcile library2 SHA256
`750614491b30274a5fb998850a2a27275e072c2a05742b63bd2ba6e5fd465341`.
`provenance1.json` SHA256 is
`9a445adb2659788bd8cb7c8cbfcba634948a521062ce760bba6eca830faf2940`.
This manifest binds artifacts after execution; the launch record froze the
instrumented library hash and command before execution. Production sources,
selected runtime, frozen comparator results and pre-existing dirty work remain
unchanged by this diagnostic. The larger qualification goal remains open.

### Register-streaming rejection and prepared verification layout (2026-11-05)

The selected runtime already had four-column reuse for four-row Mistral
projections. Two new register/loop arrangements were tested against that
selected kernel, rather than treating reuse as a new optimization. Both keep
packed weights, FP16 coefficients, FP32 dequantization/accumulation, K-word/j
order and SIMD reduction unchanged. Scalar streaming holds the four columns'
packed coefficients and streams one input value per token. Vector streaming
keeps packed_float4 loads but visits four K values at a time. Register counts
were not measured, so the experiment does not establish an occupancy cause.

Each arrangement uses four serial processes: control/candidate and then
candidate/control, with three warmups and nine measured samples per cell.
The existing Mistral operator harness covers nine shapes, dense rows1–4 and
all sixteen four-row owner subsets:180 cells per process. Across both
arrangements,25,031,832 compared FP32 words match byte for byte and9,072
independent FP64 samples pass the unchanged original bound. Both arrangements
are rejected on performance. The scalar arrangement's first-cohort down
projection has an apparent gain that disappears in the reverse cohort; its
control timing differs substantially between cohorts. That sample remains in
the collection and is not used to select the candidate.

| Four-row projection | Scalar ratio, cohorts1 /2 | Vector-slice ratio, cohorts1 /2 |
|---|---:|---:|
| K4096/N14336 gate/up |0.689 /0.693 |0.971 /0.951 |
| K14336/N4096 down |1.402 /0.729 |0.951 /0.984 |

Ratios are control/candidate median **operator GPU durations**, not token
rates. The vector-slice prototype initially failed Metal compilation because
`half` was used as a loop variable; the corrected, separately frozen source3
uses `part`. Failed source2 produces no valid candidate timing claim. Raw
sources, binaries, outputs, manifests and rejected attempts remain under
`mistral-mlp-stream1`, `mistral-mlp-stream2` and `mistral-mlp-stream3`, with
immutable source/build siblings. Rejected patches are retained in
`test2026/metal/experiments/mistral_streamed_affine_{scalar,vector}.patch`.
`audit_mistral_streamed_affine.py` checks both successful collections.

The next native change specializes verification attention for the exact
B1/T4/C4096/H32/KV8/D128 ungated layout. The new build option
`GARNET_METAL_MISTRAL_VERIFY_LAYOUT` defaults **OFF** and requires
`GARNET_METAL_VERIFY_ATTENTION`. Two pipelines compile before readiness.
Other shapes retain the generic route. Batch, physical token width, heads,
KV heads, head dimension, capacity and ungated mode become compile-time values;
strides, slots and active flags remain runtime data. Attention reductions
traverse the actual visible prefix. The existing probability suffix is still
zeroed through capacity. This preserves up-to bounds; it does not require a
request to contain4096 input tokens. No xModel operator, weight, precision,
KV ownership or prefill policy changes.

`target_attention_layout_dispatches` records the logical specialized attention
pairs in verification statistics. The build advertises
`mistral_verification_layout_b1t4_c4096_h32_kv8_d128`. A native per-session
`verification_set_mistral_layout(bool)` diagnostic selects between the two
already prepared routes. It rejects open/undrained batches; the session wrapper
rejects closed/poisoned sessions. The API is exposed only in an enabled build,
and its capability is `verification_layout_route_control`. Route selection
changes neither graphs nor allocations and allows a comparison in one engine.

The actual-layout reference covers positions0,384 and4092, every prefix length
0–4, in-place cache-before/cache-after updates, pending queries and partial
flushes. Different chunk/batch/capacity and gated inputs exercise fallback.
The selected on-build passes21 cases,189,133,824 compared FP32 words and81
rejection checks; the off-build passes19 cases,171,275,264 words and57
rejections. Both compare probabilities, context and complete physical KV
byte for byte with ordinary sequential attention. All28 CTest cases pass
with the option on and again off. A configuration with layout enabled but
verification disabled is rejected. Unsupported inputs keep the original
validation/ownership guards before pipeline selection.

An initial target-only replay reproduces all eight full saved coding sequences
and their complete KV/hidden states. All1,218 verifier calls report32 layout
pairs and zero affine misses. Candidate target GPU spans total66,187.051ms,
versus71,293.952ms in the prior control and69,989.494ms in a fresh sequential
control. Control/candidate GPU-time ratios are1.057–1.077x in separate runs;
these are not generation speedups. The full ordinary/snapshot/resident generation collection with
selected guarded library2 then passes all24 sequences /9,156 IDs, complete
physical KV/hidden and the independent causal audit. Its pooled rates are
23.903/24.333/28.622tok/s. The prior collection was
23.985/24.201/28.488tok/s: a much smaller overall change, with collection drift
not isolated. Both collections have identical draft IDs/proposals and target
commits. The isolated timing improvement must not be substituted for this
full-stage result.

A subsequent collection holds weights, buffers and prepared pipelines in
**one engine**. Ordinary decode is compared with GPU-resident drafting using
either generic control attention or the compiled layout. Case order alternates
ordinary/control/layout and layout/control/ordinary. Both draft modes use the
same causal cross-tokenizer proposal policy and native reconciliation; neither
captures host prefix snapshots. The full1024/EOS budget and eight coding
prompts are unchanged. Every complete output, KV/hidden witness, draft token,
proposal and committed prefix matches between routes.

| Same-process native stage | Post-prefill outputs | Decode seconds | Pooled tok/s |
|---|---:|---:|---:|
| Ordinary |3,044 |127.317 |23.909 |
| Resident draft, generic control |3,044 |110.485 |27.551 |
| Resident draft, compiled layout |3,044 |108.190 |28.136 |

The compiled-layout/control ratio is1.0212x in this single collection; seven
cases improve and one regresses0.54%. Individual ratios range0.9946–1.0334.
Layout/ordinary is1.1768x. Target verification GPU spans fall
73,816.726 to71,631.980ms; draft work remains33.162 /33.050s.
Both modes make1,248 target calls, including1,218 multi-token verifications,
and accept1,796 of3,645 proposals. This is a small experimental native-stage
improvement. It is not repeated statistical qualification, an HTTP throughput
cell, improved coding quality or a2x result. Absolute rates across collections
vary; retain all collections rather than selecting the largest absolute rate.

The paired audit independently re-tokenizes causal prefixes, reconstructs the
resident history/clear counts, checks exact target commits, all32 layout pairs
for the candidate and zero for control, complete state equality, prepared
families and cleanup. Its inherited20 corruption checks and six additional
layout-specific corruptions pass. Prepared allocation is6,246,580,224 bytes;
no new graphs appear after readiness, native owner counters close to zero,
and remaining Metal allocation stays within4MiB. These are observed snapshots,
not a measured peak or proof of arbitrary driver-resource cleanup.

Reproduction tools under `test2026/metal/` are
`mistral_verification_layout_test.cpp`, `run_mistral_verification_layout.py`,
`audit_mistral_verification_layout.py` and
`test_mistral_verification_layout_audit.py`. Sources/builds/libraries1,2,3 are
immutable and separately named. Library1 is the ungated prototype used by the
saved-input replay; library2 adds the default-off build gate and is used by
the ordinary/snapshot/resident collection; library3 adds route control and is
used by the same-process collection. Timing inputs and libraries are frozen
in each launch manifest. Raw results, manifests, on/off CTest logs, rejected
configuration, residual-work proof and aggregate post-run provenance remain
under `out/research/mistral7b-local-coder/mistral-verify-layout1/`.

The official EAGLE weight tables were rechecked at commit
[`cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b`](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/README.md#eagle-3-weights).
They do not list a Mistral-7B-Instruct-v0.3 draft. This is a bounded check of
that official list, not proof that no community artifact exists. No unrelated
EAGLE model was downloaded or admitted as a matching draft.

The layout option remains off by default, and the existing serving library
and frozen vLLM/SGLang results are unchanged. Full HTTP/B4 speculative batching,
long-input policy, strict coding quality, all2x/half-TTFT gates, ANE/Gemma model
qualification and the original27B/9B/4B work remain open.

The final guarded/controlled library3 SHA256 is
`89f36d03b83614ffb1553f9b52663d69a8e42c9863b06810ecf7a3cba32cc6df`.
The aggregate `mistral-verify-layout1/provenance1.json` SHA256 is
`b0fffa2535a360847fd2b7aeb89c364aaa4e1ed71bccea5c461765d67cbecc95`.
It binds the separately frozen run artifacts and clean selected runtime files
after execution. The target weight hash, frozen comparator summary and
protected serving DLL hash are rechecked unchanged. Pre-existing dirty work
is reconstructed separately from the selected native edits and is preserved.

An additional default-off option, `GARNET_METAL_MISTRAL_VERIFY_VECTOR4`,
requires the prepared Mistral verification layout. It reads four adjacent
value columns per lane, preserving the scalar kernel's ordered FP32 sum over
the actual visible prefix. Its dispatch uses two SIMD groups per threadgroup.
The existing B1/T4/C4096/H32/KV8/D128 ungated guard remains mandatory;
other shapes and gated attention retain the original kernels. Pipelines are
prepared before readiness. The native session method
`verification_set_value_vector4(bool)` switches routes only while the command
batch is closed and drained. Verification statistics expose
`target_attention_vector4_dispatches`; both compared routes retain all32
compiled-layout attention pairs, while only the candidate records32 vector
context dispatches. Build defaults remain unchanged.

The saved-input paired replay covers eight complete sequences and6,104 token
observations, with identical full KV/hidden witnesses. Target GPU spans are
67,379.893ms scalar versus64,608.496ms vector (1.0429x); replay wall time is
71.120s versus68.270s (1.0418x). This replay excludes causal draft execution.
The subsequent complete generation collection holds one prepared engine and
alternates ordinary/scalar/vector and vector/scalar/ordinary by case. Both
draft modes use identical resident Qwen0.6 inputs and proposal policy.

| Same-process native stage | Post-prefill outputs | Decode seconds | Pooled tok/s |
|---|---:|---:|---:|
| Ordinary |3,044 |126.075 |24.144 |
| Resident draft, scalar layout values |3,044 |101.511 |29.987 |
| Resident draft, vector layout values |3,044 |98.360 |30.947 |

Vector/scalar is1.0320x and vector/ordinary is1.2818x in this single
collection. All eight vector/scalar cases improve, with ratios1.0129–1.0413.
Complete outputs, draft IDs, proposals, accepted prefixes and full KV/hidden
states match across routes. Both draft modes accept1,796 of3,645 proposed
tokens over1,248 target calls. Target GPU spans fall66,273.856 to63,110.627ms;
draft work is31.543 versus31.570s. These timings are native complete-output
stage evidence, not an HTTP/concurrency cell, repeated statistical
qualification, improved coding quality or achievement of the2x target.
Absolute rates vary between collections; compare paired routes within each
collection and retain earlier results.

Native component checks compare probabilities, context and complete KV
bitwise against sequential decode, including inactive masks, visible-prefix
endpoints, cache-before/cache-after updates, partial flushes, pending inputs,
fallback profiles and route selection. The independent causal audit rejects
20 inherited corruptions; eight additional checks reject vector and parent
layout route corruption. Prepared Metal allocation remains6,246,580,224
bytes. Owner counters reach zero and remaining Metal allocation is1,769,472
bytes after close; these are snapshots rather than a measured peak.

Reproduction tools are `run_mistral_verification_value_vector.py`,
`audit_mistral_verification_value_vector.py`,
`test_mistral_verification_value_vector_audit.py` and the extended registered
`mistral_verification_layout_test.cpp`. Raw paired replay/generation data,
pre-run manifests, native reference output and on/off build checks are
retained under `out/research/mistral7b-local-coder/mistral-verify-value-vector1/`.
The measured library SHA256 is
`c6ff3842102bcc006df217723e6936e95af51a565eb9820f544500ca7cd99db0`.
The frozen comparator engines are not rerun.

A separate Qwen0.6 single-row projection-reuse experiment is rejected:
35,561,292 FP32 words match, but its second ordered cohort regresses all six
tested projection shapes. Changing ordinary attention threadgroup size also
shows only small/inconsistent single-row gains; it is not promoted. The
retained causal trace attributes about19.0s of the preceding collection's
33.1s draft time to native decode, with17.3s GPU spans inside18.7s measured
decode-call wall time. This does not isolate CPU overhead, but it rules out
assuming that Python scheduling dominates those draft decode calls. Larger
gains require reducing draft work and/or increasing accepted depth as well as
target kernel improvements. The original performance, batching, quality and
model qualification requirements remain open.

Both vector-enabled and vector-disabled builds pass all28 CTest cases.
The expanded enabled component test covers23 cases,206,992,384 FP32 words
and110 rejection checks; a configuration enabling vector values without the
prepared parent layout is rejected. The selected native sources match the
measured library's source archive. The aggregate post-run provenance is
`mistral-verify-value-vector1/provenance2.json`, SHA256
`7dab71b32ecefac2e8abd3e9d50716c1c2c9f9a3cb0dcea3319f42401ba1017c`.
Frozen target weights, comparator summary and protected serving DLL hashes
are rechecked, and pre-existing dirty work is preserved separately.

A subsequent shared-vocabulary Medusa experiment is **rejected**. It exposes
residual features without evaluating each raw BF16 vocabulary matrix, stacks
the normalized target hidden row and three residual rows, and evaluates the
target's existing quantized vocabulary matrix once over four rows. The first
row must reproduce the already-known target token. The raw residual adapter's
training base is unverified; replacing its learned vocabulary projection with
the target projection is an approximation, not an exact weight-sharing
transformation. Native greedy verification still decides every emitted token.

Three operator inputs reproduce pinned residual features and original raw
head logits exactly. The393,216 packed vocabulary words match four separate
target projections bitwise. The residual-only path loads three BF16 residual
matrices and no additional quantized target weight owners. Bounded two-row
features match the one-row results, seven invalid profiles are rejected, and
owner counters reach zero. These arithmetic checks do not establish useful
acceptance, which is measured separately.

Two complete frozen coding cases, merge-closed-intervals and
stable-unique-generator, are then collected in one prepared engine with
ordinary/raw/shared and shared/raw/ordinary order. Each retains its1024/EOS
budget. This is a feasibility collection over two cases, not the eight-case
generation campaign or the full HTTP qualification.

| Native feasibility stage | Post-prefill outputs | Decode seconds | Pooled tok/s | Accepted/proposed |
|---|---:|---:|---:|---:|
| Ordinary |796 |32.479 |24.508 |— |
| Raw BF16 Medusa heads |796 |27.692 |28.745 |346/1,350 |
| Shared target vocabulary |796 |44.828 |17.757 |25/2,313 |

Observed mean draft time falls8.26 to2.91ms per call, but acceptance falls
25.63% to1.08%, increasing target calls450 to771. The different accepted
trajectories mean this is not an isolated equal-input kernel comparison.
The cheaper draft does not compensate for the weak proposals. All six
complete token streams and full KV/hidden witnesses match ordinary decode;
the causal/state audit also rejects12 corruptions. Allocation is6,778,339,328
bytes at readiness and1,769,472 bytes after close, with zero remaining native
owners. These are allocation snapshots. The raw-head rates here supersede
neither prior collections nor frozen comparator results; they use the newer
prepared four-row target kernels and cover only these two cases.

The residual-only native change and research adapter are retained solely in
`test2026/metal/experiments/mistral_medusa_shared_head.patch`. It applies to a
clean isolated source archive of the preceding checkpoint. It is not part of
the admitted runtime API. Reproduction tools are
`probe_mistral_medusa_shared_head.py`, `run_mistral_medusa_shared_head.py` and
`audit_mistral_medusa_shared_head.py`; the latter accepts a campaign directory
and optional new output path. Raw inputs, library, manifests and independent
audit results remain in
`out/research/mistral7b-local-coder/mistral-medusa-shared-head1/`. Aggregate
`provenance2.json` SHA256 is
`968481fbb47474fff0d3e8db459d433323fe830e1b85f1b2d9074eb71970ed52`.
No shared-head shortcut is enabled, and the performance goal remains open.

The subsequent Core ML study in [coreml-ane-plugin.md](coreml-ane-plugin.md)
adds observed model-labelled ANE activity, CPU-only/LUT controls and an offline
2048-channel/64x range profile. In the final direct two-cohort layer0 MLP
comparison, the original ANE path is17.587 /17.353ms and the new profile is
14.310 /14.121ms:1.2291x /1.2289x operator speed. Relative RMS error against the
Metal operator falls7.236% to1.104%. Metal remains faster at10.358 /10.364ms.
The128 normalized embedding rows are an operator proxy. This does not establish
full-model ANE gain, coding quality, ANE KV/batch integration or the2x goal.
The optional offline flags preserve existing defaults; native C++ inference,
xModel operations and the protected serving DLL are unchanged. Full retained
traces, staged/final campaigns, source equivalence and qualification limits are
documented in that report.


A subsequent thin FP32 tensor-matmul experiment is **rejected for runtime
routing**. It compares ordinary and existing four-row blocked affine kernels
with MPP cooperative operands at M16/N32/K16. Caching dtype-aware fragment
coordinates and capacities outside the K loop substantially improves the MPP
implementation; its original coordinate queries were inside every reduction
iteration. No FP16 fragment-layout assumption is used for FP32. A second
serial run retains both forms and adds the supported K32 reduction tile.
The installed SDK rejects K64 when both inputs are cooperative tensors; the
failed compilation and its raw diagnostic are retained separately.

The final run interleaves all five variants, three warmups and15 measured
samples, over five actual Mistral projection dimensions. All40,550,400 checked
FP32 words pass the unchanged independent full-output FP64 bound
`2e-6 +4e-6 * sum_abs_products`;225,280 unique reference outputs are computed.
Cached K16 and the original MPP K16 retained outputs are bitwise identical for
all shapes and both cache-pressure conditions. MPP itself changes reduction
order relative to ordinary decode. This is arithmetic evidence over synthetic
inputs, not full-model identity or coding quality.

| N / K, four input rows | Selected native GPU ms | Cached MPP K16 GPU ms | Cached MPP K32 GPU ms |
|---|---:|---:|---:|
|4096 /4096 |0.1123 |0.3693 |0.4042 |
|1024 /4096 |0.0494 |0.2891 |0.3388 |
|14336 /4096 |0.3499 |0.7905 |1.0904 |
|4096 /14336 |0.3821 |0.9428 |1.2013 |
|32768 /4096 |0.6855 |1.7195 |1.8850 |

These are medians from the256 MiB sweep condition in one interleaved process;
the sweep is not proof of a cold cache. Hot samples and all raw timing vectors
are retained too. The selected control is the production ordinary route for
N1024/K4096 and the existing four-row blocked route for the other shapes.
The fastest MPP candidate still takes2.26–5.86x the selected control's GPU time;
larger K32 tiles regress. No production kernel, native API, xModel operator or
serving DLL is changed, and no full-model performance run is justified by this
result. Fixed-shape preparation alone does not establish a gain for this kernel.

The standalone reproduction source is
`test2026/metal/experiments/mistral_thin_mpp_probe.mm`, with build and serial
run instructions in its header. Raw source, executable hashes, JSONL, binary
outputs, unsupported-configuration diagnostic, source-equivalence proof and
`summary1.json` remain under
`out/research/mistral7b-local-coder/mistral-thin-mpp-hoist1/`. Pre-existing dirty
work is hash-checked unchanged. The2x generation/throughput and half-TTFT goals
remain open; frozen vLLM/SGLang baselines are not rerun.


The next draft-cost study prepares two **research-only proposal heads** from
Qwen3-0.6B's existing quantized embedding rows. Selection is frozen before any
new inference and depends only on the tokenizer JSON: retain printable ASCII
strings (plus tab/LF/CR), all256 single-byte tokens, and all26 added tokens.
The second profile additionally limits multibyte ASCII tokens to original
IDs below65536. The resulting output heads contain94,510 and62,709 rows,
versus151,936 in the full head; packed head payloads are54,437,760 and36,120,384
bytes. Every existing input-embedding/decoder tensor remains bitwise unchanged,
and every selected weight/FP16 scale/bias row is independently re-read and
checked bytewise. Output indices must be explicitly remapped to original token
IDs before decoding text or feeding a subsequent draft step. These are not
generic Qwen checkpoints or changes to the target model. They can change draft
proposals; only native target verification determines emitted tokens.

Six native head tests (three synthetic inputs at batch1/2) check1,414,971
compact FP32 logits against the exact selected full-head logits and verify
stable mapped argmax results. All match bitwise. The standalone head timing
includes host input preparation and download of the complete logit vector;
the resident session instead selects argmax on the GPU. These timings are
retained but are not GPU-only projection measurements and cannot predict
complete-generation performance or acceptance.

The first complete-generation attempt exposed an existing native restriction:
`reconcile_dense_tokens` bounded input token IDs using the vocabulary head's
output columns. This rejects valid high input IDs when a proposal head has a
smaller untied output vocabulary. The failed attempt, its completed full-head
case and zero-owner cleanup are retained; it is not a valid comparison.

Token-prefix preparation now resolves the immutable **input embedding** domain
from checkpoint metadata, validating positive bounded vocabulary, width and
typed byte layout (or the explicit affine INT4 layout). It does not upload a
second embedding table. Reconciliation uses this prepared input bound; head
output dimensions continue to govern prediction. Graphs with no input embedding
metadata retain their previous output-head bound. Closing a session clears the
prepared domain. Existing operators and public API signatures are unchanged.
All28 native CTests pass. The original head passes seven real-model dense
reconciliation cases, and each compact head passes ten including input IDs
151643/151645/151935 above its output size. Each also passes13 rejection checks
with unchanged KV/hidden on failure, plus append/change/truncation replay,
row/session isolation and zero native owners after close.

Raw assets, static-selection manifests, operator checks, failed first run,
isolated native build and real-model reconciliation reports are retained under
`out/research/mistral7b-local-coder/mistral-draft-head-profile1/`. Generation
qualification and speed decisions must use the completed campaign and its
independent causal/tokenizer/target-state audit, not the projection test.


The second complete campaign finishes all eight cases at the original1024/EOS
budget, running full/ASCII/ASCII64k and the reverse order on alternating cases.
All9,156 target IDs and full KV/hidden witnesses match the frozen ordinary
controls. An independent Hugging Face tokenizer audit checks causal draft
conditioning, token-map membership, resident LCP/counters/clears, cross-tokenizer
proposal translation, native verifier acceptance/rollback, output budgets,
profile/asset hashes and cleanup; it rejects22 corrupted records. The existing
hybrid fixture still rejects dense rewind twice, then restores its recurrent/KV
snapshot and prediction exactly. All native owners reach zero after close.

| Draft output head | Post-prefill tokens | Decode seconds | Observed tok/s | Accepted / proposed | Target calls |
|---|---:|---:|---:|---:|---:|
| Full151,936 rows |3044 |98.492 |30.906 |1796 /3645 |1248 |
| ASCII94,510 rows |3044 |98.081 |31.036 |1796 /3645 |1248 |
| ASCII64k62,709 rows |3044 |99.470 |30.602 |1777 /3704 |1267 |

The larger compact head preserves every recorded draft ID and proposal, but
its observed total speed difference is only0.42%; draft time falls31.433 to
30.636s while target time varies67.012 to67.399s. The smaller head loses19
accepted tokens and adds19 target calls, offsetting its cheaper projection.
These timings are **diagnostic only**: this is one prepared process, and late
snapshots report roughly4GB of compressed process memory. They do not establish
a repeatable speed improvement, and the larger head regresses one case by6.5%.
The original full head remains the default; neither compact profile is promoted.
A cleaner repeated collection would be needed before any small gain claim.

The memory inventory is7,111,294,976 bytes at readiness and1,769,472 bytes after
close, with zero native owners; these are snapshots rather than peak/unified
memory proof. Exact target identity preserves the existing strict0/8 coding
result. Full serving/batch throughput, half TTFT and the2x goal remain open.
The prefix bound correction is the admitted native change; no compact policy
is enabled and the protected serving DLL and frozen comparators are unchanged.

Reproduction tools are `prepare_mistral_draft_head_profile.py`,
`probe_mistral_draft_head_profile.py`, `run_mistral_draft_head_profile.py`,
`audit_mistral_draft_head_profile.py` and
`check_dense_reconcile_compact_head.py` under `test2026/metal/`. CLI docstrings,
`generation-manifest2.json`, `native-build-command2.json` and
`postrun-manifest2.json` retain exact commands and hashes. The measured native
library, failed first run, all binary assets, raw traces, `audit2.json` and
`summary2.json` remain in the campaign directory above. The operator collector
uses full-logit host returns, so its timings must not be reported as resident
GPU-only projection speed.


The original **eight-row operator portion is invalidated**. A later source
review found that the collector's outer replacement changed its own kernel
transformation from `replace("row<4","row<8")` into a no-op
`replace("row<8","row<8")`. Local arrays expanded to8 but the generated row
loops still stopped at4. The reused output buffer retained correct baseline
values in unwritten rows, so the numerical comparisons did not prove that the
candidate wrote all eight rows. The old64,880,640-check claim and associated
operator speed table must not be used as eight-row evidence. The raw source,
binary and results are retained with `operator-invalidated1.json` under
`mistral-verify8-feasibility1/`; they are superseded by the poisoned-output
study below. This affected the standalone collector, which was never enabled
as a native target8 path. Existing native target4 and confidence qualification
use separate kernels and complete runtime checks.

The replacement collector uses checked, kernel-body-only transformations and
poisons the complete output before every dispatch. A deliberately broken
four-row writer labelled as8 must be rejected before positive measurements.
The repo reproduction file now contains that corrected collector. The earlier
causal acceptance experiment remains separate from this invalid operator test.

A separate native Qwen0.6 proposer then measures deeper **causal acceptance**.
The target is a frozen ordinary-token oracle, not a running target GPU engine.
The proposer receives only original user messages and already-committed target
text. An independent tokenizer audit checks all prompt IDs, prefix LCP/history,
zeroed suffix bytes, prepared prefill chunks, emitted/proposed token accounting,
EOS/output budget and cleanup. The depth3 trajectory matches every draft ID,
proposal and commit in the preceding real native target-verification campaign.
This validates the oracle accounting; it does not qualify a native eight-row
verifier, target KV rollback, batch service or generation speed.

| Proposal policy | Projected target passes | Accepted proposals | Draft decode steps | Missing draft prefill tokens |
|---|---:|---:|---:|---:|
| Fixed depth3, max8 draft tokens |1248 |1796 |2014 |2501 |
| Fixed depth7, max16 draft tokens |968 |2076 |4615 |1962 |
| Causal adaptive depth3/7 |1098 |1946 |2998 |2239 |

Each policy covers the same eight cases and3044 post-prefill target IDs. The
adaptive rule is frozen before its collection: start at depth3; use depth7
only when the previous round accepted at least3 proposals, otherwise use
3. This removes12.0% of projected target passes but adds48.9% draft decode
steps. Fixed depth7 removes22.4% passes but requires2.29x draft decode steps.
Draft-only timing also excludes the target's cache eviction and workload.
None of these results establishes a net speed improvement, and no eight-row
or adaptive policy is enabled. Attention, KV backup/rollback and commit costs
must be included before promotion. A blanket relaxation of existing four-row
guards would be incorrect, especially for hybrid/recurrent profiles.

Both fixed and adaptive oracle audits reject11 corrupted records each, and
all native owners reach zero after close. The first oracle attempt failed an
assertion because the preparation compile counter was not reset after warming;
it is retained, with no admitted records. The measured second collector records
preparation compile time, resets the counter at readiness and independently
asserts complete output identity and zero new graphs/compilation during work.

Reproduction sources are `experiments/mistral_verify8_projection_probe.mm`,
`run_mistral_proposal_depth_oracle.py` and
`audit_mistral_proposal_depth_oracle.py` under `test2026/metal/`. The base collector
uses `LIB SOURCE INPUT OUTPUT`; its target-oracle scope is explicit in the
output. Apply `experiments/mistral_adaptive_proposal_depth.patch` to an isolated
copy of the collector for the measured adaptive policy. The audit takes the
campaign directory and optional `--adaptive`. The retained patch is checked to
recreate the measured adaptive source exactly. Raw operator binaries/outputs,
source snapshots, static/adaptive pre-run manifests, failure, independent audits
and `feasibility-summary1.json` are under
`out/research/mistral7b-local-coder/mistral-verify8-feasibility1/`. This study
changes no native runtime, public API, xModel operation, protected serving DLL
or frozen comparator. The full2x/half-TTFT and coding-quality goals remain open.


The next study adds **default-off native prediction confidence** without
changing xModel operations or the existing greedy result. Enable
`GARNET_METAL_PREDICTION_CONFIDENCE` only with native token-prefix and per-row
final-hidden support. Invalid configuration is rejected. A GPU reduction returns
four internal FP32 words: lowest-index winner, nonfinite-error flag, maximum
logit and largest logit at a distinct index. The first two match the old argmax
ABI exactly; the existing two-word kernel/callers remain intact. Equal top logits
produce gap0. A logit gap is not a calibrated probability.

The optional session API is prepared before requests, after
`configure_token_prefix(...)`, using `configure_prediction_confidence()`.
`run_tokens_confidence(graphs, hidden, positions, slots, active, 0, epochs, ids)`
accepts the existing tracked **base single-token rows**, with rank2 hidden input.
For chunked prefill, use existing tracked prefill without prediction and then
`predict_rows_confidence(epochs)` to predict from private last-active hidden
states. Both return one `[head_index, top1, top2, gap]` list per active row and an empty
list for inactive rows. Row prediction requires at least one selected current
epoch. The fresh full head and reduction execute in the same locked GPU command;
shared head logits are never read later after another session can overwrite
them. Only4 words per row cross to the CPU. The private Bx4 workspace is allocated
before readiness and included in the native memory inventory. MTP auxiliary
sessions are excluded. Existing `run_tokens`/`predict_rows` outputs stay unchanged.

The new CPU-reference GPU test passes211 cases over24,245,894 input logits,
including batch1/2/4, vocabulary tails, the2^24 index bound, distinct winners,
ties, signed zero, extreme finite values and NaN/infinities. It checks the old
ID/error fields bitwise, top-two values against independent CPU selection and
six bounds/ownership/abort rejections. All29 enabled CTests and all28 disabled
CTests pass. The real packed Qwen3 B2/C64 session test checks five steps against
full CPU logit reductions and unchanged ordinary IDs/full KV/hidden; partial
rows, overwritten shared heads, replay, epochs and12 rejected accesses are
covered. The existing seven-case dense reconciliation and hybrid snapshot
regressions also pass. Native owners reach zero after close.

A complete native-target collection then freezes a single experimental stopping
threshold (gap<1.0), with no tuning from benchmark confidence values. After each
appended draft token, it first honors the three-target-token proposal limit;
otherwise, when at least one aligned target proposal exists, low gap stops
further drafting. The max8 draft-token and original1024/EOS target budgets
remain unchanged. Each case runs original/no-score, score/no-stop and score/stop
modes, with order reversed on alternating cases; all use the same target4/vector4
verification profile and one prepared target/draft pair.

| Native mode | Post-prefill tokens | Decode seconds | Observed tok/s | Draft steps | Target passes |
|---|---:|---:|---:|---:|---:|
| Original speculation |3044 |107.900 |28.211 |2014 |1248 |
| Confidence capture, no stopping |3044 |109.476 |27.805 |2014 |1248 |
| Confidence capture, gap1.0 stop |3044 |107.062 |28.432 |1380 |1309 |

All9,156 complete target IDs and full KV/hidden witnesses match the frozen native
controls. Capture alone preserves every recorded draft ID/proposal/commit.
The independent tokenizer/causality audit validates score IDs, finite top-two
values and gap arithmetic, the first permitted stop, terminal predictions,
request-prefix LCP/clears, prepared shapes, acceptance/rollback and exact target
state. It rejects24 corruptions. The stopping policy reduces draft decode steps
31.5%, but loses61 accepted proposals and adds61 target passes. Draft time falls
34.330 to29.521s while target time rises73.505 to77.479s. Its pooled observed
speed difference is only+0.78%, with per-case ratios0.942–1.120. Capture alone
is0.986x the original in this collection. One serial prepared collection and
roughly432MB maximum compressed-process snapshots do not establish a repeatable
speed improvement or isolated reduction overhead. No stopping policy is promoted.

Allocation is6,246,580,224 bytes at readiness and2,031,616 bytes after close;
these are snapshots rather than a measured peak, and all native owners close.
Exact output identity preserves the existing strict0/8 coding result. The full
2x decode/throughput, half-TTFT, serving and coding-quality goals remain open.
The native API is optional infrastructure for further experiments, not an
admitted speed policy. Frozen comparator results and the protected serving DLL
are unchanged. No Rust code or new model operator is introduced.

Reproduction sources are `prediction_confidence_test.cpp`,
`check_prediction_confidence_session.py`, `run_mistral_prediction_confidence.py`
and `audit_mistral_prediction_confidence.py` under `test2026/metal/`. The runner
uses `LIB SOURCE INPUT OUTPUT`; audit flags are `--result`, `--manifest`,
`--output`, `--fixture`, `--reference`. All raw native/operator/session results,
pre-run build/policy/generation manifests, enabled/disabled build checks,
independent audit and `summary1.json` are retained under
`out/research/mistral7b-local-coder/mistral-draft-confidence1/`. The first build failed
because the confidence include was nested under the fused-head flag; its log
is retained. The measured build uses the corrected independent include. Existing dirty work is
preserved by an exact three-way residual proof; the measured build uses the
selected clean native sources.

The measured confidence-enabled native library SHA256 is
`ee55a9f074237f7fa46c18d1b4574aa23c205e36ab495f12ea65aade913cbb5d`.


The corrected **indexed occupancy study** supersedes the invalid eight-row
operator table. Kernel transformations operate only on the shader body and
assert the exact number of replacements. Each complete physical output is
filled with a specific NaN payload before **every** warmup/sample/capture,
outside measured spans. Active outputs must be finite, meet the original
`2e-6 +4e-6 * sum_abs_products` FP64 bound and match native-control bits;
inactive rows must retain the exact poison bits. Noncontiguous selected-row
permutations and small N1/7/17/33 tails are included alongside the five actual
Mistral projection dimensions. A deliberately four-row writer declared as8
fails with an unwritten active row, reproducing and catching the old gap.
The negative guard passes twice for the corrected collectors. An initial
replacement-count preflight error is retained; no positive run starts until
the known-bad writer is rejected for missing writes.

Three serial fresh processes cover45 shape/occupancy cases each, occupancies
1/2/3/4/8, two cache-pressure conditions, four variants and three warmups/15
samples per cell. Cohort1 explores max8 templates for smaller occupancies.
Cohorts2/3 repeat the same binary/source and use the exact original max4 template
for occupancies up to4, with a separate true-max8 template. The controls retain
the existing ordinary indexed kernel, or the admitted four-row blocked kernel
where appropriate; eight-row controls execute two4 projections in one command.
All584,527,104 physical output checks pass, including inactive poison checks;
1,014,804 FP64 reference outputs are computed per collection. The auditor
verifies every retained binary, source/executable manifests and coverage/timing
counts and rejects30 corrupted records/binaries.

Three-active-row tile4 projections show a candidate worth native testing:

| N / K | Control / candidate GPU ms, cohort2 | Control / candidate GPU ms, cohort3 | Operator ratios |
|---|---:|---:|---:|
|4096 /4096 |0.1193 /0.0989 |0.1195 /0.0962 |1.206 /1.242x |
|14336 /4096 |0.3979 /0.2981 |0.3927 /0.2977 |1.335 /1.319x |
|4096 /14336 |0.4356 /0.3173 |0.4395 /0.3227 |1.373 /1.362x |
|32768 /4096 |0.7423 /0.5895 |0.7894 /0.6029 |1.259 /1.309x |

These are standalone MSL4/safe-math sweep-condition operator medians; a256MiB
sweep is not proof of a cold cache. They do not qualify the native runtime's
Metal build, actual target verification, batching or serving performance.
Native admission still needs its own prepared-profile and complete-state
checks. Narrow KV outputs and other occupancies are not admitted by this study.

Correctly writing all eight rows gives a much smaller benefit than the old
invalid collector claimed. Tile4 ratios against two4 are1.097 /1.092x for
4096/4096,1.097 /1.105x for14336/4096,0.993 /1.013x for4096/14336 and
1.160 /1.222x for32768/4096; the narrow1024/4096 shape is1.433 /1.413x.
These do not establish a model gain or justify a native target8 path, especially
given the separately measured increase in deeper draft work.

The existing confidence GPU reference test now also poisons both output
matrices before each case. It again passes211 cases over24,245,894 input logits,
with six rejection checks. This is a test hardening change, not a change to the
confidence kernel/API or its earlier native generation results. The native
runtime, original weights, protected serving DLL and frozen comparator data
remain unchanged; the full2x/half-TTFT and coding-quality goals remain open.

The repo's `experiments/mistral_verify8_projection_probe.mm` now contains the
corrected occupancy collector. Run it with `NEW_OUTPUT 15`; the optional third
argument `negative` must fail for the deliberately missing-row writer. The
independent `audit_mistral_occupancy_tiles.py` takes the campaign directory,
`--cohorts 1 2 3` and a new `--output` path. Full corrected source, binaries,
pre-run manifests, raw samples/outputs, negative controls, confidence test
rebuild, audits and `summary1.json` remain under
`out/research/mistral7b-local-coder/mistral-occupancy-tiles1/`. The old raw
collector is preserved separately with an explicit invalidation and a
superseding `feasibility-summary2.json`; it must not be used as valid8-row
operator evidence.


### Native three-active-row projection qualification

The corrected indexed-operator study is now exercised through the production
MSL3.1/safe-math Metal build. `GARNET_METAL_MISTRAL_THREE_ROW_REUSE` is a
separate default-off option requiring `GARNET_METAL_MISTRAL_DECODE_REUSE`.
It selects the existing max-four-row/output-tile-four shader for exactly three
active rows at K/N 4096/4096,4096/14336,14336/4096 and4096/32768. Other
occupancies, narrow KV projections and all other shapes retain their controls.
Profiles still compile all admitted occupancies before readiness; no xModel
operator, tokenizer or public inference API changes. Native verification stays
bounded to four physical token rows. The candidate library SHA256 is
`18418fe111a766aab3a543357a8ce55732cbd64eca450984b624c5591095fb2e`.

Thirty native CTests pass. The added component test runs with both the unchanged
control and candidate libraries:532 checks/16,121,820 FP32 output words and21
invalid-index rejections per engine, every four-row active mask and permutation,
NaN-poisoned outputs, exact scalar-reference bits, zero inactive outputs,
no request-time profile compilation and zero remaining owners. Its initial
coefficient-type mistake was rejected before data execution and corrected to
explicit FP16; that failed log remains in the research folder.

Two fresh serial complete-generation cohorts reverse engine order. Every
cohort/engine runs all eight frozen coding contexts to their original1024/EOS2
limits with the same resident draft policy. The independent audit checks
12,208 target IDs, full32-layer KV/hidden hashes and all draft proposals against
frozen ordinary-generation witnesses;60 corrupted evidence variants are
rejected. No vLLM/SGLang baseline is rerun.

| Serial cohort | Control native decode tokens/s | Three-row candidate tokens/s | Candidate/control |
|---|---:|---:|---:|
|1:control then candidate|31.967|30.523|0.955x|
|2:candidate then control|29.600|30.063|1.016x|

This does not show a consistent full-generation improvement. Only3 verification
passes per engine use three valid tokens, compared with1212 four-token passes
and3 two-token passes;30 additional target calls use ordinary one-token
fallbacks. The verification vocabulary head always projects all four physical
rows, so partial control passes still have one blocked head dispatch. Three-row
candidate passes have160 blocked body projections plus that head. In normal
three-request decode the vocabulary head is itself indexed, and the control has
zero blocked projections versus161 for the candidate. Those scopes must not
share a counter expectation.

Evidence is retained under
`out/research/mistral7b-local-coder/mistral-three-row-native1/`, including
`generation-manifest3.json`, `generation-audit3.json`, original failed assertions
and raw outputs. The first collector guard incorrectly expected zero blocked
heads for partial verification and stopped; it was corrected using frozen
native route evidence. The first CPU audit used a record-stride variable later
shadowed by a chunk-width loop; a separately saved corrected validator audits
the original unchanged native results. Neither failed attempt is pooled.

The continuous B4 early-finish test has four distinct original coding prompts:
row0 finishes at2 generated tokens, leaving126 native decode commands for the
other three rows through128 tokens each. It verifies all386 generated IDs,
full per-request KV/hidden hashes and the inactive row's full KV before/after.
Steady command timing excludes prefill, scheduler, tensor preparation and hash
checks; it is not matched HTTP aggregate throughput. Rejected test-setup attempts
are preserved: fixed scheduler plans omit `prefill_chunk`, requiring the
configured128 fallback rather than1, and preparation compile time must be
recorded then reset once before checking request-time compilation.

The two reverse-order serial early-finish cohorts have the following aggregate
rates for their126 three-survivor native decode commands (378 tokens per row
of this table). They exclude the initial four-owner decode command.

| Serial cohort | Control command tokens/s | Three-row candidate command tokens/s | Candidate/control |
|---|---:|---:|---:|
|1:control then candidate|58.040|62.283|1.073x|
|2:candidate then control|56.018|63.537|1.134x|

The independent batch audit passes all1544 generated IDs, exact paired
32-layer KV/hidden hashes and both inactive-row checks per process;56 corrupted
evidence variants are rejected. Each process prepares12,824,494,080 Metal
bytes (a readiness snapshot, not a peak) and closes with no native owners.
The configured feature-off source is also rebuilt and passes the532-case
projection test; enabling the feature without its parent Mistral engine is
rejected at configuration. The original24-sequence confidence audit still
passes with the generalized validator, including its corruption checks.
`batch-manifest4.json`/`batch-audit4.json` bind the fresh measurements; the
separate passing functional preparation run is not pooled into them.

The feature remains default-off. The observed7.3%/13.4% command-rate gains do
not establish a matched HTTP serving improvement. These native studies do not
establish the requested2x decode/aggregate or half-TTFT targets, HTTP statistical
admission or coding-quality admission; the strict coding score remains0/8.

### Upstream Mistral v3 protocol and coding-quality diagnosis

A separate profile checks an asset-format issue without changing any frozen
baseline. The converted checkpoint's default chat template emits
`<s> [INST] USER [/INST]`; upstream Mistral v3 emits
`<s>[INST] USER[/INST]`. The extra spaces produce two extra token IDs in all
eight original coding prompts. The pinned upstream source is
[mistralai/Mistral-7B-Instruct-v0.3 at c170c708](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/c170c708c41dac9275d15a8fff4eca08d52bab71/tokenizer_config.json).
The converted tokenizer.json and SentencePiece model are byte-identical to
the upstream tokenizer assets; the drift is in chat framing, not vocabulary,
merges, token decoding or model weights. All eight old saved completions decode
to exactly the stored native text with both independent CPU decoders, including
indentation. This rejects a native whitespace-decoding explanation for the
old strict failures. Existing single-fence diagnoses also contain algorithm
errors; explanatory text is not their only issue.

`prepare_mistral_v3_protocol.py` freezes the explicit
`mistral_v3_upstream_single_user_text_v1` profile. It validates pinned upstream
assets, retains the exact eight user messages and original1024/EOS2 budgets,
and checks all prompt IDs against both upstream Jinja/Hugging Face rendering
and independent SentencePiece instructions `[BOS,INST]+encode(user)+[/INST]`.
The corrected prompt lengths are82,69,101,117,118,131,144,149. Native C++
tokenization checks those IDs again before collection. This profile qualifies
only these single-user text requests, not tools, system messages or arbitrary
conversation serialization. It does not replace the converted asset defaults.

An unchanged native Metal library (`ee55a9f0...`) and up-to128/B1/C4096 profiles
complete all eight new requests. All3,229 output IDs terminate naturally;
decoded text matches both independent CPU decoders. The independent audit
rejects16 corrupted evidence variants, verifies output/evaluator provenance,
KV cursor and state-hash completeness, bounded readiness/run allocations and
zero owners after close. State hashes are recorded; no new reference-engine
generation or full-state equality is established by this experiment.

| Coding-quality gate | Original converted template | Explicit upstream v3 template |
|---|---:|---:|
| Same strict evaluator and contracts |0/8|0/8|
| Complete requests at original budget |8/8|8/8|

The unchanged evaluator's sandbox startup passes. The corrected framing does
not solve the coding-quality failure: all eight completions still violate the
complete-code output contract. No prose removal, code repair, task filtering or
threshold change is used to promote a score. Inference operators, tokenizer
implementation and model artifacts remain unchanged.

Artifacts are retained under
`out/research/mistral7b-local-coder/mistral-v3-protocol1/`, including pinned
upstream assets, `fixture1`, `native1`, `legacy-decoding-audit1.json`,
`quality1.json` and `audit1.json`. Existing vLLM/SGLang results are preserved and
apply only to their original frozen prompts; this changed profile has no
matched reference-engine speed or quality comparison. Mistral remains
unqualified as the requested reliable local coder, and the2x/half-TTFT and
broader model-qualification goals stay open.


### Packed Qwen2.5-Coder draft on unchanged Mistral target

The C++ packed-checkpoint adapter now explicitly recognizes the `qwen2` dense
family, with a distinct `mlx_qwen2_affine_int4_g64_fp16`/`...bf16` IO format.
The existing Qwen2 xModel operator graph already implements projection biases
and omits Q/K normalization; no xModel operators or inference APIs change.
The supported representation remains affine INT4/group64 with typed coefficients
and mapped tensor bounds. Mistral, Qwen3 and hybrid routing retain their existing
identities. This is draft onboarding, not a change of target model.

The pinned asset is `mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit` revision
`6b16732e5af5cd9bd600186ad59fa618867ef7a4`,278,064,920 safetensors bytes.
Download sizes and LFS SHA256 are checked. Actual coefficients, norms and72
Q/K/V bias tensors are FP16, and the tied head uses the packed embedding table;
no weight conversion is applied. The configuration is hidden896/FF4864/24layers,
14 query heads/2 KV heads/dim64, vocabulary151936. At B1/C512 its analytical
FP32 KV payload is12 MiB, versus112 MiB for the existing Qwen3-0.6B draft.
These payload counts are not measured peak allocations.

Thirty native CTests pass. Small metadata fixtures run for all three dense
families:6 accepted representation cases and30 rejected invalid/ambiguous
cases. An independent actual-weight FP64 oracle checks60 projection cells
(15 matrices across first/last layers and embedding head, rows1/2/4/16),
2,760 sampled dot products with the original2e-6+4e-6*sum_abs bound. The worst
error is0.013144 of that bound;6,272 FP32 embedding words match bitwise.
Seven real B2/C64 reconciliation scenarios and13 invalid-state/index/epoch
checks pass; the5-step confidence/session test passes12 invalid transitions.
Both tests are generalized from config dimensions and rerun successfully on
the existing Qwen3 draft.

A separate serial MLX/native standalone collection covers all eight original
user messages at64 output tokens. Both ordinary and chunk16 native prefill
match every saved MLX token and decoded byte in all16 runs (1,024 native-token
comparisons against512 reference IDs). This is a bounded functional check,
not full coding quality or independent16-task coverage. MLX uses its FP16
state policy; native decode/state stays FP32. The collectors' initial Python
int-token and prefill-counter setup mistakes are retained as terminal failed
attempts; neither completed comparison data and neither is pooled.

For complete target generation, both drafts use the same new native library,
unchanged original Mistral prompts and1024/EOS2 limits, draftC512/T16, max8
draft tokens and max3 target proposals. Each process prepares target profiles
and its own draft before requests. All models execute serially; the second
cohort reverses draft order. The existing base-mode collector's
`occupancy-control` argument selects the identical target route in both cases,
not a new occupancy experiment. Draft chat prefixes come from their respective
immutable tokenizer configs. `prepare_mistral_coder_draft.py` reproduces the
measured candidate input exactly without executing a model.

| Serial cohort | Qwen3-0.6B draft: native target tokens/s | Qwen2.5-Coder-0.5B draft: tokens/s | Candidate/control |
|---|---:|---:|---:|
|1:control then candidate|30.895|34.152|1.105x|
|2:candidate then control|29.362|33.628|1.145x|

The candidate makes1,179 target calls versus1,248, accepts1,865/3,369 proposed
tokens versus1,796/3,645 and executes1,875 draft steps versus2,014. Those counts
are identical between cohorts. Draft wall time falls31.595->24.107 seconds
and32.218->24.028 seconds. Prepared Metal allocation is6,082,428,928 versus
6,246,580,224 bytes, both snapshots rather than peaks. Per-case rates are
retained:one case in cohort1 is1.3% slower; no claim that every request improves.

The independent causal audit retokenizes every draft prefix/proposal, checks
acceptance/commit/rollback counters and all12,208 target IDs plus full32-layer
KV/hidden hashes against frozen ordinary-native witnesses. Different draft
models are allowed different proposals; the old Qwen3 control still must match
its frozen proposal trace. The audit rejects52 corrupted evidence variants.
The original24-sequence confidence auditor also passes with its default family
and all corruption checks. Every native owner count is zero after close.

The library SHA256 is
`4494a6a2fb7a70654eed67bf7774b8d0577fb6cf9996848e04b8d1f0168e4f77`.
Raw assets/manifests, reference outputs, numerical/state tests,
`comparison-manifest1.json`, `comparison-audit2.json` and per-case results are
retained under `out/research/mistral7b-local-coder/mistral-coder-draft-native1/`.
Frozen vLLM/SGLang artifacts are untouched and not rerun. The observed native
10.5%/14.5% gains do not establish matched HTTP throughput, TTFT, statistical
admission or the2x goal. The target's strict coding score remains0/8 by full
output identity; clipped draft results do not qualify a new main local coder.
The draft selection is not promoted to a default serving profile. The full
performance, quality and subsequent27B/9B/4B goals remain open.


A subsequent default-off `GARNET_METAL_MISTRAL_STREAMED_PREFILL` path now
streams one packed projection into a reusable FP16 buffer and executes an eagerly
prepared MPS matrix product, with FP16 operands and requested FP32 output.
Vendor internal accumulation is not separately instrumented. The native state
and decode policy remain FP32; no xModel operator or tokenizer changes are made.
The exact admitted *experimental route* dimensions are N/K14336/4096,
4096/14336 and4096/4096, with512 actual active token rows. Smaller counts and
other dimensions retain existing kernels. The up-to512 profile does not pad
short requests to512 for this path. Selected-row permutations are gathered and
scattered explicitly; existing validation rejects duplicates and bounds errors.

The session prepares three MPS plans before request leases, with one shared
scratch set across all layers/projections:117,440,512 bytes of half weights,
14,680,064 bytes of half inputs and29,360,128 bytes of float output. Total
161,480,704 bytes (154MiB), allocated under the existing recommended-working-set
budget and exposed as three distinct inventory roles. There is no persistent
expanded model cache. Encoding creates no plan or native matrix owner; abort
recovery and the existing explicit control route are retained. Private workspace
operands are rejected. A diagnostic `prefill_set_streamed_affine` switch selects
the A/B route only between drained native command batches.

Standalone operator collections first tested128/256/512 rows. The256-row MLP
down improvement was only1.064x in one sweep-conditioned collection, while128
regressed. This justified the initial512-active-row threshold, rather than a
blanket prefill replacement. The final native component collector includes
input gathering, unpacking, MPS and output scattering, with poisoned outputs,
short/empty/ragged selected counts, dense/permuted/reversed maps, a narrow KV
control and a256-row fallback. The independent full FP64 audit checked all
170,917,888 saved output values against the unchanged2e-6+4e-6*sum_abs bound,
and rejected10 deliberate missing/wrong-result corruptions. The final ON build
passes31 CTest cases; the OFF build passes30. These are native operator gates,
not full serving admission.

The native B1/C4096/T512 model study runs two fresh-process cohorts in opposite
orders, control/candidate then candidate/control, with one warmup and three
prefill measurements per case. Both routes prepare the same plans/workspace;
only encoding differs. Embedding materialization, readiness, HTTP and queueing
are outside these prefill timings. Nominal256 uses the frozen263 actual IDs;
2048 uses the frozen long-prompt IDs unchanged. The512-token prefix and appended
newline2049 case are additional native probes. All inference is serial, and no
vLLM-Metal/SGLang baseline is rerun. Median prefill wall durations are:

| Actual prompt tokens | Cohort1 control / candidate s | Ratio | Cohort2 control / candidate s | Ratio |
|---|---:|---:|---:|---:|
|263 |1.037 /1.208 |0.858x |1.262 /1.247 |1.011x |
|512 |1.610 /1.412 |1.140x |1.873 /1.326 |1.412x |
|2048 |7.472 /5.844 |1.279x |8.089 /5.972 |1.355x |
|2049 |7.775 /6.108 |1.273x |8.226 /6.046 |1.360x |

The independent trace audit verifies all2048 generated IDs across16 sequences,
and exact full32-layer KV/hidden witnesses at both prefill and decode boundaries;
11 deliberate trace/inventory corruptions are rejected. All processes report
zero native owners and at most4MiB device allocation after close. Readiness
allocation is7,967,358,976 bytes in both routes (a snapshot, not peak/free-RAM or
paging evidence). The263-token fallback shows no consistent speed gain, and
larger-input gains are1.28–1.36x, below the half-TTFT goal. These descriptive B1
native results establish neither HTTP throughput nor B4/prefix/continuous-batch
admission of this route, and do not change the strict coding score of0/8.

The first four native traces are explicitly invalidated for full-trace admission:
the research recorder retained the mutable slot list, so its prefill slot
metadata was later changed by decode. Original records remain untouched;
corrected records copy the list and all four processes are rerun using the final
library. Earlier native version1 source/library and the failed build/preflight
logs are preserved. Only the corrected version2 model audit is used above.
Final numerical data is also regenerated/audited with the final native binary.

Reproduction sources are `streamed_affine_test.cpp`, `audit_streamed_affine.py`,
`run_mistral_streamed_prefill.py`, `audit_mistral_streamed_prefill.py` and the
standalone `experiments/mistral_streamed_prefill_probe.mm` with
`audit_streamed_prefill_probe.py`, under `test2026/metal/`. The standalone probe
compiles with C++17/ARC and Foundation, Metal and MetalPerformanceShaders
frameworks; it takes a fresh output directory and9 repetitions. Probe audits
accept `--collection operator1|operator2`; native model audits require
`--revision2` (written as `--revision 2`). Raw commands, libraries, fixtures,
full audits, invalidation and manifests remain under
`out/research/mistral7b-local-coder/mistral-streamed-prefill-native1/`; initial
operator studies remain under `mistral-prefill-fp16-cache1/`. Default-off routing
and the overall performance/quality goal remain unchanged pending further
qualification. ANE is separate and is not used for these new MPS results.


The next qualification retains the same final native library from the streamed
prefill checkpoint, with no C++/xModel/Metal runtime change. Native B4/C4096/T128
leased prefill and ordinary greedy generation now run against the frozen C4
prompt IDs in two serial fresh-process cohorts, control/candidate then
candidate/control. Each case has one warmup and three measured prefills followed
by one128/EOS2 generation per request. Embedding construction, readiness and
state witnesses are outside native timing; these are not HTTP/queue-inclusive
numbers. The nominal256 inputs actually contain263/266/263/264 tokens, and the
nominal2048 inputs2048/2051/2048/2049. Every full512-active-token command must
use160 streamed projections in the candidate; smaller ragged tails retain the
existing kernels. Readiness exposes the same three plans/154MiB scratch set
and11,190,976,512 Metal-allocated bytes in both routes (snapshot, not peak/free
RAM/paging evidence). There are no request-time profile builds.

| Nominal input / B4 | Cohort1 control / candidate prefill s | Ratio | Cohort2 control / candidate prefill s | Ratio |
|---|---:|---:|---:|---:|
|p256-c4-o128 |3.051 /2.553 |1.195x |3.717 /2.610 |1.424x |
|p2048-c4-o128 |30.434 /22.120 |1.376x |31.602 /22.427 |1.409x |

All4096 generated IDs across32 request sequences and complete32-layer private KV
and visible hidden hashes agree at prefill/decode boundaries. The independent
auditor rejects12 trace corruptions. Descriptive native aggregate rates from
the final prefill wave plus ordinary decode are49.678 /49.016tok/s for the
short candidate and13.260 /13.102tok/s for the long candidate. These exclude
HTTP/readiness/state-witness overhead. Frozen HTTP reference medians are still
45.235/42.194tok/s (short vLLM-Metal/SGLang) and21.179/16.337tok/s (long); they
are not rerun. The native rates are not a matched serving admission, and the
90.470/42.358tok/s2x targets remain unmet. Prefill alone improves1.38–1.41x
on the long B4 case, rather than the required2x.

A separate B4 native token-prefix and continuous-scheduler study uses128/512
prefixes, a384MiB accounted host-cache budget and16 entries. It stores64/128/512
checkpoints, uses the longest stored matching boundary (a change at token300
restores128), checks exact return, append, changed suffix, shorter query,
replacement and stale epochs, and compares14 lookup-free request references
per process. All private per-request KV hashes, current decode hidden hashes
and emitted IDs agree for matching precision histories. Four full B4 prefill
commands in the direct episode plus one in the scheduler episode exercise the
MPS path, while individual replay/sparse-row commands retain original kernels.
Fourteen before/after survivor KV checks per process establish that row0
release, reuse, restore and prefill leave B/C/D unchanged. Five invalid native
operations are rejected without state/cache mutation, including stale dispatch
and late preparation. Native A-to-E-to-A scheduling includes three mixed ticks
and one head-only exact hit, with different request IDs and four distinct epochs
on physical row0. Prefill/decode remain separate native commands; packed token
segments are scheduler metadata, not a fused mixed kernel. Cache eviction is
observed within the384MiB accounted bound; these snapshots do not prove a peak
host-memory bound. All processes close with zero native owners and at most4MiB
Metal allocation. The prefix auditor rejects12 additional corruptions.

There is an explicit precision limitation. The first prefix control collection
fails the cold full-state gate for a scheduled append. Native policy captures
finished decode histories, so the actual hit is129 tokens:128 prompt tokens
computed through FP16 prefill plus one generated input computed through FP32
decode. A cold130-token FP16 prefill is a different precision history. A focused
lookup-free probe proves complete KV/current-hidden bit equality for both
cached129 versus the same FP32-decode history, and cached128 versus cold FP16
prefill; all four routes emit the same two IDs. Across the two different
histories, complete state hashes differ. Diagnostic samples at tokens0/127
agree, while token128 has0.0227% relative RMS difference and token1290.1578%
(maximum sampled absolute difference0.021513). These are samples, not a full
numerical-error qualification. Both original failed data and focused probe
remain untouched. The subsequent two collectors preserve the cold comparison,
require cold token agreement, and separately verify lookup-free matching-history
state. They do not turn the failed cold bit-equivalence gate into a pass.

Thus this checkpoint establishes B4 streaming parity and cache/scheduler
correctness for the original precision history. Cold FP16 equivalence of
FP32-generated-history cache entries, cached HTTP admission, coding quality
and the overall performance goal remain unproven. Strict coding score remains
0/8; no quality fixture is changed. The new path stays default off. The maximum
configured snapshot budget remains512MiB; a complete Mistral2048-token FP32 KV
snapshot needs512MiB before hidden/metadata, so it cannot fit that limit. This
last bound is derived from the native layout/configuration, not a new runtime
capture result. FP16 KV/state policies or precision-aware cache contracts need
separate qualification.

Reproduction tools in `test2026/metal/` are `run_mistral_streamed_batch.py`,
`audit_mistral_streamed_batch.py`, `run_mistral_streamed_prefix.py`,
`audit_mistral_streamed_prefix.py` and `audit_mistral_prefix_precision.py`, plus
`experiments/mistral_prefix_precision_probe.py`. Collectors take LIB SOURCE
FIXTURE NEW_OUTPUT and control/candidate; the focused probe omits the mode.
Raw fixture/commands, both cohorts, prefix revisions, focused sample data,
full audits and scope limits are in
`out/research/mistral7b-local-coder/mistral-streamed-batch-prefix1/`.
Authoritative audits are `batch-audit1.json`, `prefix-audit2.json` and
`precision-audit2.json`; the original prefix control1 fails cold state parity
and remains explicitly excluded from broad admission. No frozen comparator,
protected serving library or model artifact is replaced. ANE is not used here.


A subsequent native experiment adds default-off
`GARNET_METAL_MISTRAL_PACKED_PREFILL`. For the three Mistral projection shapes
(N,K)=(14336,4096), (4096,14336), (4096,4096), it gathers selected FP32 input
rows once into compact FP16 storage, then performs inline Q4-to-half weight
dequantization and strict MPP multiply with FP32 accumulation/output. Packed
uint4 loads amortize coefficient reads, half4 input loads/shared-weight stores and
float4 output stores preserve alignment, and128/256-row tiles use256 threads
with K staging128. The per-accumulator K16 traversal and arithmetic are
unchanged. Other shapes, decode, and fewer than128 selected rows retain the
previous routes. No persistent expanded-weight cache, ANE execution, Rust, or
xModel operator change is introduced.

The up-to512 preparation creates three shape/stride plans, each with two tile
pipelines, and one14,680,064-byte half-input workspace per session. Actual
selected counts remain runtime data; the new kernels do not pad logical input
to512. Pipeline compilation and workspace allocation occur before request
leases. Gather parameters use an explicit16-byte uint4 ABI. New optional
session APIs expose `packed_prefill_dispatches`, `packed_prefill_plans` and
`prefill_set_packed_affine`; inventory identifies `packed_prefill_half_input`.
The dedicated packed counter is separate from the legacy MPP/reuse counters;
common affine-profile coverage still totals224 body projections. This
checkpoint builds packed ON/streamed MPS OFF: the half workspace is14MiB
versus154MiB for the prior MPS route, saving140MiB. Readiness snapshots are
7,820,558,336 Metal bytes for B1 and11,044,175,872 for B4, not peak-memory or
paging proofs. No request-time plan growth is observed.

Primary-source inspection uses installed/pinned MLX0.32.4 headers and Git blob
052dd761c48ff21f750ea6ff2203202a0be1f25e for its quantized dispatcher.
Strict vector IO alone improved selected operators by about1.1–1.27x;
larger tiles alone did not win. Compact half input produced the stronger
signal. The corrected standalone thirteen-variant collection passes full FP64
checks on524,812,288 saved values, including NaN-poisoned full output, one-writer
counts, redzones and exact half-input conversion checks. An earlier exploratory
half-input probe supplied12 bytes for a Metal uint3 constant requiring16; it
is explicitly excluded. The corrected probe and production use16 bytes.

The native operator campaign uses signed FP16 coefficients, ragged/permuted and
dense row maps, counts0/4/127/128/129/255/256/257/383/384/511/512, a narrow-output
fallback and a256-physical-row profile. All57 cells preserve control bits,
clear inactive outputs and remain finite/stable after poisoning. Guard tests
cover duplicate/out-of-range rows, unready inputs, abort/recovery, selection or
preparation inside live batches, and preservation of the explicit64-row control
API. Owner counts and prepared-plan counts remain constant during execution,
and return to zero after close. The independent unchanged-bound FP64 audit
checks339,738,624 retained values and rejects corrupt outputs. Thirty other
configured native CTests pass. The operator executable is byte-identical before
and after the later decoder-session preparation guard correction, so that
full operator audit binds the final build too. A separate default-OFF build
and its thirty tests pass.

A pre-existing telemetry issue is retained explicitly: `PredictRowsUnlocked`
submits the head but does not refresh session `gpuMs_`, so collectors reading
`head_gpu_ms` receive the preceding body duration again. Reported batch GPU
sums/ratios (including older streamed collections) are therefore unqualified.
The new auditor omits those comparison fields and marks the raw GPU aggregate
unqualified. Wall-clock prefill and aggregate throughput, token/state evidence,
and direct MetalContext operator GPU timers are unaffected. Correcting session
head telemetry is required before subsequent GPU-time admission.

The first prefix control catches a real lifecycle omission: the pre-existing
managed-row preparation guard was compiled only with streamed MPS. The final
library extends that guard to either streamed or packed prefill, before graph
registration or mutation. The failed record is retained. Final library2
74280b0d839a5159febc4a9a950ff239f1cbfc8e11005213d6e71a7f363a3e5f
then repeats both fresh-process B1 and B4 control/candidate cohorts, including
reversed order. Below are final-library medians across three prefills per case,
following one warmup. Embedding construction, readiness and state hashes are
outside timing. Each request continues128/EOS2 output tokens.

|B1 case|Control prefill s|Packed prefill s|Paired speedup range|
|---|---:|---:|---:|
|short263|1.195–1.287|0.916–0.924|1.304–1.392x|
|prefix512|1.828–1.916|1.129–1.142|1.620–1.677x|
|long2048|8.096–8.289|5.186–5.295|1.561–1.565x|
|ragged2049|8.270–8.438|5.213–5.273|1.586–1.600x|

|B4 case|Control prefill s|Packed prefill s|Paired speedup range|
|---|---:|---:|---:|
|p256-c4-o128|3.822–3.839|2.249–2.255|1.700–1.703x|
|p2048-c4-o128|32.284–32.333|19.333–19.501|1.655–1.672x|

All2048 B1 and4096 B4 generated IDs, complete32-layer private KV hashes and
visible hidden hashes agree at both prefill/decode boundaries. The independent
model and batch auditors reject11 and12 corruptions respectively. These native
rates are not matched HTTP results; frozen vLLM-Metal/SGLang campaigns remain
unchanged and are not rerun. The2x decode/aggregate and half-TTFT goal is not
achieved, and the prior strict coding score0/8 is not a new-route quality pass.

The final prefix campaign tests returning A, append, a change at token300,
shorter queries, E replacing A on row0, and unchanged B/C/D survivor states.
Fourteen lookup-free references per process match IDs and complete native state
for the same precision history. Fifty-one candidate prefill commands (including
single-request128-row replay and full B4 work) exercise the packed route.
Native continuous scheduling retains three mixed ticks, one head-only exact
hit, fresh row epochs, stale-epoch rejection and bounded384MiB accounted host
cache storage with eviction. Twelve deliberate trace corruptions are rejected.
Cold FP16 equivalence of cached FP32-decoded histories remains false and
explicitly unadmitted; this optimization does not normalize that difference
away. Prefill/decode still execute as separate native commands.

One earlier standalone tool issue is also corrected: its reuse64 control used
256 threads even though that kernel requires128. A deliberate wrong-launch
atomic-writer probe identifies917,504 duplicate output words. Native runtime
reuse64 already used128 and is unaffected. Historical standalone reuse64
latencies are unqualified; prior best-native selections used the separately
correct reuse128 route. The corrected legacy tool passes a fresh full-output
FP64 audit and now requires an explicit128-thread stamp, rejecting old
unstamped collections. Historical raw results are retained.

Reproduction tools are `packed_affine_test.cpp`, `audit_packed_affine.py`,
`run_mistral_packed_prefill.py`, `audit_mistral_packed_prefill.py`,
`run_mistral_packed_batch.py`, `audit_mistral_packed_batch.py`,
`run_mistral_packed_prefix.py` and `audit_mistral_packed_prefix.py` in
`test2026/metal/`. Raw sources/commands, both library revisions, failed prefix
control, full output payloads, all model cohorts, audits and scope limits are
retained in `out/research/mistral7b-local-coder/mistral-packed-prefill-native1/`;
standalone vector probes remain in `mistral-vector-prefill1/`. Final artifacts
are `model-audit2.json`, `batch-audit2.json`, `prefix-audit2.json`,
`full-native-audit1.json` and `corrected-legacy1/audit1.json`. The packed path
stays default off. Serving admission still needs explicit worker/profile/auditor
accounting for the new route, compact-owner checks, matched HTTP measurements
and coding qualification. No protected serving library/model or frozen
comparator result is replaced.


The next checkpoint corrects row-head telemetry and qualifies the packed route's
uncached HTTP contracts. `PredictRowsUnlocked` now refreshes session `gpuMs_`
from the just-completed MetalContext command before returning either ordinary
or confidence prediction. It does not change kernels, input precision, KV,
outputs or xModel operators. A synchronous regression reproduces the old bug
(the old head reports its preceding464ms body for an8ms head
call), then requires the new GPU span to fit its wall interval, with1ms clock
slack. Full, one-row, two-row and full confidence heads preserve complete32-layer
KV/hidden hashes and ordinary token IDs. The independent auditor checks old/new
prefill state equality, rejects six corruptions and binds both native libraries.
Initial test-harness setup failures remain retained and excluded; only final
head-old4/head-new4 records qualify. All31 configured native CTests pass.
Earlier saved batch GPU aggregates remain unqualified; they are not rewritten.

Serving profiles now explicitly declare `mistral_packed_prefill`, its128-row
minimum, three prepared plans and14,680,064-byte workspace. The worker validates
backend compatibility and saved session inventory before ready. New prepared
and dispatch trace fields distinguish160 eligible packed body projections from
legacy reuse64/reuse128 routes. Generic MPP coverage combines packed and legacy
counts (224 body projections), while native public legacy counters retain their
existing meanings. The independent HTTP auditor derives eligibility from native
request plans, checks exact stage/row/workspace coverage, and rejects moving
packed work into legacy counters while preserving the total. It also checks the
full saved inventory. Replaying the immutable prior reuse128 campaign with the
new auditor preserves its legacy guarantees; no comparator inference is rerun.

Final native library SHA256 f2f97532c767d9a7e95c7e6d708d18cf9894214b09287f69d7922921d08d7459
serves the same frozen six performance cells and eight coding cases through
HTTP/SSE, B4/C4096/global512, per-request prefill chunk128, prefix cache off,
ordinary greedy decode, identical model and request artifacts. Compact-owner
B1/B2 families are prepared before readiness. This is one fresh native process,
no warmups and one wave per cell; comparator columns remain the frozen15-wave
medians. The table is descriptive, not repeated2x performance qualification.
Garnet retains FP32 KV/decode while the Apple reference backends use FP16 state.

|HTTP cell|Garnet output tok/s|vLLM-Metal frozen|SGLang frozen|2x best target|
|---|---:|---:|---:|---:|
|p256-c1-o128|18.972|22.021|22.042|44.084|
|p256-c2-o128|35.208|31.464|32.127|64.254|
|p256-c4-o128|54.359|45.235|42.194|90.470|
|p2048-c1-o128|9.405|12.345|12.518|25.036|
|p2048-c2-o128|12.361|16.536|14.482|33.071|
|p2048-c4-o128|15.172|21.179|16.337|42.358|

The native single-request short/long TTFT is0.864/4.471s and decode21.634/13.917
tok/s. The frozen half-TTFT targets are0.344/2.079s and2x decode targets
50.134/41.502tok/s. All2x aggregate, single decode and half-TTFT targets remain
unmet. Short C2/C4 single-wave throughput exceeds the corresponding reference
medians, while C1 and all long cells remain slower; this is not a causal paired
performance claim under identical system conditions. No frozen reference raw
result or profile is changed.

All22 requests complete, including4-way activity and compact-owner transitions.
The independent auditor checks SSE/native/frozen token identities, all eight
frozen native coding controls, row epochs, scheduling, prepared plans, dispatch
coverage and clean shutdown. It rejects41 corruptions, including five dedicated
packed profile/workspace/route/accounting faults. There are4844 emitted IDs
across performance and coding requests, zero native owners after close and
1,507,328 remaining Metal bytes. Request/trace validation does not independently
prove GPU KV contents, dynamic arrivals or peak memory; previous native state
qualification and head regression have their own narrower scopes.

Strict coding evaluation remains0/8 with the unchanged suite and evaluator
(675ea667c357f361ce39e3863b61193dc6b32f4a229eb7d29a17803b75408786 and
a73809aab308180338ab1df644afdd8c4970bc136d6b0bc93c049c8862ebcc7a).
The evaluator sandbox probe passes. Exact reproduction of the weak native
coding controls is an integrity result, not coding-quality admission. The packed
path remains default off; no general local-coder, cached-HTTP, cold-FP16-state,
2x or ANE admission follows from this campaign.

Reproduction/evidence is retained under
`out/research/mistral7b-local-coder/mistral-packed-http1/`: clean source overlay,
build commands, both head-regression library bindings, failed harness attempts,
`head-audit1.json`, `admission1/` engine/profile/tools manifests, full saved
inventory, raw HTTP/native streams, memory/paging samples, accounting/native
negative audits, coding responses and `quality.json`. `admission1/comparison.json`
and `summary1.json` keep actual sampling scope and unmet targets explicit.
New repository tools `run_mistral_head_timing.py` and
`audit_mistral_head_timing.py` accompany the serving worker/auditor updates.
Further work must target measured prefill/decode costs, repeated matched native
performance, coding quality and the retained cache precision limitation. The
existing isolated-op profiler disables fusions and splits submissions, so its
latency must not be substituted for the optimized serving path.


### Optimized GPU counter diagnostics (2026-11-06)

The M5 capability probe supports stage-boundary timestamps, not dispatch-boundary
sampling. Empty marker encoders produced zero samples; independent no-op markers
failed to bracket the actual work because Metal can overlap them. Their rejected
raw results are retained. The selected diagnostic attaches counters to actual
native compute encoders without disabling graph fusions or adding submissions.
It does not observe MPS-owned encoders and never sums overlapping intervals as
elapsed GPU time.

`metal_decoder_session.prepare_gpu_counters()` is explicit, diagnostic-only and
must run once on a ready session before row leases. Preparation allocates one
32 KiB/4096-sample buffer and reserves bounded metadata. Sample overflow rejects
the diagnostic. Clock calibration pairs, every raw interval and command index
are emitted after completed submission. The serving default remains disabled;
xModel operators and tensor/weight precision are unchanged.

Evidence: `/Users/shawn/CantorAI/out/research/mistral7b-local-coder/mistral-fused-gpu-profile1/`.
A fresh counters-off process followed by counters-on used the same frozen B4
short/long prompts, C4096, up-to128 chunks, packed prefill and ordinary greedy
decode. All 64 generated IDs, all 32 layers of full KV state, hidden state and
route/submission counters matched. The audit covers 36 batches and 29,380 direct
encoders, rejects 11 corruptions, and validates raw GPU/CPU clock conversion.
All 31 CTests passed. No vLLM or SGLang inference was rerun.

For 14 full long-prompt chunks (offsets256 through1920), median total GPU span
was1149.0 ms; the median union of packed Q4 prefill encoder intervals was640.1 ms,
pointwise110.3 ms, direct attention97.7 ms, remaining affine projections71.9 ms
and state kernels56.4 ms. These category unions can overlap and must not be
added. Median direct-encoder coverage was84.6%; MPS work and gaps are not assigned.
Counters-on long prefill wall time17.819 s versus counters-off16.940 s demonstrates
instrumentation overhead. These are diagnostics, not new serving benchmarks.
The published HTTP TTFT/throughput comparison, strict coding0/8 result and unmet
2x targets remain unchanged.

Next optimization hypothesis: preload two K16 operand fragments and use vector
half4 right-fragment reads while preserving each accumulator's K order. Test
full FP64 operator outputs and writer coverage before model/HTTP admission.


### K32 preload screen: mixed, not admitted (2026-11-06)

`/Users/shawn/CantorAI/out/research/mistral7b-local-coder/mistral-k32-prefill1/`
retains baseline, vector-right-load and two-K16 preload shader/source/binaries,
serial9-repetition hot/swept data and all output arrays. Both candidates preserve
each accumulator's arithmetic order. The full independent FP64 audit passes
121,110,528 values; every retained word matches the existing kernel exactly.
Nine shapes cover128/256/512 actual rows and all three whitelisted projections,
using the production128/256 tile policy. Packing is included in timed commands.
The audit rejects16 corruptions. Two experimental source-generation failures
were rejected by Metal compilation before dispatch; their diagnostics and
corrected fresh runs are preserved.

Gains and regressions are mixed, with no reliable gain for the dominant B4
512-row mix. No candidate was added to the native runtime, and no model/HTTP or
writer-map admission is claimed. These results do not change serving TTFT or
throughput. Next investigate the actual compiled scratch/lifetime plans and
larger real prefill batches within measured working-set limits, rather than
assuming another fragment scheduling change will deliver2x end-to-end speed.


### Compiled pooled memory profile (2026-11-06)

The existing lifetime/alias, hidden ping-pong and fully-written attention-output
pool features are now qualified together for the strict packed Mistral profile.
They are explicit build/runtime profile options; defaults and xModel operators
are unchanged. The tracked initial-cache preset and worker profile are:

- `test2026/metal/profiles/mistral7b_packed_pooled_memory.cmake`
- `test2026/metal/profiles/mistral7b_packed_pooled_memory.json`

Use the CMake preset with `-C` in a fresh dedicated build directory and supply
local xlang3 SDK/runtime paths. All118 Garnet cache settings match the measured
build. Disabling lifetime reuse while retaining aliases is rejected. The
profile still has an up-to512 global physical-token-row bound, B4/C4096/T128,
three packed projection plans and one14MiB half-input workspace. Actual shorter
chunks and inactive rows remain runtime data; requests are not padded to512.

Evidence root:
`/Users/shawn/CantorAI/out/research/mistral7b-local-coder/mistral-profile-memory1/`.
With224 ordinary prefill graphs, prepared Metal allocations fall from
11,044,175,872 to9,346,269,184 bytes (10.29 to8.70GiB,15.4% less).
The independent audit verifies224 alias groups/448 slots,91 distinct shared
scratch buffers and14 private hidden ping-pong buffers. Scratch/hidden buffers
do not alias private KV or checkpoint weights. Both full32-layer KV states,
current hidden and all64 generated IDs match a fresh control; the audit rejects
10 corruptions. All32 CTests pass. These are snapshots, not peaks or paging-free
claims.

The complete HTTP profile also prepares the14 compact owner families. Its
prepared Metal snapshot falls from12,839,174,144 to9,954,541,568 bytes
(11.96 to9.27GiB,22.5% less). The same frozen14 plans/22 requests run in one
fresh process with no warmups and one measured wave per cell. All4,844 generated
IDs, prompt IDs, request bodies, finish reasons and rendered text match the
preceding packed HTTP checkpoint exactly. The native route/SSE audit rejects
49 corruptions and verifies explicit alias preparation and32 hidden bindings
per prefill body. The first new audit checked the wrong event order and failed;
its raw failure is retained. The corrected audit consumes a pool event after
the completed affine profile, rejects missing/duplicate events, and reuses the
original inference data without rerunning it. Accounting and unchanged quality
evaluation pass; strict coding remains0/8.

| Nominal prompt | Concurrent | Garnet aggregate tok/s | Garnet TTFT s | Frozen vLLM-Metal TTFT s | Frozen SGLang-MLX TTFT s |
|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 19.935 | 0.761 | 0.687 | 0.732 |
| 256 | 2 | 35.732 | 1.188 | 1.417 | 1.285 |
| 256 | 4 | 54.523 | 2.259 | 2.496 | 2.240 |
| 2048 | 1 | 9.237 | 4.446 | 4.219 | 4.159 |
| 2048 | 2 | 12.145 | 9.314 | 6.814 | 6.450 |
| 2048 | 4 | 15.225 | 17.607 | 11.570 | 10.884 |

Reference medians retain their15 frozen waves; neither comparator was rerun.
Garnet retains FP32 KV/decode while references use FP16 state. The pooled
profile does not show a long-prefill serving speed gain, and all2x throughput,
2x single decode and half-TTFT targets remain unmet. It is qualified here as a
memory/correctness improvement, not repeated performance or coding admission.

The same pooled library also passes the native A-to-E-to-A return/append cache
campaign against the frozen control: full survivor KV/hidden, longest stored
prefix, stale epochs, host-budget eviction,14 lookup-free references, three
mixed scheduler ticks and one head-only hit agree. Ten corruptions are rejected.
The previously documented cold FP16 versus decoded FP32 snapshot-state mismatch
is preserved; only matching precision histories are qualified. Broad cached
HTTP remains unadmitted.

Next evaluate compiled residual/packed normalization fusion with these alias
lifetimes, and separately examine larger physical prefill profiles within the
measured working-set budget. Memory savings alone do not establish speed.


### B4 normalization fusion and shared-layout screening (2026-11-06)

The optional packed residual/RMS primitive and graph matcher now admit physical
batch4 (the previous cap was2). They retain all original residual, reshaped and
normalized outputs, explicit incompatible-layout/alias checks and the existing
compiled chunk bound. No xModel operator or default/profile flag changes.

Evidence:
`/Users/shawn/CantorAI/out/research/mistral7b-local-coder/mistral-pooled-norm-fusion1/`.
The expanded native oracle covers batches1/2/3/4, chunk2/24/64/80/128,
width7/1031/4096, two offsets and four signed input scales:480 packed and120
residual cases. Every residual/layout/normalized output matches the unfused
path bitwise;124,653,520 normalized words also meet an independent FP64 bound.
Fused outputs start as NaNs; downstream consumers, begin/abort counters,
alias/shape/epsilon/uninitialized guards and rejection of batch5 are checked.
All32 CTests pass. Stage1 source/logs are retained; final audit2 supersedes its
smaller geometry coverage.

Three serial fresh processes (candidate, control, candidate) use the unchanged
B4 short/long fixtures and pooled profile. Full32-layer KV/current hidden and
all128 candidate generated IDs match the control. Every prefill body uses32
packed norm fusions; every decode body uses32 residual norm fusions. The
independent audit rejects nine corruptions. Head-only snapshots retain legacy
last-body norm counters; only immediate body telemetry is treated as fusion
coverage. This is not a head-counter, HTTP or quality admission.

Long prefill wall times are15.574/16.126 seconds for candidates versus15.484
for control; short times are1.903/1.846 versus1.926. These diagnostics show no
consistent overall gain, so the measured pooled serving profile remains unchanged.
No new vLLM/SGLang or HTTP results are claimed;2x and strict coding targets remain
unmet.

A separate XOR-swizzled weight-stage screen is retained at
`/Users/shawn/CantorAI/out/research/mistral7b-local-coder/mistral-swizzled-prefill1/`.
Two XOR maps reduce stage storage from17,408 to16,384 bytes while keeping K128,
256 threads and the original arithmetic/tile selection. Independent enumeration
proves all8,192 shared half words receive one aligned vector-store write and
remain in bounds. GPU full-output bit parity and121,110,528 independent FP64
checks pass. Both variants are slower in every swept condition; an isolated hot
speedup is not consistent. Both are rejected. Smaller per-threadgroup storage
alone is not proof of higher occupancy; occupancy was not measured.

The225 projection weights and FP16 coefficients occupy4,001,366,016 distinct
payload bytes in the validated inventory. Apple publishes153GB/s for this M5
([Apple specifications](https://support.apple.com/en-us/125405)). Under an
explicit one-full-weight-stream-per-single-decode-step assumption, ignoring KV,
compute and software overhead, the bandwidth-only estimate is38.24 tok/s. The
short-case2x goal50.13 tok/s would require200.60GB/s in that model. This is an
analytical hypothesis, not measured DRAM traffic, achieved performance or proof
of an absolute limit: cross-step caching/compression is not independently
measured. It motivates correctly verified multi-token/sequence weight reuse.
Existing history-only lookup replay and deeper-draft feasibility results were
reviewed rather than rerun; they do not establish a2x runtime gain.

Next screen a512-thread M128/N128/K64 cooperative prefill tile with strict K
order and explicit resource/one-writer coverage. Separately investigate batched
speculative verification for decode. Neither path is admitted by these results.


### 512-thread prefill and strict batched FP32 screening (2026-11-06)

`/Users/shawn/CantorAI/out/research/mistral7b-local-coder/mistral-wide512-prefill1/`
contains strict M128/M256 by N128/K64,512-thread prototype kernels, compared
with the current production128/256-row,N64/K128,256-thread tile selection.
All three whitelisted matrix shapes and actual128/256/512 rows are measured
serially with three warmups and nine repetitions, hot and256MiB swept conditions.
The sweep does not prove cold cache. Packing is timed and its16-byte parameter
ABI is checked. Actual512-thread pipelines fit the measured resource limits;
candidate shared storage is18,432 bytes versus17,408 for control.

Atomic GPU writer maps verify60,555,264 output words exactly once, check redzones
and all staged FP16 input words. A deliberately short grid leaves1,835,008
poisoned outputs; its saved writer map independently confirms the fault.
Every retained result is bit-identical to control, and121,110,528 saved values
meet the original independent FP64 bounds. Sixteen audit corruptions are rejected.
Both wider tiles slow the dominant feed-forward cells. An isolated narrow-cell
gain does not establish a model improvement. Neither kernel is integrated.

`/Users/shawn/CantorAI/out/research/mistral7b-local-coder/mistral-batched-verify16-screen1/`
then tests streaming input fragments for strict FP328/16-row projections with
one/two/four output columns per SIMD group, plus a preload control. The reference
uses two or four distinct native4 input/output slices in the same command;
retained row groups are independently checked to prevent the old repeated-slice
oracle mistake. Per-row packed-word, FMA and SIMD-reduction order is unchanged.
All outputs begin as NaNs. The in-process full FP64/bit checks cover162,201,600
words, and the independent saved-output audit covers13,516,800 words against
FP64 with no FP16 rounding of input or dequantized weights. All native4 output
bits match. Fifteen corruptions are rejected.

| Rows | Exploratory best-per-shape projection proxy, hot | Swept |
|---:|---:|---:|
| 8 | 1.177x | 1.134x |
| 16 | 1.024x | 1.013x |

This proxy weights two Q/O, two K/V and two gate/up projections per layer,
one down projection and one final head. It omits attention, KV, normalization,
drafting/acceptance, scheduling and actual-model execution; it is not measured
model throughput or an admitted dispatch policy. Streaming improves on the
especially costly16-row preload prototype, but most16-row cells remain slower
than four existing native4 projections. Native verification, KV and scheduler
row bounds therefore remain unchanged. No serving or coding result changes,
no comparator inference is rerun, and the2x goal remains unmet.

The next numerical experiment will use an explicitly separate compensated
BF16 GPU tensor profile with FP32 input/output and accumulation, informed by
[Henry et al.](https://arxiv.org/abs/1904.06376) and the
[PyTorch precision documentation](https://docs.pytorch.org/docs/main/generated/torch.set_float32_matmul_precision.html).
It will compare three/six/nine products under the original full-output FP64
bounds, including adversarial rounding cases and per-row exact-FP32 fallback
when decomposition fails. This is a hypothesis, not an implementation or a
precision admission. Reassociated tensor arithmetic must not be claimed
bit-identical to ordinary FP32. A distinct target/cache profile, single-row
versus batched consistency, actual KV rollback, generated IDs and coding quality
are required before speculative serving. The existing hybrid bounds remain
protected; this research does not enable a CUDA-only PyTorch option on Metal or
use CoreML ANE.

### Input staging and grouped value context follow-up (2026-11-06)

`/Users/shawn/CantorAI/out/research/mistral7b-local-coder/mistral-staged-input-prefill1/`
stages both compact FP16 input and dequantized FP16 weights in shared memory.
M128/K64 uses27,648 bytes and M256/K32 uses25,600 bytes, versus the current
17,408-byte weight-only layout. Both keep the original K16 accumulation order.
The K32 packet reads exactly one packed uint, including the final packet.
CPU staging-ownership maps, atomic GPU output maps and the deliberately short
grid fault check pass. All121,110,528 retained values meet independent FP64
bounds and match baseline bits;16 corruptions are rejected. The main FFN cells
are slower. The isolated M128 square gain does not justify integration.

The next implemented experiment instead targets decode value-context reuse:
`/Users/shawn/CantorAI/out/research/mistral7b-local-coder/mistral-grouped-value-context1/`.
Two query heads sharing a KV head reuse each FP32 value load while maintaining
separate ordered accumulators. The optional native route accepts four physical
rows,32 query heads,8 KV heads,D128,capacity up to4096 and ungated context.
Other geometries, partial physical batches and gated context keep the existing
route. Positions, row ownership, probability/softmax computation and KV storage
are unchanged; no xModel operator or public runtime API is added.

The first screen used Metal4 safe math. The native bridge correctly rejected
those saved baseline bits because the existing context library uses Metal3.1
with default fast math. `operator2` repeats the full screen with those native
compiler settings; its outputs match the actual native bridge bit for bit.
`audit3.json` is the successful native-settings audit; the failed `audit2`
negative-control test is retained (its purported mutation initially left the
metadata unchanged). Safe-mode results remain a separate, non-native experiment.

| Four rows, capacity4096, visible tokens per row | Current context GPU ms | Two-head reuse GPU ms | Kernel speedup |
|---:|---:|---:|---:|
| 256 | 0.104 | 0.070 | 1.48x |
| 2048 | 0.835 | 0.525 | 1.59x |
| 4096 | 1.777 | 1.060 | 1.68x |

These are nine-repetition medians after three warmups and a256MiB sweep;
the sweep does not prove cold cache or establish model throughput. Both compiler
screens retain2,752,512 fully checked output values across40 cases, with
independent saved-input FP64 references and candidate output bit parity.
Coverage includes ragged visible lengths, poisoned unused tails, padded strides,
gating, invalid-slot fallback and a zero-probability inactive row. CPU enumeration
proves each candidate output has one owner. The native-settings audit rejects14
corruptions; the actual native bridge checks458,752 saved-reference words.

The isolated candidate passes all33 native tests, including a new regression
for FP64 context bounds, ragged poisoned tails, inactive rows, partial-batch and
gated fallback, pending-output rejection, abort recovery and owner cleanup.
Two candidate/control/candidate fresh-process model checks preserve all128
candidate output IDs and full32-layer KV/hidden hashes after prefill and decode.
The first candidate logs448 bounded native dispatches; its trace is disabled in
the second candidate. The independent model audit rejects seven corruptions.
CoreAnalytics warnings are retained separately; native owner counters reach zero.

For2048-token prompts, seven diagnostic decode steps have median wall times
105.91/114.07/106.24ms for candidate/control/candidate, roughly7% lower for the
candidate. Short-prompt medians are49.56/50.03/49.10ms. Prefill varies between
runs; no TTFT gain is established. These are diagnostics, not repeated HTTP
serving or coding qualification. The latest admitted HTTP TTFT/throughput table
is unchanged, coding remains0/8 and the2x targets remain unmet.

`GARNET_METAL_MISTRAL_GROUPED_VALUE_CONTEXT` is disabled by default and requires
the prepared four-row Mistral decode engine. The next step is a frozen-fixture
HTTP measurement and prefix/survivor parity check before enabling it in a serving
profile. Wider verification and compensated tensor arithmetic remain research
hypotheses rather than admitted runtime changes. Unrelated working changes are
preserved by reversing only this patch in an external copy and reproducing their
original hashes exactly.

### Grouped value context HTTP follow-up (2026-11-06)

The same optional candidate now completes the frozen22-request HTTP campaign.
All4,844 generated token IDs, prompt IDs, finish reasons, request bodies and
rendered text match the preceding pooled control. Accounting has no issues;
the native auditor checks all eight coding-control outputs and rejects50
corruptions, including a missing/mismatched grouped-value compiled policy.
Strict coding remains0/8. The prefix/survivor campaign also passes with the
frozen control:51 packed commands,14 lookup-free oracles,14 survivor checks,
three mixed scheduler ticks and one head-only hit; ten corruptions are rejected.
Decoded FP32 snapshot versus cold FP16 prefill-state equivalence remains
explicitly unqualified.

The copied immutable HTTP tools retained the historically superseded pool-event
checker, so its first native audit failed after inference/accounting succeeded.
The corrected committed checker plus the new grouped-policy assertion audits
those same raw results successfully (`native-audit2.json`). Original tools,
manifest, failed log and all runtime output remain untouched; inference is not
rerun to fix an auditor. `audit-correction1.json` records the distinction.

Each engine cell below is aggregate output tokens/s followed by median TTFT
seconds. Garnet is one new wave; references remain the frozen15-wave medians.
Native FP32 KV/decode and reference FP16 state differ. No repeated-performance,
quality, half-TTFT or2x admission follows from this table.

| Nominal input | Concurrency | Garnet tok/s / TTFT s | vLLM-Metal tok/s / TTFT s | SGLang-MLX tok/s / TTFT s |
|---:|---:|---:|---:|---:|
| 256 | 1 | 19.983 / 0.686 | 22.021 / 0.687 | 22.042 / 0.732 |
| 256 | 2 | 34.520 / 1.230 | 31.464 / 1.417 | 32.127 / 1.285 |
| 256 | 4 | 52.481 / 2.390 | 45.235 / 2.496 | 42.194 / 2.240 |
| 2048 | 1 | 9.116 / 4.698 | 12.345 / 4.219 | 12.518 / 4.159 |
| 2048 | 2 | 12.288 / 9.012 | 16.536 / 6.814 | 14.482 / 6.450 |
| 2048 | 4 | 15.702 / 17.694 | 21.179 / 11.570 | 16.337 / 10.884 |

The long four-request cell improves aggregate throughput by3.1% over the prior
15.225tok/s observation, while its TTFT is effectively unchanged. Short four-
request throughput falls3.7% in this wave. This mixed serving result does not
justify enabling the kernel by default. It remains an optional building block.
Prepared Metal allocation remains9,954,541,568 bytes (9.27GiB).

The reusable `mistral7b_grouped_value_context.cmake` preset matches all115 Metal
cache settings in the qualified candidate build. Its JSON profile and the worker/
auditor explicitly bind the new compiled capability; all earlier profiles keep
that capability disabled. The first code checkpoint is7abe4f08a, and staged
profile bindings preserve unrelated work by a second inverse-patch hash proof.

The next performance direction is a memory-budgeted larger joint-prefill
profile, measuring real1024/2048-token projection shapes and attention-plan
storage before admitting a wider native profile. The current512-token global
prefill bound must remain intact until that separate profile, its scratch/layout
bounds and scheduler policy are explicitly compiled and qualified. Comparator
inference remains frozen; this is a proposed experiment, not a measured gain.

### Larger joint-prefill feasibility and long-prefix storage (2026-11-06)

`/Users/shawn/CantorAI/out/research/mistral7b-local-coder/mistral-large-joint-prefill1/`
compares single1024/2048-row packed projections with two/four distinct512-row
input/output slices in one command, including each input pack. The existing
M128/M256 tiles keep their original Metal4 safe-math/K16 accumulation order.
All484,442,112 retained outputs match control bits and independent FP64 bounds;
GPU atomic maps check242,221,056 output words, and a short grid leaves the
expected1,835,008 poisoned outputs. The successful `audit2.json` rejects23
corruptions, including repeated-input-piece oracles. The earlier audit's variant
sort error is retained. Its inherited raw half-check counter is explicitly
corrected by `metadata-clarification1.json`: direct scratch checks cover
196,083,712 words, including only the final512-row scratch of each split control.
Full-output FP64 checks still cover every distinct input row.

| Projection N/K | Rows | Distinct512-row control GPU ms | Joint M256 GPU ms |
|---|---:|---:|---:|
| 14336/4096 | 1024 | 9.598 | 9.624 |
| 14336/4096 | 2048 | 19.007 | 20.288 |
| 4096/14336 | 1024 | 10.412 | 10.248 |
| 4096/14336 | 2048 | 20.877 | 23.239 |
| 4096/4096 | 1024 | 3.060 | 3.100 |
| 4096/4096 | 2048 | 5.991 | 6.377 |

These are nine-repetition swept medians, not a cold-cache or model-speed claim.
Larger joint work offers no useful improvement. Native packing, affine,
scheduler and attention row bounds therefore remain unchanged at512 tokens.
Read-only native MPS row-pitch queries reproduce the current21-plan scratch
allocation exactly:1,044,162,560 bytes. A proposed27-plan wider profile would
need4,670,138,368 bytes of MPS payload alone; B4/T256 and B4/T512 exceed the
current per-plan512MiB guard at616/1360MiB. The wider profile is not admitted.

The useful follow-up is in native prefix storage:
`/Users/shawn/CantorAI/out/research/mistral7b-local-coder/mistral-long-prefix-budget1/`.
A2048-token Mistral row needs512MiB of FP32 KV plus hidden/key/owner metadata.
It cannot fit the original512MiB maximum host budget. The new compiled
`GARNET_METAL_TOKEN_PREFIX_HOST_MIB` option allows512(default),1024 or3072MiB.
The larger engines check the proposed cache plus a replacement reserve before
configuration, then check the actual additional payload before capture. Both
Metal-allocation and task-footprint snapshot ceilings include a256MiB margin;
these checks are not a combined physical-memory upper bound, measured peak, or
system-free-memory guarantee. Arithmetic uses checked subtraction/division.
Invalid compiled sizes, missing prefix support and above-cap API budgets reject.
No tokenizer, xModel operator, KV dtype or public Xlang API signature changes.

The1GiB native campaign retains a2048-token A checkpoint plus128-token fallback.
The3GiB campaign retains all four distinct2048/2051/2048/2049-token roots and
A's128-token checkpoint:2,182,245,828 accounted bytes. Two candidate processes
and a fresh legacy512MiB control preserve full32-layer KV, per-row predictions,
all254 arrival-history token observations and complete post-decode hidden hashes.
Other-row KV/position/prediction remains stable during reset/restore/prefill;
`Hidden()` is compared only after B4 decode, because it describes the latest
forward tensor rather than the persistent per-row hidden bank. The initial
failed assertion and corrected oracle are retained. Shorter queries select the
stored128-token checkpoint; changed-first-token queries miss; stale leases
reject. Other request roots restore exactly. The3GiB audit rejects18 corruptions.

The first transport smoke (`http1`) used short coding-suite bodies despite its
inherited2K title; its correction note excludes it from2K claims. `http2` and
later campaigns explicitly use the frozen2048-context performance bodies.
The full HTTP workload exposed two cache-policy issues that direct restore tests
could not: decoded completion snapshots made the original prompt old in the LRU,
and2051/2049-token prompts each stored an almost identical2048-token intermediate
snapshot. The latter duplicates about1GiB of KV in this cohort.

Larger compiled profiles now refresh the exact original-prompt checkpoint before
and after completion capture, without incrementing request hit/miss counters.
They also skip intermediate power-of-two checkpoints within64 tokens of the
full prompt boundary. Default512MiB engine policy is unchanged. Touch-only
`http3` still misses A and is retained as a failed admission. Final `http4`
restores all2048 tokens, executes a head-only first prediction, and overlaps
A/E/A row reuse with ongoing B/C/D decoding under fresh epochs.

| HTTP return, all inputs at least2048 tokens | Fresh serial TTFT s | Cached concurrent TTFT s | Restored tokens |
|---|---:|---:|---:|
| Exact A return | 4.667 | 0.334 | 2048 |
| Appended instruction | 4.992 | 7.017 | 0 |
| Changed instruction | 4.900 | 3.850 | 1024 |

These are individual Garnet observations in a separate cached workload.
The fresh controls are reused unchanged from `http2`; only cached phases rerun
when native cache policy changes. All2026 output-token observations across the
paired fresh/cached streams match, including surviving requests. The independent
HTTP audit rejects nine corruptions, checks client-triggered A/E/A overlap,
fresh row epochs, actual head-only work, bounded host accounting and genuine
request cache counters. Final counters are two hits/six misses for eight arrivals;
maintenance does not forge hits. Maximum observed accounted host storage is
3,204,450,522 bytes, below3GiB. This is not measured peak RAM. Native close leaves
1,507,328 allocated Metal bytes and zero native owners.

All34 native tests pass for the final engine, including exact-key/isolation/LRU
maintenance, unchanged counters, budget boundaries, checkpoint-spacing edges,
overflow cases and existing KV/prefix/attention/packed-kernel checks. CPU budget
checks also pass with512MiB and1GiB compiled limits. The reusable
`mistral7b_prefix_host3g` CMake/JSON profile matches all117 Metal cache entries in
the final candidate configuration. Existing dirty work is preserved by reversing
only this patch in an external copy and reproducing its original hashes.

The0.334s result is warm returning-request HTTP TTFT; it is not a comparison with
the frozen uncached vLLM/SGLang timings. Cold prefill,2x decode/aggregate throughput,
half-cold-TTFT and coding qualification remain unmet (strict coding remains0/8).
Appended prompts can still miss useful checkpoints under the bounded full-snapshot
LRU. Next investigate shared immutable dense-KV blocks with separate recurrent/
hidden checkpoints, retaining exact stored-boundary and precision-lineage rules.
No arbitrary truncation of recurrent state or full4K cache qualification is claimed.

Direct-model1/3GiB state evidence precedes the serving-policy revisions.
`capture-restore-binding1.json` verifies that final-source configuration,
capture and restore function bodies, native Metal GPU source, compiled-graph
execution and decoder-body execution remain byte-identical to the3GiB direct
model candidate. Final34-test and `http4` evidence covers the changed scheduler/
LRU integration; HTTP logs alone are not independent GPU KV-content observations.


## Immutable shared dense-prefix blocks and exact chunk checkpoints

Evidence: `mistral-shared-prefix-blocks1` under the local research root.
This is an optional native C++ cache-storage/scheduler change. It does not change
xModel operators, GPU kernels, operand precision, the512-token global prefill
limit, or the128-token per-request chunk. `GARNET_METAL_PREFIX_SHARED_BLOCKS`
defaults OFF. The new `mistral7b_prefix_shared3g` CMake/JSON profile explicitly
selects3GiB accounted host storage,128 entries and64-token immutable dense-KV
blocks; its118 Metal cache entries match the measured build.

Previously, saving a1024-token checkpoint and a2048-token checkpoint retained
both full copies. The new snapshots share owning dense blocks only after exact
byte comparison against current live KV. Recurrent/convolution payloads and
final hidden state remain private to each stored boundary. Matching tokens alone
never proves numerical lineage. Common-prefix hints only select candidate
resources to compare; they do not authorize partially restoring a later recurrent
state. Request restoration still chooses the longest complete stored checkpoint.
It validates every block and allocates bounded assembly scratch before resetting
any row. Storage blocks have64 tokens; this profile captures completed128-token
prefill chunks, so a matching prefix of2033 tokens restores1920, not2033.

Native cache accounting charges each immutable resource identity once, plus each
checkpoint's private payload, token key and metadata. Eviction removes a resource
charge only when its last cache entry retires. Outstanding owning hits remain
valid across eviction/clear. Object/control metadata is an accounting estimate,
not allocator-reserved or peak RAM. Existing physical-task/Metal envelope guards,
replacement reserve and256MiB transient allowance remain. No total-system free-RAM
or peak-memory guarantee is claimed. The legacy full-snapshot path stays default.

The direct native model sequence retains66 checkpoints for four frozen2K roots
in2,153,344,552 accounted bytes:4,279,848 private and2,149,064,704 unique-resource
bytes across8,320 dense blocks. Two independent candidate processes and a fresh
legacy control preserve full32-layer KV, predictions,254 arrival-history token
observations and post-B4-decode hidden hashes. Stored1024-token boundaries restore
exactly; changing the first token misses; stale epochs reject; the other roots
restore without changing survivors. The first harness run used the inherited
16-entry limit, evicted old checkpoints and failed its shorter-prefix assertion;
its raw evidence remains. The corrected intended128-entry profile passes.

A second fresh/control pair replays the exact appended/edited HTTP input IDs.
Both restore1920 tokens, prefill the changed tail and match complete KV,
predictions,127 arrival-history IDs and post-decode hidden state. These are actual
Mistral dense-model checks. CPU tests additionally prove distinct recurrent/hidden
boundary payloads with shared resources; actual hybrid Qwen model qualification
is still separate work. `run_mistral_shared_prefix.py` is the native regression
runner: pass library directory, clean source root, frozen fixture JSON (four2K
roots plus `shared_prefix_arrivals` with append/changed prompt IDs), fresh output
directory and `control`/`candidate` mode to xlang3. The control uses the retained
512MiB engine; the candidate uses the new shared3GiB engine. All35 native CTests
pass, including resource-union accounting, eviction lifetimes, malformed resource
rejection, boundary isolation and exact group-major FP32 assembly.

The serial HTTP collection reuses the previous fresh controls unchanged and runs
only the new cached engine. The8-request A/B/C/D, E, A-return, append, changed
sequence preserves all2026 paired output-token observations, complete input IDs,
text and finish reasons. A/E/A share a physical row with fresh epochs while B/C/D
remain live; the exact return executes a head-only prediction. Final counters are
three genuine request hits/five misses,94 entries, no evictions, and2,996,048,262
accounted bytes (6,578,566 private;2,989,469,696 unique resources). The HTTP audit
rejects11 corruptions; each independent direct-model audit rejects22.

| Individual HTTP observation | Retained fresh serial TTFT s | Previous full-snapshot concurrent TTFT s | Shared-block concurrent TTFT s | Shared restored tokens |
|---|---:|---:|---:|---:|
| Exact A return | 4.667 | 0.334 | 0.279 | 2048 |
| Appended instruction | 4.992 | 7.017 | 1.049 | 1920 |
| Changed instruction | 4.900 | 3.850 | 0.957 | 1920 |

The first four cold requests regress: their individual concurrent TTFTs rise
from16.479/17.098/16.552/17.015s in the previous full-snapshot campaign to
18.304/18.879/18.245/18.938s with shared storage. This implementation still reads,
copies and compares the existing prefix at every chunk checkpoint. It improves
warm reuse/retention, not cold prefill. Preserve this cost and keep the profile
optional; investigate comparing immutable blocks directly against completed
native storage and copying only new blocks. The table is not a matched warm-cache
vLLM/SGLang comparison or repeated speed admission. Frozen cold baselines stay
unchanged;2x decode/throughput, half cold TTFT and strict coding (0/8) remain unmet.


## Direct completed-storage prefix capture

Evidence: `mistral-direct-prefix-capture1`. The optional
`GARNET_METAL_PREFIX_DIRECT_CAPTURE` profile compares existing immutable blocks
directly against completed native KV, and copies only new or changed blocks.
It requires shared prefix storage and defaults OFF. The reusable
`mistral7b_prefix_direct3g` CMake/JSON preset matches all119 Metal cache entries of
the measured build; the worker rejects a profile/backend capture-policy mismatch.
The native context validates device, initialized/completed state, row bounds and
logical stride before lending a read-only pointer inside C++. Nothing retains
that pointer or borrows mutable GPU storage after capture.

Direct-mode blocks record immutable finite status at construction. Reuse requires
matching geometry, a complete block inside the common-token hint, finite stored
words and exact byte equality with live KV. Therefore an equal block is already
proved finite; changed or new blocks are copied and checked again. NaN/infinity,
changed FP32 history, signed-bit comparisons, partial tails and distinct recurrent
boundaries retain the previous safety rules. Recurrent/convolution state and
final hidden state remain complete private checkpoint payloads. No partial
recurrent truncation, kernel/math changes or xModel operator/API additions occur.
The native Metal context differs only by this CPU bridge/include;50 other native
Metal implementation/header files remain byte-identical to the preceding engine.

All35 CTests pass, including actual padded-MPS-row capture versus legacy compact
readback, immutable ownership after source reset, survivor isolation, closed/
uninitialized/invalid row checks, nonfinite rejection, exact strided comparisons,
changed-lineage fallback and new/reused-block copy counters. Both native Mistral
model sequences match the retained fresh controls: full32-layer KV, predictions,
254 arrival-history token observations and post-decode hidden hashes. Exact and
appended-token requests restore2048 tokens; the separately replayed edited and
appended HTTP prompts restore1920 and prefill the tail. The direct-model audits
reject24/22 corruptions. These are dense Mistral checks; actual hybrid Qwen
qualification remains separate.

For the four-root direct model capture sequence, counters exactly match the
independently derived capture sites:2,148,532,224 copied bytes,20,402,143,232
compared bytes,8,320 new blocks and77,952 reuses. The previous shared capture path
copied22,550,675,456 bytes into full-prefix temporaries plus2,148,532,224 into new
blocks. Direct capture eliminates those full-prefix temporary copies. These are
host payload operation counts, not measured DRAM/GPU traffic or peak RAM. Current
arm64 direct-mode accounting includes56-byte block objects plus the existing
16-byte control estimate; legacy shared-block layout stays unchanged with the
new flag OFF. A final duplicate2048-token capture is observed at15.62/14.82ms in
the two direct model sequences versus54.05/54.11/54.27ms in the retained previous
candidate sequences. These micro timings are not HTTP TTFT or matched repeated
performance admission.

The serial8-request dynamic HTTP replay passes all2026 paired output-token
observations, input IDs, text and finish reasons. A/E/A reuse one physical row
under fresh epochs while B/C/D continue. The exact return executes a head-only
prediction; append/edit reuse1920-token boundaries. Final cache:94 entries,
three hits/five misses, no evictions,2,996,143,066 accounted host bytes. Diagnostics
show2,988,703,744 copied bytes,27,615,297,536 compared bytes,11,968 new blocks and
105,344 block reuses. The direct HTTP audit rejects15 corruptions; the immediately
preceding old-engine replay passes its unchanged11-corruption oracle. No vLLM/
SGLang inference is rerun.

| Individual concurrent HTTP observation | Previous shared capture TTFT s | Direct capture TTFT s | Restored tokens |
|---|---:|---:|---:|
| A-return | 0.276 | 0.312 | 2048 |
| append | 1.035 | 1.048 | 1920 |
| changed | 0.937 | 0.951 | 1920 |

The first dynamic HTTP pair does not establish a cold win: the first four TTFTs
are17.757/18.196/17.582/18.251s previously versus18.043/18.597/18.091/18.578s with
direct capture. Retain this negative observation instead of selecting only the
faster recapture diagnostic.

A separate frozen native-only cold experiment uses the same four2K input bodies,
eight output tokens each and six fresh processes in fixed old/direct/direct/old/
old/direct order. No warmup inference occurs; both engines prepare identical GPU
profiles before health. The independent audit verifies all sixteen512-token
packed prefill rounds per process, four active rows, zero cache hits, exact
paired outputs (192 token observations), native dispatch/close accounting,
nonoverlapping process lifetimes and eight rejected corruptions.

| Cold B4 capture policy | Fresh processes / request observations | Pooled median TTFT s | Process median TTFTs s |
|---|---:|---:|---|
| Previous shared capture | 3 / 12 | 19.299 | 17.827 / 19.447 / 19.724 |
| Direct capture | 3 / 12 | 18.089 | 17.359 / 18.153 / 18.510 |

The pooled median is about6.3% lower in this series. Process ranges overlap and
vary substantially; the script does not establish thermal/paging control or a
formal confidence interval. This eight-output-token cache-policy experiment is
not a replacement for the frozen128-output-token vLLM/SGLang workload. Preserve
both the negative dynamic pair and the modest repeated cold result. The profile
remains optional, with lower capture-copy volume and exact native state proved;
2x decode/aggregate throughput, half cold TTFT, full hybrid qualification and
strict coding (0/8) are still unmet. Next profiling should distinguish GPU
prefill, cache-index work and capture costs before choosing another kernel or
memory/scheduler change.


## Prefill stage attribution and rejected prepared-codebook candidate

Evidence: `mistral-prefill-stage-profile1`, `mistral-codebook-prefill1` and
`mistral-current-serving-c9911`. Native library remains the qualified c991906e6
engine (`80923f7d...`); no GPU implementation, xModel operator or native precision
changes in this checkpoint. The existing `trace_stage_latency` switch remains
OFF by default. Explicit tracing now records monotonic boundaries for admission,
scheduling, embedding, native prefill/decode/head calls, progress/cache commits
and token delivery. Native complete-chunk wrappers are installed only when tracing
is enabled. Head tracing reads only refreshed GPU elapsed time and call wall time;
prepare/encode/wait fields belong to the preceding graph call and are not reused.
No operation serialization, GPU counter sampling or fusion changes are enabled.

Three fresh processes run serially: trace off with direct prefix capture, trace
on with direct capture, trace on with prefix capture disabled. They preserve all
96 paired output-token observations, input IDs, text, finish reasons and native
row/epoch/dispatch/clean-close invariants against retained controls. Each uses
four approximately 2K inputs, eight output tokens, sixteen packed 512-token
prefill rounds and the four-token tail. No cache hits occur. This diagnostic
workload is separate from the 128-output-token serving comparison.

`audit_mistral_prefill_stages.py --study STUDY --output FRESH_JSON` reproduces the
independent audit. The study uses `plans1.json`, `status1.json` and the three named
phase directories; optional `--fixture` and `--controls` override retained sibling
fixtures. Its eight corruption tests reject missing commits, inconsistent clocks,
impossible GPU times, stale head times, overlapping wall categories, changed
output IDs/epochs and unsupported partial submissions. Trace-off emits no timings.
CPU wall categories are exclusive; native GPU/prepare/encode/wait details are
nested and must never be added to them. Command-buffer spans are elapsed GPU
intervals, not hardware utilization or per-kernel active time.

| Diagnostic observation | Direct prefix capture | Cache disabled |
|---|---:|---:|
| Worker prefill window, seconds | 18.031 | 17.352 |
| Prefill call wall, seconds | 16.694 | 16.918 |
| Progress/cache commit wall, seconds | 0.943 | 0.00008 |
| Embedding wall, seconds | 0.130 | 0.130 |
| Scheduling wall, seconds | 0.088 | 0.129 |
| GPU command spans within window, seconds | 16.703 | 16.934 |
| GPU spans / window | 92.6% | 97.6% |

These are individual observations in different processes, not a causal speedup
estimate. They show that eliminating host cache work cannot halve uncached cold
TTFT; most of this window is spent inside GPU command spans. Earlier fused
stage-counter evidence identified packed Q4 projections as a major GPU cost.

The next standalone operator experiment prepares all 16 decoded binary16 values
for each affine group64 in advance, replacing repeated FMA dequantization with
one of two exact lookup paths: device-memory lookup or a 4096-byte threadgroup
table per K128 stage. It retains production M128/M256 tile selection, N64/K128
staging, 256 threads, Safe Metal math, FP16 operands and original K16 FP32
accumulation order. Gathering FP32 activations into FP16 is timed. Table
preparation is deliberately outside timing, making this an optimistic steady-state
screen; each table adds N*K/2 bytes, as large as that matrix's packed Q4 weights.

Nine shapes run three warmups and nine measured repetitions per variant, with
rotating/reversed order under hot and 256 MiB sweep conditions. A sweep does not
prove a cold cache. Every run checks finite, stable outputs and exact baseline
bits; the independent FP64 audit checks all 121,110,528 retained output words with
the unchanged `2e-6 + 4e-6*sum_abs` bound. Signed scales, subnormal coefficients,
signed zero, cancellation and finite coefficients causing decoded-half overflow
are covered by a separate 196-group / 3136-value binary16 table oracle. Overflow
is tested only in this edge oracle, never admitted into inference. Eighteen
corruptions are rejected. The initial audit passed all numerical checks but
mutated the edge-metadata record instead of a cell in its wrong-tile self-test;
`audit2.py` selects cells explicitly and passes. Both attempts are retained.
There is no writer-map or model/runtime qualification for this standalone screen.

| M | N | K | Baseline hot GPU ms | Device lookup ms | Shared lookup ms |
|---:|---:|---:|---:|---:|---:|
| 128 | 14336 | 4096 | 1.453 | 1.658 | 1.707 |
| 256 | 14336 | 4096 | 2.595 | 2.789 | 2.918 |
| 512 | 14336 | 4096 | 4.975 | 5.180 | 5.525 |
| 128 | 4096 | 14336 | 1.644 | 1.921 | 1.680 |
| 256 | 4096 | 14336 | 2.792 | 3.286 | 3.092 |
| 512 | 4096 | 14336 | 5.558 | 5.822 | 6.300 |
| 128 | 4096 | 4096 | 0.569 | 0.688 | 0.722 |
| 256 | 4096 | 4096 | 0.911 | 0.985 | 1.020 |
| 512 | 4096 | 4096 | 1.513 | 1.687 | 1.880 |

Both candidates are slower on the main projection shapes. Preserve this negative
result and do not integrate or allocate codebooks in native inference. The
experiment disproves this implementation's speed hypothesis; it does not prove
which hardware limit dominates. Next GPU work must address actual projection or
attention execution rather than assuming dequantization removal alone wins.

### Current cold serving refresh at c991906e6

The native-only six-cell refresh uses the frozen request bodies, cache-off policy,
128 output tokens, unchanged GPU profiles and no tracing. Its 14 completions
preserve all 1792 output IDs and text/finish reasons against retained native
controls. Accounting passes and four corrupted records are rejected. No vLLM or
SGLang inference is rerun. Garnet has one measured wave per cell; reference values
are frozen 15-wave medians. Native FP32 KV/decode differs from FP16 references;
this is not a matched precision or repeated statistical speed admission.
TTFT is seconds, decode is median tokens/s/request excluding TTFT, aggregate is
output tokens/s including prefill across the concurrent wave.

| Approx input | Concurrent | Engine | TTFT s | Decode tok/s/request | Aggregate tok/s |
|---:|---:|---|---:|---:|---:|
| 256 | 1 | Garnet | 0.898 | 20.40 | 17.93 |
| 256 | 1 | vLLM-Metal | 0.687 | 24.87 | 22.02 |
| 256 | 1 | SGLang-MLX | 0.732 | 25.07 | 22.04 |
| 256 | 2 | Garnet | 1.253 | 18.52 | 31.51 |
| 256 | 2 | vLLM-Metal | 1.417 | 18.85 | 31.46 |
| 256 | 2 | SGLang-MLX | 1.285 | 19.00 | 32.13 |
| 256 | 4 | Garnet | 2.624 | 15.24 | 46.61 |
| 256 | 4 | vLLM-Metal | 2.496 | 14.37 | 45.24 |
| 256 | 4 | SGLang-MLX | 2.240 | 12.88 | 42.19 |
| 2048 | 1 | Garnet | 4.985 | 12.28 | 8.34 |
| 2048 | 1 | vLLM-Metal | 4.219 | 20.75 | 12.35 |
| 2048 | 1 | SGLang-MLX | 4.159 | 20.74 | 12.52 |
| 2048 | 2 | Garnet | 9.699 | 9.69 | 11.19 |
| 2048 | 2 | vLLM-Metal | 6.814 | 15.47 | 16.54 |
| 2048 | 2 | SGLang-MLX | 6.450 | 11.75 | 14.48 |
| 2048 | 4 | Garnet | 19.865 | 6.86 | 13.31 |
| 2048 | 4 | vLLM-Metal | 11.570 | 10.48 | 21.18 |
| 2048 | 4 | SGLang-MLX | 10.884 | 6.31 | 16.34 |

Short B4 aggregate throughput is slightly above both retained references in this
single refresh; long-input TTFT and throughput remain materially behind. All
2x decode/aggregate, half cold-TTFT and strict coding gates remain unmet. Exact
native output identity preserves the previous strict coding 0/8 result, not a
new coding-quality pass. Qwen hybrid/multimodal, ANE and Gemma qualification
remain separate unfinished work. Baselines, model weights and protected native
libraries are unchanged.


### Supported tensor-fragment screen: no consistent gain

Evidence: `mistral-wide-fragment-prefill1`, parent checkpoint `f4d140f90`.
An M16/N64/K16 cooperative-input prototype is rejected before dispatch by the
runtime Metal compiler: cooperative inputs permit only16/32 for M/N/K and require
one dimension32. The same failed attempt also used the SDK example's obsolete
`get_mask` name; the active toolchain exposes `is_valid_element`. Source, full
compiler diagnostics and terminal failure remain separate from measured data.

Three supported serial standalone collections retain the production outer
M128/M256 by N64 tile policy, K128 weight staging, ordered K16 accumulation,
FP16 operands, FP32 outputs, Safe math and timed half-input gathering. All nine
matrix/row geometries run three warmups/nine samples under hot and256MiB sweep
conditions. No frozen comparator inference is repeated; a sweep does not prove
cold cache. The first implementation uses documented coordinate/mask accessors
for M16/N32 and M32/N32 fragments. Both are much slower than direct register
indexing. This tests that implementation, not an abstract tensor-API limit.

A separate GPU layout capture enumerates all3328 valid per-lane operand and
result positions for both fragment shapes. Independent formula checks match
all positions and prove each of the six logical operand maps is bijective.
This binds the subsequent direct-index prototypes to the observed M5/toolchain
ABI, not a portable layout guarantee or atomic output writer-map admission.
The first direct M32 prototype uses128 threads and the same number of
accumulator objects as the256-thread M16 baseline. It remains mostly slower,
with particularly large regressions on M256 outer tiles. No register spilling
or occupancy cause is claimed without hardware evidence.

The final M32 prototype retains256 threads and halves accumulator-object count
(one/two versus two/four for M128/M256), keeping per-thread output-word count
and17,408-byte shared stage equal to baseline. It removes the large earlier
regressions but gives no consistent gain. A renamed byte-equivalent M16 control
is included in each direct-access collection. Below are baseline/candidate GPU
median ratios from this final individual screen; greater than1 means faster.

| Actual M | N | K | Hot ratio | Swept ratio |
|---:|---:|---:|---:|---:|
| 128 | 14336 | 4096 | 0.992x | 0.991x |
| 256 | 14336 | 4096 | 1.001x | 1.044x |
| 512 | 14336 | 4096 | 0.994x | 0.942x |
| 128 | 4096 | 14336 | 1.053x | 0.988x |
| 256 | 4096 | 14336 | 1.013x | 0.976x |
| 512 | 4096 | 14336 | 0.996x | 1.002x |
| 128 | 4096 | 4096 | 0.834x | 0.984x |
| 256 | 4096 | 4096 | 1.012x | 1.016x |
| 512 | 4096 | 4096 | 0.994x | 1.104x |

Absolute baseline timings drift substantially across shapes/conditions and
collections; no controlled thermal/paging state or confidence interval is
established. The retained post-run VM/thermal snapshot cannot establish what
happened during a measurement. Do not select an isolated gain as admission.

Every repetition checks finite/stable outputs and baseline bit identity.
The independent audit covers all363,331,584 retained FP32 output words across
the three supported collections, with signed scales and the original
`2e-6 + 4e-6*sum_abs` FP64 bound. All words also match the corresponding
baseline exactly. Forty-two corruptions are rejected. Output writer-map,
real-weight/model/HTTP qualification is deliberately absent: the speed screen
is negative. No candidate kernel enters Garnet, no profile changes, and no
additional persistent weight memory is allocated. The c991906e6 serving table,
strict coding0/8 and all unmet2x/half-TTFT gates remain unchanged. Continue with
measured GPU execution and algorithmic weight reuse; these results do not justify
another native tile change.


## Tensor-memory screening and compiler-layout startup validation

Evidence: `mistral-tensor-memory-prefill1` and `mistral-packed-layout-guard1`,
parent `22978e173`. Direct tensor-memory operands are screened separately from
the earlier manually assembled cooperative fragments. Device FP16 activations
and a17,408-byte decoded shared-weight stage feed eight-SIMD-group MPP operations
with the original M128/M256 by N64 outer tiles. K16/32/64/128 reductions retain
Safe math, FP16 operands and FP32 accumulation/output. Gathering remains timed;
no persistent expanded-weight storage is introduced. Static full-tile variants
use explicit fixed-extent views only on geometries proved to contain whole tiles;
ragged native inputs are not qualified by these screens.

Nine shapes, three warmups and nine measured repetitions per variant run serially
under hot and256MiB sweep conditions. Dynamic K16/K32 are slower. K128 provides
small gains on some M128 cases and regresses on important M256 gate/up/down
projections. Static views do not remove those regressions. The mixed pattern
is not runtime or end-to-end performance admission. Three API compilation
failures are retained separately: const-element deduction, temporary rather
than named tensor views, and an SDK-example `static_slice` member absent from
the active runtime compiler. All have zero dispatched benchmark cells.

A subsequent four-variant process compares runtime/offline compilation of both
the existing manual kernel and the public static K128 tensor-memory kernel,
using the same inputs, packed weights, output allocation and interleaved timing.
Offline build uses the recorded SDK, Metal4.0, explicit safe math, fast FP32
function selection and O3. This is an explicit compiler configuration comparison;
it does not prove all internal compiler optimization defaults are identical.
No offline candidate consistently improves the important projection mix.
Selected valid runtime K128 ratios below are baseline/candidate GPU medians in
that final process; above1 means faster, not a serving-speed claim.

| Actual M | N | K | Runtime K128 hot ratio | Runtime K128 swept ratio |
|---:|---:|---:|---:|---:|
| 128 | 14336 | 4096 | 1.065x | 1.058x |
| 256 | 14336 | 4096 | 0.987x | 0.969x |
| 512 | 14336 | 4096 | 0.957x | 0.934x |
| 128 | 4096 | 14336 | 1.053x | 1.055x |
| 256 | 4096 | 14336 | 1.071x | 1.061x |
| 512 | 4096 | 14336 | 0.919x | 0.918x |
| 128 | 4096 | 4096 | 1.047x | 1.038x |
| 256 | 4096 | 4096 | 1.064x | 1.054x |
| 512 | 4096 | 4096 | 1.032x | 1.086x |

### A measured incompatible compiler layout

The offline manual kernel is numerically invalid, despite successful compilation
and finite/stable writes. All18 offline-baseline shape/pressure cells fail the
unchanged FP64 bound. The independent auditor checks524,812,288 retained words;
484,442,112 words in the other variants pass the original
`2e-6 + 4e-6*sum_abs` bound and match runtime-baseline bits. Worst valid error is
0.037143 of the bound; the invalid offline route reaches30349.60 times it.
`audit2.json` explicitly records18 numerical rejections and
`all_candidates_numerically_qualified=false`. Its overall evidence-audit pass
must never be read as qualification of the bad route. Forty-seven corruption
checks include an actual flipped false-parity promotion; the initial audit1
only asserted the original parity declaration in that self-test and is retained
as superseded. No numerical bound is relaxed.

A separate GPU capture compares192 per-lane operand/result maps across both
compiler paths. All128 input-operand lane maps change;64 result maps are unchanged.
For M16/N32/K16, the observed left capacity changes8 to16 and right coordinates
change. The M32 operand capacities/maps also differ. The raw SDK-offline library,
AIR, source, compiler commands and both layout captures are retained. Layout is
compiler-specific; old direct register indexing cannot be assumed portable.
The public tensor-memory variant remains numerically correct under both measured
compiler paths, but its mixed timing does not justify native routing.

### Preparation-only guard in the existing native packed profile

Native `PreparePackedAffineProfile` now appends a small layout-validation kernel
to the SAME library as its existing packed kernels and executes it before
publishing the library, workspace or plans. Every one of32 lanes validates the
qualified FP16 M16/N32/K16 capacities, masks and coordinates for left, right and
FP32 result fragments. A128-byte shared status buffer starts poisoned; host
validation requires exactly32 zero words. Missing writes, nonzero lanes, failed
commands or an incompatible ABI cause explicit preparation failure. An unknown
layout is an unsupported packed configuration, not an automatic portability fix.

Validation runs once when preparing that packed library; later plan creation
reuses it. Temporary guard resources are released after the drained startup
command. No request-time guard, new model operator/API, Rust, precision change,
default flag change or existing inference-shader modification is introduced.
Compilation success alone is no longer enough to admit manually indexed kernels.
A guard failure occurs before publishing the private packed workspace or plans.

All36 native CTests pass, including the full existing packed operator oracle and
a new live-layout/capacity-fault test. The host status predicate rejects both
failure and unwritten poison at every lane, plus null/wrong status counts. The
same guard is separately compiled through the actual incompatible SDK path;
all32 lanes return17 (left-capacity and right-coordinate mismatch), and the exact
native host predicate rejects it. This is a real compiler mismatch check in
addition to injected faults, not a passing-label assertion on known bad math.

One fresh native Mistral B4/C4096 process then runs the four retained approximately
2K requests with eight output tokens and prefix caching/tracing disabled.
All32 output IDs, complete input IDs, text and finish reasons match retained
native controls. The audit validates active rows, epochs, dispatch coverage,
zero cache hits and clean shutdown, and rejects five corruptions. No vLLM/SGLang
inference is rerun. The existing inference kernel source strings remain unchanged;
this model check covers startup integration and bounded generation, not a new
full1024-token quality evaluation or full-state/hybrid qualification.

Neither tensor-memory timing nor the startup guard establishes a new serving
speed result. The last128-output-token c991906e6 comparison remains the published
serving table. Prior strict coding0/8 remains unresolved, with no new quality
claim. All2x/half-TTFT, Qwen hybrid/multimodal and broader model goals remain open.


## Compensated BF16 verification screen: correct variants slower

Evidence: `mistral-compensated-bf16-verify1`, parent `4b34ca0f0`.
The retained Qwen2.5-Coder draft trajectory spends24.107s drafting and64.994s in
1179 target calls; only48 calls skip drafting because of the512-token draft
capacity. These are earlier native diagnostic timings, not a new comparison.
The next experiment therefore targets FP32 verification projections.

Following the decomposition in [Henry et al.](https://arxiv.org/html/1904.06376v1),
the standalone prototype splits FP32 inputs and exactly dequantized FP32 weights
into three BF16 components, evaluates three/six/nine tensor products with separate
FP32 significance accumulators and combines corrections before the main sum.
This is an explicit alternative numerical profile; reassociated arithmetic is
not assumed bit-identical to ordinary FP32. The original
`2e-6 + 4e-6*sum_abs` full-output FP64 bound remains unchanged.

The fixed physical tile is M16/N128/K32 with128 threads. Three decoded BF16 weight
stages use27,648 shared bytes; there is no persistent expanded-weight cache.
A prepared16-row input workspace holds three BF16 components. Actual logical
counts remain1/4/8/16. Every timed alternative command includes GPU input splitting,
row classification, compacted row-list/indirect-argument construction, tensor
projection and any required ordinary-FP32 fallback. Safe Metal math and original
FP32 source activations/F16 affine coefficients are retained; the input or
dequantized weight is never silently rounded to one FP16/BF16 operand.

Eligibility is deterministic per input row: nonzero exponent fields47..207
(unbiased -80..80), with negative zero and nonfinite/extreme-range rows routed
out. The test uses independently validated finite F16 coefficient patterns.
Ineligible input rows get zero tensor operands. GPU compaction builds a sorted
ordinary-row list and two indirect grids; it can skip the tensor grid when all
rows require ordinary arithmetic. The fallback reuses original packed-word/FMA/
SIMD reduction order on distinct logical input/output rows. It does not select
one arithmetic mode for the whole batch. No native route/profile/cache identity
is enabled by this experiment.

Fifteen geometries include actual8/16 rows over all five Mistral projection
shapes, an adversarial midpoint, mixed extreme/negative-zero rows, all-fallback,
one-row isolation and N1031/K64 tails. Three warmups/nine samples interleave four
variants under hot and256MiB sweep conditions. The sweep is not proof of a cold
cache. Complete outputs start as NaNs on every dispatch; redzones and repeated
bit stability are checked. This is coverage checking, not an atomic one-writer
proof. Existing native verification, hybrid state and scheduler bounds stay
unchanged.

The independent audit reconstructs original FP32 inputs through closed-form
LCG arithmetic, checks signed F16 coefficients, and computes unrounded FP32-weight
FP64 dot references. It checks10,884,320 retained output words and3,545,088 GPU
BF16 component words against independent ties-to-even conversion. All GPU row
flags, compact IDs and both indirect grids match independently computed values.
Fallback outputs match ordinary FP32 bits exactly, including the all-fallback
zero-tensor-grid case. The same eligible row on its own and inside a mixed
16-row batch matches bitwise for every variant. Thirteen corruptions are rejected.
Atomic writer maps and actual model/KV/rollback/generation gates remain absent.

Three products fail four shape/pressure cells: both the midpoint and the short
K64 tail case exceed the original bound. The midpoint uses input and weight
1.00390625; three products produce4128 instead of4128.0625,3.785 times the allowed
error. Six/nine products recover that correction exactly and pass every retained
full-output bound; their worst fraction is0.055654 of the unchanged limit. The
initial auditor used the wrong data-root level. The next expected-failure gate
then caught the additional short-dot failures. Those attempts and the diagnostic
are retained. Final `audit4.json` requires all controls and six/nine-product
results to pass, and explicitly records all four three-product rejections. No
bound is loosened or failed output relabelled as qualified.

| Rows | Projection command proxy, hot: BF3 / BF6 / BF9 | Swept: BF3 / BF6 / BF9 |
|---:|---|---|
| 8 | 0.538x / 0.291x / 0.171x | 0.498x / 0.303x / 0.177x |
| 16 | 1.085x / 0.605x / 0.371x | 1.035x / 0.599x / 0.351x |

The proxy is a32-layer weighted sum of two Q/O, two K/V, two gate/up and one down
command per layer plus one head, compared with distinct native4 row slices.
It omits attention, KV, norms, drafting/acceptance, scheduling and actual model
execution. BF3 is invalid despite selected16-row gains. The numerically passing
six/nine-product implementations are slower on the important projections. This
screen rejects runtime integration; it establishes neither a tensor-hardware
limit nor a model/serving gain. No additional writer-map or model benchmarks are
run for this negative speed candidate.

All native sources/defaults, model assets, protected libraries and frozen
vLLM/SGLang results remain unchanged. There is no new serving measurement, and
the published c991906e6 table still applies. Coding qualification and all2x/
half-TTFT goals remain open. Further acceleration needs a measured reduction in
target/draft work rather than promotion of these inaccurate or slower kernels.


## Prepared affine coefficients and explicit FP16 verification: rejected screens

Evidence: `mistral-prepared-affine-coefficients1` and
`mistral-half-verification-profile1`, both based on `8799162cc7`. Neither screen
changes native inference, numerical defaults, xModel operations or serving.

The prepared-coefficient experiment stores an FP32 pair per64-weight group:
either original scale/bias or16*scale and bias-16*scale. The latter interprets a
nibble through `bitcast_float(0x3f800000 | (q << 19)) = 1 + q/16`. GPU preparation
checks all16 original and transformed FP32 FMA result bits; a canonical NaN in
the adjusted bias marks groups requiring original arithmetic. The prepared
pair costs N*K/8 extra bytes, one quarter of the earlier16-half codebook. The
probe holds **both** candidate tables simultaneously; the planning estimate of
838,860,800 bytes across160 selected Mistral projections is for one table and
is not a measured native allocation. Preparation time is recorded separately
and excluded from projection timing; input packing remains timed.

Nine M128/256/512 gate/down/QO geometries retain production outer tiles,
256 threads and K16 accumulation order. All121,110,528 retained output words
pass the unchanged full FP64 bound and exact baseline bit parity. An independent
IEEE FMA oracle checks196 coefficient edge groups, including signed zero,
subnormals and finite extremes:165 transform exactly and31 are marked for
fallback. Eighteen corruptions are rejected. All normal GEMM groups are eligible;
this is **not** actual fallback-GEMM, atomic-writer-map or model coverage.

| Projection / logical rows | Original-coefficient pair: hot / swept speed ratio | Transformed pair: hot / swept speed ratio |
|---|---|---|
| Gate M128 | 1.103x / 1.017x | 1.109x / 1.008x |
| Gate M256 | 1.001x / 0.998x | 1.004x / 0.996x |
| Gate M512 | 1.001x / 1.001x | 0.988x / 0.991x |
| Down M256 | 1.006x / 1.014x | 1.012x / 1.007x |
| Down M512 | 1.001x / 1.003x | 0.978x / 0.995x |
| Q/O M256 | 0.986x / 1.053x | 1.063x / 1.054x |

Ratios above1 mean faster than the original standalone kernel. Gains are mixed
and small on important M256/M512 projections; the selected M128 hot result does
not survive sweeping. The extra persistent memory is not justified by this
screen. Neither transform enters native inference.

The second screen considers an explicitly different FP16-operand policy:
RNE-half inputs and dequantized weights with FP32 accumulation/output. Original
FP32 native controls keep the unrounded oracle. Scalar FP16 and two tensor
variants use their separate rounded FP64 oracle; the original
`2e-6 + 4e-6*sum_abs` bound is never relaxed. Negative zero or inputs whose half
conversion is nonfinite route the whole row through original FP32 arithmetic.
Other rows use the rounded policy independently; tiny inputs may round to zero.
No alternate target/profile/cache identity is enabled.

The physical tensor tile has16 rows but only1/2/3/4 logical rows. N64/K128 and
N128/K64 variants use17,408 and18,432 threadgroup bytes respectively. Actual
FP32 controls use production point routes for1/2/3 rows and the existing four-row
blocked route for admitted wide shapes; narrow K/V retains the point route.
Every timed alternative includes input rounding, padding, row classification,
compaction, indirect planning and ordinary fallback. Fixed16-row workspaces
include128-aligned K padding, including the K64 tail. There are25 geometries:
five Mistral shapes at each logical row count plus midpoint, mixed, all-fallback,
row-isolation and N1031/K64 tails. All outputs have NaN initialization/redzones
and repeated bit stability checks; this is not an atomic one-writer proof.

The independent audit checks4,551,904 retained output words and2,166,784 half
component words, reconstructs inputs/coefficient bits, row flags, compact IDs
and indirect grids, and verifies fallback bits. Half variants preserve the first
row across logical occupancies1..4; the eligible row alone versus inside the
mixed batch also matches. Ordinary FP32 and scalar-half controls pass every
retained full-output bound. Both tensor tiles fail the midpoint under both
pressure conditions: input/weight1.00390625 produce4128.015625 instead of the
exact4128.0625. Their error is2.83846 times the unchanged rounded-operand bound.
This is an accumulation failure against the **distinct half** reference, not a
comparison with the old FP32 operand policy. Four tensor cells remain rejected.

The first command resolved the virtualenv interpreter symlink and lost its
package path; no numerical audit ran. The corrected command's initial all-pass
gate caught these four failures. `diagnostic3.json` retains all cells;
`audit4.json` requires all controls/scalar cells to pass, records exactly these
four tensor failures and rejects15 corruptions, including false midpoint
promotion. A passing audit means the rejection evidence is valid, **not** that
the tensor implementations are numerically qualified.

| Logical rows | N64/K128 projection proxy: hot / swept | N128/K64 projection proxy: hot / swept |
|---:|---|---|
| 1 | 0.362x / 0.347x | 0.398x / 0.355x |
| 2 | 0.374x / 0.364x | 0.367x / 0.347x |
| 3 | 0.512x / 0.453x | 0.555x / 0.475x |
| 4 | 0.508x / 0.408x | 0.543x / 0.431x |

The proxy is the32-layer weighted projection-command sum used by the preceding
screen; it excludes attention, KV, norms, acceptance, scheduling, model and HTTP
execution. These variants are slower as well as numerically rejected. They
receive no native/model/quality/speed admission, and no additional model benchmark
is warranted for these rejected candidates. The saved c991906e6 serving table
still applies; all2x/half-TTFT and coding-quality goals remain open.

## Bounded residency: ownership census and public-API feasibility only

Evidence: `mistral-bounded-residency1`. The saved B4/C4096 prepared inventory has
8,819,633,184 declared unique buffer bytes. Actual device allocation is
9,954,541,568 bytes; the remaining1,134,908,384 bytes are outside the inventory's
declared graph/session owners, which explicitly excludes some context/driver
resources. Private FP32 K/V is4,294,967,296 bytes, exactly the32-layer geometry.
Immutable affine payload/coefficient allocations total4,001,366,016 bytes.
The inventory has no default graph K/V inputs with the private-state shape.
Inspection confirms the Mistral resident constructor already skips host K/V
arrays and default graph buffers are retired. The suspected duplicate host-KV
allocation is therefore not a supported optimization hypothesis.

The same retained snapshot reports5,674,647,552 compressed task bytes. It is a
snapshot, not proof that weights were paged, that compression caused latency,
or that all private-state bytes were physically resident. Large zero/padded
cache regions are not distinguished by this snapshot. No active process's
current system memory state is relabelled as conditions of that saved run.

The local clean vLLM-Metal checkout `a44be48731005932a24dd30b55c21e992dc90b9a`
calls its wired-limit helper during worker initialization; that helper selects
MLX's device-recommended wired limit. The installed MLX-LM source has a matching
helper. This is source inspection, not proof of a particular frozen process's
actual wired residency. Garnet's tracked native Metal context has no explicit
residency-set policy. [MLX documents the wired limit](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html)
as controlling resident bytes. [Apple's residency-set documentation](https://developer.apple.com/documentation/metal/simplifying-gpu-resource-management-with-residency-sets)
provides public preparation/queue-registration APIs; requests can be postponed
under competing memory needs. Neither mechanism by itself proves a speed gain.

A standalone Objective-C++ feasibility helper uses only public Metal APIs. It
validates a matching device/queue, standalone buffer allocations, unique backing
identities and an explicit budget no larger than the recommended working set
minus256MiB. It charges committed residency-set size including any internal
allocation, registers the set on its queue and supports explicit renewal. Close
appends/waits a queue marker before detaching, ending residency and removing /
committing allocations. It does not change system settings or any native model
profile. Callers must serialize access and exclusively own the queue.

On this M5, the2MiB probe passes exact GPU-copy bits, alias deduplication,
16 close/reprepare cycles,18 renewals and10 invalid-operation rejections. A weak
resource reference becomes nil after RAII destruction, proving resource release
on the exercised success path. Close also drains a previously submitted copy.
This validates the small API/lifetime experiment only. Different-device handling
is implemented but not exercised on this single-device Mac; failed-drain and
out-of-memory paths, full-model owner lifetime, native integration and long-lived
idle behavior remain unqualified. No resource remains in the closed plan.

The next runtime experiment should opt in only the immutable affine buffers,
retain current FP32 math/KV/context/scheduling, enforce a process/profile byte
envelope and test failure cleanup before a serial native-only comparison.
It must prove unchanged complete model IDs/state and measure an actual benefit
before promotion. No vLLM/SGLang inference is rerun for this inspection/probe.


## Native bounded affine-weight residency: correct, no demonstrated speed gain

Evidence: `mistral-native-affine-residency1`, parent `88431ed1e`. The native
extension is compile-time opt-in (`GARNET_METAL_AFFINE_WEIGHT_RESIDENCY`, default
OFF), and even that build does nothing until explicit engine preparation.
`decoder_session.prepare_affine_residency(budget_bytes)` prepares existing
immutable packed/scales/bias buffers before scheduler attachment; it rejects
nonpositive budgets, closed/poisoned sessions and attached request schedulers.
`affine_residency_stats_json()` reports allocation accounting and renewals,
explicitly not proof of physical residency. This changes no xModel operation,
tokenizer, model weight, arithmetic, KV representation or scheduler policy.

The session gathers affine owners from its existing layer/head graphs. Native
Metal code validates device/queue and standalone allocations, deduplicates
backing buffers, enforces the explicit byte budget and the process/device
recommended-working-set envelope minus256MiB, commits a public Metal residency
set, attaches it to the session queue and requests residency. Each subsequent
native command-batch begin renews the request. There is no persistent weight
copy or expanded representation. Context teardown aborts/drains pending GPU
work before detaching, ending residency and removing/committing allocations.
The plan's RAII cleanup is idempotent. The test-only fault template is a distinct
specialization, avoiding an ODR difference between test and production classes.

The clean archive builds and all37 CTests pass. The new test rejects17 invalid
operations, including insufficient budget, empty inputs, heap allocation,
repeated preparation, preparation during an open native batch, and injected set
creation / committed-accounting failures. It checks alias deduplication,
16 close/reprepare cycles,18 direct renewals, draining an already submitted GPU
copy, weak-reference release, default-off behavior and exact native projection
bits across three command batches. The single-device test cannot exercise a
foreign device, and does not force a real driver out-of-memory or GPU failure.
The first successful build was followed, before tests/inference, by the template
specialization repair and a bitwise comparison strengthening; both build logs
and the earlier source fragments are retained.

Four fresh native engines run strictly serially in off/on/on/off order. Each
receives the same four frozen approximately2K prompts, C4096/B4,128 outputs,
chunk128 / global up-to512 token-row profile, no prefix reuse and no tracing.
Both conditions use the exact same native library, INT4/group64/F16 coefficient
checkpoint, FP32 decode/KV and current kernel code. Enabled preparation holds
675 unique buffers,4,001,366,016 committed bytes under an8GiB budget; final
renewals are148 in both enabled runs. No frozen comparator inference is rerun.

| Native condition, two fresh waves each | TTFT median of wave medians (s) | Decode median (tokens/s/request) | Aggregate median (tokens/s) |
|---|---:|---:|---:|
| Residency off | 16.753 | 8.201 | 15.841 |
| Residency on | 16.596 | 8.184 | 15.906 |

TTFT changes by-0.94%, decode by-0.21% and aggregate by+0.41%. The wave ranges
overlap; two observations per condition do not establish a stable gain. The
first off-run ready snapshot has3.387GB compressed task memory; both on-runs
and the final off-run show zero. That variation is not a causal page-fault
measurement. This collection does not support paging as the main latency cause,
and residency remains disabled by default. Its bounded native API can be used
for further explicitly qualified profiles, but no serving performance promotion
is made from this result. Absolute rates in this new collection must not be
presented as an implementation speedup over older runs under other conditions.

The independent audit binds client timing arithmetic to raw SSE monotonic
timestamps, checks each full prompt/output/text/finish against retained native
controls, reconstructs scheduler leases/chunks/dispatch coverage, checks resource
policy events before readiness and before clean shutdown, validates the exact
675-buffer charge and renewals, and rejects14 corruptions. All2,048 generated
IDs match controls. Each native worker exits0 with all tracked owners zero and
1,507,328 device bytes after close. These HTTP logs do not directly inspect
full-model GPU KV contents, dynamic arrivals or coding quality. Those existing
qualification limits and all2x/half-TTFT targets remain open.

### GPU operator diagnostic directs the next optimization

One separate8-output B4/2K run enables the existing native compute-encoder
counter API before scheduler attachment. It keeps fusion and command batching;
`GARNET_METAL_PROFILE_OPS` stays0. These counter timings are diagnostic, never
serving rates. MPS-owned encoders remain unobserved. The audit partitions raw
calibrated intervals by active category, so overlaps appear in joint buckets
and are never counted twice. Unsampled command spans are not automatically
attributed to MPS or any other operator.

All32 generated IDs match retained control prefixes; native output text agrees
with SSE/client text. The initial audit incorrectly required whole-length
comparison against controls that can be longer than8 tokens. Its failure and
source are retained; the corrected audit explicitly checks the8-token prefixes.
There is no numerical divergence or relaxed comparison of selected tokens.
All27 drained command records and calibration intervals validate; seven counter
corruptions are rejected. Sixteen main-prefill commands have512 physical token
rows; one tail command has256 physical rows, eight decode commands have up to4,
and two head commands complete the trace. Physical rows do not imply that all
those rows contain active prompt tokens.

| Diagnostic phase | Important exclusive sampled category | Fraction of its command GPU spans |
|---|---|---:|
| main_prefill | Packed affine work, including its packing dispatches | 60.06% |
| main_prefill | Other affine projections | 6.54% |
| main_prefill | Observed attention prepare/merge work | 7.11% |
| main_prefill | SwiGLU pointwise work | 5.72% |
| decode | Attention scores/softmax | 40.23% |
| decode | Attention value aggregation | 22.90% |
| decode | Ordinary affine projections | 25.56% |
| decode | Indexed affine projections | 7.59% |
| decode | KV updates | 0.29% |

Main-prefill command spans total15,094.164ms; sampled interval unions cover
13,190.967ms. The unsampled remainder includes unobserved work and gaps; it is
not a utilization estimate. Decode spans total876.454ms, with861.855ms in sampled
unions. Attention scores plus value aggregation account for63.13% of those
decode spans, whereas KV updates account for0.29%. These measurements distinguish
the decode attention bottleneck from main-prefill packed affine work.

A concrete next candidate is to split independent FP32 score dot products over
bounded token chunks, followed by the same full-row softmax/reduction order.
The current kernel runs all visible positions inside one128-thread group per
query head. A split grid may expose more parallel work while retaining its
per-token dot order, existing score buffer and FP32 operands. This is a hypothesis,
not an implemented or admitted acceleration. Exact full probabilities, masked
rows/tails, output ownership, native KV/ID parity and serial timing must be
proved before changing dispatch. It addresses decode; faster main prefill still
requires a separately demonstrated packed-projection improvement.


## Split decode scores: native bit/state parity, diagnostic gain, serving unproven

Evidence: `mistral-split-score1`, `mistral-split-score-default1` and
`mistral-split-score-native1`, based on `67da03847`. The prior counter collection
identified score/softmax as40.23% of sampled decode command spans and value
aggregation as22.90%. The original score kernel serially visits all visible
cache positions inside one128-thread group per query head. This experiment
exposes independent token-dot work to a larger grid, then retains the original
full-row softmax order. It does not implement a fused online FlashAttention
reduction or change KV ownership/precision.

The local vLLM-Metal source has partitioned KV producers/reducers in
`metal/kernels_v2/pagedattention.metal` and shape-sensitive split selection in
`attention/nax_split.py`. These supplied scheduling references, not benchmark
inference. Garnet's candidate differs: token-dot chunks write raw scores into
the existing score allocation, and a second encoder loads those scores into
the original full-row shared scratch. Per-token lane multiply/add and SIMD sum,
full-row maximum, exponential, striped denominator sum, SIMD reductions and
normalization retain the original sequence. A tracked resource dependency
between the two encoders orders raw-score writes before in-place normalization.
No new data buffer or xModel operation is introduced.

The first standalone collection used explicit Metal4 safe math and compared
chunks64/128/256/512. It passed112,728,320 retained probability words against a
complete independent FP64 softmax reference and exact original-kernel bits.
It also checked112,733,440 stage-writer/padding count words, poisoned outputs,
redzones, repeated bit stability and13 corruption rejections. Atomic writer
hooks specialize away in timed pipelines; the separate instrumented pass counts
exactly one raw-score writer for each visible active token and one normalized
output writer for every logical score, including masked positive zero. Padding
has zero writers. These are separate internal versions, not a claim that the
same address is written only once over both encoder phases.

Before native integration, inspection found that the current attention library
uses `compileOptions=nil`, unlike the explicitly safe affine tensor kernels.
The first screen therefore was not a matching-native compiler measurement.
The second standalone collection preserves native nil/default options. On this
SDK its reported default language is4.0 and fast-math-enabled is true. This is
an observation of the existing policy, not a new precision relaxation. A separate
executable links the already qualified native `libgarnet_metal.a` and calls the
actual `MetalContext::EncodeAttentionScores`; all11,272,832 native probability
words across28 cases match the matching-options standalone controls exactly.
The independent matching-options audit again checks112,728,320 probabilities,
112,733,440 writer-count words, all input bits from independently reconstructed
LCG values and the unchanged native attention FP64 bound
`1e-5 + 2e-4*abs(reference)`. It rejects14 corruptions. The initial copied audit
passed the data then correctly failed its own ineffective safe-math mutation;
the corrected test flips the native-default setting instead. No numeric bound
or bit-equality gate changes.

The28 cases include B1/B2/B4 with visible128/512/2048/4096 in C4096, visible
1/31/127/129/511/513/2051 boundaries, uniform negative-zero queries, dominant
logits, cancellation, mixed active lengths with inactive NaN slots, all-inactive
rows, non-power-of-two C2053, padded strides, and extra D64/D256 controls.
Production optimization is narrower than this mathematical screen: only
H32/KV8/D128, B1/B2/B4, C512..4096 and known visible lengths at least512.
Unsupported profiles and GPU-pending controls retain the original kernel.

| Matching-options standalone, chunk128 | Hot score-kernel speed ratio | Swept ratio |
|---|---:|---:|
| B1, visible512 | 1.565x | 1.668x |
| B1, visible2048 | 2.408x | 2.244x |
| B1, visible4096 | 2.818x | 2.373x |
| B2, visible512 | 1.779x | 1.855x |
| B2, visible2048 | 2.765x | 2.495x |
| B2, visible4096 | 3.360x | 2.909x |
| B4, visible512 | 1.347x | 1.387x |
| B4, visible2048 | 2.214x | 1.772x |
| B4, visible4096 | 2.281x | 1.866x |

Ratios compare the complete score operation, including both candidate encoders,
against the original score kernel in that same standalone process. Three
warmups/nine interleaved samples are used; a256MiB sweep is not proof of a cold
cache. These are not model generation or HTTP throughput ratios. Short128-token
contexts can lose to launch overhead, which motivates the512-visible guard.

The native extension is opt-in at compilation:
`GARNET_METAL_SPLIT_DECODE_SCORES` defaults OFF. Its pipelines are compiled once
with the shared immutable pipeline set using the same attention compiler options.
Chunk128 is fixed; actual visible length bounds the dispatch grid, up to the
compiled C4096 envelope. It does not pad short prompts to the profile maximum.
The bridge's explicit `split_decode_scores` boolean selects the policy before
scheduler attachment via `decoder_session.split_score_set_enabled(bool)`.
Preparation and per-frame native dispatch counts make the selected route
observable. CPU-readable, initialized and nonpending active/slot controls must
be binary and finite/integral/in-bounds on active rows; inactive NaN slots are
ignored. Unknown controls and short/unsupported shapes use the original route.
The runtime allocates no new score/KV data and changes no tokenizer or xModel op.

All38 CTests pass. The new native test compares24 ordinary/mixed cases bitwise,
checks short-context fallback, GPU-produced pending-control fallback, profile
rejection and rejection of policy changes during an open GPU batch. The actual
model gate uses two fresh B4/C4096 engines, off then on, over the frozen short
and2K prompt sets. Complete32-layer K/V and hidden hashes match after prefill,
seven ordinary decode calls and two mixed-row calls per case. This is384 full
KV tensor-pair comparisons and six hidden-state comparisons. All72 generated IDs
per mode match retained native prefixes, including masked-row continuation.
The on-mode long case executes288 selected layer dispatches; the short case
retains the old route. Eight model-audit corruptions are rejected. Both engines
close with no tracked owners and bounded driver allocation.

### Serial serving comparison does not establish a stable gain

Four fresh engines run off/on/on/off, B4/C4096, the same approximately2K prompts,
128 outputs per request, no prefix reuse, no operator counters or stage traces,
no residency preparation and unchanged quantization/precision. All2,048 generated
IDs/text/finish match retained controls. The audit binds reported TTFT/decode
arithmetic to raw SSE timestamps and the native score counts to every kernel
frame, rejects15 corruptions, and verifies clean worker exits and zero owners.
The initial audit referenced an inherited second-build filename; the corrected
artifact check binds this candidate's actual successful first build. Numeric,
timing, state and corruption gates remain unchanged.

| Condition | Fresh waves | Median TTFT (s) | Median decode (tokens/s/request) | Median aggregate (tokens/s) |
|---|---:|---:|---:|---:|
| New library, split off | 2 | 33.225 | 4.435 | 8.256 |
| New library, split on | 2 | 32.783 | 4.476 | 8.344 |
| Unchanged older67da native library | 1 | 34.054 | 4.464 | 8.172 |

The two split-on decode wave medians span4.236..4.715 tokens/s, overlapping the
off range4.414..4.455. Aggregate changes by about+1.06% and decode by+0.93%; no
stable serving improvement is admitted from these two observations per mode.
The score route does not run during these main-prefill/first-token paths, so
TTFT variation is not evidence of faster prefill. All ready snapshots have the
same9,954,541,568 device allocation bytes and0 compressed task bytes; the new
kernel has no extra persistent matrix allocation.

Absolute rates in this collection are lower than the earlier native table.
The unchanged older native library is also slow on the current machine, which
is consistent with a shared execution-condition change rather than a slowdown
specific to this candidate. This one anchor does not identify its cause or
prove equivalence across all conditions. Read-only power snapshots report AC,
low-power mode0, and no recorded thermal/performance warnings. GPU frequency
was not measured; no thermal or paging cause is asserted. Older faster results
must not be relabelled as current, nor used to attribute a cross-run speedup.
No vLLM/SGLang inference is rerun or frozen raw result changed.

### Interleaved real-model diagnostic isolates the score benefit

One additional unmanaged B4 engine uses the frozen A prompt in four independent
rows, with32 consecutive decode steps in repeated off/on/on/off order. Its
existing API permits toggling only outside a drained batch, without attaching
a request scheduler. Context and query change between steps; each four-step
cycle gives modes the same average context length, not identical inputs.
This cloned-input diagnostic differs from the diverse serving workload and
uses encoder counters. Its rates are never reported as serving throughput.
All132 token observations match the retained target prefix. Raw counters confirm
32 score encoders per off-step and64 per on-step; seven corruptions are rejected.

| Interleaved diagnostic,16 steps per mode | Split off median | Split on median |
|---|---:|---:|
| Complete GPU command span (ms) | 234.606 | 196.728 |
| Score encoder interval union (ms) | 91.799 | 51.330 |
| Remaining command span (ms) | 143.207 | 145.584 |

Across eight balanced cycles, median off/on speed ratios are1.787x for score
work and1.189x for the complete GPU command. Complete-command ratios range
1.167..1.294x, all above1; other work remains roughly unchanged. This supports a
real score-kernel benefit under the instrumented native workload, with about16%
lower complete GPU-step time. It does not override the inconclusive HTTP result,
qualify a2x decode/aggregate gain, improve prefill, or pass the coding-quality
gate. Compilation remains opt-in and frozen serving profiles remain explicit.

Further work must establish repeated serving benefit under recorded conditions
and improve the remaining attention/value and packed-prefill work. A useful
next screen is profile-specific index/layout specialization while preserving
per-token arithmetic and full-row reduction order. All original Mistral/Gemma
and subsequent Qwen qualification goals remain open; this is a measured native
kernel checkpoint, not completion of the user objective.


## Bounded compiled-layout score indices (2026-11-06)

The follow-up to f3c9e9a49 tests the profile concept directly: compile the known
Mistral H32/KV8/D128/C4096 score layout and address strides, while batch, active
rows, positions and actual visible work stay dynamic. The maximum context is a
storage/dispatch envelope; it does not require padding a short input to 4096.

Standalone evidence is `mistral-score-profile-specialization1`. Five variants
run in interleaved order: unchanged original, current split128, bounded 32-bit
indices, compiled 32-bit layout, and a separate fully-unrolled dot candidate.
The original runtime `rsqrt(float(runtime.dim))` is deliberately retained, even
though D128 is known to address generation: changing its constant folding could
change rounding. Per-lane dot order and the full-row max/exp/striped denominator
sequence stay unchanged. All libraries use the native attention nil/default
compiler options, not the distinct safe-math screen from the preceding study.

All 28 cases and both hot/swept conditions pass 112,728,320 full retained
probability comparisons against the original bits and independent FP64 softmax,
and 112,733,440 per-stage writer/padding checks. All inputs are independently
reconstructed. Existing 1e-5+2e-4*abs(reference) bounds are unchanged. Short,
odd-capacity, padded-stride, D64/D256, inactive/mixed-NaN, uniform, dominant and
cancellation cases are retained. The fixed variants explicitly fall back to
split128 for incompatible layouts. Seventeen corruptions are rejected. Original
and split128 outputs match all retained native-bound control/split128 outputs;
the original kernel source and matching native compiler policy are unchanged.
This reuses the retained actual-native control; it is not a newly-run standalone
native-control program.

| Standalone condition | Index32 / split128 speed ratio | Compiled profile32 / split128 | Fully unrolled / split128 |
|---|---:|---:|---:|
| B1, visible2048, hot | 2.033x | 2.062x | 2.074x |
| B1, visible4096, hot | 1.804x | 1.895x | 1.940x |
| B2, visible2048, hot | 1.983x | 2.077x | 2.082x |
| B2, visible4096, hot | 1.922x | 2.145x | 2.101x |
| B4, visible2048, hot | 2.037x | 2.715x | 2.351x |
| B4, visible4096, hot | 1.963x | 2.324x | 2.310x |

These ratios compare the complete two-encoder score operation with the current
split128 kernel, not only the old original. Three warmups/nine samples and a
separate 256MiB sweep are retained. Sweep is not proof of cold cache. This does
not measure model decode or serving throughput. Full unrolling is not selected:
it gives no consistent advantage over the simpler compiled layout.

Native evidence is `mistral-profile32-score-native1`. The additional
`GARNET_METAL_SPLIT_SCORE_PROFILE32` compile flag defaults OFF and requires
`GARNET_METAL_SPLIT_DECODE_SCORES`. The profile pipelines compile once with the
shared kernel set and the same native attention options. The bridge's explicit
`split_score_profile32` boolean calls the native
`split_score_profile32_set_enabled(bool)` before scheduler attachment. The
context API permits changes only outside a drained batch. The existing split
policy must also be enabled for this route to run. Native preparation and
per-frame profile dispatch observations distinguish it from generic split128.

Exact query/key/output strides 4096/4194304/131072 in float elements are required.
Slot/active strides remain runtime data; every complete stride extent is proven
to fit uint32 before narrowing. Unsupported layout, short contexts and pending
controls retain the existing generic/original routes. The new Metal library has
no additional matrix/KV data allocation and changes no xModel operation,
tokenizer, precision, weights or reduction order. All preexisting shader files
and embedded original shader strings remain byte-identical to parent f3.

All 38 CTests pass, including 36 ordinary/mixed native cases with C2053 fallback,
current split128 and original probability-bit comparisons, bounded-index and
wrong-stride rejection, and pending-control fallback. Two fresh full-model
engines compare split128/profile32 on frozen short and2K prompt sets: 384 complete
KV tensor pairs and six hidden-state pairs match after prefill, seven ordinary
steps and two mixed-mask steps per case. All 72 IDs per mode match frozen native
prefixes. Both modes' full state also matches the retained original-kernel
model results. Eight state-audit corruptions are rejected; engines close with
zero tracked owners and bounded driver allocation. The first native build had
a host-header brace error; the corrected successful second build is explicitly
bound by the audit and retained alongside the failed attempt.

### Serial cold and warmed serving comparison

Four fresh native engines run split128/profile32/profile32/split128. Every engine
processes the same frozen B4/~2K/128-output request set three times in sequence:
one cold round, followed by warm1 and warm2. Prefix reuse, stage tracing,
operator counters and affine residency are OFF throughout. Each warmed round
still recomputes its full prompt and uses distinct request/row epochs. No
vLLM/SGLang inference is rerun. Raw SSE arithmetic, complete outputs, every
native route observation, round/process sequencing and clean shutdowns are
independently audited; all 6,144 generated IDs/text/finish match frozen controls.

| Serving condition | Rounds | Median TTFT (s) | Median decode (tokens/s/request) | Median aggregate (tokens/s) |
|---|---:|---:|---:|---:|
| cold, split128 | 2 | 33.909 | 4.694 | 8.377 |
| cold, profile32 | 2 | 33.118 | 5.221 | 8.890 |
| warmed, split128 | 4 | 32.962 | 4.663 | 8.489 |
| warmed, profile32 | 4 | 32.950 | 5.208 | 8.927 |

Cold median profile32/split128 ratios: decode 1.1124x, aggregate 1.0613x; TTFT time ratio 0.9767. split128 round decode medians span 4.681..4.706 tokens/s. profile32 round decode medians span 5.211..5.231 tokens/s.

Warmed median profile32/split128 ratios: decode 1.1167x, aggregate 1.0517x; TTFT time ratio 0.9996. split128 round decode medians span 4.609..4.724 tokens/s. profile32 round decode medians span 5.172..5.284 tokens/s.

These finite native-only repetitions do not establish general speed or quality
qualification. The main prefill/head does not select profile32 score kernels;
TTFT fluctuations cannot be attributed to faster prefill. References retain
FP16 KV/decode while Garnet retains FP32 state, and older comparator runs have
uncontrolled cross-run machine conditions. No 2x single-request/aggregate or
half-TTFT target is admitted. B1/B2 serving and coding quality are not newly
qualified here. Both compile/runtime profile policies remain explicit and
optional. Next work must address the remaining ordered value-context work and,
separately, the packed-affine prefill bottleneck without repeating the already
compiled/32-bit affine profile or previously rejected precision/tile experiments.
Mistral remains first under the human priority override; Gemma E2B/E4B and
Qwen 27B→9B→4B qualification remain open.


## Compiled value-cache address profile (2026-11-06)

The follow-up to f3073acc2 addresses the ordinary value-cache attention kernel.
The compiled H32/KV8/D128/C4096 layout narrows proved element indices to uint32
and uses 128 threads per group. Each output column retains the original scalar
FP32 accumulation over ascending token positions. No split reduction, precision
change, cache layout change or xModel operation is introduced. Batch remains
one, two or four rows, positions remain dynamic, and visible work uses actual
positions within the maximum context; inputs are not padded to 4096 tokens.

Standalone evidence is `mistral-value-profile-specialization1`. The original,
generic index32, compiled layout, and compiled layout with 128 threads are
compared in interleaved order, with three warmups/nine timing samples and both
hot and separate 256 MiB sweep conditions. The sweep does not establish a cold
cache. All libraries use the original native attention nil/default options;
the old grouped-value study's explicit Metal 3.1 option is not relabelled as
this compiler policy. Original native source is byte-identical to parent.

All 52 cases pass 4,718,592 complete retained output comparisons against original
FP32 bits and an independent saved-input FP64 reference. All 2,389,088 complete
writer-map words, including guards/padding, are checked. Instrumentation is
specialized off during timing and separately proves one writer per logical
output. The 23 rejected corruptions cover metadata, fallback, timings, numerical
payloads, output bits and duplicate/missing/guard writers. Inputs are reconstructed
from their integer formulas; unused cache tails and padding keep their poison
bits. Cases include B1/B2/B4, odd C2053, padded strides, gated output, mixed visible
lengths, NaN/negative/infinite/out-of-range slots, and zero-probability rows.
Fixed-layout variants fall back to the original kernel for incompatibilities.

An additional actual native control links the already qualified f3073 static
Metal library with its matching compiler definitions. All 589,824 native output
words from the 52 cases match the standalone original bits; complete native
outputs are retained and independently compared, rather than trusting only the
program's success flag. This remains operator evidence, not serving throughput.

| Standalone condition | Index32/original | Compiled layout/original | Compiled layout, 128 threads/original |
|---|---:|---:|---:|
| B1, visible 2048, hot | 1.042x | 1.028x | 1.069x |
| B1, visible 4096, hot | 1.141x | 1.128x | 1.133x |
| B2, visible 2048, hot | 1.149x | 1.141x | 1.186x |
| B2, visible 4096, hot | 1.106x | 1.125x | 1.110x |
| B4, visible 2048, hot | 1.194x | 1.396x | 1.428x |
| B4, visible 4096, hot | 1.524x | 1.495x | 1.567x |

Native evidence is `mistral-value-profile32-native1`.
`GARNET_METAL_VALUE_PROFILE32` defaults OFF; the explicit runtime `value_profile32`
boolean also defaults false. Pipelines compile once with the shared kernel set,
using the same compiler options as the original attention library. The native
policy proves dimensions and exact P/V/gate/output strides
131072/4194304/4096/4096 in float elements, ungated output, positive slot stride,
and all complete stride extents within uint32 before narrowing. An explicitly
compiled grouped-value alternative keeps its existing priority. Unsupported
layouts/gating retain their prior kernels. Slots are still GPU inputs; the
candidate preserves the original finite/range test and full-context fallback
for invalid positions. No CPU position inference is needed.

`decoder_session.value_profile32_set_enabled(bool)` configures the policy on a
ready engine before scheduler attachment. Direct context changes require a
drained batch. Preparation and per-kernel-frame dispatch counts make selection
observable. No additional matrix, KV or weight data allocation is added. All
prior shader files, embedded original shader strings and xModel sources are
byte-identical to parent; the eleven preexisting dirty files are inverse-patch
protected independently of this change.

All 39 CTests pass. The new native test has 96 exact original/candidate cases,
covering B1/B2/B4, full and short positions, C2053 and gated fallbacks, invalid
positions, zero-probability rows, index/shape guards and rejection of policy
changes during an open batch. The fresh full-model off/on test keeps the current
compiled score profile enabled in both modes. All 384 complete KV tensor-pair
and six hidden-state comparisons match after prefill, seven ordinary steps and
two mixed-mask steps per case. The two frozen short/long prompt sets produce
72 IDs per mode matching retained native prefixes; complete state also matches
the retained original-kernel results. Eight model corruptions are rejected.
The on mode selects 32 value layers in every ordinary/mixed decode call and
zero during these prefill frames. Both engines close with zero tracked owners
and bounded driver allocation.

### Incremental serving result with the score profile held fixed

Four fresh native engines run value off/on/on/off with the compiled score
profile ON throughout. Each engine executes three serial rounds of the same
frozen B4/~2K/128-output request set: cold, warm1, warm2. Prefix caching, affine
residency, operator counters and stage tracing are OFF. Every round recomputes
its full prompt and uses fresh row/request epochs. No vLLM/SGLang inference is
rerun. Complete outputs, request lifecycle/ownership, profile dispatch counts,
round/process sequencing, clean exits and raw SSE timing arithmetic are audited.
All 6,144 serving IDs/text/finish results match frozen controls.

| Condition | Rounds | Median TTFT (s) | Decode (tokens/s/request) | Aggregate (tokens/s) |
|---|---:|---:|---:|---:|
| cold, score_profile32 | 2 | 26.285 | 7.598 | 12.420 |
| cold, value_profile32 | 2 | 20.411 | 11.518 | 16.253 |
| warmed, score_profile32 | 4 | 26.159 | 7.554 | 12.381 |
| warmed, value_profile32 | 4 | 19.385 | 11.618 | 16.811 |

Cold median value-on/value-off ratios: decode 1.5158x, aggregate 1.3086x; TTFT time ratio 0.7765. score_profile32 round decode medians span 5.276..9.921 tokens/s. value_profile32 round decode medians span 11.366..11.670 tokens/s.

Warmed median value-on/value-off ratios: decode 1.5380x, aggregate 1.3578x; TTFT time ratio 0.7410. score_profile32 round decode medians span 5.209..10.040 tokens/s. value_profile32 round decode medians span 11.362..11.940 tokens/s.

This finite native-only collection does not qualify general performance or coding
quality. No selected value-profile dispatch occurs in main prefill/head, so TTFT
fluctuations do not prove faster prefill. Older frozen references retain FP16
state while Garnet retains FP32, and cross-run machine conditions are uncontrolled.
Both compile/runtime policies remain optional. No 2x single-request/aggregate or
half-TTFT target is admitted, and B1/B2 serving and coding quality are not newly
qualified. The packed-affine prefill bottleneck remains the next priority for
TTFT; its dimensions/strides and indices are already compiled/32-bit, so that
existing optimization must not be rediscovered as new work. All Mistral/Gemma
and subsequent Qwen qualification goals remain open.

### Mixed execution conditions prevent a general serving-gain claim

The unchanged value-off control moves from cold TTFT
33.369 s / decode
5.276 tokens/s to
19.202 s /
9.921 tokens/s at the final control process.
Main prefill selects no new value-profile dispatch, so its large shift cannot
be attributed to this kernel. The blended cold/warmed median ratios above are
retained descriptive arithmetic; the apparent 54% warmed decode gain is NOT
admitted as a causal or stable serving gain. GPU frequency was not measured;
read-only power snapshots record no thermal/performance warning. System swap
usage is recorded separately and is not proof of the cause. Other apps and the
protected WorldSense processes were not changed.

A justified additional test, `interleaved-diagnostic1`, uses one unmanaged engine
and 64 growing-context decode steps in repeated off/on/on/off order, with the
compiled score profile fixed ON. The frozen A prompt is cloned into four
independent rows; this differs from the diverse serving inputs. Every four-step
cycle balances the modes' mean context, not the exact queries/contexts. Encoder
counters and op tracing are OFF. Native dispatch observations confirm 32 score
profile layers on every step and 32 value-profile layers only on enabled steps.
All 260 generated token observations match the full retained prefix. Twelve
corruptions are rejected; native owners return to zero on shutdown.

| Same-engine diagnostic | Value off median | Value on median |
|---|---:|---:|
| Complete GPU command (ms) | 84.836 | 71.631 |
| Host step wall time (ms) | 85.888 | 72.752 |

Across 16 balanced cycles the median off/on complete-command speed ratio is
1.1819x, with range
1.1742..1.1919x;
all cycles favor the candidate. The median wall-time ratio is
1.1802x. This supports an incremental native
model-step benefit under this diagnostic, not a serving throughput ratio,
prefill gain, coding-quality pass or 2x qualification. `performance-assessment1.json`
records the condition shift and claim limits explicitly. Compilation stays
opt-in, and the full goal remains open.


## Prefill staging, paired projections and exact nibble masks (2026-11-06)

This continuation keeps the human Mistral-first priority and the frozen vLLM /
SGLang assets. No comparator inference is rerun. The previous reporting turn
only restated saved measurements; this turn revalidates HEAD, the eleven dirty
paths and engine inactivity before taking the next operator experiments.
These are standalone synthetic screens from `d5262163bf0e`; production/native
runtime, xModel operations, profile limits, precision, current serving library
and the comparison table above are unchanged.

### Narrower groups and pipelined weight staging: negative

Evidence: `mistral-narrow-pipelined-prefill1`. N32/128-thread groups retain the
M128/M256 tile policy and ordered M16/N32/K16 FP32 accumulators. A second variant
preloads alternating weight stages with one stage-retirement/publish barrier per
K128 step. Shared storage is respectively 8,704 and 17,408 bytes, versus the
original N64/256-thread 17,408 bytes. Neither stages input in shared memory or
adds global expanded weights. The runtime guard checks the actual cooperative
ABI in BOTH timed and writer-instrumented libraries before dispatch.

The nine gate/down/QO geometries include M128/M256/M512. Complete output bits
also equal the earlier retained native control. All 121,110,528 saved output
words meet the original FP64 bound, and all 60,558,720 complete writer-map words
(including guards) pass. Twenty-four corruptions are rejected, including an
intentionally underdispatched M512 grid. Most dominant FFN shapes are slower;
the small M128 Q/O improvement does not justify integration.

| Shape | N32 original staging / original | N32 double staging / original |
|---|---:|---:|
| M128, N14336, K4096 | 0.982x | 0.974x |
| M256, N14336, K4096 | 0.971x | 0.983x |
| M512, N14336, K4096 | 0.962x | 0.922x |
| M128, N4096, K14336 | 1.007x | 0.978x |
| M256, N4096, K14336 | 0.835x | 0.826x |
| M512, N4096, K14336 | 0.924x | 0.805x |

These ratios are hot operator times, not full prefill or TTFT ratios. The sweep
condition touches 256 MiB between commands; it does not prove cold caches.
An illustrative weighted projection sum is not promoted to model latency.

### Paired gate/up projection: small positive, native admission still required

Pinned installed Apple/MLX NAX code confirms decoded-weight staging and input
fragment reuse, using the already studied cooperative primitive. Existing
Garnet BF16 joined-MLP graph matchers do not apply to affine-int4. The new
standalone candidate computes both independent affine-int4 gate/up outputs
using one gathered input and reuses each left fragment across both weight
streams. It preserves every output's original K16 accumulation order and
FP32 coefficient FMA-to-half conversion. SiLU remains separate. No new xModel
operator or persistent weight table is introduced.

The narrow pair uses M128/M256, N32, K128, 256 threads and 17,408 shared bytes.
Each SIMD handles both projections at a smaller row tile, retaining the old
per-thread accumulator count. Both projections use distinct packed-code and
coefficient sequences; coefficients include signs, signed zero and subnormals.
The input/output strides have eight padding elements, rows are gathered in
reverse physical order, and two inactive physical rows remain guarded. Nine
actual row lengths 1/127/128/129/255/256/257/511/512 test ragged tiles inside the
up-to512 envelope; they do not pad every request to the maximum.

Evidence is `mistral-paired-gate-up-prefill1`, followed by two independent
collections with additional wide variants. Each timed condition uses three
warmups and nine rotating-order repetitions, counting the two original gathers
versus one candidate gather. Separate writer-instrumented libraries have
same-library ABI guards; every physical output and every writer word is saved.
A deliberately short grid leaves 256 rows absent in BOTH outputs; the audit
independently reconstructs their reversed physical locations and poisoned data.
The old reference kernel strips exactly back to parent source. Later collections
also check original and narrow-pair outputs against the first saved collection.

| Narrow pair / two original projections | First hot | First sweep | Second hot | Third hot |
|---|---:|---:|---:|---:|
| M128, N14336, K4096 | 1.021x | 1.030x | 1.021x | 1.041x |
| M256, N14336, K4096 | 1.048x | 1.048x | 1.051x | 1.055x |
| M512, N14336, K4096 | 1.027x | 1.027x | 1.028x | 1.030x |

This is a modest repeatable synthetic operator improvement. It does not establish
native model benefit, TTFT, serving throughput, coding quality or the 2x goal.
Runtime graph matching, alias/readiness/lease validation, whole-model KV/token
parity and serial native serving qualification are still required before any
production integration or performance promotion.

### Wider paired tiles and masked coefficient decode: negative

`mistral-paired-wide-prefill1` uses N64/K64 and 18,432 shared bytes, holding more
input/accumulator fragments to reuse input across gate/up. The M256 case needs
eight accumulator tensors per SIMD and is slower; this observation is not proof
of occupancy or spills. `mistral-paired-wide128-prefill1` tests the justified
follow-up of keeping that variant at M128 for every row length. It removes most
of the M256 regression, but still fails to beat the narrow pair.

`mistral-masked-nibble-prefill1` separately tests the pinned MLX `dequantize`
bits4 idea of high-nibble masks and an exponent-scaled coefficient. Four nibble
positions use exact FP32 scale multipliers 1/16^j, with the original FP32 FMA
and half destination. No half arithmetic, new coefficient table or relaxed math
is used. Its main shapes do not show a consistent material gain. The M512 hot
collection also shows changing execution times; neither that change nor the
sweep is used to infer frequency, thermal state or memory behavior.

| Candidate / two original projections | M128 hot | M256 hot | M512 hot | M512 sweep |
|---|---:|---:|---:|---:|
| Wide pair, M128/M256 | 1.004x | 0.804x | 0.802x | 0.800x |
| Wide pair, fixed M128 | 1.005x | 1.001x | 0.971x | 0.960x |
| FP32 masked nibble decode | 1.011x | 1.007x | 0.976x | 1.000x |

| Collection | Saved output words checked | Complete writer words checked | Rejected corruptions |
|---|---:|---:|---:|
| `mistral-paired-gate-up-prefill1` | 377,648,832 | 125,887,552 | 27 |
| `mistral-paired-wide-prefill1` | 566,473,248 | 188,831,328 | 27 |
| `mistral-paired-wide128-prefill1` | 566,473,248 | 188,831,328 | 27 |
| `mistral-masked-nibble-prefill1` | 377,648,832 | 125,887,552 | 27 |

Counts include complete physical padding/inactive rows and repeated saved
hot/sweep/instrumented payloads; they are not counts of unique model outputs.
All logical outputs meet the unchanged `2e-6 + 4e-6 * sum_abs` FP64 bound and
match original native bits. Every full writer map includes front/back redzones.
Per-stream swapped weights, wrong values, missing/duplicate/guard writes,
geometry, packing ABI, math/timing modes and incomplete-run corruptions are
rejected. Each study keeps raw JSONL, complete outputs/maps, tool/status files,
source and executable hashes, independent oracle, primary-source hashes and
preparation scripts in `out/research/mistral7b-local-coder/<collection>`.

The masked study preserves its initial failed auditor and a repair record. Its
full numeric/writer checks passed before a negative test incorrectly changed
the two-gather metadata to the already-valid value2. Only that injected value
is corrected to1; raw operator files are unchanged and inference is not rerun.
The corrected auditor must reject the bad value before admission.

The wider/masked candidates are rejected. The narrow pair is retained only as
an operator-qualified candidate. No candidate from these screens is added to
the production runtime, and no new comparison result is reported. Mistral,
Gemma, ANE and the subsequent Qwen qualification requirements remain open.


## Native paired affine-int4 prefill admission (2026-11-06)

Evidence: `mistral-paired-prefill-native1`, from `8898b1349409`. This advances
the N32/K128 gate/up pair from the standalone screen into guarded native graph
execution. It does not establish a serving performance gain. The human
Mistral-first priority remains in effect; all broader model and 2x goals remain
open. vLLM/SGLang inference and their frozen raw assets are not rerun or changed.

`GARNET_METAL_PAIRED_PACKED_PREFILL` defaults OFF and requires strict packed
prefill with FP16 checkpoint coefficients. The additional runtime
`paired_prefill` boolean defaults false. The native
`paired_prefill_set_enabled(bool)` method requires a ready, unpoisoned session
before any request-row claims. Context policy changes require a drained command
batch. The implementation is C++/ObjC++/Metal; Python additions only select the
research profile and record native observations.

The native graph optimizer considers adjacent transposed affine-int4 projections
with the same input and compatible indexed-row policies. Both complete FP32
outputs remain materialized, so existing SiLU and any other consumers retain
their inputs. Unsupported shapes/layouts or aliases decline the pair without
encoding, retaining the original two projection calls. Operator tracing also
retains the original path. This adds no xModel operator and changes no tokenizer,
KV layout, weights, precision or per-output accumulation order.

Gate/up pipelines compile eagerly only for K4096/N14336 prepared layouts, with
M128/M256, N32, K128, 256 threads and 17,408 shared bytes. Actual selected rows
remain runtime data inside the prepared envelope of up to 512 rows; fewer than 128 rows
fall back. There is no request-time pipeline compilation or new matrix/data
allocation. Both projections reuse the existing 14,680,064-byte packed-input
scratch. Logical packed/affine/indexed counts still count both projections;
a separate `paired_prefill_dispatches` observation counts physical layer pairs.

All controls and BOTH output spans are validated before any zero fill or gather:
shape/device, complete uint32 stride extents, prepared maximum rows, input
readiness, output pending ownership, span-write readiness, row uniqueness/range,
byte extents, alias overlap and private-workspace ownership. Inactive physical
rows are cleared in both outputs. Both pending outputs are tracked through the
existing lifetime/abort machinery. The eleven preexisting dirty paths are
preserved separately from owned edits; the dirty root is never built.

### Native correctness and build qualification

All 40 serial CTests pass. The new test has 21 complete original/candidate cases:
empty/4/127 fallback,128/129/255/256/257/511/512 active rows, permuted/reversed
maps, dense execution, inactive output clearing, numerical samples at the
unchanged FP64 bound, disabled defaults, invalid indices, aliased/unsupported
outputs, unready inputs, policy locking, abort/recovery and scratch/plan bounds.
Both outputs are compared in full; no request-time matrix allocation appears.

The clean full 32-layer off/on runs keep the existing compiled score and value
profiles ON in both modes. All 384 complete KV tensor and six hidden-state
comparisons match, as do 72 generated IDs per mode, seven ordinary and two mixed
active-mask decode calls per prompt set. The two frozen short/long sets retain
original native state and ID controls. Eight corruptions are rejected. Actual
observations select 32 layer pairs in each eligible prefill frame, zero in
fallback tails, and zero in every ordinary/mixed decode frame and head call.
Both engines close with zero tracked owners and bounded driver allocation.

The immutable compiled qualification and admitted code are both `source1`.
Unsupported FP16-coefficient-OFF full configurations are rejected by existing
Mistral decode-reuse and FP16 operand guards. Packed already requires reuse128,
which explicitly requires FP16 coefficients and operands. A redundant guard
prototype in `source2` is retained only as an unadmitted build attempt. Its
negative-test assertions initially expected later diagnostics; the full CMake
configurations correctly rejected earlier. `build-guard-proof1.json` records the
verified dependency chain and qualified binary SHA. No production source changes
after inference qualification. All previous shader files, three embedded original
shader strings and xModel sources stay identical.

### Serving result: no stable incremental gain

Four fresh engines run pair off/on/on/off, each with cold/warm1/warm2 rounds of
the frozen four 2K requests and 128 outputs/request. Score and value profiles stay
ON, while cache reuse, affine residency and tracing remain OFF. All 6,144
IDs/text/finish results match frozen native controls; epochs, chunk selection,
per-frame pair/score/value/affine counts, raw SSE timing, serial process/round
boundaries and clean exits pass. Twenty-six corruptions are rejected.

| Condition | Rounds | TTFT (s) | Decode (tokens/s/request) | Aggregate (tokens/s) |
|---|---:|---:|---:|---:|
| cold, separate_prefill | 2 | 17.069 | 12.880 | 19.017 |
| cold, paired_prefill | 2 | 17.759 | 12.523 | 18.297 |
| warmed, separate_prefill | 4 | 18.075 | 12.485 | 18.093 |
| warmed, paired_prefill | 4 | 18.236 | 12.362 | 17.917 |

The first and last unchanged control phases shift from cold TTFT 15.762 s /
decode 13.574 tokens/s to 18.375 s / 12.185 tokens/s. The paired kernel has zero decode
dispatches, so that decode shift is not caused by pairing. GPU frequency is not
measured; no thermal, paging or clock cause is inferred. The blended medians
above are descriptive. They do not qualify a stable positive serving/TTFT gain,
and the runtime policy remains OFF by default.

### Controlled identical late-prefill replay

A justified additional diagnostic uses one UNMANAGED engine so policies can
change between drained batches without weakening managed-session policy guards.
Four diverse frozen prompt prefixes are seeded through 2048 tokens. Every replay
uses the exact same prepared input/position/active tensors and slots 1920..2047.
One warm ABBA cycle precedes six measured ABBA cycles, yielding 24 measured
steps. Embedding, head prediction, hashes, preparation and mode changes are
outside the timed span; encoder counters and operator tracing remain OFF.

| Fixed late chunk | Pair off median (ms) | Pair on median (ms) |
|---|---:|---:|
| Complete GPU command | 1188.191 | 1181.465 |
| Host run wall time | 1191.548 | 1184.676 |

The six-cycle median complete-command ratio is 1.005747x, with range
1.004701..1.006630x. This is about 0.6% improvement on one
identical late-prefill replay, far smaller than the standalone gate/up gain.
Every replay records 160 logical packed projections and 224 affine projections,
with 32 layer pairs only when enabled, zero misses and no partial submission.
All 28 hidden witnesses match; before/after full KV byte hashes (64 tensors) and
hidden hash match, and the untimed final A prediction matches its frozen native
ID. Twelve corruptions are rejected. Cleanup returns all tracked owners to zero.
This is not a full-prompt TTFT, serving throughput or coding-quality result.

The candidate is retained as an experimental native route with exact arithmetic,
not promoted as a solution to the prefill bottleneck. The next prefill work must
address a larger share of full-command cost; this experiment rules out expecting
a substantial TTFT gain from input-fragment sharing alone. No 2x single-request,
aggregate, half-TTFT or new strict coding-quality target is admitted. ANE/Gemma
and subsequent Qwen requirements remain open.


## Inclusive MPS prefill census and native SiLU32 screen (2026-11-06)

This follows `f61fb8384374`. It adds research replay patches and evidence, with
no production runtime/profile change and no new serving table. The previous
paired route remains optional; all Mistral/Gemma/ANE and subsequent Qwen goals
remain open. The frozen vLLM/SGLang engines and raw baselines are unchanged and
are not rerun. All GPU phases run serially in isolated clean archives; the
eleven preexisting dirty paths remain untouched.

### Inclusive MPS query/key and probability/value intervals

`mistral-prefill-mps-envelope-diagnostic1` extends the temporary counter build
to bracket each original pair of cached/live MPS products. Internal MPS encoders
remain unobserved. A one-thread marker uses volatile integer read/write of the
same first word of each relevant input/output matrix. This preserves bits while
establishing a tracked resource hazard; an unrelated timestamp encoder alone
would not prove the MPS group had finished. The marker before each group touches
its already-produced inputs, and the marker after it touches the produced
outputs. Both products remain together, in their original order. Markers may
add serialization/overhead, so these are inclusive diagnostic intervals rather
than pure MPS kernel times or uninstrumented cost fractions.

The clean build passes 41 serial CTests. The additional test compares original
attention output bits with counters OFF then ON across nine complete-write
cases, including gapped/ragged, overwrite, gated, pending controls, tails and
inactive inputs. Its independent FP64 samples and unchanged read-only KV checks
pass. The actual 32-layer diagnostic then seeds four diverse frozen prefixes
through 2048 tokens and replays exactly slots 1920..2047 with pair OFF and current
score/value profiles ON. One warmup and three measured steps per counter policy
use identical prepared input/position/mask tensors. All eight hidden witnesses
and 64 before/after full KV hashes match. The untimed A prediction matches its
frozen native ID; cleanup returns tracked owners to zero.

Five complete counter records cover the four counter-enabled replay calls and
one untimed final prediction. Every record has 64 owned MPS envelopes, 128 real
identity markers and 96 pack/softmax/merge encoders. The auditor ties interval
endpoints to the actual boundary sample indices, checks calibrations and native
routes, and partitions the union of all intervals without double-counting.
Twenty corruptions are rejected. Overlapping categories stay explicit, and
unsampled gaps are not labeled as MPS or utilization.

| Exclusive category | Fraction of instrumented GPU span |
|---|---:|
| Packed affine projections | 55.275% |
| Other affine projections | 5.798% |
| MPS query/key group envelope | 8.031% |
| MPS probability/value group envelope | 7.808% |
| Native attention softmax | 8.702% |
| SiLU | 5.465% |
| Identity markers themselves | 0.021% |

The complete fixed-chunk GPU medians are
1058.908ms without counters and
1080.072ms with counters.
OFF precedes ON; the difference is descriptive, not an isolated causal estimate
of probe overhead. The MPS envelopes do not hide the main packed-GEMM cost.
These measurements are not TTFT, serving throughput or GPU utilization.

The exact diagnostic-only replay patch is
`test2026/metal/experiments/mistral-mps-envelope-diagnostic.patch`. It reproduces
the isolated source changes from the parent; it is not applied to production.
The corresponding command/status files, unit/model logs, counter JSON, original
program hashes and full state witnesses are retained in the study directory.

### SiLU layout/indexing screens and timing-control qualification

The native pointwise kernel computes 64-bit row division/remainder and row-stride
indices for every element. The bounded screens preserve the original FP32
expression and nil compiler options while testing maximum-thread and 128-thread
original calls, generic 32-bit indices, contiguous scalar 32 and contiguous float4.
Both finite and nonfinite/edge corpora include ten layouts with 1/4/128/256/512
rows, width 14336/4096, padded strides and a ragged width 14335. The complete
writer maps include front/back guards and physical padding; deliberately
underdispatching the vectorized M512 case leaves the expected half poisoned.

`mistral-silu-layout-profile1` and `mistral-silu-finite-profile1` each audit
398,326,980 complete physical output words and 132,775,660 complete writer words.
All candidate outputs match the actual old native API in bits. Finite inputs
meet `2e-6 + 4e-6 * abs(FP64 reference)`. Existing original-native IEEE class
differences for nonfinite inputs are counted explicitly; the edge qualification
preserves native bits and does not claim new IEEE nonfinite compliance.
The initial IEEE-class assertion and later path-binding auditor failures are
retained. Only auditor scope/path fixes are made; raw collections are unchanged.
The final independent auditors are `audit4.json` in both directories.

The copied-source standalone control and actual native API have materially
different timing conditions at some widths. In particular the standalone loop
rewrites poisoned output buffers on the CPU before each timed pass, while the
separate actual-native hot collection does not. Compilation context also
differs. The cause of the timing gap is not isolated. The copied-source 2-5x
ratios therefore are NOT promoted as Garnet speedups, despite exact output bits.

### Actual native same-context index32 result

`mistral-silu-native-screen1` resolves that control ambiguity with the actual
MetalContext and its complete-command GPU timestamps. It compiles the candidate
into the same main Metal library as the preserved old SiLU function, alternates
off/on order with equal buffer handling, and checks three warmup/nine measured
runs per mode. The temporary C++ policy requires drained batches; it is not
exposed to XLang, sessions or serving. It selects only bounded rows <= 512 and
width 4096/14336 with complete stride extents within uint32. The candidate uses
the same FP32 expression, generic 32-bit indexing and 128 threads. Unsupported
width 14335 retains the old route.

| Finite corpus, width 14336 | Original native GPU median (ms) | SiLU32 median (ms) | Component speed ratio |
|---|---:|---:|---:|
| 128 rows | 0.354375 | 0.144708 | 2.449x |
| 256 rows | 0.769625 | 0.329250 | 2.338x |
| 512 rows | 1.713250 | 0.858750 | 1.995x |

All 20 finite/edge native cases pass 1,274,308,560 runtime word comparisons and
106,192,380 retained full-output word comparisons against the original native
API payloads. Eleven corruptions are rejected. Prepared owner counts stay
constant during calls and return to zero after teardown. Source padding is
stripped to logical values for native Upload, so those native cases verify data
semantics rather than additional padded native view layouts; the standalone
writer/oracle screens cover their physical padding explicitly.

The kernel prototype is saved exactly as
`test2026/metal/experiments/mistral-silu-native-profile-screen.patch`, independently
replayed onto its parent files and checked against qualified source hashes.
Both research patches retain all original embedded shader strings and xModel
sources. They are text artifacts, not enabled runtime changes.

This is a worthwhile native component candidate, not a 2x decode/throughput or
half-TTFT result. SiLU occupies about 5.5% of this instrumented late-prefill span;
its improvement alone cannot meet the end-to-end target. Full 32-layer KV/hidden
and complete generated-ID checks, a guarded optional session policy and serial
serving measurements are still required before production admission. No new
coding-quality or single-request-serving qualification is claimed.


## Bounded native SiLU32 profile admission (2026-11-06)

This follows `f917f94e635d`. The clean isolated archive in
`out/research/mistral7b-local-coder/mistral-silu-profile32-native1` adds an
optional native SiLU profile. Compile option `GARNET_METAL_SILU_PROFILE32` and
runtime `silu_profile32` both default OFF. It selects F32 matrices with actual
rows 1..512 and width 4096/14336 only after proving each complete row-stride
extent fits uint32. Row counts are dynamic within the bound, with no padding to
512. Unsupported shapes retain the old path. The expression and ordinary main
Metal library compiler options are unchanged; the candidate uses uint32 indices
and 128 threads. The eager pipeline has no request-time compilation or owners.

The session setter requires a ready, unpoisoned engine before request rows are
claimed. The context setter requires drained commands. Begin/Abort reset the
observation count independently. Backend capability, preparation and per-frame
dispatch observations expose whether the optional route actually ran. The
worker prepares it before row claims. xModel operations, scheduler, weight/KV
storage, all prior shader headers and three original embedded shader strings
are byte-identical. Source scope covers seven modified and three new native/
worker/test files, plus this separate documentation addition.

### Correctness evidence

All 41 serial CTests pass. The new actual-native test covers 24 finite/edge
cases, including rows 1/4/128/256/512, both supported widths, unsupported width
14335 and rows513 fallback. It compares all 41,086,720 logical output words
bitwise against the preserved original native API. 3,071 finite FP64 samples
meet `2e-6 + 4e-6*abs(reference)`. Nonfinite evidence preserves native bits; it
does not assert new IEEE-class compliance. Host-only stride tests include
padded and overflow extents; the current native Upload cases are logical
matrices, not an additional physical-padding native view qualification. The
previous standalone writer/oracle screen retains the padding evidence.
Unready input, alias, bad shapes, abort/recovery, flush/policy locks, default
OFF, unrelated Add/Norm and zero retained owners are checked.

Two serial full-model runs use the frozen short/long four-row prompts. Score
and value profiles are ON in both runs and paired prefill is OFF; only SiLU32
varies. Every prefill body, including tails, and ordinary/mixed decode selects
32 SiLU kernels when enabled; prediction heads select zero. All 384 full KV
tensor hash comparisons, six complete hidden-state comparisons, 72 generated
IDs per mode and mixed active/inactive predictions agree. They also agree with
the retained original native state and ID control. Eight deliberately corrupted
reports are rejected. These checks preserve target behavior, not coding quality;
the existing strict coding result remains 0/8.

### Serial serving and fixed late-prefill replay

Four fresh native engines run OFF/ON/ON/OFF, each with cold/warm1/warm2 rounds
of B4, approximately 2048 input tokens and 128 output tokens per request. Cache
reuse, stage tracing, operator counters and extra weight residency stay OFF.
All 6144 generated-ID observations, prompt IDs, text, finish reasons, native
leases/epochs, chunk coverage and cleanup pass independent audit against the
retained controls. TTFT and decode arithmetic are rebound to actual SSE
content/usage timestamps. The new SiLU observations are adjacent to the native
frame and distinguish prefill/decode from head. Thirty report corruptions are
rejected. Raw engine/worker/SSE events, commands, statuses, bound source and the
exact library are retained. No vLLM/SGLang inference is rerun.

| Engine / cold condition | TTFT seconds (lower better) | Per-request decode tok/s | Aggregate tok/s including prefill |
|---|---:|---:|---:|
| Garnet native SiLU OFF | 17.276 | 12.717 | 18.771 |
| Garnet native SiLU32 ON | 17.209 | 12.187 | 18.470 |
| Frozen vllm-metal | 11.570 | 10.484 | 21.179 |
| Frozen sglang-mlx | 10.884 | 6.307 | 16.337 |

Native entries are medians from two cold waves per policy; frozen references
retain their older 15-wave medians. Native FP32 KV/attention versus reference
FP16 and uncontrolled machine execution conditions limit cross-engine claims.
Warmed TTFT medians are 18.089s OFF and 17.599s ON (2.71% lower); warmed
aggregate medians are 18.080 and 18.362 tok/s (1.56% higher). Cold TTFT differs
by only -0.38% and cold aggregate is 1.60% lower. Off control wave ranges
overlap candidate results; no consistent serving improvement is established.
Every individual wave remains in `audit_http1.json`.
These measurements do not establish any of the 2x/half-TTFT targets.

A separate same-engine diagnostic seeds the four frozen prefixes through2048,
then replays identical slots1920..2047 with unmanaged `session.run`, one warm
ABBA cycle and six measured cycles. No row claims or scheduler policy guard is
weakened. Prepared inputs and complete KV state are fixed. All 28 hidden
witnesses and 64 before/after full KV hashes match; the untimed A prediction
matches its frozen ID and owners return to zero. Hashing, embeddings and heads
are outside the timed span. Twelve corruptions are rejected.

Complete late-chunk GPU medians: original 1220.744ms,
SiLU32 1176.920ms. Median within-cycle GPU speed ratio is
1.03860x, range 1.03544..1.03954x;
corresponding host ratio is 1.03858x. This is full-chunk replay,
not serving throughput, TTFT or isolated SiLU kernel time. Component speedups
from the prior native screen are not promoted as end-to-end gains.

Final review caught the new header nested inside the packed-prefill include
guard. Corrected `source2` uses sibling guards so SiLU32 can compile with packed
prefill OFF. All other source bytes match measured `source1`. Qualified Release
preprocessing differs by exactly one empty line between two closed header
namespaces: every C++ token and shader literal is unchanged, without path or
token normalization. The initial stricter exact-byte assertion and failed status
are retained. A separate complete native library build with packed/pair OFF and
SiLU32 ON and its actual-native SiLU test pass. Serving raw data and the measured
library remain bound to immutable `source1`; the correction is bound separately
by `include-fix-proof1.json` and `native-source2.json`. No timing is transferred
from the packed-OFF build. The final commit takes native files from `source2`.

The route remains optional and defaults OFF. `performance-assessment1.json`
records the measured serving and replay findings. More improvement must come
from larger prefill/decode costs; SiLU alone cannot satisfy the original goals.
Mistral/Gemma/ANE and subsequent Qwen qualification remain open.

An earlier preparation assertion stopped before context methods were written.
A prematurely started build then failed at link time. Both terminal attempts,
their statuses and partial source/build are preserved as attempt1; no inference
ran from them. A fresh clean archive, corrected independent Abort reset and
successful rebuild produced all admitted evidence above.


## Actual native packed-projection loop-control screen (2026-11-06)

This follows `693baa05783b`. The clean isolated study
`out/research/mistral7b-local-coder/mistral-packed-loop-control-native1` tests
inner-loop controls used by the pinned MLX implementation. Its retained
`quantized_nax.h` has `STEEL_PRAGMA_NO_UNROLL` at line1125, a volatile compiler
barrier declaration at1130 and a discarded read at1147. The local source hash
and exact lines are bound in `plan1.json`; this is source inspection, not a new
vLLM/SGLang inference run.

Three diagnostic variants add inner `unroll(disable)`, the volatile barrier
alone, or both to the existing M128/M256 packed kernels. They compile next to
the preserved original kernels in the actual native context's same strict
Safe Metal4 library. All three retain original code/accumulation order,
function-constant dimensions/strides, FP16 input/coefficient dequantization,
FP32 accumulators/output, dynamic actual row counts, grid and 256 threads.
The existing cooperative ABI validation still runs before any plan is
published. Eager variant pipelines must have the same static threadgroup memory
as their original. No xModel operation, scheduler, KV/weight layout or scratch
allocation changes. A temporary drained-context C++ selector chooses already
prepared pipelines; it is not exposed through XLang, sessions or serving.

Source audit covers all26213 tracked parent files. Only three research native/
CMake paths are modified, with two new diagnostic headers/test files. Removing
only the declared pragma/barrier controls and renaming variants reproduces both
original kernel bodies byte for byte. All prior shader headers and the three
original embedded shader strings are unchanged. The11 preexisting dirty paths
remain untouched. Both diagnostic observation counters reset on Begin/Abort.

All41 existing CTests pass serially. The native operator collection contains
62 cells: all three Mistral projection dimensions, an unsupported1024-output
fallback, 128/256/512 physical rows, empty/tiny/tile-boundary/tail selections,
permuted and reversed indices and dense calls. Every call poisons output first,
checks full output finiteness and inactive zero rows, then compares every word
with the original nonpacked native control. All four variants also remain
repeatable. Invalid policy, command/flush policy locks, duplicate rows, abort
unreadiness/recovery, constant prepared plan/owner counts and clean teardown
are checked. Actual native route counts bind each mode to each call.

The independent offline auditor checks2,105,540,608 runtime word comparisons
and709,885,952 retained full-output words. All candidate bits match the original
native output. 6960 independent FP64 samples reconstructed from retained raw
inputs, packed Q4 weights and FP16 coefficients pass the original
`2e-6 + 4e-6*sum_abs` bound; maximum bound fraction is0.019113. Twelve report/
payload corruptions are rejected, including a changed active output bit,
missing writer poison, altered row mask, missing mode, wrong route count and
false oracle. Physical native matrices are tightly allocated logical views;
this does not add a new external-padding/redzone qualification.

The timed cells use three warmup and nine measured calls per mode in rotating
order, identical buffers/poison handling and full-command native GPU timestamps.
These are actual native component times, not copied-source standalone controls.
Full-output downloads, oracle checks and raw saves occur outside GPU timestamps.
Dense-call medians in milliseconds are:

| Actual rows | Output columns | Inner dimension | Original native | No-unroll | Volatile barrier | Both |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 14336 | 4096 | 1.4421 | 1.4463 | 1.4559 | 1.4252 |
| 128 | 4096 | 4096 | 0.4071 | 0.4058 | 0.4055 | 0.4070 |
| 128 | 4096 | 14336 | 1.6719 | 1.6559 | 1.7169 | 1.6412 |
| 256 | 14336 | 4096 | 2.5983 | 2.5902 | 2.5995 | 2.6282 |
| 256 | 4096 | 4096 | 0.8441 | 0.8564 | 0.8337 | 0.8462 |
| 256 | 4096 | 14336 | 3.1423 | 3.1224 | 3.1748 | 3.1845 |
| 512 | 14336 | 4096 | 4.8980 | 4.9002 | 4.8951 | 4.9264 |
| 512 | 4096 | 4096 | 1.8439 | 1.7007 | 1.7421 | 1.8215 |
| 512 | 4096 | 14336 | 5.4670 | 5.7485 | 5.6572 | 5.5488 |

The broad result is negative: wide gate/up is approximately flat and the long
512-row down projection slows. The512-row4096x4096 dense ratio of medians looks
8.4% faster with no-unroll, but its median paired-sample ratio is only1.0153x
with6/9 wins and range0.9174..1.1836x. Its indexed counterpart has4/9 wins and
ratio of medians0.9377x. That single dense median is not a qualified profile
selection. Pipeline maximum-thread limits stay1024 for these cells; this does
not measure physical occupancy, register counts, clock frequency or utilization.

No candidate is promoted and no model/serving run is performed for it. The
existing SiLU32 serving comparison remains current, and no new TTFT, decode,
throughput or coding-quality result is claimed. Strict coding0/8 and all unmet
2x/half-TTFT, Mistral/Gemma/ANE and subsequent Qwen requirements remain open.
The negative result rules out this small compiler-control change as a general
solution to the dominant projection cost.

The exact local-only replay patch is
`test2026/metal/experiments/mistral-packed-loop-control-native.patch`. It is
independently applied to the exact parent files and checked against the
qualified hashes. Only that text artifact and this evidence record enter the
production branch; the research selector/kernels are not enabled there.


## Bounded bulk-projection amortization feasibility (2026-11-06)

This follows `74a0b6471`. After the negative inner-loop screen, this experiment
changes the work schedule: project several known prefill chunks together and
amortize one decoded weight across the larger product. It is a native component
feasibility study for an eventual prepared layer/operator-major prefill plan,
not a completed model schedule. Evidence lives in
`out/research/mistral7b-local-coder/mistral-bulk-projection-native1`.

The research C++ interface prepares dense actual-row MPS plans up to8192 rows
for the three existing Mistral projection dimensions. It decodes Q4 weights
with the unchanged native FP32 coefficient FMA-to-FP16 helper, casts actual F32
input to FP16, and requests an F32 MPS result directly into the output owner.
Vendor internal accumulation is not independently instrumented. Each measured
call includes unpacking and input casting; no expanded weights are prepared
outside timing or retained for free across calls. Shared workspace is fixed at
117,440,512 half-weight storage bytes plus234,881,024 half-input bytes,
352,321,536 total (336MiB). Existing owned F32 buffers provide storage and
lifetime accounting; their internal views carry half data. Prepared MPS plans,
pipelines and matrix owners remain constant during collection.

The control is the actual current native packed-affine path, with the same
weights and16 distinct512-row input chunks for the largest case. All chunk
calls run inside one native command rather than charging16 host submits. The
candidate runs one product on their exact dense concatenation. This avoids the
old repeated-slice oracle error and copied-source timing ambiguity. Both
policies use the same native context, prepared resources and output-poison
handling. Actual route observers require `chunks` packed calls and zero bulk
calls for control, or zero packed calls and one bulk call for candidate. All
buffers/plans exist before warmup. The dimensions are actual counts within the
bound; the512/2048 cases do not execute8192 padded rows.

All41 existing native CTests pass serially. Nine projection cells cover actual
rows512/2048/8192, meaning1/4/16 packed chunks, with three warmup and nine measured
calls per mode in alternating order. Each call starts with poisoned output,
requires every result finite and deterministic, and checks no new prepared
pipeline/matrix owners. Native preparation during a command, unready inputs,
abort unreadiness/count reset and recovery are checked after saving outputs.
Owners return to zero after all contexts close. Dense complete initialized
inputs are required by this research API; sparse row masks, padded views,
chaining pending bulk outputs, attention, request leases and scheduler fairness
are not qualified by it.

The independent audit checks5,813,305,344 runtime output-word observations and
484,442,112 retained output words. Every one of the242,221,056 paired candidate
output words is bit-identical to the native chunked control. 660 independent
FP64 samples reconstructed from raw inputs, Q4 codes and FP16 coefficients pass
the unchanged `2e-6 + 4e-6*sum_abs` bound, with maximum bound fraction0.009793.
This is not a new full-output FP64 oracle qualification. Twelve corruptions
are rejected, including missing-writer poison, wrong output, altered shape/
chunk schedule, workspace bound, route/order, timing, word count and oracle.
All prior shader headers, three embedded shader blocks and xModel sources are
unchanged. The eleven preexisting dirty paths are untouched.

| Actual rows | Output columns | Inner dimension | Native chunked GPU ms | Bulk GPU ms | Component speed ratio | Paired wins |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 14336 | 4096 | 4.884 | 6.303 | 0.7748x | 0/9 |
| 2048 | 14336 | 4096 | 18.966 | 18.937 | 1.0015x | 5/9 |
| 8192 | 14336 | 4096 | 75.272 | 70.646 | 1.0655x | 9/9 |
| 512 | 4096 | 14336 | 5.541 | 6.916 | 0.8012x | 0/9 |
| 2048 | 4096 | 14336 | 21.027 | 20.820 | 1.0099x | 9/9 |
| 8192 | 4096 | 14336 | 83.418 | 77.683 | 1.0738x | 9/9 |
| 512 | 4096 | 4096 | 1.530 | 1.800 | 0.8500x | 0/9 |
| 2048 | 4096 | 4096 | 6.011 | 5.964 | 1.0079x | 8/9 |
| 8192 | 4096 | 4096 | 23.456 | 21.541 | 1.0889x | 9/9 |

At8192 rows all nine paired samples favor bulk for each projection. Paired ratio
ranges are1.0573..1.0677x for gate/up,1.0729..1.0761x for down and
1.0874..1.0965x for Q/O. At512 rows all9 pairs regress;2048 rows are approximately
flat to modestly positive. The result supports amortization only for larger
known work groups, not replacing the current small-chunk route. These are
complete native projection command times, not TTFT or full-prefill speed.
They do not isolate measured DRAM traffic, GPU frequency or utilization.

This is a modest6.5-8.9% large-projection component gain, below the overall
2x/half-TTFT goal even before unmeasured full-model costs. A serving path needs
an explicitly bounded activation tape and prepared operator/layer-major plan,
per-layer causal KV progression, state/hidden parity, lease/prefix correctness
and a policy that preserves existing decode fairness. No such model schedule
is implemented here. No full-model, HTTP, decode, throughput, coding-quality or
ANE measurement is added. Strict coding0/8 and all Mistral/Gemma/ANE and
subsequent Qwen goals remain unresolved; the prior SiLU32 serving table remains
current and frozen vLLM/SGLang are not rerun.

The source replay is
`test2026/metal/experiments/mistral-bulk-projection-native.patch`. It is
independently applied to the exact parent files and checked against qualified
hashes. Only the research text patch and this record are committed; the
prototype C++ API is not part of the selected production inference route.


## Native layer-major prefill with bulk FFN: correct but slower (2026-11-06)

This follows `b6f8683ab`. The previous component result now has a real native
full-model schedule in the isolated research build
`out/research/mistral7b-local-coder/mistral-layer-bulk-prefill-native1`. It is
not selected for production because balanced model timings regress. The frozen
vLLM/SGLang assets and current SiLU32 serving table remain unchanged.

### Prepared execution and memory contract

`GARNET_METAL_DIAGNOSTIC_LAYER_BULK` defaults OFF. The research build requires
packed affine, private prepared spans, last-active hidden and owner verification.
A compiled graph must contain exactly one structurally matched Mistral FFN:
RMS norm, gate/up affine projections, SiLU multiply, down projection and residual
add. Captured dimensions, affine weight identities, normalization parameters,
cache shapes, graph generations and permitted suffix operations are validated.
The32 graph owners, head and shared scratch are locked through completion.
No xModel operation or tokenizer change is made.

The B4/chunk128 plan retains16 known chunks, with8192 total token rows. It runs
each layer's attention prefix in increasing chunk/slot order against that
layer's private K/V state, copies each residual into prepared tape storage,
then performs one bulk FFN over the concatenation. Attention and its causal
cached/live selection remain chunked; no full-context score matrix is added.
Two hidden tapes alternate across layers. Every chunk has a separate initialized
control/derived-control bank, so a CPU update never overwrites controls still
needed by earlier GPU encoders. Prefix-operation masking and all temporarily
bound slots are restored even on exceptions. The ordinary current path is
retained as the in-context control, using the same prepared resources.

Additional workspace is2,446,393,344 distinct buffer bytes, including two tapes,
residual/normalization/gate/up/SiLU/down storage, control banks, output hidden
banks and the352,321,536-byte bulk helper workspace. The complete estimate is
checked against the device recommended working-set budget before allocation;
all generic native allocations retain the existing256MiB driver margin.
Prepared source2 allocated11,792,662,528 device bytes; after first-call ordinary
input materialization both modes report11,801,051,136 bytes and unchanged native
owner counts through measured runs. These are snapshots, not peak/system RAM
or absence-of-paging claims. Process compression changes are recorded and are
not treated as a measured causal explanation of performance.

The prototype accepts a fresh unmanaged dense text prefix only. It validates
and privately snapshots all host input/control tensors before resetting its
isolated session. Preparation rejects an attached MTP workspace. Every run rejects configured
token caches and managed request rows; the qualified build has MTP disabled. It does not reset a leased serving context. Sparse/tail prefill, cached
bulk execution, serving admission and decode fairness of a managed bulk group
remain unimplemented; existing serving continues through its previous path.

### Numerical, continuation and rejection evidence

Both source versions pass42 serial CTests: the prior41 plus a new generic
prepared-view test. The latter verifies all1024 output words, offsets, shared
readiness, bounds, open/partial-command preparation rejection, abort unreadiness
and complete-parent restoration. The bulk projection reader now accepts inputs
pending only in its own current command, preserving the existing ownership rule
for a norm/SiLU producer feeding MPS in the same batch.

The corrected source2 model passes complete32-layer KV and hidden hashes for
the frozen four distinct prefixes through2048 tokens each. They also match the
retained original native witness from `mistral-paired-prefill-native1` exactly.
Actual control routes report2560 packed projections and zero bulk products;
candidate routes report1024 packed prefix projections and96 bulk FFN products.
Every mode returns identical four prefix prediction IDs. Eight independently
corrupted first-probe reports are rejected.

Untimed generation then compares1024 token-ID observations (128 per row/mode),
127 ordinary decode calls per mode, and two alternating mixed-active-mask calls.
All IDs,192 complete post-prefill/decode/mixed KV tensor hash comparisons and
three full hidden comparisons agree. Rows A/C are complete original2048-token
prompts and512 ID observations also match their unchanged historical128-token
native controls. B/D use explicit2048-token prefixes of longer prompts; no full
historical B/D-output claim is made for the truncated inputs. Four malformed
prefix cases per mode are rejected before reset, with full prefix state
unchanged after the guard group. A managed call is rejected with all four
original leases releasable and complete KV state unchanged. Ten corrupted
generation/guard reports are rejected. These are deterministic native parity
gates; strict coding remains0/8, not a new quality pass.

### Balanced full-model measurements

One native process warms OFF and ON, runs three OFF/ON/ON/OFF measured cycles,
then performs two full KV gates after all timing collection. Complete hidden
and prediction checks occur after every call. Large full-KV witness allocations
are deliberately kept after measured collection to avoid perturbing the next
measured mode. The independent audit requires12 measured calls, two warm calls,
two post-gate calls, fixed routes/owners, raw timing finiteness and native
original-state identity; eleven corruptions are rejected.

Every input embedding is prepared before these measurements. The body includes
the four-row vocabulary prediction; native API wall additionally includes
input snapshot and reset work. HTTP queueing, tokenizer/embedding preparation,
SSE and ordinary generation speed are not measured here. These are not TTFT.

| Metric | Current control median s | Layer bulk median s | Median within-cycle control/candidate speed ratio |
|---|---:|---:|---:|
| Summed GPU commands | 17.416 | 18.128 | 0.96073x |
| Native body wall | 17.474 | 18.244 | 0.95780x |
| Complete native API call wall | 17.652 | 18.376 | 0.95966x |

All three measured GPU cycle ratios are below1, ranging0.95766..0.96118x.
The median ratio0.96073x corresponds to about4.1% more GPU time; body wall
regresses similarly. The6.5-8.9% projection-only gain therefore does not justify
this model schedule. Neither constant clocks, DRAM traffic nor a causal thermal/
compression diagnosis is established. The path stays a research experiment;
no serving/TTFT/decode/throughput comparison is updated and no2x goal is met.

The first build failed on an ambiguous XLang `Get(0)` overload; explicit integer
indices fix it. Source1 then completed its control but failed at the final
last-active-hidden guard because the last tape span has a nonzero offset.
Its source, library, partial trace, failure status and cleanup remain preserved.
Source2 uses the already copied independent offset0 hidden bank for last-row
selection and the head, preserving every guard. The corrected42-test build and
all qualified model traces bind to source2/library2.

The exact replay is
`test2026/metal/experiments/mistral-layer-bulk-prefill-native.patch`, independently
applied to the exact parent and checked against all ten qualified research paths.
Only that text artifact and this record enter the branch. All prior shader
headers, the three original embedded shader blocks, xModel sources and the
11 preexisting dirty paths remain unchanged. The next proposed memory/lifetime
fusion is explicitly unimplemented in `next-fusion-design1.json`; it has no
measured gain. Mistral/Gemma/ANE and subsequent Qwen requirements remain open.


## Native bulk FFN memory fusion: exact parity and a small prefill gain (2026-11-06)

This follows `fad0fb71f`. The prior correct but slower layer-major experiment
now writes the FP32 SiLU result directly into its prepared FP16 down-projection
input and reuses dead normalization storage for the down result. The isolated
research build is `out/research/mistral7b-local-coder/mistral-layer-bulk-fusion-native1`.
All evidence binds to source1/library1. No xModel operation is changed. This
path remains unmanaged research and has no serving, cache or quality admission.
Frozen vLLM/SGLang results and the SiLU32 serving comparison remain unchanged.

### Native computation, memory and lifecycle

`bulk_silu_half` uses exactly the original FP32 expression
`(gate / (1.0f + exp(-gate))) * up`, then explicit half conversion, compiled
beside the original scalar SiLU with the same main-library math options. It
accepts1..8192 rows and4096/14336 columns. Host bounds prove complete row
extents fit uint32 indices; devices, F32 carrier metadata, input readiness,
shape, strides, output aliasing and parent-span write readiness are validated.
The eager pipeline requires SIMD32 and at least128 threads, dispatched128.
No new xModel dtype or operation represents the private half carrier.

The existing prepared bulk MPS down product consumes that exact half carrier.
The same ordered gate/up products finish reading normalization before the down
product overwrites the same normalization allocation. The tracked command
encoders preserve these resource hazards. The separate8192x14336 F32 SiLU buffer
(469,762,048 bytes) and8192x4096 down allocation (134,217,728 bytes) are removed.
Distinct extra workspace drops from2,446,393,344 to1,842,413,568 bytes: a
603,979,776-byte/576MiB reduction. No change to chunked causal attention, private
per-layer KV ownership or its precision is made. One successful fused prefix
reports1024 packed projections,96 bulk products and32 direct-half SiLU calls;
the control reports2560 packed projections, zero bulk products and zero fused
SiLU calls.

Reset restores complete unique diagnostic root buffers, including the prepared
private half scratch, before any reused span writes. Shared norm/down identity
is deduplicated. This maintains existing parent readiness and abort rules;
a partial view cannot initialize an unreadied parent. The same reset precedes
each mode and is outside the reported native body/GPU timing; complete API wall
includes it. Full preflight validation still precedes reset. The path rejects
managed rows and configured token caches; it must never reset leased serving
rows. An actual managed call is tested for rejection without lease/KV damage.

The repeated process prepared11,188,682,752 Metal device bytes and recorded
11,197,071,360 after ordinary input materialization, stable through measured
calls in both modes. These are device snapshots, not peak/system RAM or a
no-paging guarantee. Both modes retain the same1.842GB research workspace,
which differs from ordinary serving without this prepared plan. Machine state
is uncontrolled and measured times drift upward; within-cycle ratios are used.
No occupancy, bandwidth, thermal or compression cause is inferred.

### Half-bit and full-model correctness

All43 serial native CTests pass. The added actual-native test retains outputs
from the old FP32 SiLU plus the real native GPU half cast and from the fused
kernel, across20 finite/edge cases including row counts1,4,128,129,257,511,512,
8192 and both permitted widths. All350,396,416 paired half words are identical.
All350,207,072 finite reference words also match independent IEEE half rounding
of the retained FP32 SiLU result. A further2559 sampled FP64 checks pass the
specified2e-6 +4e-6*abs(reference) bound, with maximum error/bound0.03012.
Seven corruptions are rejected. Nonfinite native bits are preserved; this is
not a new IEEE nonfinite-class guarantee. Stride/index policy is tested on the
host, while actual retained native cases use tight logical matrices; no new
padded-view/redzone claim is made.

The first model probe compares all64 per-layer K/V hashes, complete hidden
state and four predictions exactly. They also match the frozen original native
2048-token prefix witness, as checked by the repeated audit. Eight corrupted
first-probe reports are rejected. Untimed continuation compares1024 generated
ID observations (128 per row/mode),127 ordinary decode calls per mode and two
alternating active masks. All512 generated-ID pairs,192 full-KV hash pairs and
three complete hidden pairs match. Rows A/C additionally match512 historical
native128-token ID observations for their complete2048-token prompts. B/D are
explicit2048-token prefixes of longer fixtures, with no historical full-prompt
claim. Eight malformed input requests are rejected before reset with state
unchanged after each guard group. Managed refusal preserves all four original
leases and complete KV; all leases remain releasable. Ten corrupted generation
reports are rejected. Cleanup leaves zero native owners and at most4MiB device
bytes. These parity checks do not change strict coding0/8.

### Balanced native prefill timing

After one warm OFF/ON pair, a single process runs three OFF/ON/ON/OFF cycles
(12 measured calls), then two complete KV gates after timing collection.
Hidden/predictions are checked after each call. Large full-KV host witnesses
are kept after all measurements to avoid perturbing subsequent timings. The
independent audit checks order, routes, bounds, state, finite measured times,
stable owners and cleanup; eleven corruptions are rejected.

| Metric | Ordinary control median s | Fused layer-bulk median s | Median within-cycle control/candidate speed ratio |
|---|---:|---:|---:|
| Summed GPU commands | 17.469 | 17.034 | 1.02724x |
| Native body wall | 17.530 | 17.144 | 1.02575x |
| Complete native API wall | 17.761 | 17.310 | 1.02909x |

All three measured GPU cycles favor fusion:1.02707..1.03072x, median1.02724x.
This is about2.65% less GPU time within matched cycles. Native body and complete
API time show similarly small benefits. The preliminary cold pair regressed
slightly (15.243s versus15.079s GPU), so no universal/cold-start claim is made.
The result reverses the prior unfused experiment's direction in this process,
but cross-process absolute times must not be subtracted as a fusion gain.

Inputs are embedded before timing. Native body includes the final four-row
vocabulary prediction; API wall includes snapshots/reset. HTTP queueing,
tokenization/embedding preparation, SSE and generation throughput are absent.
These are not TTFT, decode speed or aggregate serving throughput. No serving
comparison cell changes, and the2x targets remain unmet. Before serving use,
a non-resetting managed execution API needs native epoch/fresh-prefix checks,
sibling-state preservation, ragged/tail requests, prefix-cache compatibility
and scheduling/fairness qualification.

The exact13-path replay is
`test2026/metal/experiments/mistral-layer-bulk-fusion-native.patch`, independently
applied to the exact parent and matched to audited source hashes. Only that
research text and this record enter the feature checkpoint. All original shader
headers, three embedded shader blocks, xModel sources and11 preexisting dirty
paths are preserved. Mistral/Gemma/ANE and subsequent Qwen requirements remain
open; this checkpoint is useful progress, not goal completion.


## Native tracked leased bulk append with cache reuse (2026-11-06)

This follows `15dc5ad10`. The prior fused layer-major prefill now has a separate
native non-resetting entry point, `diagnostic_leased_bulk_run`, in the isolated
research build `out/research/mistral7b-local-coder/mistral-leased-bulk-native1`.
Actual Mistral7B state,128-step continuation, request replacement and cached
prefix append match the existing ordinary `RunTokens` path. Qualified evidence
binds to source4/library4 and44 serial native CTests. This is not yet connected
to the serving scheduler; HTTP TTFT, throughput and the frozen comparison table
remain unchanged. Strict coding stays0/8.

### Lease/history contract and profile bounds

The new C++ API takes hidden/control chunks, physical-row epochs, token chunks
and the control/candidate selector. It requires configured native token-prefix
tracking and a prepared B4/context4096/chunk128 Mistral plan, with at most16
actual chunks per call. Every nonzero epoch must still match its native row.
Every selected row must have valid tracked input history. Active slots and all
three text position axes must continue from that row's native history length;
IDs must lie in the immutable input vocabulary. Inactive controls are canonical
zero, and each row's active tokens form a contiguous prefix of the new group.
Rows can end in different chunks. Empty calls/trailing chunks, context overflow,
stale/noninteger epochs, gapped activity, malformed tensors/tokens and untracked
histories are rejected before any KV/hidden/history-content mutation. Input
hidden/control tensors are privately copied, token IDs are retained, and native
history capacity is reserved before device execution.

The session mutex covers validation, execution and GPU completion. Candidate
graph/head/weight identities and owner generations are checked and held with
ordered graph locks; shared scratch is locked while used. There is no session
or request-row reset in the API. A cached row continues from the state restored
by the existing native prefix matcher, while unrelated rows retain their KV,
last-active hidden and tracked history. Existing MTP capture is explicitly
unsupported and rejected. Hybrid Qwen/recurrent bulk execution is not introduced.

The profile is an upper bound: separate MPS plans and root-buffer views are
prepared for every actual chunk count1..16. Runtime groups use their actual
length; a1-chunk call does not execute16 chunks. Gate/up share one MPS shape,
and down uses a second shape per length, requiring32 bounded plans. Owned
workspace is unchanged at1,842,413,568 bytes. These are prepared MPS descriptors
and operators, not proof that all device compilation occurs eagerly. Per-row
tails still use the existing fixed128-token chunk with inactive lanes.

Layer attention follows each row's validated slots against its existing private
KV. The fused bulk FFN operates on only the actual chunk group. At the end,
last-active hidden is selected across every actual chunk, preserving rows that
did not participate and rows whose last token occurred before the final chunk.
The initialized/completed-bank guard remains unchanged: each tail copy/selection
is submitted and completed before the next chunk updates that bank. Only after
successful device completion are selected tracked histories extended
and final-hidden readiness committed. The ordinary native prediction head runs
for selected rows and returns-1 for inactive rows. Partial execution failures
poison the session and require existing full-reset recovery. No new xModel
operation or tokenizer behavior is added.

### Actual-weight lifecycle evidence

Each mode runs the same sequence, with complete per-row32-layer K/V hashes,
full4096-word last-active-hidden hashes and observed native tracked counts:

| Phase | New tokens A/B/C/D | Actual chunks | Checked behavior |
|---|---|---:|---|
| Fresh ragged prefixes |259 /131 /257 /1 |3 | Different row endings, correct final hidden/history |
| Append A/C |17 /0 /129 /0 |2 | B/D unchanged |
| Ordinary tracked decode |128 each |128 base calls |1024 generated-ID observations across both modes, all512 pairs equal |
| A return |259 cached |0 inference | New epoch, complete original A prefix restored, B/C/D unchanged |
| Changed A tail |3 /0 /0 /0 |1 | Cached prefix plus changed tokens, siblings unchanged |
| B replacement plus append |1 /129 /0 /2 |2 | New B epoch, C unchanged |
| Full-bound append |2048 each |16 | Existing tracked state plus maximum group length |

The A cache key contains the original259-token prefix and a changed three-token
suffix. Restore returns259 and exactly reproduces the original A KV/hidden and
prediction. Stale A/B epochs fail after both rows have been released and
reclaimed. All final current leases remain releasable. The full-bound append
finishes at native counts2311/2177/2562/2179. An additional C context-overflow
request is rejected before touching any row. An ordinary untracked C update
then makes its history ineligible; a subsequent bulk append to C is rejected
with complete state unchanged.

Thirteen malformed requests per mode (26 total) are rejected. The independent
auditor checks all10 lifecycle phases and compares13 complete four-row state
witnesses across modes:3328 row-KV tensor hash pairs and52 whole-row hidden hash
pairs. It separately checks all inactive/survivor preservation, guard before/
after equality, native tracked counts, generated IDs, cache reuse, epochs,
actual group sizes and routes. Fourteen deliberately corrupted reports are
rejected. Clean close leaves zero native matrix/weight/pipeline owners and at
most4MiB Metal device bytes. Actual model group lengths1,2,3,16 are covered;
the prepared4..15 variants are not all independently model-qualified here.

### Preserved failures and remaining admission work

The first build attached the package decoder source to the standalone Metal
kernel target, which lacks XLang SDK headers. The source was moved to the
existing `garnet` package target; the failed CMake text/log/status are retained.
Source1 then hit the prior27-plan table bound during preparation. Source2
raises the bound to exactly32 without changing workspace/context limits, passes
44 tests and runs the ordinary ragged prefix, but its observer incorrectly
called an internal scheduler method through an absent package binding.
Source3 exposes the existing native count method as a diagnostic binding and
completes all ordinary lifecycle phases, but candidate tail selection correctly
rejects updating the bank again while its previous update is still pending.
Source4 preserves the existing completed-bank guard and completes each actual
chunk tail copy/selection before starting the next. It retains the repaired
harness, all44 passing tests and complete model evidence. All failed model
outputs, stderr, source versions and libraries are preserved.

All GPU/model runs are serial. No reference engine was rerun. The lifecycle
trace time fields are diagnostic observations surrounded by large full-state
host readbacks; they are not a performance benchmark or TTFT. The separate
balanced timing study follows below. Before serving
admission, the scheduler needs bounded group selection, cancellation and decode
fairness, plus HTTP overlap/row-reuse/token parity and measured memory checks.
Cached-policy and cold-reference comparisons must remain distinct. No2x target,
new coding-quality result, ANE/Gemma or later Qwen requirement is complete.

The exact16-path research replay is
`test2026/metal/experiments/mistral-leased-bulk-native.patch`, applied to the exact
parent and independently matched to all qualified source hashes. Only the
research replay and this record enter the local feature checkpoint. All prior
shader headers/embedded blocks, xModel files and11 preexisting dirty paths are
preserved; serving behavior is unchanged.


### Balanced leased-prefill timing after lifecycle qualification

A separate process uses the qualified source4/library4, one warm OFF/ON pair,
three measured OFF/ON/ON/OFF cycles and two full-KV postgates after timing
collection. Every call claims four fresh rows under unique native epochs,
appends exactly2048 tokens per row through the tracked API, checks native
history length, complete hidden/row-bank hashes and frozen native prediction
IDs, then releases the original leases. Full32-layer K/V hashes occur only
in the two postgates and match the unchanged original native2048-token prefix
witness. All64 leases are distinct and released, with stable prepared native
owners and clean shutdown. Fifteen corrupted timing/lifecycle reports fail
independent audit. The repeated process never captures prefixes into its cache;
this is fresh tracked-prefill timing, not a cache-hit speed claim.

Claims and releases are measured separately from API/body/GPU time. Their
interval sum below includes their own calls and prefill, excluding the gaps
used for diagnostic readbacks. Embeddings/controls/token lists are prepared
before these calls. It is not total user latency, HTTP TTFT or serving
throughput. Both modes retain the same prepared1.842GB research workspace.
System conditions remain uncontrolled; no constant-clock or peak-RAM claim
is made.

| Metric | Ordinary tracked control median s | Leased bulk median s | Median within-cycle control/candidate ratio |
|---|---:|---:|---:|
| Summed GPU commands | 17.864 | 17.384 | 1.02777x |
| Native body wall | 17.923 | 17.506 | 1.02424x |
| Complete native prefill API | 17.927 | 17.509 | 1.02451x |
| Sum of claim/API/release intervals | 17.986 | 17.649 | 1.02325x |

Measured GPU cycle ratios range1.02371..1.03219x; median
1.02777x. This is a native prefill
observation with native ownership/tracking, not the2x/half-TTFT target or an
update to any vLLM/SGLang comparison. The earlier unleased1.02724x measurement
is a different experiment and must not be substituted for this result.

The lifecycle observer's first guard-label list remains shared with the later
capacity/history labels, so the retained final JSON lists all13 there. The
auditor requires that exact order and the separately retained before/after
witnesses for the initial11, capacity and untracked guard groups; no rejection
is counted twice. A future harness with a copied label list is retained but is
not substituted for the executed harness. The auditor's initial assumption
that every ordinary request has a positive packed count was also corrected:
native `TryEncodePackedAffine` rejects counts below128 active tokens, where
indexed affine is expected. The final auditor derives exact packed counts
from the frozen active-token geometry. Raw failed auditors and reasons remain
alongside the immutable model trace.


## Native scheduled bulk prefill: lifecycle/HTTP correct, serving gain unqualified (2026-11-06)

This follows `eb0f64a4d`. The isolated build
`out/research/mistral7b-local-coder/mistral-scheduled-bulk-native1` connects the
native leased bulk path to `MetalContinuousScheduler` and the portable C++
scheduling policy. Source1/library1 passes45 native CTests, two actual-weight
scheduler lifecycle studies and a serial HTTP OFF/ON/ON/OFF campaign. All
outputs match retained ordinary native controls, but cold serving regresses
and warm aggregate throughput remains nearly flat. The route stays
experimental/default-off; no production serving library is replaced.

### Native scheduling, execution and disabled-retention mode

`set_bulk_prefill(16)` configures a separate8192-token group bound while
ordinary chunk work retains its512-token physical budget. The scheduler may
select16 chunks of128 tokens only when all four physical rows have at least
2048 remaining text prompt tokens and no decoder is ready. Vision, position
deltas, short rows, exact cache-head hits and mixed decode/prefill retain the
ordinary bounded path. A group does not hide an earlier prompt end. The
prepared upper bound is distinct from a single graph's128-token input shape.
The previous native API supports other actual group counts, but this serving
profile selects only16 and uses ordinary fallback otherwise.

`next_json` returns native row epochs and a unique positive bulk ticket.
`run_bulk_prefill(hidden_chunks,ticket)` accepts only that current unexecuted
ticket. The native allocator derives token IDs, slots, all three position axes
and active controls from its immutable request records; the bridge supplies
embeddings and transports events. The native runtime revalidates placement,
logical progress, token tracking and profile bounds before executing the
already-qualified causal attention/bulk FFN schedule. A cancelled old ticket
cannot apply embeddings to a replacement row with a new plan. This adds no
xModel operation and uses no Rust inference implementation.

The normal completion APIs reject acknowledging a selected group before
execution or with the wrong token count. Cancelling before execution releases
the target lease and abandons the untouched survivors' scheduled metadata,
putting them back in the queue without clearing their state. Cancelling after
execution clears only the target row; survivor progress remains pending for
normal acknowledgments. Tickets and epochs prevent stale execution/completion
from mutating a replacement. Native mutexes serialize execution and GPU
completion. An executing group remains non-preemptible; mid-group cancellation
latency and arrivals during that interval are not newly qualified here.

`configure_token_tracking(identity)` validates the same native vocabulary,
position and owner contract but creates a zero-entry, zero-byte prefix index.
Histories are tracked for progress checks without retaining snapshots. Exact
(0,0) is accepted only in the research feature; other invalid entry/byte
combinations retain their guards. Stats expose `retention_enabled:false`.
Both HTTP modes disable prefix retention: ordinary control uses its existing
untracked cold path; bulk uses tracking-only admission. All client cached-token
counts and retained/copy/hit counters stay zero. Persistent cached serving
through the new scheduler still needs its own admission; the previous direct
native cache-restore/append gate is preserved.

### Correctness, ownership and fairness evidence

The new portable test covers the8192 total group budget,128-token physical
chunks, short tails, ready-decoder fallback, exact cache heads, vision/delta
fallback, abandoned plans, cancellation and row epochs. Its first fixture used
a4096-token prompt inside a4096-token context; reservation correctly refused
it because at least one output token is required. The corrected4095-token
fixture passes without changing capacity guards. The failed test text/log and
repair rationale are retained. All45 final serial CTests pass.

Two actual-weight studies use the same native engine, first ordinary prepared
families and then all14 compact owner families. Each schedules four2048-token
prompts, rejects an in-flight next tick and malformed hidden count, and rejects
premature completions. A is cancelled before execution, E reuses its physical
row with a new epoch/ticket, and the old ticket is rejected with full KV state
unchanged. The real native group exactly matches the original complete
32-layer K/V state, complete output hidden, last-active bank and four predicted
IDs. Duplicate execution and incorrect progress acknowledgment are rejected.
E is then cancelled after execution, with all B/C/D KV and hidden preserved.
F reuses row0 and refills in16 ordinary128-token ticks while B/C/D decode first.
The remaining decode drain contains no bulk selection while decoders are ready.
Each study completes143 regular ticks and512 generated IDs, exactly matching
the retained ordinary native128-token outputs. Seventeen report corruptions
per study are rejected. No snapshot is retained and all scheduler requests,
leases and native owners drain. These1024 new model ID observations are parity checks. The last observed
strict coding result remains0/8; no new coding-quality evaluation was run.

Four fresh HTTP engines run serially OFF/ON/ON/OFF, each with cold/warm1/warm2
rounds of the unchanged four ~2048-token requests and128 output tokens. All48
complete streams/6144 IDs and full text match the frozen native controls.
The transport audit reads raw events directly and separately checks aggregate
bulk execution against its pending physical plan; it does not synthesize
ordinary kernel events for bulk work. It retains existing per-frame checks
for ordinary attention, affine/packed/indexed routes, split/profile32/value
score policies, row epochs, progress and cleanup. It also validates group
bounds/tickets, disabled snapshot retention and content/usage times against
raw SSE. Twenty-three corrupted reports fail audit. All four workers exit
cleanly with no forced kill, zero native owners and at most4MiB device bytes.
Prepared device allocations are9,954,541,568 bytes for ordinary mode and
11,796,955,136 bytes for bulk, an exact1,842,413,568-byte workspace increase.
Each close reports1,507,328 bytes. These sampled device values are not peak
process/system RAM or a no-paging guarantee.

### Latest experimental serving observation

The table uses two cold native waves per mode, versus unchanged15-wave
reference medians. Native KV/attention remain FP32; references use FP16.
System conditions are uncontrolled. These are full serving measurements,
including prefill in aggregate throughput, unlike the earlier native-only API
result. TTFT is first content minus submission; per-request decode is127 divided
by the first-to-last usage interval; total throughput is512 divided by the
four-request completion window.

| Engine | Cold TTFT s | Decode tok/s per request | Total tok/s including prefill |
|---|---:|---:|---:|
| Ordinary native chunks | 17.156 | 12.891 | 18.994 |
| Experimental native bulk | 18.145 | 12.145 | 17.821 |
| Frozen vLLM-Metal | 11.570 | 10.484 | 21.179 |
| Frozen SGLang-MLX | 10.884 | 6.307 | 16.337 |

Cold candidate/control ratios are1.05764 for TTFT,
0.94216 for decode speed and
0.93825 for total throughput. The cold
control TTFT range is15.642..18.670s, illustrating the run-to-run variation.
Across four warm rounds per mode, TTFT is18.216s ordinary versus
18.012s bulk, decode is12.412 versus
12.077 tok/s, and aggregate is
17.969 versus17.872 tok/s.

The separate native API gain does not establish an HTTP gain. No individual
thermal, memory, embedding or scheduling cause is isolated by these traces.
Bulk remains off by default; no2x throughput/decode or half-TTFT target is met,
and no new single-request or coding-quality qualification is made. All vLLM/
SGLang binaries and raw results are unchanged; no comparator inference is rerun.

The next unimplemented design examines attention-side norm/Q/K/V/O projection
amortization while retaining per-chunk causal KV commits. Its first required
gate is actual-native K/V-width1024 output-bit/FP64 comparison before any model
schedule change. Extending only metadata or rerunning this unchanged HTTP path
is not a performance fix. Whole-model/continuation/lease/cache/fairness, bounded
memory and serial HTTP gates remain required before adoption. No gain is
assigned to that design. ANE/Gemma and subsequent Qwen requirements remain open.

The exact24-path research replay is
`test2026/metal/experiments/mistral-scheduled-bulk-native.patch`, applied to the
exact parent and matched to every audited source hash. Only that research text
and this record enter the local feature checkpoint. All original shader
headers/embedded blocks, xModel sources and11 preexisting dirty paths are
preserved. The latest stable serving artifact stays unchanged.


## Native Mistral bulk K/V projection: exact state, measured full-prefill scope

Research parent `f024d483f45374901a370ba435cd780dfaff2476`; artifacts `out/research/mistral7b-local-coder/mistral-kv-bulk-prefill-native1` and operator gate `mistral-bulk-kv-projection-native1`. This is isolated native Metal/MPS research. Stable serving selection and xModel operations are unchanged. The runtime route remains disabled by default; legacy unmanaged diagnostic explicitly refuses this selector.

The operator gate keeps the existing indexed 1024-output FP16-operand/FP32-accumulator control unchanged. Two finite/edge sets at 512/2048/8192 actual rows match every paired FP32 word. Clean serial 8192-row operator medians show 2.37–2.40× speed ratios. The first numerical collection overlapped a source audit and its timings are explicitly excluded; the second serial collection alone supports the operator timing. These are component observations, not model or serving speedups.

The leased model path precomputes only K/V projections into private buffers after one actual-group RMS normalization. Immutable graph matching validates input mask/norm, named weights, projection shapes and consumers. Original per-chunk Q, RoPE, causal attention, KV commits, O projection and residual order remain. Rebinding/resident-operation masks restore on success and exceptions. Ragged/masked groups retain original K/V projections. Native lease/history validation completes before mutation. Selector changes refuse live row owners. Prepared MPS products use 48 bounded shapes; actual group counts 1–16 use views without maximum padding. Two F32 K/V roots add 67,108,864 bytes, for 1,909,522,432 total prepared workspace bytes; this is not a peak host-RAM claim.

All 46 native tests pass. Scheduled full-prefix state matches 64 original KV hashes, complete hidden and row bank, plus 512 original generated IDs; before/after cancellation, stale-ticket rejection and decode-first refill pass. Leased ragged/cache-return/dense append tests retain 3,584 paired row-KV hashes and 56 row-hidden hashes, 576 paired generated IDs, 26 malformed-request rejections and cache restoration at 259 tokens. Actual 1/4-chunk K/V studies add 192 full-KV comparisons, 6 hidden comparisons and 64 paired continuation IDs. Complete-state gates and corruption tests are retained in `scheduled-audit1.json`, `leased-audit1.json` and `small-audit1.json`. An inherited auditor's mutable-list expectation was corrected to match 11 initial rejection labels plus the two separately recorded later guards; raw native evidence did not change.

Full-model prefill timing uses the same library/weights/cache precision: B4, 2048 input tokens per row, 128-token physical chunks, FP32 KV/attention, tracking without retention. Three warm calls precede three measured ABC-CBA cycles (18 measured calls, 6 per mode); three full-KV postgates follow timing. Large KV hash snapshots and source hashing are excluded from measured collection. Claims/releases are outside API/body/GPU timing and retained separately. Uncontrolled system conditions limit generalization.

| Native mode | GPU seconds | Body seconds | API seconds |
|---|---:|---:|---:|
| Ordinary chunks | 18.1033 | 18.1619 | 18.1659 |
| Bulk FFN | 17.5454 | 17.6594 | 17.6634 |
| Bulk FFN + K/V | 16.8979 | 17.0407 | 17.0446 |

K/V candidate versus ordinary chunks: median cycle GPU ratio `1.072155`, API ratio `1.067582`. Ratios greater than 1 favor the K/V candidate. Full per-cycle/pair values, owner stability and 192 original full-KV hash comparisons are retained in `repeated-audit1.json`.

K/V candidate versus bulk FFN: median cycle GPU ratio `1.037735`, API ratio `1.036856`. Ratios greater than 1 favor the K/V candidate. Full per-cycle/pair values, owner stability and 192 original full-KV hash comparisons are retained in `repeated-audit1.json`.

No new HTTP TTFT/decode/aggregate-throughput, ANE performance, multimodal or coding-quality measurement is admitted. Frozen vLLM-Metal/SGLang artifacts were not rerun or modified. Their FP16 KV/attention remains distinct from native FP32. The previous 0/8 strict coding result is unchanged historical evidence, not a new evaluation. Neither the 2× model/serving targets nor the broader local-coding goal is complete.


## Native bulk K/V: HTTP correct, serving gain unqualified

This follows `5023aa05c`. Evidence is in `out/research/mistral7b-local-coder/mistral-kv-scheduled-http1`. It uses the immutable qualified native library from `mistral-kv-bulk-prefill-native1` (SHA256 `d127ee1b473face371969534d1b4f382d55038bae8859cd044edab1696b0ec65`). No native kernel, xModel operation, model artifact or comparator implementation is changed for this campaign.

A separately frozen HTTP worker validates an explicit boolean `native_bulk_kv` profile field. Grouped mode selects K/V before any request row is owned. Each actual 16-chunk group must report 64 K/V bulk calls, 160 total bulk products, 32 fused SiLU-half calls and 1,909,522,432 prepared workspace bytes. The ordinary control uses the same library without grouped preparation. Both modes disable prefix snapshot retention; grouped mode tracks native histories for progress validation. Score32/value32 stay enabled, paired-prefill/SiLU32 stay disabled. The inherited profile.scope sentence mentions an older SiLU experiment; the saved scope note identifies this metadata label and the authoritative actual profile fields without changing frozen evidence.

Four fresh native engines run OFF/ON/ON/OFF, strictly serially. Each serves cold/warm1/warm2 rounds of four concurrent requests, approximately 2,048 input and exactly 128 generated tokens per request, C4096/T128 with a 512-row physical chunk budget. The grouped path retains the existing 8,192-row up-to group bound. No inference, heavy source hashing or builds overlap these timed engines. Start/finish records, graceful worker exit and zero final native owners are audited; no forced process kill is accepted.

All 48 requests preserve complete input/output IDs, output text and finish reasons against the retained native controls: 6,144 output-token observations. Raw SSE content and token-usage timestamps independently support TTFT and decode arithmetic. The audit also checks native tickets/epochs, compact tails, decode-first scheduling, score/value routes, exact cache-disabled policy and grouped dispatch counts. It rejects 27 deliberate corruptions. This HTTP audit does not independently read GPU KV contents; the full native state, cached-return, cancellation and continuation gates remain separately bound to this exact library. It adds no coding-quality evaluation.

Cold values below are medians of two fresh-wave medians per native mode. Each frozen Apple comparator has 15 retained waves; no comparator inference is rerun. Generation speed is per request under B4. Aggregate throughput includes prefill and the full completion window. Native FP32 cache/attention and comparator FP16 remain distinct; system conditions are uncontrolled, so absolute cross-campaign values do not establish a causal or universal gain.

| Engine | Cold TTFT s, lower better | Decode tokens/s/request, higher better | Aggregate output tokens/s, higher better |
|---|---:|---:|---:|
| Garnet ordinary control | 17.411 | 12.716 | 18.700 |
| Garnet bulk FFN + K/V | 17.796 | 12.014 | 17.973 |
| Frozen vLLM-Metal | 11.570 | 10.484 | 21.179 |
| Frozen SGLang-MLX | 10.884 | 6.307 | 16.337 |

Cold candidate/control ratios: TTFT `1.022127` (lower favors candidate), decode `0.944758` and aggregate `0.961148` (higher favors candidate). Full native wave ranges and cold/warm round metrics are retained in `audit_http1.json`; warmed values pool four rounds per mode.

Warmed candidate/control ratios: TTFT `0.953943` (lower favors candidate), decode `0.970972` and aggregate `1.015711` (higher favors candidate). Full native wave ranges and cold/warm round metrics are retained in `audit_http1.json`; warmed values pool four rounds per mode.

Cold TTFT is 2.2% higher and aggregate throughput 3.9% lower for the candidate in this series. Warmed TTFT is 4.6% lower and aggregate throughput 1.6% higher, with overlapping wave ranges. The serving collection does not establish a robust win. The next diagnostic should distinguish ordinary unprepared execution, ordinary prepared-but-unused execution, and actual grouped execution with stage timing. Preparation also retires graph storage, so its effect must not be labeled pure RAM pressure.

The new native prefill observation does not prove 2× single-request decode, 2× aggregate throughput, half TTFT or coding qualification. The older strict coding result remains historical 0/8 and was not reevaluated here. Persistent-cache HTTP admission, mid-group cancellation latency, full ANE/Gemma and later Qwen requirements remain open. Keep the production default unchanged; preserve this candidate and evidence for the next measured change. The separate primary-source notes flag that upstream INT8 activation staging changes precision and needs its own quality gate rather than being treated as an exact optimization.


## Preparation and grouped-submission attribution: separate eight-output diagnostic

This follows `07f0b8d0e`. Evidence is `out/research/mistral7b-local-coder/mistral-kv-preparation-stages1`. The native library remains the exact qualified K/V library `d127ee1b...`; no C++/Metal kernel, model, precision, xModel operation or comparator changes occur. A separate frozen worker adds prepared-but-unused mode and one grouped stage-trace event. Regular native stage tracing is reused. The new grouped event binds its native ticket, actual 8192 rows, summed GPU time and native body wall directly to the existing bulk execution record; it does not reuse stale per-graph prepare/encode/wait counters.

Six fresh engines run ordinary/prepared/grouped/grouped/prepared/ordinary, strictly serially. Each has one traced B4/~2K-input round with eight output tokens per request. All 192 output IDs match prefixes of retained 128-token native controls, and text matches across all six modes. Native scheduling/dispatch/epoch/close policy checks pass, all final native owners are zero, no forced kill is accepted, and 13 deliberate corruptions are rejected. This is not the 128-output workload and must not replace the saved vLLM/SGLang comparison table.

Prepared-but-unused mode builds the same 16-chunk bulk workspace but keeps ordinary scheduling, no tracking-only index and no grouped execution. Preparation also retires graph storage and updates hidden-pool preparation, so this control does not isolate RAM pressure alone. Ready Metal allocations are 9,954,541,568 bytes for ordinary and 11,864,064,000 for both prepared/grouped, an exact 1,909,522,432-byte difference. These are snapshots, not peak RAM. Ready compressed bytes vary from 0 to 2,896,478,208 across these processes; no page-fault or causal memory claim is made.

The worker window starts at first prefill scheduling and ends when all four first tokens have been delivered. Exclusive CPU wall categories are clipped to that interval and checked for overlap. GPU command spans are nested inside native API time and are never added as a separate wall category. Unassigned gaps stay unlabeled. The trace changes event overhead, conditions are uncontrolled and there are only two fresh observations per mode.

| Diagnostic mode | Worker window s | Native API wall s | Nested GPU spans s | Embedding s | Scheduling s |
|---|---:|---:|---:|---:|---:|
| Ordinary, unprepared | 17.2057 | 16.8863 | 16.7417 | 0.1297 | 0.1127 |
| Ordinary, prepared but unused | 17.7156 | 17.3276 | 17.1276 | 0.1290 | 0.1824 |
| Grouped FFN + K/V | 17.2984 | 16.8064 | 16.2572 | 0.1316 | 0.3468 |

The grouped GPU span is lower than the prepared control, but its API/GPU gap is larger. The first decode inside the worker window takes about 0.245s in grouped mode versus 0.062s in ordinary/prepared; this is an observed transient, not a causal hardware explanation or steady decode benchmark. Grouped first scheduling is about 0.347s. Embedding stays about 0.13s, so this diagnostic does not identify embedding as the main missing gain. Prepared and ordinary wave variation overlaps. The aggregate grouped worker window is nearly flat against ordinary unprepared, consistent with retaining the negative/mixed serving observation.

The next native experiment should prepare the existing immutable per-chunk control banks for all 32 layers before opening a command, then encode the complete group in one submission if resource/readiness/alias validation permits. It must preserve per-chunk causal cache writes, exact state and output bits, graph-binding restoration, poison/abort recovery, and the completed-bank barrier before last-active-hidden updates. This is an unimplemented design, not a speed claim. A default-off native selector, up-to actual group-count tests, native lease/cache/cancellation gates and balanced timings must precede any HTTP promotion. No extra GPU workspace or xModel operation is proposed. The 2x/quality/ANE/Gemma and later Qwen requirements remain incomplete.


## Native grouped prefill submission screen: exact state, no meaningful speed gain

This follows `77c9ffc5c`. Evidence is `out/research/mistral7b-local-coder/mistral-bulk-submission-native1`. A clean archive plus the qualified K/V replay adds a native C++ research selector, `diagnostic_bulk_single_batch_set_enabled`. Fresh session objects default to disabled. Closed/unprepared/poisoned sessions and changes with owned request rows are refused. The original unmanaged diagnostic refuses this configured policy instead of implying it exercised the new path. The selector is native runtime configuration; there is no xModel operation or Rust inference change.

When selected for leased grouped work, the runtime prepares all 32 immutable layer control banks before opening the session command. Every bank was already separately allocated within the existing profile. It then encodes all 32 layers into one command instead of submitting/waiting once per layer. Q/K/V math, MPS products, RoPE, per-chunk causal attention/cache-write order, temporary binding/mask restoration, native lease/history validation and poison/abort handling are preserved. The single layer batch completes before the established per-chunk last-active-hidden bank updates. Therefore a 16-chunk call still has 16 completed tail batches and its head prediction; this is not a claim of one batch for the entire API.

Prepared GPU workspace stays 1,909,522,432 bytes, with zero additional GPU allocation. Actual groups remain up to 16 chunks without maximum padding. Ragged/masked groups keep the original K/V fallback while using the selected FFN submission schedule. Serialized per-operation profiling is explicitly rejected before state mutation because it would split the selected layer batch. Stats report configured/effective policy, actual outer layer batches, partial submissions, tail batches and separate CPU control-preparation/encoding/submit-wait intervals. All exercised candidate groups have one layer batch and zero partial submissions.

All 46 native tests pass. The existing real-GPU view test now also checks 32-layer pending ping-pong chains at 1/4/16 chunks, complete output regions, rejection of pending host reads, aborted-work parent invalidation, refusal of partial-view repair and full-parent recovery. No real driver failure or out-of-memory event is forced. Complete native Mistral gates pass: 192 small-group full-KV comparisons and 64 continuation pairs; scheduled 64 original KV hashes plus hidden/bank identity and 512 frozen continuation IDs; leased/cache-return/ragged/dense append 3,584 paired row-KV hashes, 56 paired row-hidden hashes, 576 continuation pairs and 27 malformed-request rejections. The new serialized-profile rejection preserves the complete preflight state. Independent auditors reject 12/19/16 small/scheduled/leased corruptions, including wrong batch/partial-submission claims. Device-failure recovery and reuse of a closed/reinitialized research session are not newly admitted.

The same-library native timing uses identical B4/2048-token input rows, FP32 KV/attention, current K/V bulk path, prepared buffers and zero retention. Two warm calls precede three measured ABBA cycles (six measured calls per policy); two complete original-KV postgates follow timing. Large snapshots, builds and source hashing do not overlap timed collection. Claims/releases remain outside API/body/GPU timing and are separately retained. Native owner counts stay stable and cleanup returns owners to zero. Both final states match all 128 original KV hash comparisons; another 17 timing/route corruptions are rejected.

| Submission policy | GPU spans s | Native body s | Native API s |
|---|---:|---:|---:|
| 32 layer batches | 17.0523 | 17.1765 | 17.1804 |
| One layer batch | 17.0797 | 17.1293 | 17.1332 |

Median cycle API speed ratio is `1.002303`, with cycle ratios `[0.9973537437781913, 1.0027556111512135, 1.0023033470405427]` and `5/6` individual pair wins. GPU median cycle ratio is `0.998717`. Ratios above 1 favor the one-batch candidate. The approximately 0.23% API observation is not a meaningful or universal gain; one cycle regresses and GPU time is slightly worse. Control-bank preparation is only a few milliseconds, and reduced submission count does not resolve the model's dominant GPU work.


Keep this policy disabled and retain the negative evidence. No new HTTP campaign is justified by this result, so the previous 128-output TTFT/decode/aggregate table is unchanged. Frozen vLLM/SGLang inference is not rerun and protected artifacts remain identical. The historical strict coding 0/8 result is not reevaluated. The 2x single-request decode/aggregate target, half TTFT, quality, ANE/Gemma and later Qwen requirements remain incomplete.

The next design returns to full-group Q/O projection amortization with existing norm/residual roots, after explicit immutable graph and read-before-reuse validation. Q/O share one 4096x4096 MPS shape, potentially taking the prepared descriptor bound from 48 to 64 across 16 actual chunk counts without another GPU buffer. This is an unimplemented design, not a measured improvement. Existing actual-native projection word/FP64 results motivate a fresh alias/writer gate, followed by full-model/cached-return/scheduler and balanced timing checks before any serving promotion. Preserve the current reference path and do not silently select the unhelpful one-batch policy in that new experiment.


## Native full-group Q/O projection with bounded buffer reuse

This follows `6e75f943a`. Evidence is `out/research/mistral7b-local-coder/mistral-qo-bulk-native1`. The final candidate is immutable `source2/library2`, distinct from retained initial `source1/library1`. A clean archive plus the qualified K/V replay adds full-group Q/O projection without changing xModel operations, original shader headers, embedded Metal shader text, FP16 projection operands/FP32 accumulation, FP32 KV/attention, model or comparator artifacts. The previous single-batch experiment is deliberately not stacked into this candidate. Fresh sessions default to Q/O disabled; selectors refuse unprepared, poisoned or row-owned changes, and Q/O requires K/V enabled. The unmanaged legacy diagnostic refuses this selector.

The compiler retains logical alias identities rather than inferring semantics from reused physical scratch. A strict immutable descriptor validates Q/O weights, shared normalized input, ungated 32/8-head RoPE, causal attention and its two original cache commits before O, the original residual and identity layouts. Dense actual groups up to 16 chunks use bounded MPS products; ragged/masked groups fall back. Full-group normalization and Q precede per-chunk original RoPE, attention and KV commits. Each context replaces consumed norm data; O replaces consumed Q data. Post-O Add writes a distinct norm root and FFN buffer roles rotate. Prepared workspace remains 1,909,522,432 bytes: zero additional GPU buffers. Prepared descriptor bound is 64 across actual group counts, without maximum padding. The final candidate removes exactly five validated norm/layout/mask producers per layer/chunk, only after proving their outputs have no other live prefix consumers. Independent position/control producers remain. Resident masks and temporary graph bindings restore on success and exceptions.

Two development failures are retained. A map-type assignment caused the first build failure and was repaired with iterator insertion. The first alias gate rejected an unsupported in-place Add. Its safety guard was preserved; buffer-role rotation repaired the schedule. Source2 is separately rebuilt from a clean archive and repeats all native/model gates. No false passing result replaces these failed logs.

All 47 final native tests pass. The actual-native guarded alias/projection gate covers finite/edge data and 1/4/16 chunks, checks 352,321,536 paired stage words, 98,304 guard words and 880 independent FP64 samples within the unchanged numerical bound, and rejects 12 corruptions. An intentional missing context writer is detected. Pending host reads, partial-abort invalidation, refusal of partial-view repair and full-parent recovery are exercised; no real driver or out-of-memory failure is induced. Source1 and source2 stage payload hashes match.

Full-model small-group gates preserve 192 full-KV comparisons, six hidden comparisons and 64 continuation pairs. The native scheduler gate preserves 64 original full-KV hashes, full hidden/bank and 512 original generation IDs across cancellation, stale/duplicate tickets, epochs and decode-first refill. Leased ragged/cache-return/dense append gates preserve 3,584 paired row-KV hashes, 56 row-hidden hashes and 576 continuation pairs with malformed-input rejection. At full 16-chunk density the native stats require zero packed calls, 224 bulk products (64 Q/O and 64 K/V), 32 fused SiLU-half calls and 2,560 skipped input operations. Ragged groups require zero effective Q/O products.

An additional untimed return study explicitly caches a 128-token dense prefix produced by Q/O, releases A while B/C/D decode eight steps, restores A under a new epoch, changes A's appended tail, appends 128 tokens to all rows and decodes 16 more steps. Same-library ordinary controls match 1,280 full-row KV hashes, 20 row-hidden hashes, 88 continuation IDs and nine prefix predictions. Survivors stay unchanged on A's restore; final histories are [272,280,280,280]. All final native owners return to zero, and 13 corrupted reports are rejected. This is a direct native cache-return qualification, not persistent-cache HTTP or coding-quality admission.

Native timings use the same library/model/precision and identical B4/2048-token inputs. All three modes prepare the same workspace, retain no prefixes, and track native lease histories. Three warm calls precede three ABC-CBA measured cycles (six calls per mode); three full-KV postgates follow timing. Large state readbacks, builds, hashing and other inference do not overlap collection. Claims/releases are outside API/body/GPU timing and retained separately. Every timed call retains exact hidden/bank/prediction witnesses and stable native owners; postgates retain all 192 original full-KV comparisons. Seventeen timing/route corruptions are rejected. These GPU spans are nested within API time, never an additive wall category.

| Native mode | GPU spans s | Native body s | Native API s |
|---|---:|---:|---:|
| Ordinary chunked | 18.2083 | 18.2702 | 18.2741 |
| Bulk FFN + K/V | 16.9792 | 17.0883 | 17.0922 |
| Bulk FFN + Q/K/V/O; dead input removed | 16.5442 | 16.6581 | 16.6620 |

Q/O candidate versus ordinary: median cycle API speed ratio `1.097232`, GPU ratio `1.101066`, with `6/6` API pair wins. Ratios above 1 favor the candidate. Full cycle/range values remain in `repeated-audit2.json`.

Q/O candidate versus K/V bulk: median cycle API speed ratio `1.025785`, GPU ratio `1.026062`, with `6/6` API pair wins. Ratios above 1 favor the candidate. Full cycle/range values remain in `repeated-audit2.json`.

The roughly 1.10x native prefill result justifies a separate serial serving experiment. It is not HTTP TTFT, steady decode, aggregate throughput or a universal gain, and it must not be inserted into the frozen serving comparison table. Production selection remains unchanged. Native FP32 cache/attention differs from frozen Apple references' FP16; conditions are uncontrolled. vLLM-Metal/SGLang are not rerun. The historical strict coding 0/8 result was not reevaluated. The 2x single-request decode/aggregate target, half TTFT, coding quality, persistent-cache serving, ANE/Gemma and subsequent Qwen qualifications remain incomplete.

The complete 26-path research replay is `test2026/metal/experiments/mistral-qo-bulk-native.patch`, bound to the exact parent and every owned source hash. Only this replay and research documentation enter the feature checkpoint. All 11 preexisting dirty paths remain byte-identical; the stable serving library is unchanged.


## Full-group Q/O: separate serial serving comparison

This follows `74a3509d3`. Evidence is `out/research/mistral7b-local-coder/mistral-qo-scheduled-http1`. No native kernel, model artifact, xModel operation or comparator changes occur during this campaign. It uses the previously qualified, immutable `mistral-qo-bulk-native1/source2/library2`, with all 47 CTests and native operator/model/cache-return gates separately bound. A frozen HTTP worker adds an explicit boolean Q/O profile selector requiring K/V grouping; both are selected before any native row lease is owned. Each effective full 16-chunk Q/O group must report zero packed projections, 224 bulk products, 64 Q/O calls, 64 K/V calls, 2,560 removed input producers and unchanged 1,909,522,432 prepared workspace bytes. K/V-only requires 1,024 packed projections, 160 bulk products and zero Q/O/removed inputs. Ordinary control is unprepared for bulk execution. This comparison therefore measures complete serving choices, not an isolated same-workspace microbenchmark.

Six fresh engines run ordinary/KV/QO/QO/KV/ordinary, strictly serially, with cold/warm1/warm2 rounds each. All modes use B4, approximately 2,048 input tokens and exactly 128 output tokens per request, C4096/T128. Score32/value32 stay enabled, paired/SiLU32 disabled; stage tracing and prefix retention are off. Grouped modes track native histories, with the existing bounded non-preemptible 16-chunk scheduling policy. The first engine and every subsequent one close gracefully before the next starts. No builds, heavy source hashing or other inference overlap timed collection. Ready/close resource observations are snapshots, not measured peak memory.

All 72 complete requests preserve their full inputs, generated IDs, text and finish reason against retained native controls: 9,216 output-token observations. Actual native dispatch/ticket/epoch/compact-tail/score/value/tracking policies and zero final owners pass. Raw SSE content/usage timestamps independently support TTFT, per-request generation speed and aggregate completion-window arithmetic. Thirty-two deliberate corruptions are rejected, including wrong Q/O selector, missing input removal, wrong packed route and fake Q/O preparation. HTTP logs do not directly read GPU KV contents; separate actual-native full-state and return/cancellation gates remain bound to the exact library. There is no new coding-quality evaluation.

The table uses the median of two fresh cold-wave medians per native mode; warmed metrics and full wave ranges remain separate in `audit_http1.json` and `comparison1.json`. Frozen vLLM-Metal/SGLang have 15 retained waves each and are never rerun. Native FP32 KV/attention differs from reference FP16, and system conditions are uncontrolled. Cross-campaign absolute values do not establish a universal or causal gain. Generation speed is per request under B4; aggregate throughput includes prefill and the full completion window.

| Engine | Cold TTFT s, lower better | Generation tokens/s/request, higher better | Aggregate output tokens/s, higher better |
|---|---:|---:|---:|
| Garnet ordinary control | 18.628 | 12.144 | 17.819 |
| Garnet bulk FFN + K/V | 19.264 | 11.417 | 16.866 |
| Garnet bulk FFN + Q/K/V/O | 18.828 | 11.220 | 16.975 |
| Frozen vLLM-Metal | 11.570 | 10.484 | 21.179 |
| Frozen SGLang-MLX | 10.884 | 6.307 | 16.337 |

Cold Q/O versus ordinary, candidate/control: TTFT `1.010769` (lower favors candidate), generation `0.923904` and aggregate `0.952671` (higher favors candidate).

Cold Q/O versus K/V, candidate/control: TTFT `0.977363` (lower favors candidate), generation `0.982678` and aggregate `1.006482` (higher favors candidate).

Warmed Q/O versus ordinary, candidate/control: TTFT `0.991219` (lower favors candidate), generation `0.912996` and aggregate `0.960206` (higher favors candidate).

Warmed Q/O versus K/V, candidate/control: TTFT `1.046969` (lower favors candidate), generation `0.932007` and aggregate `0.939031` (higher favors candidate).

Q/O cold TTFT is 1.1% higher and aggregate throughput 4.7% lower than ordinary control. Warmed TTFT is 0.9% lower and aggregate throughput 4.0% lower. Cold generation is 7.6% lower and warmed generation 8.7% lower; no decode kernel changes occur in this campaign. Wide overlapping wave ranges and uncontrolled conditions preclude assigning a cause to these differences. The native prefill gain has not translated into a robust serving gain. Keep production selection unchanged. Neither the 2x single-request decode/aggregate-throughput targets nor half TTFT or coding qualification is established. This is a B4 observation, not B1 decode evidence. The historical strict coding 0/8 result was not reevaluated, and ANE/Gemma, persistent-cache HTTP, and subsequent Qwen qualifications remain incomplete. The complete prior raw serving and native manifests are rechecked unchanged before this record is committed. The new replay applies only the HTTP bridge delta to the admitted native source; the 11 preexisting dirty paths stay byte-identical.


## Bounded shared attention arena: actual memory and native correctness admission

This follows `d2b98b2de`. Evidence is `out/research/mistral7b-local-coder/mistral-attention-arena-native1`; the preceding CPU descriptor study is `mistral-attention-arena-feasibility1`. This is a clean-archive native C++/ObjC++ research build with compile-time opt-in `GARNET_METAL_CHUNK_ATTENTION_ARENA`, default OFF. It keeps the qualified Q/O path, all shader headers and embedded Metal text, math, FP32 KV/attention, quantization, model, xModel operations and frozen vLLM/SGLang artifacts unchanged. No new Rust or xModel operation is added.

The exact shape envelope is up to four owners, H32/KV8/D128, C4096, chunks up to 128, at most 512 physical token rows. Twenty-one logical B1/B2/B4 by T2/4/8/16/32/64/128 plans retain their actual descriptors, row pitches, buckets and request-span products. Two-token scores retain the native 16-byte pitch; they do not use a padded maximum128-token computation. Other shapes keep their separate original scratch. The seven query/key/value/cache-score/live-score/live-probability/live-contribution roles share seven bounded initialized roots. Declared scratch falls from 1,044,162,560 to 306,184,192 bytes, a reduction of 737,978,368 bytes (703.791 MiB). The existing native shared scratch mutex spans all command continuations through completion or abort. No runtime GPU allocation is added.

Every arena-backed encoding prevalidates all parent readiness before marking the complete roots pending. Output/partial-span guards remain intact. Abort invalidates the seven parents. Partial repair and unprepared replay are refused; preparation under the same native lease can restore only the complete private parents. Scratch accounting deduplicates physical backing buffers, while logical plans and actual compute bounds stay unchanged. The backend exposes an explicit arena tag. The diagnostic root handles are native research/test APIs, not xModel or Python operations.

All 48 native tests pass serially. The new actual-native test prepares all 21 shapes, alternates large/small shapes through 25 dense/gapped/overwrite/pending-control/all-inactive cases and two full-role poison patterns, preserves read-only KV and stable owners, and exercises partial flush, pending host-read refusal, abort invalidation, refused partial repair, refused unprepared replay and exact full-parent recovery. The same harness is separately compiled against the unchanged prior native static archive. Independent audit compares all 31,342,592 saved output-word pairs bit-for-bit, 1,516 paired independent FP64 samples under the unchanged bound, and 4,480 raw boundary-guard words. Thirteen corrupted variants are rejected. The operator's prepared device snapshot is 738,017,280 bytes lower; this is a snapshot, not peak memory.

All four whole-model gates pass again: actual1/4-chunk nonzero-history append and ordinary continuation; native scheduled full16-chunk execution with original fullKV/hidden/bank and 512 frozen IDs, cancellation/tickets/epochs and decode-first refill; leased ragged/dense/replacement/cache-return histories; and return of an actual Q/O-produced dense128 prefix while survivor rows advance. Beyond same-new-library controls, all 11,382 original full-state hash observations and 215 complete token/prediction/history fields across these four studies exactly match the prior library's saved results. This does not double as a new coding-quality evaluation.

The scheduled model's ready device allocation changes from 11,864,064,000 to 11,126,161,408 bytes, 737,902,592 bytes lower. Whole-device driver allocations are not identical to the isolated operator census. The initial audit incorrectly required their deltas to match; that assumption was corrected while retaining exact seven-role scratch accounting, complete state/token equality, positive observed allocation reduction and the working-set margin. The first build also retained a new-test metadata-field compile error, repaired using the existing identity/length API; the independent auditor's all-inactive first record needed a nonempty oracle-sample mutation target. These failures and repairs are retained. No runtime safety or numerical tolerance was weakened.

The native timing follow-up uses four fresh engines in old/arena/arena/old order, identical full-Q/O work, all 21 compact attention profiles, B4/2048 inputs, FP32 KV/attention and zero prefix retention. Each engine has one warm and two measured native calls. Full KV readbacks occur only after the last measured API, never between measured calls; builds, hashing and other inference do not overlap collection. Claims/releases are outside API/body/GPU time and retained separately. Full original KV postgates cover 256 hash comparisons; hidden/bank/prediction witnesses and owner stability cover all 12 calls. Sixteen deliberate timing/route corruptions are rejected. Conditions remain uncontrolled and there are only two engines per policy.

| Interval | Separate plans s | Shared arena s | Native speed ratio |
|---|---:|---:|---:|
| GPU span | 16.0359 | 16.3826 | 0.978834 |
| Native body | 16.1802 | 16.5525 | 0.977507 |
| Native API | 16.1846 | 16.5569 | 0.977513 |
| Lease + API + release | 16.3061 | 16.7521 | 0.973376 |

Opposite-order engine API pair ratios are `[0.942891121742853, 1.0117568408228255]`; ratios above 1 favor the arena. Complete ranges/engine observations and nested timing scope remain in `timing-audit1.json`. These are native prefill intervals, not HTTP TTFT, decode speed or serving throughput.


Native API median-of-engine-medians is 16.1846s for separate plans and 16.5569s for the arena, about 2.3% slower. The opposite-order pairs disagree, so no consistent native speed benefit is qualified. This observation does not justify another serving campaign. Preserve the memory benefit, keep the arena disabled, and profile inclusive GPU envelopes of the current bulk MPS products and attention before choosing another speed optimization. Additional workspace borrowing is not presumed to improve speed. Production selection remains unchanged. The previous serial serving table is still authoritative; this experiment does not replace it or rerun frozen references. Native FP32 KV/attention differs from reference FP16. Historical strict coding 0/8 was not reevaluated. The 2x single-request decode/aggregate target, half TTFT, coding qualification, persistent-cache serving, ANE/Gemma and subsequent Qwen goals remain incomplete. A serving follow-up needs to be justified by these measurements; further workspace borrowing from phase-disjoint grouped gate/up storage needs a separate lifetime/lease/offset/readiness gate.

The full 27-path source replay is `test2026/metal/experiments/mistral-attention-arena-native.patch`, independently applied to the exact parent and matched to every owned source hash. Only that replay and this record enter the local feature checkpoint. All 11 preexisting dirty paths remain byte-identical; prior native and 343-file serving evidence is rechecked unchanged.


## Current full-Q/O prefill GPU census directs work toward matrix products

This follows `17fb81d73`. Evidence is `out/research/mistral7b-local-coder/mistral-bulk-prefill-census1`. The isolated clean source uses the qualified Q/O stack with the unhelpful arena disabled. It adds diagnostic tracked-resource timestamp envelopes and stage names, not a new performance policy. Model artifacts, all xModel operations, original shader headers/embedded programs, arithmetic, quantization and FP32 KV/attention are unchanged. The frozen vLLM/SGLang engines and raw measurements are not rerun or replaced.

The earlier FP32 attention QK/PV marker technique now also brackets the actual bulk matrix product's two F16 inputs and F32 output. Each boundary performs one-thread volatile integer identity writes to actual resource words; this preserves both half values in a word and establishes tracked hazards with MPS. Resource type/count, region/layer/shape, native encoder samples, endpoint indices and complete timestamp calibrations are retained. Cached/live attention products stay paired in their original order. MPS internal encoders remain unobserved. Probes can add synchronization and perturb execution; intervals are inclusive diagnostic groups, not pure MPS kernel times, GPU utilization or uninstrumented cost fractions. Marker time, overlaps and unsampled remainder remain explicit. No unrelated timestamp interval is relabeled as MPS.

All 49 native tests pass serially. The additional attention gate checks old/native complete output bits and FP64 samples with counters OFF then ON. The new actual half-input bulk gate covers K/V, Q/O, gate/up, ordinary down and the largest 8,192-row fused SiLU-half/down product. All 51,904,512 F32 output-word pairs and 440,401,920 complete staging-word pairs are identical OFF/ON; 136 independent FP64 GEMM samples reconstruct packed weights and use actual saved F16 input under the unchanged bound. This is not a new independent SiLU oracle: the existing exact native SiLU-half gate remains separately required. The auditor rejects 12 corruptions, including missing typed half-input hazards, changed staging bits, false endpoints and oracle changes. Pending output reads and full-parent recovery are exercised. The first new test's fused replay was correctly refused after abort invalidated its private half-input parent; the repaired test asserts that refusal, restores the complete parent and requires exact output replay. Failed logs and the prior test source are retained. No native safety or numerical guard is weakened.

The model study uses four fresh native engines in counterOFF/ON/ON/OFF order, strictly serially. All prepare the same 21 compact attention profiles and current full-Q/O route, B4 with 2,048 input tokens per row, T128/C4096, native FP32 KV/attention and zero prefix retention. Each has one warm and one observed prefill. Counters are prepared before any row claim, as required by the existing native API. Large full-KV readbacks occur only after the final observed prefill; source hashing, builds and other inference do not overlap collection. Subsequent complete 16-token output prefixes match frozen native IDs. All eight calls preserve complete original hidden/bank/prediction witnesses; four full-KV postgates match 256 original hashes. All 256 continuation-prefix token observations and native history cursors match. Owners return to zero on every close.

There are 226 complete raw counter frames: two instrumented engines each emit 98 prefill frames (two calls with 32 layer, 16 completed-bank tail and one head frame), followed by 15 ordinary decode frames. The main census uses only the observed second prefill in each counterON engine. Every layer has seven actual bulk products, 16 original cached/live QK pairs and 16 cached/live PV pairs, with 78 uniquely owned identity markers. Thus the selected two prefills contain 448 bulk products, 1,024 QK groups, 1,024 PV groups and 4,992 identity markers. The independent auditor ties all boundaries to actual sample indices, typed resource counts, operation order, layers/shapes and the native API's summed GPU span. It partitions interval unions without double-counting. Twenty-four deliberate model/frame corruptions are rejected.

| Exclusive category | Fraction of instrumented GPU span |
|---|---:|
| FFN matrix products | 51.99% |
| Other dense projections | 12.01% |
| Attention QK/PV groups | 12.95% |
| Attention softmax | 6.11% |
| Other work, overlap and unsampled | 16.94% |

All dense projections together occupy `64.00%` of this instrumented span. FFN products alone occupy `51.99%`. The observation supports prioritizing a new supported native matrix-product mechanism. It does not establish hardware utilization, memory causality, a forecasted serving gain or a universal cost mix. The aggregate instrumented span is `31.503740s` across the two observed calls; `36.371582ms` remains unattributed. Raw detailed overlap partitions, including SiLU/unpack/layout interactions, remain in `model-audit1.json`.

Observed native API medians are `15.626821s` with counters OFF and `15.872459s` with counters ON. GPU medians are `15.460063s` and `15.751870s`. These fresh-process observations include uncontrolled condition differences and cannot isolate counter overhead or replace a serving comparison.


No new HTTP TTFT, single-request decode speed, aggregate throughput, coding quality or ANE result is added. The last serial serving table remains authoritative. Native FP32 KV/attention still differs from frozen references' FP16. Historical strict coding 0/8 is not reevaluated. The 2x decode/aggregate target, half TTFT, reliable local-coder qualification, persistent-cache serving, ANE/Gemma and subsequent Qwen goals remain incomplete. The next candidate must be grounded in the exact pinned Apple/MLX matrix code and prior Garnet negative tile/layout screens, then pass independent operator/lifetime/full-model/batching/cache and balanced timing gates before new serving measurements.

The complete 28-path diagnostic replay is `test2026/metal/experiments/mistral-bulk-prefill-census.patch`, independently matched to admitted native source hashes. Only that research replay and this record enter the local feature checkpoint. All 11 preexisting dirty paths and the prior native/343-file serving manifests remain byte-identical.


## Actual native bulk result orientation: exact words, negative speed screen

This follows `ebb17e02a`. Evidence is `out/research/mistral7b-local-coder/mistral-bulk-orientation-native1`. The preceding census attributed about 64% of instrumented native prefill GPU span to dense products, motivating a different large-group matrix layout screen. The existing `transpose_half_prefill_probe.mm` is a distinct earlier right-weight-storage experiment with scaledBF16 weights, FP32 activations and 22..80 rows. The new experiment changes the full product/output orientation for the current affineINT4/F16-staged native path at 512/2048/8192 actual rows. No reference inference is rerun.

A clean archive plus the qualified uninstrumented Q/O source adds native C++ research methods to prepare and encode `W @ A.T`, with explicit physical F32[N,M] output, alongside unchanged `A @ W.T` F32[M,N] control. Both include original Q4 unpack and actual-input F16 casting inside each native GPU command, reuse exactly 352,321,536 existing private staging bytes and request the same F32 MPS result policy. Internal MPS accumulation/order is not independently instrumented. The original row-major preparation and encoding function bodies are byte-identical to the qualified control. All old shader headers and embedded programs, xModel operations, model artifacts and selected runtime defaults stay unchanged. This is not a same-layout drop-in replacement and does not count any CPU canonicalization as a free GPU transpose.

The prepared result view explicitly supports N1024/4096/14336, actualM in512-token-row multiples up to 8192, initialized dense F32 parents, bounded offsets and original device/storage/readiness rules. Original generic view bounds are preserved. Transposed product preparation must follow the existing immutable original staging profile and refuses open/undrained commands. Encoding refuses unprepared/unsupported shapes, unready input, input/output/private-workspace aliases, foreign pending output and invalid partial parents. A separate actual-route counter resets on Begin/Abort. No new xModel/Python operator or Rust implementation is added.

All 48 native tests pass serially. The added test exercises four native projection shapes with guarded nonzero-offset input/outputs, malformed views/profile preparation, private-workspace alias rejection, pending host-read refusal, foreign-context pending-output refusal, partial flush/abort invalidation, refused partial repair/replay and exact complete-parent recovery. Complete finite/poison writes and unchanged read-only input/guards are checked; this is not an instrumented one-writer proof inside opaque MPS kernels. A deliberate omitted producer leaves the retained poison artifact and is rejected. Treating physical[N,M] bytes as[M,N] is also explicitly rejected.

The complete actual-native collection has 24 cells: both ordinary finite and finite-edge/signed-zero/subnormal coefficient/input sets; four projection shapes; actual 512/2048/8192 rows. Each variant gets three warmups and nine rotating-order measured calls in the same context with identical weight/input bindings and private buffers. Full output downloads, guard/oracle checks, raw saves and CPU canonicalization occur outside native GPU timestamps. No builds, source hashing or other inference overlap collection. Native owner/plan counts are stable and all owners close; these counters do not establish peak driver/physical-RAM use. GPU duration comes from the unchanged native command timestamps, not new hardware utilization counters.

Independent audit verifies all 506,462,208 logical output-word pairs after the declared layout mapping, 774,144 parent-guard words and 704 independently reconstructed FP64 projection samples under the original `2e-6 + 4e-6*sum_abs` bound. Every canonicalized output word is bit-identical; maximum oracle-bound fraction is 0.073652. Fifteen corrupted evidence variants are rejected, including wrong layout, changed active word, missing producer poison, overwritten guard, false conversion/writer scope, altered oracle/shape/route/order and unclean close. Original packed weights, coefficients, full input/output parents, all timings and native command/source bindings are retained.

Below are finite-corpus native GPU medians. The final column is median paired-sample original/candidate ratio; above 1 favors the explicitly transposed result. Edge-corpus data, ratio-of-medians, pair ranges and pair wins remain separate in `operator-audit1.json`.

| Actual M | Output N | Inner K | Original GPU ms | Explicit N,M GPU ms | Median paired ratio |
|---:|---:|---:|---:|---:|---:|
| 512 | 14336 | 4096 | 6.0079 | 5.9977 | 1.006419 |
| 2048 | 14336 | 4096 | 18.8174 | 18.9925 | 0.992056 |
| 8192 | 14336 | 4096 | 70.5655 | 70.4175 | 1.001859 |
| 512 | 4096 | 14336 | 6.9262 | 6.6558 | 0.995897 |
| 2048 | 4096 | 14336 | 20.7375 | 20.8338 | 0.994673 |
| 8192 | 4096 | 14336 | 80.0128 | 80.0742 | 0.999563 |
| 512 | 4096 | 4096 | 1.7750 | 1.7566 | 1.010252 |
| 2048 | 4096 | 4096 | 5.7611 | 5.8186 | 0.994393 |
| 8192 | 4096 | 4096 | 21.3161 | 21.2760 | 1.001657 |
| 512 | 1024 | 4096 | 0.5406 | 0.5445 | 1.004073 |
| 2048 | 1024 | 4096 | 2.1055 | 2.0639 | 1.033267 |
| 8192 | 1024 | 4096 | 6.5754 | 6.5881 | 0.991902 |

For the important 8192-row gate/up/down/QO shapes across both corpora, median paired ratios range `[0.9961773222164421, 1.0018594017273328]`. This is approximately flat and provides no consistent material large-group gain. The smaller 512-row down-product gain does not justify promoting a new end-to-end layout. No FFN consumer is integrated and no model/HTTP campaign is justified for this unchanged result.


Keep this path unselected. No new serving TTFT, decode, aggregate throughput, ANE or coding-quality result is added, and the prior serial serving table stays authoritative. Frozen Apple references use FP16 KV/attention while current native uses FP32; no precision comparison is hidden by this layout screen. Historical strict coding 0/8 is not reevaluated. All 2x/half-TTFT and broader Mistral/Gemma/ANE/Qwen qualifications remain incomplete. Any next explicit output-dtype experiment must preserve the existing FP32 control/bound and qualify a distinct rounding/profile contract rather than silently lowering precision.

The exact 27-path research replay is `test2026/metal/experiments/mistral-bulk-orientation-native.patch`, independently matched to every admitted source hash on the exact parent. Only that replay and this record enter the local feature checkpoint. All 11 original dirty paths, the prior 121-file GPU census and 343-file serving manifests remain byte-identical.


## Explicit F16 MPS result: rounding evidence, no justified serving change

This follows `532a1308f`. Evidence is `out/research/mistral7b-local-coder/mistral-bulk-half-result-native1`. The current full-prefill census places dense products at about 64% of instrumented GPU span; the preceding physical result-orientation screen found no useful large-group gain. Earlier `mistral-prefill-fp16-cache1`/streamed studies used F16 operands with F32 results, and `mistral-half-verification-profile1` tested 1–4-row decode with F32 results. This screen instead tests an explicit F16 MPS result for the actual bulk geometry. Those older studies do not prove this new precision safe or fast. No comparator inference is rerun.

A clean archive plus the qualified uninstrumented full-Q/O research source adds C++ methods `PrepareDiagnosticBulkHalfResult` and `EncodeDiagnosticBulkHalfResult`. An initialized F32[M,N/2] word carrier backs an explicitly prepared F16[M,N] MPS result view. Generic native Matrix ownership/download/update semantics remain F32 word storage; no half-typed public matrix is passed to APIs that assume four bytes per element. Prepared bindings retain the carrier, include exact buffer/offset/shape identity, are bounded at 64, refuse private staging aliases and must follow original F32 profile preparation. The existing multiplication plan, affine Q4 unpack and F16 input cast are reused; unpack/cast remain inside each timed GPU command. The original F32 preparation and encoding bodies are byte-identical. Shared private staging remains 352,321,536 bytes, with no additional GPU staging. There is no xModel operation, Rust code, shader/math/default-selection or model/cache precision change. Opaque MPS internal accumulation policy is unobserved; result dtype alone does not establish it.

Native validation rejects malformed carrier dimensions, open-batch preparation, unready inputs, unprepared result bindings, input/output/private-scratch aliases and foreign pending outputs. Output writes use existing whole-parent readiness tracking. Pending host reads fail; flush/abort invalidates the complete carrier owner, partial repair and replay fail, then full-parent restoration reproduces every output bit. Route counters reset at Begin/Abort. The binding count and native owners remain stable during execution. All 48 native CTests pass; this means the control/safety contract is valid, not that a lower-precision result satisfies the original F32 arithmetic contract.

The serial collection covers 24 cells: ordinary finite and finite-edge/subnormal/signed-zero input/coefficient corpora, four projection shapes and 512/2048/8192 actual rows. Both variants use the same context, immutable inputs/weights and private scratch. Three warmups and nine rotating-order measurements per variant include actual native unpack/cast/product, with full poison initialization before every call. Downloads, independent oracle work and raw saves occur outside GPU timestamps. No builds, source hashing, other GPU tests or inference overlap timed collection. Complete result parents and input guards are retained. A deliberately omitted producer remains poisoned and is detected. MPS internal single-writer maps are not instrumented, and GPU timestamps are the existing native command spans, not utilization counters. No GPU F16-to-F32 conversion or real downstream nonlinear consumer is measured or claimed free.

The independent evidence audit checks 506,462,208 complete paired logical output words, 884,736 external parent-guard words and 704 reconstructed FP64 projection samples. Every candidate half word equals independent RNE-half of the corresponding original native F32 control. Original F32 results retain the unchanged `2e-6 + 4e-6*sum_abs` bound; maximum bound fraction is 0.073652. The distinct F16 result fails that original F32 bound on 362 sampled outputs, as expected from lower output precision; those failures are retained and never masked by relaxing the bound. One candidate sample differs from direct half rounding of the reconstructed FP64 sum even though it matches half rounding of the F32 native result. This is evidence of the rounding distinction, not exact-real-sum half qualification or model quality. Seventeen corrupted evidence variants are rejected, including false F32/accumulator/conversion claims, changed active bits, missing producers, guards, wrong oracle, altered order/route, invalid time and unclean close.

Finite-corpus timings below are native GPU medians; above-one paired ratios favor the explicit half result. Full edge-corpus timings, pair ranges/wins and ratio-of-medians remain separate in `operator-audit1.json`.

| Actual M | Output N | Inner K | Original F32 result ms | Explicit F16 result ms | Median paired ratio |
|---:|---:|---:|---:|---:|---:|
| 512 | 14336 | 4096 | 6.0447 | 6.1135 | 0.999625 |
| 2048 | 14336 | 4096 | 18.8340 | 18.7034 | 1.006762 |
| 8192 | 14336 | 4096 | 81.0255 | 78.1596 | 1.045024 |
| 512 | 4096 | 14336 | 6.9984 | 6.9252 | 1.012377 |
| 2048 | 4096 | 14336 | 20.7798 | 20.9251 | 0.994693 |
| 8192 | 4096 | 14336 | 77.8457 | 78.2378 | 0.996989 |
| 512 | 4096 | 4096 | 2.1896 | 2.0777 | 1.056010 |
| 2048 | 4096 | 4096 | 5.8522 | 5.7734 | 1.013303 |
| 8192 | 4096 | 4096 | 21.3632 | 21.4172 | 1.006575 |
| 512 | 1024 | 4096 | 0.5372 | 0.5378 | 1.002010 |
| 2048 | 1024 | 4096 | 2.0971 | 1.9774 | 1.069108 |
| 8192 | 1024 | 4096 | 6.5236 | 6.5614 | 0.982947 |

Across both corpora, important 8192-row gate/up/down/QO median paired ratios span `[0.9956113974582861, 1.0450243364429794]`. The gate/up ratio-of-medians can look more favorable than its paired ratios because timings drift; the opposite-signed down/QO results do not establish a consistent material speed gain. This does not justify integrating a half FFN consumer or running another model/HTTP campaign. Result-dtype switching is not a qualified speed path.


Keep the production selection unchanged. The last serial serving comparison remains authoritative. Native FP32 cache/attention still differs from the frozen Apple references' FP16; this component screen neither changes that policy nor creates a new matched-precision baseline. No new TTFT, decode, aggregate throughput, coding quality, ANE or Gemma result is added. Historical strict coding 0/8 is not reevaluated. The 2x decode/aggregate, half-TTFT, quality, continuous-batching/cache serving and broader Mistral/Gemma/ANE/Qwen requirements remain incomplete.

The exact 27-path research replay is `test2026/metal/experiments/mistral-bulk-half-result-native.patch`, independently matched against the parent and every admitted source hash. Only this replay and the research record enter a local feature checkpoint. All 11 preexisting dirty paths, the 188-file preceding orientation manifest and the 343-file serving manifest remain byte-identical. No main switch or push occurs.


## Compact-input 128-thread packed geometry: exact native output, slower

This follows `e31fa7f91`. Evidence is `out/research/mistral7b-local-coder/mistral-packed64-native1`. Source inspection reconfirms that pinned MLX qmm NAX uses M64/N64/BK64 with four SIMD groups for these large shapes. Its `nax.h` itself calls MPP `matmul2d`; the sixth descriptor argument is the SDK's relaxed-precision policy, not a destination-layout flag. Earlier relaxed-precision, K-staging, wide-fragment and loop-control studies remain negative. The earlier native reuse64 uses F32 indexed input loads and scalar/bounded unpacking; vector-stage64 prototypes retain M128 and 256 threads; wide-fragment variants retain the larger outer tiles and shared stage. This new combination instead tests compact F16 input, vectorized affineQ4 unpack and an M64/N64/K64 stage with 128 threads. It is geometry inspired by pinned MLX, not a reproduction of every upstream policy. No reference import or inference rerun occurs.

The clean research source keeps `packed_affine_shader.h` byte-identical. A separate candidate header copies only the original M128/K128 kernel and changes its name/parameter type, actual output-row tile to 64, K stage to 64 and packet ownership from 256 to 128 threads. The ordered M16/N32/K16 multiply sequence, FP16 operand conversion, FP32 accumulators/results and direct register mappings remain identical. Candidate static shared storage is checked at 9,216 bytes versus the original 17,408. Native context preparation eagerly specializes the extra candidate under the existing diagnostic build flag, with the same function constants, Safe Metal4 library and unchanged measured cooperative ABI guard. The original bounded 14,680,064-byte compact-input owner is reused; no GPU buffer is added. All old shader headers, raw embedded programs, xModel operations, model files, selected runtime and public serving defaults remain unchanged.

A drained-batch-only C++ selector defaults off. Ordinary and indexed packed encoding use 64-row tiles/128 threads only when selected; the original 128/256-row/256-thread policy is preserved otherwise. Original preparation and index/layout/alias/readiness validation remain. The engine profile still means up to 512 physical token rows, not padding to a maximum. Counts below 128 and unsupported shapes retain the original fallback; tails gather/store only actual selected rows. A separate route counter resets at Begin/Abort. There is no xModel or Python serving selector, production promotion or Rust change.

The first C++ build passes, but native Metal compilation during CTest correctly catches a duplicate parameter-struct definition in the appended shader. Candidate type naming is fixed without weakening the compiler/layout/numerical guards. Those failed logs and original source are retained. Before any operator collection, the recorder is also corrected to retain actual downloaded GPU parent buffers rather than rebuilding guard rows in a saved array. No performance data are collected from either unfinished recorder version.

The final build passes all 48 serial native CTests. The complete collection covers 62 cells over 10 prepared fixtures, three large projection shapes at actual 128/256/512 rows, the 512-row narrowK/V fallback, dense/indexed/reversed/permuted row maps and counts 0/4/127/128/129/255/256/257/383/384/511/512. Input/output views have real nonzero offsets and complete initialized parents with guards; full input parents remain unchanged and external output guards survive every call. Native results match the lookup-free nonpacked control bit-for-bit, all inactive rows are positive zero, output poison is overwritten and repeated outputs are stable. Invalid duplicate rows and selector changes in open/partially submitted work are rejected; abort invalidates the full partial-parent owner, partial repair/replay is refused, and whole-parent restoration recovers exact output. Plan/pipeline/owner counts are stable, and all owners close. Internal atomic writer maps are not instrumented.

The independent audit verifies 354,942,976 retained output values across two modes, or 177,471,488 logical pairs, including every inactive word; 1,052,770,304 value observations are checked natively during repeated execution. There are 1,740 independently reconstructed FP64 reference points and 3,480 original/candidate value comparisons under the unchanged `2e-6 + 4e-6*sum_abs` bound; maximum bound fraction is 0.019113. Sixteen evidence corruptions are rejected, including changed output bits, poison/missing writers, overwritten actual parent guards, false geometry/writer claims, altered masks/route/order, invalid timings and unclean close. Complete payloads, commands, admitted source/library and the failed first build/test logs remain separate.

Each timed cell uses three warmups and nine rotating-order original/candidate native calls. Timed GPU commands include the unchanged actual selected-input gather and projection. Poison updates, full downloads/guard checks, oracle work and saved payloads are outside GPU timestamps. No builds, source hashing or other inference overlap collection. The dense-cell table below is descriptive native component timing; above-one original/candidate ratios would favor the candidate. Indexed full-row timings remain separate in `operator-audit2.json`.

| Actual rows | Output N | Inner K | Original GPU ms | M64/128-thread GPU ms | Original/candidate |
|---:|---:|---:|---:|---:|---:|
| 128 | 14336 | 4096 | 1.5788 | 1.6310 | 0.968016 |
| 128 | 4096 | 14336 | 1.8105 | 1.8321 | 0.988174 |
| 128 | 4096 | 4096 | 0.5272 | 0.5238 | 1.006444 |
| 256 | 14336 | 4096 | 2.6074 | 2.8379 | 0.918777 |
| 256 | 4096 | 14336 | 3.2899 | 3.5404 | 0.929245 |
| 256 | 4096 | 4096 | 1.0025 | 1.0725 | 0.934732 |
| 512 | 14336 | 4096 | 4.8879 | 5.5289 | 0.884057 |
| 512 | 4096 | 14336 | 5.5842 | 6.5774 | 0.849003 |
| 512 | 4096 | 4096 | 1.7884 | 1.9109 | 0.935893 |

At 512 dense rows the candidate is approximately 13% slower for gate/up, 18% slower for down and 7% slower for Q/O. The isolated 128-row Q/O ratio around 1.006x does not justify selecting this geometry or another full-model/HTTP campaign. Smaller threadgroups and shared stages are not inherently faster. Keep this option unselected and preserve the negative result.

The existing 0/8 strict coding score is not rerun or changed: prior saved answers fail formatting requirements, some extracted implementations fail functional contracts, and three earlier MLX/native outputs diverge. This kernel's exact current-control outputs do not qualify coding quality. No new TTFT, single-request decode, aggregate throughput, ANE or Gemma result is added; the last serial serving table remains authoritative. Native FP32 KV/attention differs from the frozen references' FP16. The 2x/half-TTFT and complete Mistral/Gemma/ANE/Qwen goals remain open. A possible next fixed-profile aligned-tile branch specialization requires inspection of prior variants first; it must preserve arbitrary actual-count tails and original numerical/state/quality gates rather than treating an upper bound as mandatory request padding.

The exact 28-path replay is `test2026/metal/experiments/mistral-packed64-native.patch`, independently matched to every admitted file against the exact parent. Only this replay and the research record enter a local feature checkpoint. All 11 original dirty paths, the preceding 191-file half-result manifest and the 343-file serving manifest remain byte-identical. No main switch or push occurs.


## Compiled manual full-tile specialization and native Instruments trace

This follows `f2ff9fbc5`. Evidence is `out/research/mistral7b-local-coder/mistral-packed-fulltile-native1`. The preceding compact-input 128-thread geometry regresses the important projection mix. Prior static full-tile tensor-memory screens changed the tensor API and K reduction shape; older fixed-shape BF16 work targeted decode. This experiment instead preserves the current manual M128/M256 cooperative packed kernels and removes only proven-absent actual compact-row load/store checks. It addresses the up-to engine-profile concept without assuming requests always equal the profile ceiling.

A clean archive plus the qualified uninstrumented Q/O source keeps the original `packed_affine_shader.h` byte-identical. A separate diagnostic header contains the same M128/M256, N64, stageK128, 256-thread kernels with the six input row predicates and six output row predicates removed. Input gathering, affine Q4 decoding, FP16 operands, ordered K16 multiplication, FP32 accumulation/output, register indexing and static 17,408-byte shared stage remain unchanged. Native preparation eagerly specializes both candidate pipelines with the existing dimension/stride constants, Safe Metal4 policy and cooperative ABI startup guard. Shared compact-input ownership stays at 14,680,064 bytes; no GPU buffer is added. The original bulk projection preparation/encoding bodies, all other shader headers/raw programs, xModel operations, model/tokenizer artifacts and serving selection are unchanged. No Rust is introduced.

The native selector defaults off and changes only between drained batches. After existing shape/index/layout/readiness/alias validation, a candidate is selected only if the actual selected compact row count is an integer multiple of the original chosen 128/256-row tile. Otherwise the original tail-safe kernel remains. Counts below 128 and unsupported projection shapes keep the original fallback. A 384-row gate/up or down selection does not become a full 256-row-tile call; 384-row Q/O can fill three 128-row tiles. Indexed row permutations remain valid because the unchanged gather produces the actual compact logical array. The up-to-512 profile is never substituted for actual selected rows, and there is no request padding. Actual native route counters reset at Begin/Abort.

All 48 native CTests pass serially. The added CPU register-coordinate proof enumerates six whole-tile/actual-count combinations within the profile and 131,072 output coordinates, proving bijective writes and compact-row bounds for the original observed cooperative ABI. This is not atomic GPU writer-map instrumentation. The actual-native collection covers 62 cells over 10 prepared fixtures: three main projection shapes at actual 128/256/512 rows, narrow K/V fallback, dense/indexed/reversed/permuted mappings and active counts 0/4/127/128/129/255/256/257/383/384/511/512. Native observers independently require original fallback for partial tiles. Nonzero-offset input/output spans retain complete parent guards, read-only input bytes stay unchanged, inactive rows are positive zero and poison/repeated-output gates pass. Duplicate rows, late selector changes and partial-parent repair/replay after flush/abort are refused; complete-parent restoration recovers exact output. Native plan/pipeline/owner counts remain stable and all owners close.

Independent audit verifies 354,942,976 retained output values, or 177,471,488 logical original/candidate pairs. All bits match both the original packed route and its lookup-free nonpacked control. Native execution checks 1,052,770,304 repeated value observations. There are 1,740 reconstructed FP64 reference points and 3,480 value comparisons under the unchanged `2e-6 + 4e-6*sum_abs` bound; maximum bound fraction is 0.019113. Eighteen corruptions are rejected, including an illegally promoted partial tile, altered coordinate proof, actual guard overwrite, changed result, missing writer, false geometry/writer claims, bad masks/route/order/time and unclean close. The first auditor run correctly catches that its copied false-geometry self-test had become a no-op after the new candidate returned to 256 threads. The final auditor changes only that mutation to 128; failed script/logs remain, all native data are unchanged, and no inference is rerun for this auditor correction.

Each timed cell has three warmups and nine rotating-order original/candidate calls. GPU commands include the real selected-input gather and projection; poison uploads, raw readbacks/guards, oracle work and saving are outside the native GPU timestamps. No source hashing, builds or other inference overlap collection. Results below are dense native component GPU medians, not HTTP latency, token rates or hardware utilization. Both ratio-of-medians and median paired ratios are retained; above one favors the candidate.

| Actual rows | Output N | Inner K | Original ms | Full-tile ms | Ratio of medians | Median paired ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 14336 | 4096 | 4.8868 | 4.9193 | 0.993402 | 0.993792 |
| 512 | 4096 | 14336 | 5.4056 | 5.7422 | 0.941391 | 0.965753 |
| 512 | 4096 | 4096 | 1.8142 | 1.8397 | 0.986139 | 1.014857 |
| 128 | 14336 | 4096 | 1.5949 | 1.5414 | 1.034681 | 1.041608 |
| 128 | 4096 | 14336 | 1.8526 | 1.7876 | 1.036338 | 1.033179 |
| 128 | 4096 | 4096 | 0.5302 | 0.5114 | 1.036744 | 1.038700 |
| 256 | 14336 | 4096 | 2.6081 | 2.6177 | 0.996339 | 0.996276 |
| 256 | 4096 | 14336 | 3.2040 | 3.1402 | 1.020328 | 0.985523 |
| 256 | 4096 | 4096 | 1.0242 | 1.0025 | 1.021654 | 1.028540 |

The 128-row cases show about 3–4% component gains, but larger important cases do not improve consistently. At 512 rows gate/up is approximately flat, down is slower, and Q/O differs in sign between ratio-of-medians and paired ratios. This does not justify selecting the path, full-model integration or another serving campaign. Preserve the negative result and do not infer that removing a source predicate necessarily removes a material GPU bottleneck.

A subsequent independent diagnostic uses the installed `xctrace` Metal System Trace template to launch the same native executable, with a fresh diagnostic output directory and a 30-second bound. The launched process exits successfully; no reference engine or unrelated application is launched, stopped or changed. Although the target is Garnet, the template also records global device/process metadata. Analysis resolves XML references and filters GPU scheduling/shader-list rows to the exact launched PID. Only those target rows are analyzed. The full trace, command, target output, raw table-of-contents/export and target-filtered analysis remain under `instruments1/`. Traced observations are explicitly excluded from all benchmark medians.

The trace contains 599 matching target GPU compute intervals and 772 matching target shader metadata rows, including the preserved and full-tile kernel names. The default counter metadata exposes only `RT Unit Active`, and the dedicated counter-profile table is empty. Zero compiler-spill events do not prove no register spilling; compute utilization, register occupancy and stall reasons are unobserved. Asking the CLI for an instrument named `GPU Counters` reports that the name cannot be found. This is a diagnostic capability limit, not a failure of native inference or a blocker for the overall task. A configured shader/counter capture is needed before attributing the cost to occupancy, bandwidth or ALU limits. The successful default trace does not establish internal MPS-kernel performance or a new speed result.

Keep both production selection and the last serial serving comparison unchanged. The historical strict coding 0/8 score is not reevaluated; native FP32 KV/attention still differs from frozen reference FP16. No new TTFT, decode, aggregate-throughput, ANE, Gemma, multimodal or coding-quality admission is made. The 2x/half-TTFT and complete Mistral/Gemma/ANE/Qwen goals remain open. Further work should use actual configured hardware/shader evidence before another tile/precision/layout guess.

The exact 28-path replay is `test2026/metal/experiments/mistral-packed-fulltile-native.patch`, independently matched against the exact parent and every admitted source hash. Only this replay and the research record enter the local feature checkpoint. All 11 preexisting dirty paths, the previous 234-file packed64 manifest and the 343-file serving manifest remain byte-identical. No main switch or push occurs.


## Prepared native GPU captures for counter profiling

This follows `0bd2b1211`. Evidence is `out/research/mistral7b-local-coder/mistral-gpu-shader-profile1`. The preceding result-dtype, smaller-threadgroup and full-tile screens do not improve the important large projection mix consistently. Further guesses about occupancy, ALU limits or bandwidth are not justified by timestamp envelopes alone. This checkpoint prepares actual GPU capture evidence using the unchanged qualified native library; it does not select another inference kernel or add a serving-performance result.

Apple's [Metal performance analysis guide](https://developer.apple.com/documentation/xcode/analyzing-the-performance-of-your-metal-app/) documents that performance counters are off by default and identifies the Performance Limiters selection in Recording Options. Read-only inspection of the installed template had found actual boolean shader-profiling controls. A private copy changes exactly `shaderprofiler` and `shaderprofilerinternal` from false to true; independent archive comparison checks every other object/field unchanged. No numeric counter-profile enum is guessed, installed Apple template is edited or system preference is changed. The actual exported table-of-contents reports `shader-profiler=1` for both counter tables. The launched native target exits successfully and exact-PID analysis retains 599 own GPU compute intervals and 858 own shader metadata rows. The exported shader-interval table has 1,056 rows, but the instruction-PC sample table has zero rows. Only `RT Unit Active` counter metadata is available and the dedicated counter-profile table is empty. Private interval identities are not guessed into kernel attributions; global device counters are not attributed to Garnet. These observations do not establish utilization, register occupancy or stall causes, and no traced timings enter benchmark medians.

The computer-use tool twice reports that the Mac is locked. User unlock is requested once while CLI work continues; no UI action, credential entry, security change or unrelated app modification occurs. A native read-only Metal capability query independently reports only the `timestamp`/`GPUTimestamp` counter set, stage-boundary sampling support and no dispatch/blit-boundary sampling. This is this device/runtime's observed public counter capability, not a claim that Xcode cannot obtain private profiling counters. `xcrun --find gpudebug` fails in this installation. Apple documents that command-line debugger in current material; its absence here is recorded rather than assuming it is available or installing another Xcode.

The new standalone ObjC++ experiment uses Apple's [documented programmatic capture API](https://developer.apple.com/documentation/xcode/capturing-a-metal-workload-programmatically?language=objc). It links the exact qualified full-tile-study library with the optional selector held off, prepares the original native affine/packed profiles, warms three calls, then records one original packed projection inside a `MTLCaptureManager` GPUTraceDocument scope. `MTL_CAPTURE_ENABLED=1` is set only for that child process. The first wrapper correctly refuses the unsupported destination query before any default Metal device is initialized. Version two initializes and retains the default device before querying the capture manager; the query then succeeds and actual trace documents are produced. Both wrapper sources/binaries, the failed pre-device attempt, correction, compile definitions/commands and logs remain. No capability check, numeric bound, native safety guard or original library is weakened or rebuilt.

Three captures run strictly serially at actual 512 rows, with deterministic finite affineQ4/FP16-coefficient fixtures of the admitted Mistral projection dimensions. These are representative operator fixtures, not trained-weight or complete-model captures. Original compact F16 gathering, vector Q4 dequantization, ordered K16 multiplication and FP32 accumulator/output policy remain. Native counters require one original packed dispatch, zero full-tile-candidate dispatches and zero profile misses. Complete actual output parents match a lookup-free nonpacked native control bit-for-bit before and under capture. Nonzero-offset input/output guards, all finite output writes, full input immutability and fixed native owners/plans/pipelines are checked. Every native owner closes; post-capture Metal allocation is an observed snapshot, not a peak or proof of general driver-memory release.

| Capture geometry | Actual M | Output N | Inner K | GPU trace document bytes | Max original FP64 bound fraction |
|---|---:|---:|---:|---:|---:|
| Gate/up-sized | 512 | 14336 | 4096 | 87,736,946 | 0.008778 |
| Down-sized | 512 | 4096 | 14336 | 87,612,043 | 0.007599 |
| Q/O-sized | 512 | 4096 | 4096 | 42,965,376 | 0.008778 |

The independent auditor checks all 11,534,336 logical original/captured output-word pairs, 135,168 external parent-guard words and 72 reconstructed FP64 projection samples under the unchanged `2e-6 + 4e-6*sum_abs` bound. Maximum bound fraction is 0.008778. Nine corrupted records/payloads are rejected, covering a missing capture, wrong geometry/route, retained owners, changed output, unwritten poison, overwritten guard, false oracle and truncated document extent. The audit binds the actual source/binary and frozen library, hashes every capture document file and retains complete input/weight/coefficient/reference/captured payloads. The public API scope and successful native command establish capture creation; private capture binary internals are not hand-decoded and replay/detailed counter inspection is still pending. No instrumented atomic writer map or model correctness/quality result is claimed from these operator witnesses.

The three `.gputrace` documents are `up512-2.gputrace`, `down512-1.gputrace` and `qo512-1.gputrace` in the evidence directory. Their detailed Xcode counter/shader inspection remains pending while the Mac is locked. The shader-template trace and all GPU captures are diagnostic-only and excluded from the serial serving comparison. Current API-visible timestamp counters, zero instruction samples and missing CLI debugger do not justify inventing a GPU bottleneck explanation or another tile change.

The reproducible native capture helper is `test2026/metal/experiments/packed_projection_capture.mm`. It is a standalone research executable, not a normal build target. It requires the already-qualified clean full-tile research headers/library and their compile definitions; the exact final command is retained as `capture-compile2-command.json`. Invoke it as `NEW_CAPTURE.gputrace NEW_WITNESS_DIRECTORY M N K`, with `MTL_CAPTURE_ENABLED=1` for the child. Bounds are explicit: M128/256/512 and the three admitted N/K pairs. Only M512 captures are observed here; allowed argument values do not qualify the other profiles. Do not link it against an unrelated or dirty root library. Native xModel operations, engine API, selected runtime, Rust policy, weights, tokenizer and cache precision are unchanged.

The last serving TTFT/decode/aggregate table and frozen vLLM/SGLang results remain unchanged; no comparator inference is rerun. Historical strict coding 0/8 is not reevaluated or promoted by capture parity. Native FP32 KV/attention still differs from reference FP16. The 2x/half-TTFT, reliable local coder, persistent-cache HTTP, ANE/Gemma and original Qwen27B/9B/4B requirements remain open. Actual counter/shader evidence is needed before choosing the next optimization, followed by complete operator/model/state/cache/batching/quality and balanced serving gates.

Only this record and the final native capture helper enter a local feature checkpoint. All 11 preexisting dirty paths, the previous 1,000-file full-tile manifest and the 343-file serving manifest remain byte-identical. No main switch or push occurs.


## Opt-in native prefill-only prefix snapshots

This follows `42ee307f8`. Evidence is `out/research/mistral7b-local-coder/mistral-prefill-prefix-policy1`. Detailed hardware-counter inspection remains pending; the last computer-use observation was a locked Mac. Independent cache work addresses a previously observed precision-history distinction: the current Mistral prompt-prefill and ordinary decode paths do not have interchangeable computation histories. Earlier retained native evidence shows that a prefix containing a decode-produced token can differ from cold prompt prefill. Token equality alone does not justify silently treating those histories as equivalent. This change is a conservative optional snapshot-production policy, not a new arithmetic kernel or a proof that all prefill chunk shapes are numerically identical.

A clean parent archive plus the qualified uninstrumented Q/O research overlay adds `GARNET_METAL_PREFIX_PREFILL_ONLY`, default off. Native C++ tracks each leased physical row's total token count and leading prefill-produced count, with the existing context bound of 4096. An ordinary decode creates a history gap; later prefill does not erase it. Only a complete, nonempty, entirely prefill-produced snapshot is admitted when the option is selected. The runtime returns false before allocating/copying a mixed-history candidate and exposes a skipped-capture count. It retains earlier eligible snapshots for actual token-prefix matching. It does not slice a recurrent snapshot at an earlier boundary or reconstruct recurrent state from KV alone.

The final source determines the producer from immutable compiled graph metadata, graph rank and the actual verification call. Rank three alone is insufficient: explicit speculative-verification graphs also use rank-three inputs and have their own FP32 computation policy. A source review catches that exception before the local checkpoint; the final policy excludes both verification calls and graphs carrying verification metadata. The initial source, earlier witnesses and superseded replay remain retained as incomplete coverage. The corrected source is rebuilt and all final witnesses are recollected. No numerical bound is changed.

Both ordinary/compact execution and optimized native bulk commit update the same row metadata. Row reset/release clears it; whole snapshot restore validates its pure producer boundary before GPU state writes; dense reconciliation truncates it only after the existing successful state clear. Existing recurrent-rewind rejection remains. A native suffix in the execution-profile identity separates the optional producer domain. The research worker requires the frozen profile's explicit `prefix_capture_policy: "prefill_only_v1"` to match the compiled extension. Default-off engines retain the previous all-history behavior. Matching, eligibility, leases and state ownership remain inside C++; Python checks the selected profile rather than implementing cache matching. All xModel files, old shader headers, embedded programs, quantization and inference math remain unchanged. No Rust is added.

The pure C++ metadata test exhaustively compares phase sequences through length ten against an independent vector reference, checks truncation/append, four-row isolation, restore/reset, invalid prior lengths and the 4096 boundary. Final Mistral profiles pass all 48 native CTests in each mode; the separate supported hybrid profiles pass all 24 in each mode. A legacy tiny hybrid profile correctly rejects the dense Mistral shape guard. Constructing the valid diagnostic hybrid profile catches an unconditional research benchmark registration and several inconsistent inherited option combinations; the final CMake registration is guarded by its actual extension and every existing native guard remains. Early integration scripts also correctly hit invalid synthetic vocabulary, cross-session scratch-family and live-lease selector guards. Failed logs/scripts remain; final fixtures use valid token IDs, separately owned prepared graph families and drained selector changes.

The synthetic two-layer hybrid fixture retains complete convolution, recurrent, KV and hidden-output witnesses. A four-token prefill snapshot is saved, followed by a decode token and another prefill token. With the policy selected, both mixed candidates are skipped and returning A under a new epoch restores four tokens; default off preserves the six-token longest snapshot. B/C/D remain byte-identical, stale epochs are rejected and recurrent rewind is refused twice without state mutation. A separate pure-prefill candidate exceeds its small host budget and is refused with zero mixed-history skips, preventing a false budget pass. All native owners close.

The real Q4/group64 Mistral gate uses B4/C4096, prepared up-to-128 prompt families and the unchanged selected score/value policies. It saves pure A128, consumes one ordinary decode input, advances B/C/D by eight steps and returns A under a new epoch. Selected policy restores 128 tokens; default off restores its saved mixed 129 tokens. Whole KV/hidden hashes and prediction match the corresponding saved state. Compact suffix execution matches a lookup-free run with exactly the same producer history. The test deliberately does not equate the different default-off/on histories. A later optimized full Q/K/V/O bulk append preserves A's decode gap while B/C/D stay pure. A compact two-token append to physical row C updates C's metadata and state while A/B/D remain unchanged. Dense A rewind removes the mixed suffix, then compact replay yields an eligible pure snapshot. This is an untimed native state gate, not a new serving benchmark.

A separate real-model session prepares ordinary B4/T128 prefill and explicit B4/T1 verification without compact families, respecting the existing guard against combining those workspaces. After pure prefill128, one accepted verification input advances A to129 but must leave its leading prefill count128. The selected policy skips that mixed candidate and returning A restores128; the default-off control can retain/restore129. B/C/D are unchanged and all owners close. Verification and compact coexistence remains unsupported; these separate tests do not qualify that combination.

The independent final evidence audit checks 5,720 complete-row KV/hidden SHA256 witnesses across the two real-model campaigns and 10,144 retained synthetic hybrid values. These are full native-state hash witnesses, not retained raw multi-gigabyte model caches or an independent numerical oracle. Sixteen deliberately corrupted records are rejected, covering mixed-capture admission, wrong restored length, recurrent/KV/hidden changes, another row, false budget pass, wrong policy or bulk/verification metadata, retained owners and false completion. Exact states before policy-dependent reuse and in the later identical-history episodes agree between off/on modes. This qualifies the exercised native producer-policy contract; it does not prove coding quality or numerical interchangeability of distinct producer histories.

No HTTP retention, new TTFT/token-speed/aggregate-throughput, coding quality, ANE, Gemma or multimodal result is added. Historical strict coding 0/8 remains unchanged, and native FP32 KV/attention still differs from the frozen Apple references' FP16. The latest serial serving comparison and its frozen vLLM/SGLang observations stay authoritative. Persistent-cache HTTP, the 2x/half-TTFT target and the broader Mistral/Gemma/ANE/Qwen requirements remain incomplete. The optional policy is not selected for production by this checkpoint.

Only the exact final 28-path research replay, `test2026/metal/experiments/mistral-prefill-prefix-policy-native.patch`, and this research record enter the local feature checkpoint. The replay is independently applied against the exact parent and compared to every admitted source hash. All 11 preexisting dirty paths, the 343-file serving manifest, the 998-file preceding capture manifest and pinned assets remain byte-identical. No main switch or push occurs.


## Prefill-only prefix reuse through native HTTP serving

This follows `60e389d9e`. Evidence is `out/research/mistral7b-local-coder/mistral-prefill-prefix-http1`. The preceding checkpoint qualifies the optional producer policy inside native execution, including hybrid snapshots and verification. This campaign exercises that same validated Mistral library through the loopback HTTP bridge and continuous scheduler. It is a serving-correctness admission for the exercised profile and arrivals, not a new performance or coding-quality result. No native library is rebuilt, production default changed or comparator inference rerun.

Two fresh engines run strictly serially: uncached control, then retained prefill-only prefixes. Both use the exact qualified library, Q4/group64 Mistral weights, FP32 KV/attention, B4/C4096 and prepared up-to-128 prompt chunks with the existing packed/split-score/value32 policies. The optional compiled policy is explicit in both profiles; only the latter enables retention and native tracked chunk progress. Its bounded cache has 32 entries and a 384 MiB host budget. Each engine receives the same nine frozen greedy, native-tokenized requests. Initial A/B/C/D have 211-token prompts, with output budgets 8/192/224/256. E queues after A's first token, followed by exact A return, append, late edit and early edit as short requests finish. Actual native token common-prefix lengths, not string assumptions, determine expected hits.

| Returning request | Common token prefix with A | Actual cached tokens | Native action |
|---|---:|---:|---|
| Exact A return | 211 | 211 | Restore whole saved state and run head only |
| Appended instruction | 209 | 128 | Restore earlier saved prefix, prefill remaining suffix |
| Late edit | 198 | 128 | Restore earlier saved prefix, prefill changed suffix |
| Early edit | 3 | 0 | Prefill from an empty row |

All 18 HTTP streams complete at their requested output lengths, 752 generated tokens per engine. Complete prompt IDs, response IDs, incremental usage and final text agree between transport, native worker and frozen fixtures. Every request's generated token IDs match its uncached-control counterpart. Exact A return also reproduces initial A's eight-token output prefix. These synthetic coding-like requests test streaming/cache behavior; they are not the historical strict coding-quality suite.

The raw native admission event proves E enters while all initial rows are live and before A completes. Its first scheduled prefill follows A's completion, reuses A's physical row under a new epoch, and occurs while B/C/D continue. The later returning/edited requests use successive new epochs on that row while B/C/D remain active. Native plans never exceed four live rows or overlap distinct jobs on one lease. The exact-hit return's native plan has zero prefill tokens and the full cached offset. Request identities are new; native token-prefix matching establishes reuse.

A separate research worker observer performs read-only full-state snapshots around `next_json` only when a pending newcomer can claim free capacity. It checks all 32 layers' complete K/V row hashes plus final hidden and, for tracked execution, token/producer counts for B/C/D. Five restores/admissions per engine retain those survivors byte-identically. The independent audit verifies 3,900 before/after row digest observations in total; raw multi-gigabyte device caches are not retained. This observation perturbs timing and is excluded from benchmark comparison. Matching, leases, restore and capture stay in the unchanged native C++ implementation; the observer does not implement a Python cache.

The retained run records three hits and six misses, 14 entries, 347,852,398 accounted host bytes, 1,664 shared resources and no evictions. Nine completed decode histories are skipped by the native prefill-only capture policy; eligible prompt snapshots survive. This campaign does not qualify eviction/replacement pressure, longer prompts, other models, verification/compact coexistence or broader cache-security domains. Those require their own evidence. Both native workers exit zero, release all native owners and satisfy the existing 4 MiB post-close Metal snapshot bound. The bridge receives the expected graceful interrupt and is not force-killed; no unrelated application is stopped.

The independent audit rejects seven corrupted records: false cached usage, changed response ID, changed surviving hidden hash, missing restore witness, retained native owner, false mixed-capture count and an illegally nonzero exact-hit prefill plan. Earlier tokenizer preparation's unsupported close call is preserved; the final preparation exits successfully with the same recorded native token IDs. No guard, EOS behavior or numerical bound is weakened. The final auditor cleanup only removes unused mutation placeholders; retained native collection is unchanged and no inference is repeated for that cleanup.

The exact read-only observer replay is `test2026/metal/experiments/mistral-prefix-http-survivor-observer.patch`. Apply it after the previous native-policy replay, against its admitted worker source; the evidence binds both worker hashes and verifies byte-identical replay. The actual HTTP bridge, controller, profiles, fixtures, client/server raw events, graceful shutdown records and audit remain frozen with their source/library/model bindings. Only this research record and observer replay enter a local feature checkpoint. All 11 preexisting dirty paths, the preceding 483-file native-policy manifest, earlier 343-file serving and 998-file capture manifests and pinned assets remain unchanged. No main switch or push occurs.

The exercised short-prompt persistent-cache HTTP contract now has direct evidence. It does not establish general prefix-history equivalence across shapes or an end-to-end xMind integration. The last serial uncached serving table remains authoritative; no new TTFT, decode-speed or aggregate-throughput comparison is admitted. Native FP32 KV/attention still differs from frozen reference FP16. Historical strict coding 0/8 is unchanged. The 2x/half-TTFT, coding quality, longer-context/cache profiles, ANE/Gemma, multimodal and original Qwen27B/9B/4B requirements remain open.


## Current compiled serving profile: unchanged coding failures, exact native regression

This follows `68c780fb6`. Evidence is `out/research/mistral7b-local-coder/mistral-current-coding-quality1`. The earlier 0/8 strict score had remained historical while native kernels, memory plans and serving paths changed. This collection reevaluates the original eight tasks on the current qualified library and HTTP scheduler, then compares complete new outputs with saved native/MLX evidence and saved reference HTTP text. No vLLM, SGLang or MLX inference is rerun. Original baseline artifacts and grading rules are unchanged.

The fresh native engine uses B4/C4096, prepared up-to-128 prompt chunks, the existing packed FP16-operand/FP32-accumulator profile, FP32 KV/attention and selected split-score/value32 paths. Prefix retention, bulk grouping and stage instrumentation are disabled. Eight HTTP clients submit together; the native scheduler admits them through at most four leased rows. Frozen prompts contain 71–151 tokens, use greedy generation and retain the original 1024-token budgets and actual checkpoint EOS. The unmodified worker is used; the preceding read-only cache observer is absent. No native library is rebuilt or production selection changed.

| Check | Current native result |
|---|---:|
| Complete requests, all natural EOS | 8/8 |
| Output tokens retained and audited | 3,052 |
| Original strict code-only contracts | **0/8** |
| Separate single-fence functional diagnosis | 1/8, diagnostic only |
| Complete output IDs equal to saved native B1 answers | 8/8 |
| Complete output IDs equal to saved MLX answers | 5/8 |

The current answers reproduce every saved native token and complete text exactly. Strict grading still rejects surrounding prose/examples under the unchanged complete-code contract; extracting one unambiguous Python fence remains a secondary diagnosis, not an alternate pass score. Some extracted implementations fail functional contracts, and ambiguous multiple-block outputs are not stitched together. The evaluator's independent sandbox startup probe passes. A byte-escape display hypothesis is checked using actual AST byte values and a correct big-endian parser; the original binary-frame contract is valid and is not edited. The observed failures are not reclassified as infrastructure errors.

MLX output differences remain the same three tasks: merge intervals, LRU and JSON Pointer diverge after 3, 61 and 140 matching tokens respectively. Earlier fixed-continuation evidence found an FP16 top-logit tie at the earliest difference while native FP32 favored the other token. That remains a limited diagnosis, not proof that all discrepancies are harmless. This campaign does not collect new logits or change tie-breaking/precision to force a match. Exact identity to eight prior native answers proves the exercised regression contract, not general coding quality or complete reference parity.

Saved vLLM-Metal and SGLang-MLX HTTP files lack output token IDs, so new token-ID parity is not claimed for them. Their raw text is compared byte-for-byte: zero complete texts match literally. Six vLLM and five SGLang answers differ from current native text only by one leading ASCII space; those cases are recorded separately without editing raw outputs or replacing the literal comparison. The remaining two/three answers have different bodies. All four historical strict reports—native, standalone MLX, vLLM and SGLang—remain 0/8 with passing sandbox probes. Equal failing scores do not establish useful coder qualification.

An independent collection audit verifies every frozen prompt ID, incremental/final response ID, EOS, output count, native completion and strict report response hash. Native plans never exceed four live rows or reuse a live lease for another job. Prepared affine observations have zero misses. Both transport and worker terminate cleanly under the expected graceful bridge interrupt; the native worker exits zero, all native owner counters are zero and the existing 4 MiB post-close Metal snapshot bound passes. These snapshots are not a measured peak or a general driver-memory-release proof.

Eight corrupted evidence variants are rejected: false quality pass, failed sandbox relabeled as valid, truncated completion, changed output ID, cached uncached-control output, retained owner, false reference-ID parity and a false leading-space count. The collection-integrity audit explicitly retains `quality_gate_passed: false`. The new portable audit helper is `test2026/metal/experiments/audit_mistral_current_quality.py`; pass the absolute study directory as its argument. It checks retained evidence and starts no inference or generated-code execution. The original sandbox evaluator and the single-fence diagnosis remain frozen separately with their actual commands and logs.

A fresh computer-use observation still reports that the Mac is locked, so detailed Instruments counter/shader inspection remains pending. No UI action, credential entry or security change occurs. This is a profiling-step restriction; native quality collection was still possible and completed. Before another kernel/tile change, obtain actual bottleneck evidence rather than repeating unchanged quality generations. The present results do not justify a production coder or a new model/performance claim.

Only this record and the final independent audit helper enter a local feature checkpoint. All 11 preexisting dirty paths, 32 frozen response files, original suite/evaluator, preceding native/cache/capture/serving manifests and pinned assets remain unchanged. No main switch or push occurs. The last serial TTFT/decode/aggregate table remains authoritative. The 2x/half-TTFT, reliable coding, broader context/cache/MM, ANE/Gemma and original Qwen27B/9B/4B requirements remain incomplete; the full goal remains active.


## Original packed pipeline archives and direct row-tile screen

This follows `9428f638b`. Evidence is `out/research/mistral7b-local-coder/mistral-packed-binary-inspection1`. The current coder requalification reproduces the old native answers but does not pass strict quality. This turn returns to performance diagnosis. Installed Xcode27.1/Metal32023 documentation includes a `gpudebug` manual, but `xcrun` cannot locate that executable on this macOS26.6 host; inspection of the Xcode bundle and installed Metal toolchain does not find it. The manual's documented profiling commands are not treated as an available executable. The separate `metal-binary-perf` tool measures AIR container loading, not shader execution. No compiler/toolchain, OS, preference or security setting is installed or changed.

The installed `metal-objdump` advertises AGX targets, making binary inspection worth a concrete attempt. A standalone ObjC++ helper reproduces the admitted native packed library source exactly: the original three shader defines, profile prelude, packed kernels, compiled paired-kernel text and layout guard, in the same order. Metal4/Safe options, default pipeline descriptor and function constants10–13 match the native three K/N/stride specializations. Source comparison is byte-for-byte. No old shader, xModel operation or Garnet runtime file is edited, and no dirty checkout is built. These are reproduced runtime specializations; this helper does not extract an existing pipeline object from the Garnet process.

Six original M128/M256 row-tile pipelines are compiled for gate/up, down and Q/O dimensions and serialized with the public `MTLBinaryArchive` API. The native same-library cooperative ABI guard passes. Each archive reload succeeds with `FailOnBinaryArchiveMiss`; supplying an empty archive instead fails for every specialization. That negative control prevents claiming persistence solely from a driver cache/recompilation fallback. Both row tiles report SIMD width32, a maximum supported threadgroup size1024 and 17,408 static shared bytes. Actual projection dispatches still use256 threads. Maximum supported threads are not measured residency or occupancy.

Official tools identify `air64_v28` and `applegpu_g17g` archive slices and expose complete bindings, static allocations and pipeline descriptors. Direct GPU disassembly fails to initialize, including with the advertised AGX target override. Importantly, `metal-objdump` prints that error yet exits zero; the initial metadata collector incorrectly expected a nonzero exit and stops. The final collector checks the actual failure text instead of mistaking exit zero for usable instructions, reusing successful metadata outputs without recollecting inference. A documented native-translator assembly-input attempt rejects the archive as that input type. Failed logs and earlier helpers remain. No opaque archive/capture internals are guessed into register counts, spills, occupancy or stall causes; those remain unobserved. Fresh computer use still reports a locked Mac, and manual unlock is requested for detailed profiling. Frozen GPU trace documents are never opened or modified by these CLI checks.

While UI access is pending, the validated archives permit a direct empirical comparison of two already-existing row tiles. A separate native helper loads both original pipelines without archive fallback and uses the frozen synthetic input/Q4/FP16 coefficient/control payloads from the preceding three native capture fixtures. Actual rows are512; K/N and nonzero parent offsets match their control. The unchanged native FP32-to-FP16 gather is included in each GPU command span. No new shader, dequantization, K reduction, operand/result precision, model weight or algorithm is introduced. This tests a profile's tile choice, not a new register-layout kernel or an instruction-level bottleneck hypothesis.

Three warmups and nine alternating-order calls per tile/geometrical cell run in one serial process. Full output parents are poisoned before every call. All complete outputs and parent guards match the frozen lookup-free native controls bit-for-bit, and input/weight/coefficient buffers remain unchanged. Poison uploads, full readback/checks and saving occur outside GPU timestamps. Compiler/source/fixture hashes finish before the timed process; no builds, source hashing or other owned inference overlap collection. System conditions and clock frequency are uncontrolled. These component envelopes do not replace native API or HTTP measurements.

| Actual M | Output N | Inner K | M128 tile GPU ms | M256 tile GPU ms | Median paired M128/M256 | Existing selection |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 14336 | 4096 | 6.9031 | 6.4958 | 1.003048 | M256 |
| 512 | 4096 | 14336 | 5.6659 | 5.3817 | 1.045261 | M256 |
| 512 | 4096 | 4096 | 1.5397 | 1.5696 | 0.996826 | M128 |

Above-one paired ratios favor M256. Gate/up's ratio of medians looks larger than its near-flat paired ratio because observations drift. Down favors the already-selected M256; Q/O is approximately flat and retains M128. There is no justified replacement tile, production selector change or full-model/HTTP rerun. Larger archives or source accumulator counts do not establish a register-pressure explanation. The paired distributions and both ratio definitions remain separate in the audit.

Native repeated checks cover276,824,064 logical output observations. The independent audit verifies23,068,672 retained logical output comparisons,135,168 external parent-guard words and72 independently reconstructed FP64 projection points/144 tile comparisons under the unchanged `2e-6 + 4e-6*sum_abs` bound. Maximum bound fraction is0.008778. Twelve corrupted variants are rejected, covering false register claims, archive fallback/empty-cache claims, wrong constants/shared memory/geometry, sample order/time, changed/missing outputs, overwritten guards and false observation counts. The new independent auditor remains distinct from the native recorder; final payloads are actual buffer downloads, not regenerated expected outputs. Only512 actual rows are observed here; the dynamic up-to512 engine bound does not require requests to equal512 or qualify every other count through this screen.

The reusable standalone helpers are `test2026/metal/experiments/archive_packed_shader_profiles.mm`, `screen_archived_packed_tiles.mm` and `audit_archived_packed_tiles.py`. Build against the admitted clean native headers and third-party JSON headers, using the recorded commands. The archive helper takes a new output directory; the screen takes the archive directory, frozen native capture-payload root and a new output directory; the auditor takes this study directory. No framework binary, archive or large output payload enters Git.

Only these helpers and this record enter a local feature checkpoint. All11 preexisting dirty paths, the100-file preceding coder manifest,107-file HTTP cache manifest,483-file native-policy manifest,998-file capture manifest,343-file serving manifest and pinned model/reference assets remain unchanged. No main switch or push occurs. No new TTFT/token speed/aggregate throughput, coding quality, ANE, Gemma or multimodal admission is made. The last serving table, current strict0/8 failure and native FP32 versus reference FP16 distinction remain. The 2x/half-TTFT and complete Mistral/Gemma/ANE/Qwen goals stay open. Detailed counter/shader profiling still needs accessible UI or a supported equivalent tool before attributing a hardware bottleneck or selecting another substantive kernel design.


## Xcode GPU counters and a trained-model long-context capture

This follows `566bf2a6b`. Evidence is `out/research/mistral7b-local-coder/mistral-gpu-capture-inspection2` and `mistral-trained-prefill-capture1`. The user unlocks the Mac, allowing the installed Xcode27.1 GPU debugger to inspect the previously saved captures. All 998 files in the original capture study are verified before use. Each original `.gputrace` is copied into a new study and verified before opening; the originals remain unchanged. The three copies replay and profile strictly serially. No reference inference is rerun, no native library rebuilt, and no performance observation is added to the serving comparison.

The GPU debugger's Shaders table finally provides observed register and spill fields, while its local Export Encoder Counters action provides three raw CSVs. Each CSV contains272 columns and two encoder rows. Repeated column labels are retained by column index rather than overwritten in a dictionary. The matrix encoder's exported counters are:

| Synthetic actual M/N/K | Shader | Allocated registers | Reported spills | Kernel occupancy | GPU Neural Accelerator utilization | Instruction throughput limiter | Last-level cache limiter |
|---|---|---:|---:|---:|---:|---:|---:|
| 512/14336/4096 | half_m256k128 | 229 | 0 bytes | 21.51% | 87.09% | 81.88% | 38.75% |
| 512/4096/14336 | half_m256k128 | 229 | 0 bytes | 19.58% | 82.81% | 76.02% | 33.11% |
| 512/4096/4096 | half_m128k128 | 146 | 0 bytes | 30.61% | 86.41% | 76.32% | 41.44% |

These are Xcode's counter labels and estimates for the isolated captured encoders, not independent measurements of theoretical hardware peak. The GPU Neural Accelerator is not the Core ML/ANE plugin. Register use and low occupancy are now observed, but a causal speedup from reducing registers is unproved. High matrix-engine utilization together with the instruction limiter suggests compute/instruction work is a stronger limit than external memory for these captured projection cases. It does not establish the full-model prefill bottleneck. The earlier original M128/M256 paired screen still does not justify changing the selected tiles. Replay time, performance state and external activity are preserved in screenshots and excluded from serving metrics.

To resolve that scope gap, a separate process-local ObjC++ capture bridge wraps exactly one call to the already-qualified native decoder. It uses the public default-device/MTLCaptureManager API, a fresh absolute output path, explicit integer failure returns, and an owned start/stop guard. A research copy of the qualified worker brackets only invocation16 of ordinary B4/T128 prefill; it changes neither native inference math nor xModel operations. The unchanged qualified library is loaded in two fresh serial engines: uncaptured control first, then capture. Both enable the documented process-local capture environment flag. Prefix retention, bulk grouping and stage-latency probes stay off.

The four exact saved long-context rendered prompts have actual token counts2048/2051/2048/2049. The diagnostic output budget is8, not the serving benchmark's128. An initial preparation assertion incorrectly equates nominal2048 with every actual count and stops before inference; the final preparation admits the frozen actual lengths without changing prompts or token IDs. Another early C ABI probe cannot read an empty string return through the installed XLang ctypes path; the final bridge relies on tested integer returns and stderr, with invalid-path/no-owned-capture negative controls. Neither issue changes xlang3 or Garnet APIs.

The selected call has512 active input tokens at positions1920..2047, covering all32 trained Mistral layers, including ordinary dense projections and MPS attention. Both engines complete all four requests. Complete prompt IDs match the frozen fixture, all32 generated IDs and complete texts agree between control/capture, and their capture-boundary IDs/slots/epochs agree. This is a bounded regression witness, not full KV/logit parity or an independent numerical oracle. All native owners close; post-close Metal allocation snapshots are1507328 and1769472 bytes, below the existing4MiB bound. Capture stderr includes `Context leak detected, CoreAnalytics returned false`; the native owner counters do not explain or dismiss that framework warning.

The original trained document has1639 files and10076852866 logical bytes. Its complete hashes and a separately verified copy are retained before Xcode opens the copy in `mistral-trained-prefill-inspection1`. At this checkpoint the larger replay is still in Xcode's Preparing frame activity; no trained-model counter result or terminal replay failure is claimed. Continue observing that live activity rather than restarting from an observation timeout. The original trained capture is never opened in the UI.

Reusable diagnostic helpers are `test2026/metal/experiments/native_capture_scope.mm`, `mistral-trained-prefill-capture-worker.patch`, `run_mistral_trained_prefill_capture.py`, `audit_mistral_trained_prefill_capture.py` and `analyze_mistral_gpu_counters.py`. Exact binaries, commands, profiles, logs, raw CSVs, captures, screenshot/AX witnesses and failed preparations remain outside Git in the studies. The capture driver needs a newly prepared study directory with its frozen bindings/profiles/fixture; the analysis/audit helpers take an absolute study directory. These are diagnostic tools, not production defaults.

The latest serving TTFT/decode/aggregate table remains authoritative, current strict coding stays0/8, and the native FP32 versus reference FP16 KV/attention difference remains. No new speed, coding quality, ANE, Gemma, multimodal or Qwen qualification is claimed. The2x/half-TTFT and complete Mistral/Gemma/ANE/Qwen requirements remain open. Only these research helpers and this record enter a local feature checkpoint; all11 preexisting dirty paths remain byte-identical, with no main switch or push.


## Contiguous-control softmax prototype

This follows `fba215b71`. Evidence is `out/research/mistral7b-local-coder/mistral-contiguous-softmax1`. The separate trained-model Instruments collection in `mistral-trained-metal-limiters1` completes all four requests and exits zero; every one of its32 generated IDs agrees with the earlier saved control. Its process-tagged Shader Timeline identifies `chunk_mps_softmax` as a substantial sampled-work contributor alongside matrix products. Sample Duration is not elapsed kernel time or a serving cost fraction. Device-wide raw counter export remains active in that separate study at this checkpoint; no final counter attribution or replacement serving table is claimed.

The original MPS softmax checks the same contiguous positions inside every query/head group and writes a selected-live-index array in threadgroup memory. The new standalone kernel assumes an independently established contiguous active prefix per request owner, removes those repeated scans/index writes/reads, and computes the same cache/live selection arithmetically. Its FP32 score scaling, maximum, exponential, striped/SIMD denominator reductions and divisions retain the original sequence. Dynamic threadgroup storage changes from6 to4 bytes per context-bucket entry; the selected-index argument is removed. Existing matrix products, KV update order, output layout, masks, precision and xModel operations are unchanged. This is a diagnostic prototype, not a native runtime integration or default.

A future native selector must use completed, shared, correctly strided/offset, nonpending controls and prove an active prefix followed only by inactive positions. Every active position must be a finite nonnegative integer equal to `first+j`, within the context bound. Gaps, overwrites, unknown/pending controls, unsupported profiles and other unproved cases must retain the original path. The standalone admission witness rejects25 NaN/fractional/overwrite/gap/bounds variants; it does not establish native pending-control, output-alias, row-lease or prefix-cache safety. Those runtime gates remain required. The profile remains an upper bound: actual chunks8,17 and128 are exercised without padding all requests to512 tokens.

Five actual GPU correctness cases cover one/four owners, full/ragged/empty owners, nonzero buffer offsets, larger physical cache pitches, and context buckets128/256/2048/4096. Complete native outputs and parent words match the original kernel bit-for-bit. An independent retained-buffer audit checks208,659,904 word pairs,135,043,712 guard observations,53,152,896 inactive zeros and197,595 FP64 softmax values from the actual saved input buffers. The fixed probability bound is `1e-6 + 2e-6*abs(expected)`; maximum bound fraction is0.024533. Two one-bit output/NaN-guard corruptions are rejected. All read-only input parents remain unchanged under the collector checks. These are primitive checks, not trained-model KV/logit/quality qualification.

Two fresh serial component processes run opposite initial orders, three warmups per variant and nine alternating-order measured pairs per cell. All original/candidate pipelines are prepared before the first measurement. Mutable scores/probabilities are reset before each command-buffer GPU span; complete output-parent/read-only checks occur afterward. Source/fixture hashing and C++ compilation finish before collection. Only the owned CPU `xctrace export` process is verified suspended during each screen and resumed afterward with its original PID; no recording or inference is restarted. External activity and clock state remain uncontrolled.

| Owners / actual chunk / bucket / active counts | Cohort1 median paired control/candidate | Cohort2 median paired control/candidate |
|---|---:|---:|
|1 /8 /128 /8 |1.240x |1.246x |
|4 /17 /256 /17,14,1,0 |1.216x |1.210x |
|4 /128 /2048 /128,125,1,0 |1.731x |1.249x |
|4 /128 /4096 /128,125,1,0 |1.580x |1.570x |
|4 /128 /2048 /128,128,128,128 |1.551x |2.913x |

The4K ragged-prefix cell consistently favors the candidate, with original/candidate median GPU spans6.042/3.819ms and6.022/3.886ms. The2K cells show pronounced order/drift variability, including individual slower candidate pairs; their ratios of medians are not promoted to a2x result. These isolated softmax command-buffer envelopes exclude projections, matrix attention, scheduling, prefill/TTFT and generation. No whole-model speed or causal hardware bottleneck claim follows from them. No further unchanged component repeats are needed before the more relevant native guarded integration and state/lifetime/body tests.

Reusable sources are under `test2026/metal/experiments/mistral_contiguous_softmax/`. `prepare_shader.py CLEAN_SOURCE NEW_STUDY` verifies the pinned original header/ABI and reproduces the exact combined Metal source. The C++ correctness helper takes that shader and a fresh output directory; each screen takes the shader, audited correctness payload directory and a fresh output directory. The two Python auditors take the absolute retained study directory and start no GPU/model work. Exact compile/launch commands, PID-specific study drivers, binaries, retained input/output parents, source hashes and raw samples remain in the evidence directory. The two screen sources differ only in their initial order and result filename. Original/candidate kernels use Metal4/Safe math and128 threads.

Only this record and the standalone experimental helpers enter this local feature checkpoint. All11 preexisting dirty paths and the preceding frozen source/cache/coding/capture/serving evidence remain unchanged. No dirty Garnet tree is built, no existing runtime/kernel selector or API is changed, no vLLM/SGLang inference is rerun, and no main switch/push occurs. Current strict coding stays0/8 and the last serving TTFT/decode/aggregate table remains authoritative. Native integration, complete caching/batching/state gates, 2x/half-TTFT, useful coding, ANE/Gemma and original Qwen27B/9B/4B goals remain open.


## Native contiguous-control softmax: exact state, limited body result

This follows `9b90a3d28`. Evidence is `out/research/mistral7b-local-coder/mistral-contiguous-softmax-native1`. A clean archive of that checkpoint plus the28 hash-verified admitted prefix-policy overlays reproduces the selected native source;25,539 build-relevant files are verified before applying nine experimental files. The dirty worktree is never built. Existing shader literals and xModel operators are unchanged. `GARNET_METAL_MPS_CONTIGUOUS_SOFTMAX` defaults OFF and requires prepared MPS context buckets/index32. Optional bucket pipelines compile before leases; completed host controls must prove contiguous active prefixes. Unknown, pending, gapped, overwrite, unsupported and full-request inputs retain generic execution. The read-only native session counter is `chunk_attention_contiguous_softmax_dispatches`.

The first focused test exposes an unguarded new Metal literal during compilation of the original index64 control. The retained revision2 adds the matching index32/bucket shader guards; no arithmetic changes. Both final ON and OFF builds pass50 native tests each, including independent FP64 samples, complete output words, offsets, ragged owners, fallback routing, alias/readiness rejection, abort/recovery, partial submissions and cleanup.

Two fresh engines execute OFF then ON strictly serially with the four exact frozen long prompts and eight output tokens each. All32 generated IDs/full texts per engine match the saved native control. The ON engine takes512 accelerated layer dispatches; OFF takes zero. All260 complete KV/hidden digest pairs match at body calls1,16,18,25; owners close to zero and post-close Metal allocation is1,507,328 bytes per engine. Call17 is first decode for early-ready rows; call18 is the sparse final prefill. An initial auditor misclassified call18 and is corrected without editing or rerunning raw collection. The framework's `Context leak detected, CoreAnalytics returned false` warning remains in both logs with capture disabled; bounded native ownership does not explain or dismiss it. This is bounded regression evidence, not an independent complete-model numerical or logits oracle.

A separate actual-weight body study prepares the same B4/C4096/T128 profile, seeds15 actual chunks and repeats the chunk at1920 with three warmups/nine measured forwards. Eight fresh engines are collected strictly serially in reversed variant orders, with all hashing/builds completed beforehand. Complete32-layer physical KV, whole hidden output and predictions match across variants and cohorts. All72 measured forwards have the expected prepared dispatch counts and stable allocations, and all engines close cleanly.

| Active owners | Cohort | Original body s | New softmax body s | Original/new |
|---|---:|---:|---:|---:|
|1 |0 |0.309 |0.314 |0.982 |
|1 |1 |0.373 |0.358 |1.042 |
|4 |0 |1.331 |1.321 |1.007 |
|4 |1 |1.429 |1.370 |1.043 |

These are warm identical-chunk body spans, excluding embedding/head/hash/HTTP/scheduling. Process/order/clock drift is larger than the small effect; no robust whole-model gain, default promotion, HTTP rerun or2x claim follows. The earlier1.58x isolated4K softmax result is not a model speedup. Source patches and portable saved-result auditors enter a local feature checkpoint only. All11 preexisting dirty paths remain byte-identical; no main switch or push occurs. Cached returns, cache pressure, useful coding quality, ANE/Gemma/multimodal and the original Qwen goals remain open. Strict coding remains0/8; native FP32 attention/KV versus frozen reference FP16 remains a material difference. The latest saved serving table is unchanged and no comparator inference is rerun.


## Within-engine native softmax pairing and completed Shader Timeline evidence

This follows `3667eeb38`. The new study is `out/research/mistral7b-local-coder/mistral-contiguous-softmax-paired1`. Its source is a clean archive of that checkpoint plus the28 verified admitted source overlays and nine verified native-experiment overlays. Four additional files implement a diagnostic selector and its tests. All other build-relevant files, shader bytes and xModel operators remain unchanged. The per-context selector changes only between drained command batches, under the session mutex. Open/partial batches and closed/poisoned sessions reject selection; compiled-out builds reject enabling. No pipeline compilation, buffer allocation or layout change occurs at selection. The compile-time default remains OFF. The diagnostic native API is explicitly named `diagnostic_contiguous_softmax_set_enabled`, with a read-only enabled property.

All50 ON tests and the focused OFF native test pass. Tests check per-context isolation, defaults, exact complete output words across selector changes, invalid open/partial mutations, abort recovery and stable prepared resources. Four fresh engines then run strictly serially, one/four active owners with opposite initial orders. All compilation, source/model hashing and offline profiler export finish before timed collection. Each engine prepares the same B4/C4096/T128 profile, seeds15 actual chunks, warms both routes in three alternating pairs, then collects nine alternating pairs of the same late chunk at1920. The selector is set before each timer. All72 measured body forwards have expected native counters and fixed allocation. All520 complete physical32-layer KV/hidden digest comparisons at pre/post measurement boundaries match the earlier frozen original-kernel controls; predictions agree and every engine closes to zero native owners with1,507,328 Metal bytes. Closed-session selector rejection is observed through XLang after close.

| Active owners | Cohort | Original median body s | Candidate median body s | Median paired original/new | Faster pairs |
|---|---:|---:|---:|---:|---:|
|1 |0 |0.311 |0.308 |1.021 |7/9 |
|1 |1 |0.388 |0.373 |1.035 |7/9 |
|4 |0 |1.368 |1.329 |1.018 |6/9 |
|4 |1 |1.404 |1.378 |1.027 |7/9 |

A median of paired ratios differs from a ratio of medians; the ratio column uses paired observations. Every cohort includes individual slower candidate pairs. The small descriptive body effect remains far from2x and is not a full-prefill, TTFT, decode or HTTP result. It does not justify default promotion without the remaining cache/serving/quality gates. No comparator inference is rerun. The latest saved serving comparison and strict coding0/8 remain unchanged.

The separate trained Instruments study is also resolved past its old live-export checkpoint. Recording and original detailed table export exit0. The broad raw-plus-derived counter export is intentionally cancelled to stop redundant serialization of88 counter series beyond26GiB; its partial file is absent on current inspection. The original trained `.trace`, completed first table export and previously extracted complete Shader Timeline table remain. A new narrow shader-only export from the original preserved trace exits0. Its2906 rows match the retained complete table exactly. A portable parser validates public XML schema/ID references, explicit target PID5973 and the `Sample Duration` label, rejects four corruptions, and reproduces942,702,088ns of summed sampled work. The raw shader-only export and all failure/cancellation records are retained. This is not a rerun of model inference or recording.

Largest families by summed Sample Duration are MPS float matrix products20.65%, packed half projections17.34%, affine MPP14.34% and chunk softmax13.58%. These are shares of sampled work, not elapsed kernel cost fractions, theoretical hardware utilization or a serving-speed explanation. No full device-counter aggregate or private attribution of global counters is admitted. The record supports continued attention and matrix-kernel investigation, not an attention-only2x guarantee. `analyze_mistral_shader_timeline.py STUDY` independently verifies the narrow export against the retained table without GPU work. All task-owned inference/build/profiler processes are terminal; old continuation files remain historical and new continuation records supersede their live-export observations. Original captures are never UI-opened; the preexisting Xcode project and protected WorldSense processes remain untouched. All11 original dirty paths and frozen studies remain preserved, no main switch/push occurs, and the complete Mistral/Gemma/ANE/Qwen goals remain active.


## Read-only merged-KV attention: fewer products, no long-prefix gain

This follows `802a72464`. Evidence is `out/research/mistral7b-local-coder/mistral-merged-kv-attention1`. The preceding exact softmax work produces only small body effects, so this new standalone experiment tests a larger attention transformation. A native C++/ObjC++ helper preserves the original MPS shader literal/ABI and four-product matrix composition. A second control uses the already-tested exact contiguous softmax. The candidate materializes each owner's old prefix and live K/V into temporary read-only operands, then performs one QK product, causal FP32 softmax and one PV product. Query packing and materialization stay inside the command-buffer envelopes. Original KV and every input parent remain unchanged; no dirty tree is built and no xModel operator/runtime selector is edited. Only completed contiguous-prefix controls are admitted. No in-place mutation of the generic attention inputs is permitted.

Seven GPU correctness cases cover B1/B4, actual chunks8/17/128, physical context128/256/4096 with selected extents128/256/2048/4096, gating, unequal prefixes, ragged/empty owners, nonzero buffer offsets and padded input/output rows. The initial helper exits zero. A separate capture-only revision adds full merged-K/V parent payloads for independent reconstruction; all math, controls and dispatches remain unchanged. Both compilations and both untimed collections are retained. The final independent NumPy audit verifies144,966,080 complete materialized parent words including zeroed futures and untouched tails/guards, complete output bounds, and28,672 FP64 expected values for each of the three variants. The original `1e-5 + 2e-4*abs(expected)` bound is unchanged; maximum candidate fraction is0.002560. The original and contiguous-softmax outputs match bit-for-bit. The merged candidate can change reduction grouping and output bits. Twenty-three output guard/nonfinite/inactive/truncation corruptions are rejected. These are operator checks, not complete-model KV/logits or coding qualification.

Two fresh processes run strictly serially with opposite rotating variant orders. Every pipeline/MPS shape is warmed before the first timing; all source/payload hashing and builds finish before collection. Nine three-variant rounds over seven cases yield378 GPU command-buffer observations. All timed output parents match the saved corresponding variant, and the final full read-only input/guard checks pass. The samples include query packing and candidate copy work. Ragged cases use full-row MPS products, so they are not measurements of native selected active-span execution. The B4/C4096 candidate requires134,217,984 additional physical K/V bytes including guards, approximately128MiB.

| Case, all four owners active, chunk128 | Cohort | Original4-product GPU ms | Contiguous4-product GPU ms | Merged2-product GPU ms | Median paired contiguous/merged |
|---|---:|---:|---:|---:|---:|
|1920 prefix,2K extent |0 |10.721 |9.813 |8.565 |0.974 |
|1920 prefix,2K extent |1 |9.869 |8.036 |8.799 |0.936 |
|3840 prefix,4K extent |0 |21.400 |17.442 |18.872 |0.849 |
|3840 prefix,4K extent |1 |21.459 |17.590 |19.938 |0.905 |
|0 prefix,128 extent |0 |2.039 |1.343 |0.921 |1.507 |
|0 prefix,128 extent |1 |1.929 |1.418 |1.134 |1.160 |

A median of paired ratios differs from a ratio of medians. The2K cells vary substantially; the4K full-owner candidate is slower than the improved four-product control in both cohorts, with0/9 and2/9 faster pairs. Cold-prefix cells favor the candidate but remain noisy and are not a complete-prefill or body gain. This rejects broad native integration of the materializing variant. Fewer products alone are insufficient once staging, shapes and dispatches change. No causal claim that copying alone explains the regression follows. A future no-copy graph-level rewrite must prove immutable logical cache aliases, liveness and original commit semantics before changing writer order; generic read-only attention semantics must remain intact. No unchanged component repeats or full-model/HTTP runs of this rejected variant are needed.

Reusable helpers are under `test2026/metal/experiments/mistral_merged_kv_attention/`. Portable shader preparation reconstructs the pinned original controls and appends the candidate. Both raw C++ order variants, the pinned native policy copy and independent saved-data auditors are retained. This is a local feature checkpoint only; all11 original dirty paths, frozen model/baseline/caching/coding/profiling studies and the latest serving table remain unchanged. There is no main switch/push, comparator rerun, new TTFT/decode/throughput, production default, useful coding, ANE/Gemma/multimodal or Qwen admission. Strict coding remains0/8 and the2x/half-TTFT goals remain unmet.


## Native compiler-proved KV commit hoist with unchanged attention math

This follows `7e5a16320`. Evidence is `out/research/mistral7b-local-coder/mistral-kv-commit-hoist1`. The rejected materializing variant motivates a no-copy graph rewrite, but moving KV writers must be proved before changing attention arithmetic. This stage implements only that semantic foundation. A clean archive of7e5a16320 plus28 verified admitted prefix-policy overlays reproduces25,553 build-relevant source files before11 candidate files are applied. The dirty worktree is never built. `GARNET_METAL_CHUNK_KV_COMMIT_HOIST` defaults OFF and requires prepared private in-place KV and MPS attention. Original shader literals, four-product/scalar attention math, precision, xModel operators, GPU allocations/layouts and batching APIs are unchanged.

At compile time, an immutable logical-root descriptor proves one bounded32/8-head D128/C4096 ungated attention and its two original KV writers, matching cache/control inputs and returned-state groups, unique prior Q/K/V producers and legal order. It rejects intervening/late consumers of old cache, premature reads of the new state, extra cache writers, crossed token/control writes and aliases. It is constructed before any scratch rebinding. At execution, both original writer outputs must be the exact private cache views, K/V storage must be disjoint from attention data/output, and completed host controls must prove contiguous active prefixes. Unknown/pending, gapped/overwrite, unsupported, standalone, verification and per-op diagnostic calls keep the original order. A proved call encodes the same two original KV updates immediately before unchanged attention, then skips their old positions. Native read-only `chunk_kv_hoisted_commits` resets with the existing command counters. No public scheduler/model-call API changes.

Both ON and OFF pass50 native tests. The new logical policy test accepts prior old-state readers and legal updated-state readers while rejecting40 invalid producer/consumer/alias/order/shape layouts. Native tests check exact-view identity, readiness/pending controls, gaps, overwrites, nonfinite flags, empty owners, incompatible shapes and clean owner shutdown. Existing native regressions remain unchanged. Two fresh native engines then run OFF followed by ON strictly serially, four exact frozen long prompts and eight outputs each. ON observes1,088 hoisted cache commits over17 prefill body calls; OFF and all decode calls observe zero. The sparse final prefill also exercises the original scalar attention fallback after the first decode. All32 generated IDs and full texts per engine match the saved original control. All260 complete physical32-layer KV/four-row hidden-bank digest pairs agree at calls1,16,18,25, and the new OFF state also agrees with the earlier qualified state. Native owners close to zero; post-close Metal bytes are1,507,328 OFF/1,900,544 ON, both below4MiB. The existing CoreAnalytics context-leak warnings remain in the raw logs; bounded native owners do not explain them.

The next correctness campaign reuses the frozen nine-request HTTP protocol in two more fresh serial hoist-ON engines, uncached then cached. The qualified read-only survivor observer is unchanged except for additional native route events. A/B/C/D initially occupy four rows, E queues before A finishes, then E and returning exact/appended/late-edited/early-edited A reuse A's row at increasing epochs while B/C/D continue. Cached reuse remains211/128/128/0 tokens. All752 generated IDs and full text per engine agree with the corresponding frozen original-policy outputs; all3,900 survivor hash observations remain unchanged. Native reordering is observed in12 uncached prefill body calls/768 commits and8 cached calls/512 commits. All18 streams finish, both native workers exit0 with zero owners and2,162,688 post-close Metal bytes, and the unchanged HTTP auditor rejects its seven corruption cases. Existing HTTP wrapper SIGINT shutdown after graceful native close is retained explicitly. No reference inference is rerun and all diagnostic timings are excluded from serving metrics.

This is exercised semantic/lifecycle evidence, not an independent full-model numerical/logit oracle or proof of all failure/nonfinite-model behavior. Actual reordered-model exception/partial-submission recovery, cache pressure and remaining numeric/quality/serving gates stay open. The two-product implementation is deliberately absent at this checkpoint: first strengthen those failure/state gates, then replace the unchanged four-product calculation under an explicit profile identity and independent numerical/model qualification. No speed or default promotion is inferred from moving writers. Portable source preparation, the11-file native patch, route-only worker patches and independent saved-result auditors are under `test2026/metal/experiments/mistral_kv_commit_hoist/`. All11 original dirty paths, frozen assets/studies and authoritative serving table remain unchanged; no main switch/push occurs. Strict coding stays0/8; the full Mistral/Gemma/ANE/Qwen and2x/half-TTFT goals remain open.


## Actual-model KV hoist fault isolation, full reset and immutable prefix recovery

This follows `16cb55b32`. Evidence is `out/research/mistral7b-local-coder/mistral-kv-hoist-failure1`. The preceding semantic stage left actual reordered-model failure/partial-submit recovery open. This stage adds a default-OFF, test-only `GARNET_METAL_KV_HOIST_TEST_FAULTS` option requiring the hoist experiment. A clean archive of16cb55b32 plus the28 admitted overlays and11 verified hoist source files reproduces25,557 build-relevant files before seven test-only changes. Original shader literals, xModel operators, four-product/scalar arithmetic, layouts and GPU allocations remain unchanged. Both fault-enabled and disabled configurations pass50 native tests each. No dirty worktree is built and no comparator inference is rerun.

A test-only single-shot hook executes immediately after the first eligible layer's two hoisted KV commits. Mode1 throws after encoding without submission. Mode2 flushes the command and then throws; native abort waits for it before discarding the continuation. A retained native witness records encoded/submitted/drained/completed work and pending-command state. Arming rejects invalid modes, failed/closed sessions and undrained batches; reset disarms an unconsumed request. The witness is readable after failure without exposing KV values. Its GPU span is diagnostic completion evidence, not kernel/serving performance.

The actual32-layer Mistral engine prepares B4/C4096/T128, establishes healthy128/256-token controls and retains four128-token snapshots. The first worker reaches the unsubmitted fault but stops because its observer wrongly expects a tracked-count exception; the native API returns the invalid-1 sentinel. The corrected observer explicitly checks both tracked-token/prefill-1 results, preserves partial evidence on failure and completes in a new fresh process. No native code, guard or math changes for that correction. The first failed process/logs remain. A separate saved-data auditor assumption about an explicit `head_dim` is corrected to derive128 from4096/32; no raw results or inference are rerun for that correction.

| Injected site | Encoded KV commits | Submitted | Drained/completed by abort | Follow-up API rejections | Exact restored state and continuation |
|---|---:|---:|---:|---:|---|
|Before submission |2 |0 |0/0 |11 |Pass |
|After partial submission |2 |1 |1/1 |11 |Pass |

Both modes consume the single-shot arm, clear pending outputs/command/submissions and reset ordinary dispatch counters. Failed sessions reject tracked/leased generation, prediction, state/hidden reads, row reset/release, prefix capture/restore and rearming; both metadata counters return-1. Four retained prefix entries and their metadata remain unchanged. Full reset zeros every physical KV parent and the hidden bank. Old tracked/leased epochs and uninitialized predictions reject; new leases have greater epochs. All four cached prefixes restore exactly and the next128-token continuation matches healthy complete state and predictions. Reset with a separately armed but unfired fault also permits a normal exact continuation.

The final independent audit binds all seven changed sources/library/worker/fixture and both50-test configurations. The healthy128-token state matches65 prior qualified state digests. Five full resets produce325 complete zero-state hash checks against independently computed IEEE-F32-zero byte references; five prefix restores and three recovered continuations produce520 exact digest comparisons. Fifteen stale-epoch/uninitialized-prediction rejections and seven corrupted-record checks pass. The final process exits0, all native owners close to zero and post-close Metal bytes are1,900,544. Existing CoreAnalytics context-leak warnings remain recorded and are not explained by these ownership observations.

This proves the exercised software-fault/reset contract only. Other hardware failures, cancellation, serving failure propagation, cache pressure and general nonfinite arithmetic remain open. No two-product kernel or new TTFT/decode/throughput/coding admission is made. Reusable test-only patches, worker, auditor and verified source preparation are under `test2026/metal/experiments/mistral_kv_hoist_failure/`. All11 original dirty paths, fixed assets/baselines and frozen studies remain unchanged, with no main switch/push. The fault option stays default OFF; the authoritative serving table and strict coding0/8 remain unchanged. The next larger performance stage is a separately identified zero-copy two-product profile with independent numerical/model/cache/quality/serving gates. The full Mistral/Gemma/ANE/Qwen and2x/half-TTFT goals remain active.

An additional standalone native context test uses the exact fault-enabled CMake definitions/static library. Invalid/open/partly-submitted arm changes reject while preserving the previous arm; explicit disarm and context reuse after abort succeed with zero native owners. It changes no audited native source or model inputs, and the final saved-result auditor requires this independent guard check.

## Zero-copy committed-KV attention: native/model gates pass, modest body gain

This follows `dbc893c14`. Evidence is `out/research/mistral7b-local-coder/mistral-committed-kv-attention1`. A clean archive plus the verified admitted and production-form hoist overlays reproduces 25,557 build-relevant files before nine candidate files are applied. Test-only fault injection is excluded. The dirty checkout is never built. `GARNET_METAL_COMMITTED_KV_MPS` defaults OFF and requires prepared MPS/index32/context buckets, active owners and compiler-proved hoisting. Generic public attention ABI/semantics and xModel operations remain unchanged.

The private compiler entry uses the already-committed cache, packs Q only and computes one QK/one PV product per owner/KV head. It skips both live-only products and directly unpacks the result. Existing prepared plans and GPU buffers are reused; optional pipelines compile before request leases. Sparse scalar tails retain the original path. New read-only `chunk_committed_kv_attention_dispatches` resets on begin/abort. FP32 operands/cache/softmax remain, but reduction grouping changes. Backend identity `committed_kv_mps_fp32_v1` and native prefix suffix `|committed-mps-fp32-v1` prevent cross-policy snapshot reuse. This is a new numerical policy, not an exact rewrite.

ON/OFF pass 51/50 native tests. Six primitive shapes cover full/ragged/gated B1/B4 owners, 2K/4K extents and scalar fallback. All 7,127,040 output words meet the fixed control bound; 864 independent FP64 reference values for both original and candidate meet the unchanged `1e-5+2e-4*abs(reference)` bound, maximum fraction 0.0012557. Full physical cache parents match expected commits, readonly inputs remain exact, invalid/gapped/overwrite/pending controls reject, and partial submission/abort/storage/owner checks pass. These finite synthetic cases do not establish general nonfinite or complete-model numerical behavior.

Two fresh engines run OFF then ON strictly serially on four frozen long prompts and eight outputs each. Both reproduce all 32 saved native token IDs/full texts over 25 body calls. Both observe 1,088 hoisted commits; ON takes 512 committed-attention dispatches, reducing observed MPS products from 65,536 to 32,768. Prepared plans/scratch are unchanged. The OFF control's complete KV/hidden state matches the prior qualified hoist engine. ON matches the first-chunk state but later reduction differences propagate: the four observed final-hidden banks have maximum absolute errors 0/0.006295/0.004563/0.001323. These are descriptive comparisons, not an independent full-model numerical/logit oracle. Both native processes exit 0, owners close to zero and post-close Metal allocation is 1,900,544 bytes each. Existing CoreAnalytics context-leak warnings remain. The independent saved-result auditor rejects eight deliberate corruptions.

A separate body experiment uses eight fresh strictly serial engines in two opposite-order cohorts. Each seeds 15 actual trained-weight prefix chunks, warms the offset-1920 chunk three times, then measures nine identical 32-layer body calls. Embeddings, head, source hashing and complete-state snapshots stay outside measured calls; no builds or comparator inference overlap collection. The profile is B4/C4096/T128 with one or four active owners and the admitted ordinary packed/split/value32 policies.

| Active owners | Cohort | Original body s | Two-product body s | Original / candidate |
|---|---|---|---|---|
| 1 | 0 | 0.27432 | 0.26500 | 1.0352x |
| 1 | 1 | 0.30705 | 0.29416 | 1.0438x |
| 4 | 0 | 1.14561 | 1.11947 | 1.0233x |
| 4 | 1 | 1.19468 | 1.13596 | 1.0517x |

These are ratios of process medians, not within-engine paired estimates. Next-token predictions agree across policies; complete KV/hidden hashes and values are exact across the two cohorts within each policy. The complete late-chunk hidden tensors differ across policies, maximum absolute error 0.060588 (RMS 0.000685 for one owner, 0.001370 for four). Halving product count does not halve model runtime; these observations show only a modest body effect under uncontrolled system conditions. Retain the option OFF. No complete-prefill/HTTP TTFT, decode, aggregate-throughput or new quality result is admitted, and no unchanged reference benchmark is rerun.

Portable preparation, native patch, source/library bindings, worker delta and saved-data auditors are under `test2026/metal/experiments/mistral_committed_kv_attention/`. Clean preparation replay verifies the exact nine candidate files. All 11 original dirty paths, model assets and frozen prior studies remain unchanged. No main switch/push occurs. Cached return/edited-prefix and full-model numerical/logit/coding, exception/cancellation, serving and cache-pressure gates remain open for this new numerical policy. The authoritative serving comparison and historical strict coding 0/8 stay unchanged. The 2x/half-TTFT and Mistral/Gemma/ANE/Qwen goals remain active; the small body result does not justify repeating a full serving campaign alone.

## Strict FP32 MPP online attention: register map qualified, dense long gain insufficient

This follows `04a87c28e`. Evidence is `out/research/mistral7b-local-coder/mistral-strict-mpp-attention1`. The no-copy two-product result motivated a direct test of the pinned vLLM-Metal M5 matrix/online-softmax approach. Its source uses relaxed descriptors and a matching register layout. This new standalone implementation uses public MPP `float,float,float` cooperative tensors, strict `relaxed_precision=false` and Safe Metal 4 math for both QK and PV. It keeps the existing FP32 KV and probability policy. No Garnet runtime, xModel operator, tokenizer, serving API or default is changed; no dirty tree or reference engine is built/run.

The first prototype uses four 16-query SIMD groups per threadgroup and public cooperative coordinates. Online softmax uses per-group threadgroup score storage, without global score/probability intermediates or query packing. The original four-product MPS control reads old cache plus live K/V. A second, no-copy two-product MPS control uses the same immutable committed cache as the candidate, packing Q only. CPU preparation of this cache is outside both timed paths; MPS pack/softmax/unpack remain inside. Complete MPS control payloads match the prior qualified original/merged controls bit-for-bit, on identical input bytes. This is a synthetic finite operator experiment, not trained-model profiling.

The first runtime Metal compile rejects an array of dynamic cooperative tensors and a reserved type-name variable. Four explicit compiler-created destinations and the renamed variable repair these syntax issues; failed logs remain. Seven cases cover B1/B4, T8/17/128, 2K/4K extents, ragged/empty owners, gating, actual nonzero input/output offsets, row padding and guards. Complete candidate outputs meet the original control bound, and 28,672 independent FP64 values per variant meet unchanged `1e-5+2e-4*abs(reference)`. Full input and committed-cache parents remain unchanged. The first two opposite-order process cohorts show a broad long-context regression against no-copy MPS: paired ratios 0.605/0.610 for full B4/2K and 0.648/0.651 for full B4/4K.

A targeted follow-up keeps scores in registers and performs online row reductions with SIMD XOR-1/XOR-8 shuffles, directly referring to the pinned vLLM design while retaining strict FP32. The first same-library guard correctly refuses the proposed relaxed-style raw map before any attention dispatch. A separate coordinate-only GPU probe reports flags14 in all32 lanes, intact parents and no attention launch. Capacities alone are unchanged, so checking only element counts would have missed this incompatibility. The strict map has two-column fragments with eight-column strides. Every observed valid QK-left/result and PV-left/right/result coordinate is independently reconstructed. A new exact map guard checks score, P-left and output layouts before the adapted kernel runs; it is not bypassed. A separate atomic coverage probe verifies exactly one owner for all2,560 logical elements of the six cooperative operands, with no out-of-range writes or parent damage.

The adapted register kernel passes the same seven complete-output/guard and FP64 tests at the original bound. Two additional fresh processes use opposite rotating three-variant orders. Each prototype warms every shape/variant three times before its first measurement and then collects nine triples per cell:756 GPU command observations across four serial processes. All measured complete outputs agree with the corresponding retained variant. Source hashing and C++ compilation finish before timed collection. The independent auditor verifies36,933,696 preserved MPS control parent-word pairs and rejects ten metadata corruptions, separately from the numerical auditors'23 output/guard negatives per prototype.

| Register candidate case | Cohort | No-copy MPS GPU ms | Strict register GPU ms | Median paired MPS / candidate |
|---|---|---|---|---|
| B4/T128 cold prefix | 0 | 0.7679 | 0.4670 | 1.988x |
| B4/T128 cold prefix | 1 | 0.9497 | 0.4632 | 1.997x |
| B4/T128 full2K | 0 | 7.1871 | 7.3726 | 0.979x |
| B4/T128 full2K | 1 | 7.1384 | 7.6939 | 0.937x |
| B4/T128 full4K | 0 | 15.0044 | 14.5513 | 1.030x |
| B4/T128 full4K | 1 | 15.0084 | 14.8001 | 1.019x |
| B4/T128 ragged/gated4K | 0 | 13.3947 | 8.2191 | 1.642x |
| B4/T128 ragged/gated4K | 1 | 13.6966 | 8.2501 | 1.674x |

Median paired ratios differ from ratios of medians. The register design removes most of the first prototype's broad regression, but dense long-prefill gain is absent or small. Cold/ragged component gains do not establish native serving benefit. This standalone MPS control executes full physical owners/rows, whereas native already skips empty-owner spans; native context buckets also use a shared maximum extent. The shorter useful prefix/query work inside ragged owners needs a native-matched control before any benefit is inferred. No broad native selector or full-model/HTTP campaign is justified by these results alone. Do not replace the fixed homogeneous serving targets with favorable component cases.

The exact qualified source composition, C++ helpers, map probes, collectors and saved-data auditors are under `test2026/metal/experiments/mistral_strict_mpp_attention/`; the clean replay reproduces both qualified shaders and the historical rejected-map probe source. All original11 dirty paths and frozen model/reference/native evidence remain unchanged. Task-owned GPU/compiler jobs are terminal. The existing serving table, strict coding0/8 and numerical/cache/batch/ANE/Gemma/Qwen qualifications remain unchanged; the2x/half-TTFT goal is still active.

## Actual trained attention fixtures: native transport lifetime fixed and FP64 gates pass

This follows `fada9b128`. Evidence is `out/research/mistral7b-local-coder/mistral-trained-attention-fixture1`. Previous MPP component screens used small random operands. The new test-only C++/Objective-C++ path captures real Mistral attention inputs/results from the admitted original four-product hoist engine, without changing neural math, precision, shader strings, xModel operations or row scheduling. Clean source preparation delegates the hash-verified admitted/hoist overlays;25,688 other build/xModel files are checked before the eight-file candidate is built. The profile remains B4/C4096/T128 with the original packed/split-score/value32 policies. Full-model coding, ANE/Gemma and Qwen qualifications are not inferred.

`GARNET_METAL_ATTENTION_FIXTURE_CAPTURE` defaults OFF and is qualified only with the complete selected Mistral/diagnostic layer-bulk/dense512 build recipe. The test API arms a fresh absolute output directory on a drained session/context and preallocates490,749,312 requested buffer bytes, below512MiB. One32-layer frame must have full contiguous slots1920..2047 and four active owners. Ordered read-only blits capture Q/K/V, private K/V cache, slots/active, gate and original attention output at layers0/15/31. Finish accepts only drained successful complete work, verifies guards/tails/finite data and exports27 guarded FP32 tensors plus metadata. Abort fails capture; reset/cancel discards it. No Python inference or Rust runtime is added, and no new numerical prefix domain is needed for this read-only observer.

The first implementation passes51 enabled and50 disabled native tests and collects the model fixture, but the independent close audit correctly rejects493,109,248 Metal bytes against the unchanged4MiB limit despite zero tracked owners. Capture blit encoders were created outside scoped autorelease pools and retained491,208,704 buffer bytes. First source/library/logs/raw tensors and the failed auditor remain. Revision2 scopes native capture operations in autorelease pools and strengthens the native test to check actual device allocation after partial submission/abort/cancel, successful finish and full context teardown. It passes the same51/50 suites; the native test verifies122,687,328 exported parent words, exact logical copies, nonzero source offsets, source/output/capture guards, full input immutability and invalid/open/duplicate/pending-work/abort/cancel/finish controls. Requested buffer length and observed device allocation are distinct quantities. A separate native test destroys armed and exception-failed plans without explicit cancellation, including an original unprepared-attention error after input blits are encoded. Export rejects failed work; closed device allocations475,136/1,523,712 bytes remain below4MiB. This is a software failure/destructor test, not a real driver fault or new full-model failure qualification.

A fresh revision2 engine runs the same four frozen long prompts with eight outputs each, strictly after native tests terminate. All32 IDs/full texts and260 complete physical KV/final-hidden digest pairs at calls1/16/18/25 match the original qualified engine. Captured cache bytes at layers0/15/31 exactly reproduce six corresponding state16 digests. All27 payloads match the rejected first capture bit-for-bit, while post-close Metal allocation now returns to1,900,544 bytes and tracked owners to zero. The original CoreAnalytics context-leak warnings remain; bounded device/owner observations do not explain those framework messages. Capture timing is excluded from performance admission.

The captured native attention results pass18,432 independent FP64 values at the existing `1e-5+2e-4*abs(reference)` bound. The largest bound fractions at layers0/15/31 are0.007435/0.034219/0.038745. Actual Q reaches approximately-14 and absolute K approximately19.5, materially larger than the prior random±0.3 fixtures. Full current-live K/V words agree with the captured committed cache; future cache regions are zero. The source model/library/worker, actual controls, layer order, guarded tensor layouts and original full-state proof are bound to the export; six corrupted model records reject.

A separate untimed native operator replay preserves all116,395,008 input words while adding guarded row padding. It runs original four-product MPS, no-copy two-product MPS and the guarded strict FP32 MPP register path on these actual operands, with the captured native attention as a fourth gold variant. All18,874,368 complete logical replay/gold comparisons meet the original control bound. The oracle checks18,432 FP64 values per variant,73,728 variant comparisons, with no tolerance or precision switch. Maximum fractions of the complete native-gold bound are below0.009 for original MPS,0.023 for two-product MPS and0.161 for the register path. Native-gold and replayed MPS are not asserted bit-identical: row layouts/descriptors differ. These are bounded trained-operator checks, not a complete-model/logit/quality oracle for either new numerical policy. The fixture/replay data and binary/source hashes are independently linked, and no unchanged timing or comparator campaign is repeated.

Portable native source preparation, the eight-file revision2 patch, isolated autorelease correction, capture/replay auditors and the trained replay helper are under `test2026/metal/experiments/mistral_trained_attention_fixture/`. The selected implementation remains the original runtime; no main switch or push occurs. All original11 dirty paths and frozen model/reference/native studies remain unchanged. The existing serving table and strict coding0/8 stay authoritative. Further kernels must pass these actual trained operands before any model/quality/cache/batch/failure/serving promotion; the2x/half-TTFT and full Mistral/Gemma/ANE/Qwen goals remain active.

## Compensated half MPP attention: accuracy passes, trained component performance rejects

This follows `4041b0c35`. Evidence is `out/research/mistral7b-local-coder/mistral-compensated-mpp-attention1`. The new standalone C++/Objective-C++/Metal experiment splits Q/K/V and online-softmax probabilities into high/low half pairs, using separate FP32 main/correction accumulators, FP32 inputs/cache/softmax/output and strict descriptors. It tests three products versus four products per QK/PV tile; the latter includes low×low. This differs from the prior three-component BF16 affine-verification experiment. Neither the native runtime/default nor any xModel operation changes. The dirty checkout is not built.

Q/K/V scale16 and probability scale32768 are exact binary factors. The initial probability-scale1024 draft is retained but never compiled/collected; the stronger scale is fixed before collection. Explicit prototype eligibility requires finite Q/K/V/cache magnitudes up to2048. General extreme/nonfinite inputs and a runtime fallback are unqualified. The same-library actual half/half/float capacity, atomic coordinate-coverage and score/P/output raw-map probes pass before attention launches: capacities `[8,16,16,8,16,16]` per lane, exactly one owner per2,560 logical coordinates, zero invalids, intact guards and32 zero raw-map flags. Apple's official MSL specification§8.6 is retained with source hash: float-to-half conversion preserves generated denormals. This conversion rule is not generalized into a claim about all tensor arithmetic; actual tiny-probability GPU cases remain required.

Ten synthetic cells retain the earlier cold/2K/4K, B1/B4, gated/ragged/empty-owner, offset/padding cases and add half-midpoint/tiny-probability adversaries. Four products pass complete output/control checks and33,280 independent FP64 values at unchanged `1e-5+2e-4*abs(reference)`. Three products fail midpoint case7, reaching1.873028 of the fixed FP64 bound and1.693065 of the original-control bound. Both failed and passing outputs remain. Four-product maximum FP64 bound fraction is0.485056 on the long tiny-P case. No tolerance is relaxed or failing variant relabeled.

The three trained fixtures are bit-preserving copies of the qualified Mistral layers0/15/31, B4/T128/C4096, offset1920. All four replay variants pass25,165,824 complete logical native-gold comparisons and18,432 FP64 values each; the captured native gold is checked as a fifth FP64 variant. The four-product maximum native-gold bound fraction is0.345794 and FP64 fraction0.024412. These are bounded attention-operator results, not a new complete-model/logit/quality policy. Original four-product and no-copy two-product MPS output parents remain exact across12,595,392 words against the preceding trained replay. All retained input/cache parents, output guards and numerical policies are bound to the original capture and source binaries.

Only qualified variants0/1/3 enter timing. A separate host excludes rejected three-product variant2 even though it passes the trained cells. Two fresh processes run strictly serially with opposite rotating orders. Each prepares all three layer cells/pipelines before three warmups per case/variant and nine measured triples:162 measured attention commands and54 warmups total. Every timed output must equal its retained variant exactly; full input/immutable-cache/guard checks occur outside the GPU spans. CPU preparation of the committed cache is outside both candidate and no-copy MPS timing. All builds and source hashing finish before measurement. The saved-data auditor verifies metadata/order/precision/maps and rejects14 deliberate corruptions.

| Trained layer | Cohort | Original four-MPS GPU ms | No-copy two-MPS GPU ms | Compensated four-product GPU ms | Median paired two-MPS / candidate |
|---|---|---|---|---|---|
| 0 | 1 | 9.629 | 7.195 | 21.902 | 0.3335x |
| 15 | 1 | 8.777 | 7.837 | 22.230 | 0.3496x |
| 31 | 1 | 8.881 | 9.400 | 22.088 | 0.4150x |
| 0 | 2 | 9.807 | 9.528 | 22.198 | 0.4169x |
| 15 | 2 | 8.693 | 7.205 | 22.062 | 0.3413x |
| 31 | 2 | 8.650 | 8.410 | 21.916 | 0.3877x |

Median paired ratios differ from ratios of medians. Every trained comparison regresses, so this candidate is rejected for native integration. Extra correction products are explicit in the algorithm, but the timings do not isolate hardware occupancy, stalls or each component's cost. No synthetic timing campaign or full-model/HTTP rerun is justified for this unchanged candidate. The result does not imply that plain FP16 attention, a different versioned numerical policy, is qualified by these FP32-contract checks.

Hash-pinned source preparation, both native hosts, numeric/binding/timing auditors and serial collector are under `test2026/metal/experiments/mistral_compensated_mpp_attention/`. The portable numeric replay writes separate audit2 files so original audit1 evidence remains intact. Clean preparation reproduces the exact shader and hosts without building a dirty tree. Raw tensors, failed three-product outputs, original drafts, commands/statuses, both cohorts and official-source bindings remain local. No vLLM/SGLang inference or frozen reference artifact changes. All11 preexisting dirty paths remain byte-identical; no main switch/push occurs. The authoritative serving table, strict coding0/8 and selected runtime remain unchanged. The2x/half-TTFT, useful coding, cache/batch/failure, ANE/Gemma and original Qwen27B/9B/4B qualifications remain incomplete.

## Distinct half-QKV policy: packed component gain survives GPU staging

This follows `380bffb9c`. Evidence is `out/research/mistral7b-local-coder/mistral-fp16-attention-policy1`. The previous compensated full-FP32-contract path is not repeated. A new standalone policy rounds Q/K/V once to half, retaining FP32 accumulation, online softmax, denominator and output. `fp16_qkv_p16_v1` uses one scaled half P component; `fp16_qkv_p22_v1` uses high/low half P with a separate FP32 correction. These are new numerical policies, not exact replacements for FP32 attention or claims of frozen reference bit identity. No native runtime, xModel operation, tokenizer, serving API or default changes.

Before GPU collection, an independent CPU FP64 probe quantifies Q/K/V conversion on the actual trained inputs:7,156 of18,432 sampled outputs exceed the old FP32 allowance, with maximum absolute change0.001012. This does not establish coding-quality degradation, but forbids claiming the old numerical contract. The separately stated half-QKV reference and probability-conversion budgets are fixed before collection. Base coefficients remain `1e-5+2e-4*abs(reference)`; P16 adds relative `2^-11` and scaled additive `2^-40`, P22 `2^-22`/`2^-39`, propagated through probability-weighted absolute V and visible count. Maximum absolute V bounds the complete-output comparison. The existing base remains an empirical float-math allowance, not a universal hardware-roundoff proof. No bound is tuned after observing outputs; old-policy failure counts remain in the audit.

Ten synthetic cells and three actual trained layer0/15/31 cells use the prior controls/offsets/padding/guards. Separate fixtures round only logical Q/K/V/cache values, preserving all gate/control/guard bits. Candidate outputs from original and explicitly rounded inputs must agree bit-for-bit. Both policies pass complete comparisons against rounded-input four-product MPS and51,712 independent FP64 values per candidate. Original MPS outputs preserve31,333,032 exact parent-word pairs from the preceding study. This is operator qualification under the stated new policy, not full-model logits/coding/cache/serving qualification.

Half operands with float storage and strict descriptors alone are slower than MPS. The next path directly refers to the pinned vLLM-Metal prefill source's relaxed descriptor and vectorized register loads, with actual half storage. Same-library guards additionally prove the Q/K/V raw input maps, not only the score/P/result maps; atomic coordinate coverage and parent guards remain. Q conversion occurs inside each candidate GPU command. The complete outputs of the strict and packed candidates are still exact across31,333,032 parent words, but speed changes substantially. The combined storage/load/descriptor changes are measured together; no isolated causal share is assigned to one change.

The final path moves KV conversion inside the GPU command too. It packs from the same committed float cache used by the no-copy MPS control, with zero fill for inactive/future spans. Before every candidate call, entire half Q/K/V scratch parents are poisoned outside timing. Full packed parents must match independent CPU half conversions afterwards. The final staging path preserves610,949,986 float-parent word pairs and270,209,172 half-parent word pairs against the independently qualified preceding path, including synthetic ragged/empty/gated/tail cases. The standalone control still owns float buffers; model-memory savings are not claimed.

| Plain-half implementation | GPU ms, range of trained layer/cohort medians | Median paired two-MPS / candidate, range |
|---|---|---|
| Float storage, strict descriptor | 8.85–9.05 | 0.794–0.803x |
| Packed half KV, Q conversion included | 2.69–2.82 | 2.511–2.655x |
| Q and KV conversion included | 3.54–3.91 | 1.816–2.004x |

Two-product MPS is approximately7.1ms and the original four-product control approximately8.6ms on this B4/T128/2K operator workload. The P22 variant is slower: packed5.60–5.69ms, or6.45–6.61ms with KV conversion. Each implementation has two fresh serial processes in opposite rotating orders, three warmups per layer/variant and nine measured four-variant rounds. The saved-result auditor verifies648 measured command buffers,216 attention warmups and15 deliberately corrupted timing records. Builds/source hashing finish before timing. Every timed output equals its own retained correctness payload; full read-only/packed/guard checks occur outside the GPU spans. Ratios are paired per round and differ from ratios of medians.

The first packed timing launcher accidentally supplies the prior shader. Missing-function validation rejects before any attention dispatch; the failed status and empty directory remain. Corrected cohorts4/5 provide packed measurements;6/7 include GPU KV conversion. Failed samples do not enter medians. Q/KV scratch poison uploads and CPU verification are outside GPU spans; this table does not claim HTTP TTFT, token rate, throughput, hardware utilization or the overall2x/half-TTFT target.

The gain with all GPU staging included justifies a separate opt-in native trial. Native integration must preallocate bounded profile scratch before request leases, use compiler-proved committed KV, identify the new numerical/cache namespace and preserve fallback/failure/resource safety. It must then qualify complete-model/logit/coding behavior, prefix reuse, continuous batching and serving before selection. Final shader/hosts, fixed budgets, source preparation and saved-data auditors are under `test2026/metal/experiments/mistral_half_attention_policy/`. Earlier sources, all raw/rounded payloads, commands, failed attempts and numeric/timing evidence remain local. All11 original dirty paths, selected runtime, model and frozen comparator studies stay unchanged. No comparator inference or main switch/push occurs. The serving comparison and strict coding0/8 remain authoritative; Mistral performance/quality, ANE/Gemma and original Qwen27B/9B/4B goals remain incomplete.
