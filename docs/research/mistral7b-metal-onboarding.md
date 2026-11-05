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
