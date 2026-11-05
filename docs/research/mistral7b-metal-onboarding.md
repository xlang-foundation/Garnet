# Mistral 7B native onboarding

The user moved Mistral 7B onboarding, native tokenization and a Core ML/ANE
plugin ahead of further Qwen27B performance work. Return to Qwen27B after Mistral
performs better than vLLM in a matched benchmark. The existing Qwen27B targets
and comparator evidence remain incomplete and preserved. This document records
onboarding and a small functional gate, not production or performance qualification.

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

The tested context capacity is 256. KV and intermediate native state remain
FP32; tensor operands use the explicitly selected FP16 path. This is not a
claim that every operation has the same precision as MLX. No sandboxed coding
suite, 4K context, batch 4, continuous admission, retained prefix reuse, ANE,
production xMind binding or serving-performance comparison is qualified yet.
The earlier greedy attempt that passed a whole chunk to the diagnostic head,
the missing eager-setup counter, and an incomplete source snapshot remain as
failed attempts; their results are not promoted to this gate.

## Remaining implementation and qualification

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

The first tokenizer runtime build directory was reused for FP16 loader work.
Its original provenance retains the original hashes, but that build output path
is mutable. A copied immutable library under `tokenizer-runtime-final-library1`
and a fresh 120-check `native-tokenizer-runtime-api2.log` supersede that runtime
artifact. The original five-family reference probe and per-case records remain
unchanged. The protected Qwen27B qualified library still has its original SHA256
`69491581eb6ff7a659dcfc065231823b3827e8a15c603fec4756a1a2dfdbcb34`.
