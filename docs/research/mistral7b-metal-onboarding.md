# Mistral 7B native onboarding

The user moved Mistral 7B onboarding, native tokenization and a Core ML/ANE
plugin ahead of further Qwen27B performance work. Return to Qwen27B after Mistral
performs better than vLLM in a matched benchmark. The existing Qwen27B targets
and comparator evidence remain incomplete and preserved. This document records
onboarding work, not a successful inference or performance qualification.

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

The raw FP16 loader is a prerequisite, not full FP16 quantized projection support.
No Mistral generation, coding-quality result or speed comparison is admitted yet.

## Remaining implementation and qualification

1. Generalize converted checkpoint recognition to canonical Mistral tensor names
   while preserving Qwen's prefix translation and norm-offset rules.
2. Add typed FP16 scale/bias decoding to CPU selected-row embeddings and native
   Metal affine kernels, without BF16 rounding. Preserve the existing BF16 path.
   Prepare coefficient-type-specific pipelines before encoding and include the
   representation in profile/cache identity. Validate all active projection
   paths; a decode-only kernel test cannot admit prefill.
3. Verify embedding output rounding against the actual MLX dequantizer. Matching
   coefficient expansion alone does not prove embedding semantics.
4. Admit the Mistral text adapter with full GQA/RoPE/SwiGLU parity, explicit KV
   ownership, bounded prefill, batch/continuous scheduling and prefix reuse.
   Existing dense Qwen operators provide reusable primitives, but the current
   adapter guard rejects Mistral and has not been qualified for this checkpoint.
5. Establish new pinned Mistral comparator baselines and correctness gates,
   running vLLM-Metal, SGLang and Garnet sequentially. Qwen27B baseline numbers
   cannot substitute for Mistral results. Freeze artifacts, tokenizer/chat
   rendering, prompt IDs, quantization, context, cache policy, concurrency,
   generation budgets, warmups and repetitions. Record precision differences
   and unsupported settings explicitly. Keep quality and prefix/batching tests
   alongside TTFT, prefill, decode and aggregate throughput measurements.
6. Add and measure the ANE/Core ML plugin separately. Core ML usage alone does
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

The first tokenizer runtime build directory was reused for FP16 loader work.
Its original provenance retains the original hashes, but that build output path
is mutable. A copied immutable library under `tokenizer-runtime-final-library1`
and a fresh 120-check `native-tokenizer-runtime-api2.log` supersede that runtime
artifact. The original five-family reference probe and per-case records remain
unchanged. The protected Qwen27B qualified library still has its original SHA256
`69491581eb6ff7a659dcfc065231823b3827e8a15c603fec4756a1a2dfdbcb34`.
