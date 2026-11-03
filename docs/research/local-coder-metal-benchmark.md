# Local coding Metal qualification

Status: preparation; no new baseline or speedup measured.

Order: Qwen3.8-27B, Qwen3.5-9B, Qwen3.5-4B. Current target is 27B.
Hardware: Apple M5, 24 GiB unified memory. This is separate from historical
M4 experiments. Branch: feature/qwen-metal-local-coder.

## Acceptance

Target at least 2x single-request decode tokens/s and 2x aggregate output
tokens/s versus each supported comparator, and at most half TTFT. Preserve
coding correctness and report memory limits and unsupported cases. Never infer
full-model success from component tests. Garnet executes its own native Metal
backend; comparator engines are not Garnet implementations.

## Frozen comparisons

Pin comparator commits/dependencies, model revision, every asset digest,
quantization algorithm/group size and retained tensor precision, tokenizer,
chat template, reasoning mode, generation parameters and workload before timing.
27B requires a qualified compressed-weight profile on this machine. Different
quantization or internal KV/state precision must be disclosed; label unmatched
comparisons separately. Do not compare BF16 Garnet with INT4 reference as if
they were equivalent.

Measure vLLM-Metal and the actual SGLang Apple backend once per frozen
configuration, with multiple independent processes/repetitions for that baseline.
Preserve raw data. Repeated Garnet candidates reuse those baseline results.
Rerun references only after material configuration/hardware changes or evidence
that the original cohort was invalid. Never modify or slow a comparator to
improve a ratio. An unsupported reference case remains unsupported.

## Workload matrix

Start with single-request text smoke and correctness before scaling memory.
Qualify contexts 2K/8K/16K and concurrency 1/2/4; 8 is optional if memory admits.
Use fixed seeded prompts and 128/512 output-token performance cases, reporting
actual token counts and EOS. Keep fixed-output performance separate from natural
EOS coding quality. Include distinct prompts, unequal lengths and staggered
arrivals, completion/cancellation, and independent xMind-style jobs.

Cache cohorts: cold request cache; warm identical prefix; append-only follow-up;
edit after a known prefix; eviction and recomputation. Cache identity includes
model/tokenizer/template and compatible execution state. Qwen hybrid models
require both full-attention KV and recurrent/convolution checkpoints; KV-only
restoration is insufficient. Batch row changes must preserve request identity.

Multimodal cohorts are separate: fixed image bytes/preprocessing/token budgets,
fresh image encoding and no dummy image for text. Qualify backend support before
including any vision ratio. Include screenshot-to-code tasks in quality results.

## Measurement

Run only one model engine at a time. No builds, installations, downloads or
other benchmark processes during timing. Preserve commands, process tree,
versions, hardware/OS, power/thermal state, memory-pressure and swap changes.
Record initialization, first request, and warm requests separately. Use
monotonic client streaming timestamps, distinguish first visible content from
first model token, and disclose any streaming chunks that contain multiple tokens.
Report TTFT, inter-token latency, end-to-end latency p50/p95, per-request decode
rate, aggregate completed output tokens per measured wall interval, and failures.
Report recurrent/KV/weight/workspace allocation where available and process-tree
memory without double-counting shared unified-memory buffers. Retain complete
outputs/token IDs and coding test results. A memory failure is a result, not zero
latency or an omitted slow sample.

Use at least three fresh-process cohorts, two warmups and five measured waves
per admitted performance cell. Freeze the final machine-readable cases after
tokenizer and comparator admission. Publish repetitions and dispersion, not only
the best sample. Faster decoding alone does not establish coding quality.

## Runtime work

Reuse the native scheduler/cache infrastructure. Implement backend-independent
prefix matching and lifecycle with Metal storage execution, hybrid-state
checkpoints and continuous batching. xMind supplies concurrent requests and
conversation prompts; it does not manage GPU pages. Keep xModel semantics and
hardware-specific extension policy separate.

## Preparation record (2026-11-02 Pacific)

- Active goal created for 27B; 9B and 4B follow only after completing it.
- vLLM-Metal source pinned at `a44be48731005932a24dd30b55c21e992dc90b9a`.
  Its checked-in installer selects compatible vLLM 0.32.0. Installation is in
  an isolated environment in the comparator checkout; no runtime source edits.
- SGLang source pinned at `57abc8a831abb83009ab543fc89df1d39a1306ff`.
  Follow its Apple installation packaging instructions using
  `pyproject_other.toml`, installing only `srt_mps` rather than diffusion extras.
  Original packaging is retained in evidence; runtime source is unchanged.
- Candidate shared weight artifact: `mlx-community/Qwen3.8-27B-4bit`, revision
  `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`; affine 4-bit/group64. Remote file
  metadata totals 16,081,490,933 bytes. This is not measured resident memory.
  Download and local asset hashing must finish before model qualification.
- Local environments, setup manifest and logs:
  `/Users/shawn/CantorAI/out/research/qwen-local-coder/`.
- Streaming client: `test2026/metal/local_coder_stream_benchmark.py`. Synthetic
  HTTP accounting check passes for incremental counts, final-only usage and
  HTTP failure retention. This does not establish any model result. The client
  refuses to invent a decode rate from final-only usage. Memory capture,
  full workload freezing, engine smoke checks and real timing remain pending.
- Tokenizer/template preparation generated nine text performance waves at about
  2K/8K/16K input tokens and concurrency 1/2/4, with exact per-row token counts
  saved alongside the request JSON. The client checks these against server
  usage and rejects mismatches and unexpected cache hits. These waves remain
  subject to memory admission, not an assertion all nine configurations fit.
- First SGLang install exited 2 after a 300-second uv cache-lock timeout while
  the verified-live vLLM installer downloaded the same PyTorch distribution.
  Failure log retained. A dependency controller retries after that live
  package process exits; runtime source/benchmark policy is unchanged.
- Downloaded artifact config passes the existing exact Qwen3.8 architecture
  validator. Weight loading still needs native work: MLX names use
  `language_model.model.*`/`vision_tower.*`; the current Metal loader expects
  original HF names and BF16/F32 tensors. Actual completed-shard header confirms
  packed U32 affine weights with BF16 scales and biases. Do not expand the full
  checkpoint into BF16 merely to bypass the missing compressed execution path.
- Both isolated comparator installations completed. SGLang's retry exit is 0.
  Its environment has torch 2.13.0, MLX 0.32.4 and mlx-lm 0.32.2. The unmodified
  vLLM-Metal installer selects torch 2.16.0.dev20261013, vLLM 0.32.0+cpu,
  MLX 0.32.4 and its pinned mlx-lm commit. Package freezes are retained.
  Metal/MPS device queries pass; M5 recommended GPU working set is
  19,069,665,280 bytes versus physical 25,769,803,776 bytes.
- All 15 checkpoint files verified against remote sizes and hashed locally:
  `evidence/27b-asset-hashes.json` under the experiment root. Native config
  validation passes. First vLLM server startup is in progress at loopback port
  8127 with context 4096, maximum 4 sequences, prefill budget 2048, prefix
  caching disabled and no speculation. This is smoke qualification, not a
  performance result. No SGLang or Garnet model runs concurrently.
- First vLLM startup failed (exit 1): actual model memory 16.06 GB, overhead
  1.04 GB, and the normal resident-memory probe reduced KV budget to zero after
  compression/paging. Retained server log, outcome JSON and process/system
  memory samples; not a baseline. No engine workers remain from that attempt.
  Next smoke uses the stock `--language-model-only` option, batch limit 1 and
  prefill budget 512 at the same 4096 context. This is a separate text-backbone
  profile, not an equivalent multimodal configuration or a performance result.
- vLLM text-only batch1 completed two 260-input/32-output smoke requests with
  exact prompt counts and incremental usage. Both windows had substantial
  system paging; no performance baseline admitted. Server and worker were
  verified exited before any SGLang model startup.
- SGLang's `--language-model-only` rejects this architecture before loading;
  retain that failed CLI attempt. Its documented default MLX loader (without
  that flag or `--enable-multimodal`) loads through mlx-lm and completed the
  same text smoke. A second smoke also completed; still not a baseline.
- User authorized stopping existing Cantor to free resources. Unloaded
  `gui/501/com.cantorai.cantor`, verified PID 20232 absent and launch agent
  unloaded, and retained plist plus restart command in `evidence/cantor-stop.json`.
  Existing smoke timings before this machine-state change cannot establish a
  matched baseline. A 2K-input/128-output SGLang admission request is in flight.
- Explicit user constraint reaffirmed: vLLM, SGLang and Garnet model execution
  must be strictly sequential. Verify prior server/worker exit before switching.
- SGLang's 2054-input request completed with 110 output tokens at natural EOS.
  This admission window still had large paging deltas, so it is not admitted as
  a performance baseline. Five bounded checks of its generated merge function
  passed four cases but failed the requested tuple-output contract for list
  inputs. Preserve this failure; one answer is not a broad model-quality score.
- SGLang server PID 35472 and workers 35495/35496 were stopped and verified
  absent after that request. Shut down the booted iPhone simulator between
  runs to reduce unrelated memory/GPU activity; no benchmark engine overlaps.
