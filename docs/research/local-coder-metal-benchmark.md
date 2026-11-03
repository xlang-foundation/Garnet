# Local coding Metal qualification

Status: vLLM phase1 collection complete (90 measured waves, 210 requests,
three cleanly stopped cohorts). SGLang's first frozen five-cell campaign and
the stock512MiB allocator-cache-cap active4 candidate failed repeated short
batch4 memory admission. A two-active-request profile is under qualification,
with four-client workloads explicitly queued. All failed runs remain retained.
Native INT4 package
build, eight component tests, seven benchmark/evaluator tests and independent
linear-attention/decoder references for batch1/2/4 pass. The INT4-disabled
control package builds and all seven control component tests pass. Native 27B
passes a bounded 79-input/four-output smoke twice with identical reset results;
the optimized projection/row-lifecycle build passes nine component tests.
Matched 4K performance, coding quality, continuous batching, hybrid prefix
integration and multimodal runtime gates remain pending. No comparator speedup
claim. See evidence paths in the progress notes.

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
  Complete each reference campaign, stop its server and workers, and verify
  their exit before starting the next engine. Do not overlap benchmark windows
  with builds, package installation, model downloads, or GPU component tests.
- SGLang's 2054-input request completed with 110 output tokens at natural EOS.
  This admission window still had large paging deltas, so it is not admitted as
  a performance baseline. Five bounded checks of its generated merge function
  passed four cases but failed the requested tuple-output contract for list
  inputs. Preserve this failure; one answer is not a broad model-quality score.
- SGLang server PID 35472 and workers 35495/35496 were stopped and verified
  absent after that request. Shut down the booted iPhone simulator between
  runs to reduce unrelated memory/GPU activity; no benchmark engine overlaps.
- Fresh text-only vLLM (max sequences 4, prefill 2048, context 4096) passed its
  resident-cache probe without swap/compression, admitted 12288 KV tokens with
  1.29 GiB hybrid backing, and completed four concurrent short requests with
  exact prompt counts. The actual request window still paged memory. This is
  concurrency admission evidence, not the frozen benchmark. Check lower prefill
  workspace before selecting the reference configuration.
- Phase1 frozen vLLM collection is running sequential fresh-process cohorts at
  context4096, max sequences4 and prefill512. No other engine or build overlaps
  its measured windows. Cohort1 completed all six cells and shut down cleanly.
  The read-only partial audit retains all five measured waves per cell and
  flags system swap activity in the 2K/concurrency4 cell. Its final measured
  wave lacks a memory sample after completion because shutdown follows the
  last wave; retain this instrumentation limitation rather than assert zero
  paging for an incompletely bracketed interval. Full collection and review
  remain pending. Partial artifact: `evidence/baseline-phase1/vllm-partial-audit-1.json`.
  A later read-only audit also verifies exact saved request bodies and token
  accounting. The working campaign driver now waits for one trailing memory
  sample after its final request; the active vLLM campaign retains its frozen
  original driver and will honestly retain its existing sampling limitation.
- Native affine INT4 loading, Metal kernels and 27B dispatch are unvalidated
  drafts behind an opt-in build flag. Their initial 512-token/batch1 profile
  is a correctness bring-up stage, not equivalent to the frozen 4K reference.
  The new native prefix index has checkpoint/isolation/lifetime tests prepared,
  but still requires actual hybrid-state capture/restore integration. Builds
  and native numerical tests are deferred until the reference campaign exits.
- A header-only inspection matched all 497 text projections plus the embedding
  table to the 27B xModel contract, including U32 packing, BF16 scales/biases,
  converted convolution shape, full-attention norms and linear recurrent
  parameters. These quantized tensors total 15,127,511,040 bytes including
  embedding; this is representation evidence, not an execution or GPU-memory
  result. See `evidence/27b-native-text-layout.json`.
- Frozen original coding-quality requests live in `evidence/quality-phase1`:
  eight independent Python contracts, temperature0, thinking disabled and
  output budget1024. Unlike the 128-output throughput fixtures they request
  complete implementations. Exact rendered token counts are 79..145. Preserve
  every failure/truncation and run the same tests for each engine; the bounded
  suite does not establish a general coding leaderboard score. Model responses
  and executable quality evaluation remain pending.
- Header inspection also matches all 333 BF16 vision tensors (921,460,192
  bytes). The draft native vision adapter now maps the converted root and
  transposes only its small Conv3d patch weight; an independent tiny packing
  golden is prepared. The model's vision factory still rejects admission
  until native numerical and end-to-end image gates execute. See
  `evidence/27b-native-vlm-layout.json` for header-only evidence.
- Frozen multimodal inputs are in `evidence/multimodal-phase1`: two visually
  checked 256x256 PNG screenshots (search-function repair and JSON Pointer
  examples). Their manifests pin exact PNG, font, tokenizer, template,
  preprocessor and generator hashes. A 16x16 patch grid with 2x2 merging predicts
  64 image tokens and total prompt counts132/131. These are explicit predictions
  requiring actual backend processor/grid/usage admission, not measured facts.
  Quality requests allow1024 output tokens. Unsupported image execution must
  be recorded; a text-only replacement cannot qualify the multimodal arm.
- vLLM cohort2 completed all30 measured waves across the six frozen cells and
  exited with code0/no forced kill. Its partial audit found no body/token/
  concurrency mismatch and all measured windows have bracketing memory samples;
  paging remains part of the retained results. Cohort3 is live in a new process
  after the prior server and workers exited. Collection is not yet complete.
- The saved-response quality evaluator and containment/early-exit/timeout
  tests are prepared. They use a separate bounded macOS sandbox subprocess,
  require a post-assertion completion marker, and distinguish infrastructure
  errors from code failures and generation truncation. Execution and sandbox
  containment validation are deferred until the reference campaign exits.
- Frozen vLLM CLI settings do not freeze its automatic physical cache sizing:
  cohort1 admitted7736 tokens (17 hybrid blocks), cohort2 admitted5916 (13),
  and cohort3 admitted5461 (12). Its own resident probe reduced each allocation
  under memory pressure. Preserve this geometry per cohort; external request
  concurrency4 is not proof that all four requests decode simultaneously.
  Compare qualified per-cell results conservatively and disclose queueing,
  preemption, paging and physical memory instead of attributing variability
  solely to kernels. The collection remains unchanged. Read-only power/cache
  observation is in `evidence/baseline-phase1/vllm-cohort3-power-cache-observation.json`;
  current AC power/no thermal warning does not prove earlier thermal conditions.
- Source review identified an embedding precision boundary: installed MLX
  `QuantizedEmbedding.__call__` uses `dequantize` without a dtype override, whose
  documented default infers output dtype from scales/biases (BF16 here). The
  draft native requested-row lookup now rounds to BF16 before exposing host
  FP32 rows. It leaves projection-weight FMA values/FP32 accumulation separate.
  Independent even/odd/negative tie goldens are prepared. Actual MLX/native
  row parity is still pending; source reasoning alone is not numerical proof.
- Phase1 vLLM collection is now complete:90 measured waves/210 requests, three
  clean process-group shutdowns and no request-accounting mismatch. Full audit
  records one limitation: cohort1's final wave lacks a trailing memory sample.
  Preserve the auto-sized cache differences and paging. Retained summary:
  `evidence/baseline-phase1/vllm-collection-summary.json`; full audit:
  `evidence/baseline-phase1/vllm-full-audit.json`. No speedup or broad quality
  qualification follows from this collection alone.
- Native package builds with affine INT4 enabled, and all eight CTests pass.
  The disabled-option control builds and all seven CTests pass. Existing BF16
  projection tests now explicitly check the disabled body-shape policy instead
  of requiring an opt-in pipeline unconditionally. Independent original-format
  attention/decoder references pass batch1/2/4, including three-to-one head
  grouping, seven recurrent steps and inactive-row preservation. Native full
  27B execution and actual MLX row parity have not yet run.
- All seven benchmark/evaluator tests pass, including exact collection and
  process cleanup, token accounting, paging counters, sandbox outside-write/
  loopback-network denial, early-exit rejection and timeout. The sandbox grants
  dyld read access to the literal root directory, not all descendant file data;
  unsupported Darwin address-space/data limits are replaced by a parent-side
  sampled RSS guard. Build/test logs are retained under `evidence/native-*`
  and `evidence/benchmark-tool-tests.log`. SGLang batch4 admission is next.
- SGLang stock overlap scheduling passed short-input batch4, but 2K-input
  batch4 failed with Metal command-buffer OOM at both prefill512 and128.
  The documented no-overlap candidate failed even the short wave with an
  MPS/CPU device mismatch. All three attempts and server failures are preserved;
  comparator runtime source was not changed. A subsequent stock-prefill512
  admission completed short batch4 and 2K batches1/2 with exact token counts,
  no cache hits and fully bracketed memory samples. Its process group exited
  after TERM; SGLang's own cleanup killed its root (exit-9), with no driver
  forced kill. Five supported performance cells are frozen in
  `evidence/baseline-phase1/sglang-manifest.json`; the omitted 2K/batch4 cell is
  listed explicitly as failed admission, not silently removed from the goal.
- Admission auditing exposed a caller-filename assumption in the report tool.
  It now reads the exact manifest embedded in frozen metadata and the campaign
  fixture tests a descriptive manifest filename. The three accounting/campaign
  tests pass. Audit also recognizes SGLang's periodic scheduler log format.
- The first frozen SGLang five-cell campaign completed short batches1/2 but
  failed with Metal OOM during short batch4 warmup1. The prior one-wave
  admissions therefore did not establish long-lived repeatability. Preserve
  this failed campaign at `evidence/baseline-phase1/sglang`; do not report it
  as a completed baseline. Stock SGLang exposes `SGLANG_MLX_CACHE_LIMIT_GB`
  specifically to bound freed-buffer retention before model load. A new,
  separately recorded candidate caps that cache at512MiB without runtime
  source changes; repeated short/long batch4 admission is running.
- Native 27B now executes all64 learned layers and the untied head. The first
  512-capacity smoke used79 exact prompt tokens and returned the same four
  IDs `[71093,12305,198,727]` on two reset-separated requests. It allocated
  18,054,987,776 Metal bytes and was slow (~0.72s per decode step); system
  paging was retained. See `evidence/native27b-smoke1`.
- A serialized operator diagnostic identified quantized MatMul as dominant.
  Small-row decode now uses row-specialized scalar accumulators; rows>=5 use
  the tiled FP32 contraction. All numerical component tests pass. The same
  bounded smoke then retained the exact IDs, with the second request's mean
  decode step0.134s and warm TTFT11.56s versus0.739s/48.32s previously. These
  are native iteration observations at capacity512/output4, not equivalent to
  the frozen 4K/output128 comparisons or a quality score. Artifacts:
  `evidence/native27b-op-diagnostic1` and `evidence/native27b-smoke2`; native
  libraries, smoke source and input digests are pinned separately.
- The opt-in request-row lifecycle extension builds, and its primitive test
  verifies padded-stride row isolation, rejection and reuse. It is not yet
  continuous serving, and actual mixed-request session reuse still requires
  integration gates. The native prefix index likewise still needs hybrid
  snapshot/restore and owning storage integration.
- The512MiB capped active4 SGLang admission also failed on repetition of the
  short batch4 wave, despite the cap being confirmed in its startup log. The
  next stock profile retains that cap and limits GPU-active requests to2;
  external client concurrency4 is still requested and its queueing must be
  disclosed. It cannot establish native active4 support.
- Actual native session row reuse now passes an independent mixed
  linear-attention/full-KV reference at batch4: replace A with E while B/C/D
  continue, pause/reactivate C, preserve all untouched state and hidden rows,
  reject invalid row indexes without mutation, and verify predictions both
  with and without a head. The fixtures cover seven steps. Evidence:
  `evidence/native-request-row-session-heads.log` and its persisted fixtures.
  This proves the row lifecycle primitive, not a serving scheduler or prefix
  cache integration. Both remain required work.
- The capped active2 SGLang candidate passed repeated admission at external
  concurrency4 for both short and 2K prompts. Its scheduler reported at most2
  active requests; remaining clients queued. The full six-cell, three-cohort
  campaign is now running alone at
  `evidence/baseline-phase1/sglang-active2-cache512m`. Its frozen engine and
  manifest retain the failed active4 attempts. Admission is not a completed
  baseline, and this profile does not establish active4 support.
- An opt-in compact attention index stores only the latest chunk-token index
  in a signed byte, retaining the -1 old-cache sentinel and FP32 scores. The
  4K attention scratch fits the device limit and is checked before encoding.
  Independent FP64 attention tests cover 4K, duplicate/backward slots,
  inactive rows and gated/ungated outputs. All10 context-build CTests pass;
  evidence is `evidence/native-context-ctest.log`.
- The learned 27B context4k smoke completed two reset-separated requests with
  the same260-token frozen prompt and four output tokens. Both returned IDs
  `[71093,12305,198,727]`; allocation was17,938,432,000 Metal bytes. The second
  request took38.89s to first token including reset, with mean decode0.1384s.
  See `evidence/native27b-context4k-smoke`. This proves bounded execution at
  capacity4096, not a full4K sequence, output128 benchmark or coding quality.
  The study profile remains unqualified. Continuous scheduling, hybrid prefix
  snapshots, learned batch4 admission and multimodal execution remain pending.
- SGLang's capped active2 full campaign completed all three fresh-process
  cohorts and six cells:90 measured waves/210 measured requests. The full audit
  found zero accounting or memory-bracketing issues. Every final request has a
  trailing sample, with no driver-forced shutdowns. Cohorts2/3 exited0; cohort1
  self-terminated during cleanup (exit-9). Retain the variable throughput and
  system paging rather than selecting a slow cohort to inflate Garnet ratios.
  Evidence: `evidence/baseline-phase1/sglang-active2-full-audit.json` and the
  immutable `sglang-active2-cache512m` campaign. This is an active2 baseline;
  client4 includes queueing, and active4 admission failures remain retained.
- Actual MLX/native embedding parity now passes all35,840 values from seven
  selected real checkpoint rows exactly, including the final vocabulary row.
  The oracle reads only packed selected rows and calls actual MLX dequantize
  with its inferred BF16 output. A separate native process uses ModelTensorIO.
  Evidence: `evidence/27b-embedding-mlx-oracle.json` and
  `evidence/27b-embedding-native-parity.json`. Whole-model parity remains a
  separate gate.
- The word-tile INT4 candidate loads each packed word once per tile and unpacks
  eight values, with padded transposed threadgroup storage. Expanded FP64
  contraction tests include row/column tails and learned inner widths. All10
  CTests pass in the separate `native-wordtiles-build`. Native row read/write
  primitives also pass immutable capture, cross-row restore, isolation and
  invalid-write rejection; these are not yet hybrid cache integration.
- Learned context4k word-tile smoke retained the same four IDs on both runs.
  Warm TTFT excluding reset was22.73s (reset0.048s), mean decode0.1496s/token,
  and allocation17,938,432,000 Metal bytes. Evidence:
  `evidence/native27b-wordtiles-smoke`, with retained library/source/workload
  digests. The sampler started after process initialization began, so it does
  not bracket all startup activity. These are output4 native observations;
  full output128 HTTP comparison, quality and broader admission remain pending.
- The single-active native HTTP study now completes both frozen single-client
  requests with128 sampled output tokens and exact prompt-token IDs. Short
  input260: TTFT30.35s, decode6.28tokens/s; long input2054: TTFT199.56s,
  decode4.64tokens/s. These one-wave admissions are not a full native campaign
  and are below the intended performance target. The bridge queues clients;
  it does not establish active batching. Evidence:
  `evidence/native27b-http-admission1`, including source snapshots and audits.
- All eight native coding regression tasks pass their sandboxed assertions.
  Every response stops within the frozen1024-token budget, and a separate audit
  checks every prompt ID and streamed output ID. Evidence:
  `evidence/native27b-quality1`, `native27b-quality1-evaluation.json` and its
  input-output audit. This proves only these bounded contracts. Equivalent
  reference-engine quality collection remains pending. The native supervisor
  was interrupted after collection and closed its worker stdin; both exited.
  Exact child exit code was not retained by this bridge version.
- Paired screenshot fixtures are frozen at
  `evidence/multimodal-grounding-phase1-v2`. Text prompts and rendered token IDs
  are identical; arithmetic constants differ only in the actual images. Both
  PNGs were visually checked. The first draft hit the clipping guard and is
  preserved separately; wrapping the source fixed clipping without shrinking
  the font. Image-token prediction remains distinct from runtime evidence.
- Standalone learned vision parity now passes on all333 actual BF16 tensors:
  native image patches match the independent CPU preprocessing exactly at
  grid16x16, producing64 merged image tokens. All327,680 vision output values
  pass the declared tolerance; maximum absolute error0.000210, cosine
  0.999999999997. The native graph allocated2,904,604,672 Metal bytes, so the
  bounded factory reserves3.5GB before construction. Evidence:
  `evidence/native27b-vision-reference1`, including raw tensor/payload hashes.
- Both native image-to-code grounding contracts pass their sandbox assertions.
  Each request uses the real image, native vision output and multimodal rotary
  positions. Vision closes before the text decoder loads: retained Metal bytes
  at handoff were737,280; text completion snapshots were17,938,432,000. The
  identical prompts yield the different constants shown in their images.
  Evidence: `evidence/native27b-grounding1` and its evaluation. This is a staged
  256px study, not simultaneous-residency or general multimodal serving
  admission. `create_metal_vision_study` is explicit; the general vision factory
  remains closed pending serving integration and broader qualification.
- SGLang now passes the same eight frozen coding contracts at the same1024-token
  budget. Its separate quality-only collection uses one request per task and
  no warmups; the performance baseline is unchanged. Accounting/memory audit
  reports zero issues, with evidence at `evidence/sglang27b-quality1` and its
  evaluation/audit reports. The auditor now accepts either explicit cohort
  count field while rejecting missing, nonpositive or conflicting counts;
  its three campaign/accounting tests pass. vLLM quality collection is live.
- Native request-row prefix group copies now retain only the selected dense KV
  tokens, respecting request/group boundaries and physical row stride. The
  restored destination must first be cleared; recurrent/convolution state
  still requires complete snapshots at the same boundary. Tests cover cross-row
  restore, empty future slots, invalid extents/overflow, nonfinite payloads and
  rejection without mutation. All10 CTests pass in `native-prefixrows-build`,
  retained at `evidence/native-prefixrows-ctest.log`. These primitives are not
  yet connected to native token-history matching or the serving scheduler.
