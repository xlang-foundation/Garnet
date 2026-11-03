# Local coding Metal qualification

Status: 27B qualification remains active. Frozen vLLM-Metal and SGLang Apple
reference collections each contain three cohorts, 90 measured waves and 210
requests. SGLang uses two active requests and queues four-client workloads;
failed active4 admissions remain preserved. Native continuous batching, hybrid
host token-prefix reuse, batch2/4 ragged prefill and batch2/4K concurrent coding
have correctness evidence. The latter passes all eight sandboxed coding tasks
with exact single-native token sequences. Native text-only input has no dummy
image; bounded native vision parity and two image-grounding cases pass, while
general VLM and reference multimodal qualification remain incomplete.

The most recent end-to-end native performance evidence is a single-active,
one-wave-per-cell admission, not a full repeated serving campaign. Short/long
aggregate throughput is 3.250/1.034 output tokens/s; TTFT is 20.518/101.057s and
decode is 6.732/5.582 tokens/s. Reference single-client medians are vLLM
5.761/3.382 overall tokens/s and SGLang 5.502/3.275. Native batch correctness
runs and synthetic projection timing do not fill the missing batch serving
measurements. All 2x targets remain unproven; prefill is the largest measured
gap. Reference results remain fixed and engines run sequentially.

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
  its three campaign/accounting tests pass. vLLM subsequently passed all eight
  identical contracts; its quality accounting audit also reports zero issues
  (`evidence/vllm27b-quality1`, evaluation and audit). Neither quality collection
  changes the frozen performance baselines.
- Native request-row prefix group copies now retain only the selected dense KV
  tokens, respecting request/group boundaries and physical row stride. The
  restored destination must first be cleared; recurrent/convolution state
  still requires complete snapshots at the same boundary. Tests cover cross-row
  restore, empty future slots, invalid extents/overflow, nonfinite payloads and
  rejection without mutation. All10 CTests pass in `native-prefixrows-build`,
  retained at `evidence/native-prefixrows-ctest.log`. These primitives are not
  yet connected to native token-history matching or the serving scheduler.

### Bounded 27B wider-prefill study

`GARNET_METAL_QWEN38_WIDE_PREFILL` is opt-in and defaults off. It permits
16/32/64-token graph experiments with batch1 resident shared projections and
chunk recurrence. It keeps one padded graph shape for tails and retires proven
unused graph defaults after the session owns initialized private state. The
normal workspace allocator rejects allocations above the device recommended
working set minus256MiB, conservatively charging the next allocation at64KiB
alignment. This guards these workspace allocations, not every allocation in
the process or system free memory.

The first probes rejected all three widths while compiling graphs. Early
default retirement then allowed width16; widths32/64 still rejected before
inference. Both failures remain retained, without increasing the memory limit.
The successful width16 probe generated the same four golden tokens on both
reset requests. Loaded TTFT excluding reset was24.426s then12.426s; completion
Metal allocation was18,321,817,600bytes against a19,069,665,280byte recommended
working set. These are short correctness probes, not matched benchmarks or
proof of the2x target. Evidence is in `evidence/native27b-wideprefill-study1`
and `study2`. All10 CTests and independent mixed batch4 row lifecycle checks
with/without a prediction head passed after the retirement change
(`native-wideprefill-ctest2.log`, `native-wideprefill-row-session.log`).

Width16 subsequently produced identical32-token sequences across two resets
(`evidence/native27b-prefill16-output32`). Its separate HTTP coding replay
passed all8 frozen sandbox contracts with no truncation, and the native/HTTP
input-output audit found zero issues (`evidence/native27b-prefill16-quality1`).
The worker and bridge exited before the subsequent timing admission started.
These results preserve the bounded quality gate, not the broader performance,
batching or general model qualification gates.

The sequential campaign driver's process guard now includes native Garnet
bridges and27B model workers alongside reference engines. On macOS, executable
paths must be inventoried with `comm` last in its own `ps` query: adding `args`
after it truncates that field to15 characters even with `-ww`. Separate PID
inventories are joined before matching. Both guard regression tests pass, and
a live check detected the actual native bridge and worker; the old unrelated
WorldSense XLang process is not classified as a model engine. This change does
not alter any previously frozen reference driver or result.

The subsequent width16 HTTP timing admission completed both single-client
128-output requests, with zero collection/accounting issues and a bracketed
final memory sample. Short260-token input: TTFT20.485s, decode6.325tokens/s,
total40.569s. Long2054-token input: TTFT124.863s, decode5.375tokens/s,
total148.497s. One wave per cell, no warmups and one fresh process were planned;
these are admissions, not full repeated native baseline results. System swap
was in use. Shutdown used SIGTERM with no driver forced kill and the process
guard subsequently found no model engine. Evidence:
`evidence/native27b-prefill16-http-admission2`.

A separate default-off `GARNET_METAL_RETIRE_INPLACE_DEFAULTS` experiment now
retires complete exclusively owned KV output alias groups only when native
in-place update analysis proves the session replaces them. Checkpoint owners,
inputs and hidden outputs remain protected. Standalone execution recreates
those alias groups before running; resident execution validates private cache
layout before binding it. It requires the isolated wide-prefill build. All10
CTests and the independent mixed batch4 row test passed, including standalone
rehydration, private-state isolation and subsequent resident reuse, with and
without a head (`native-retiredkv-ctest.log`, `native-retiredkv-row-session.log`).
Learned model probes remain a separate qualification gate.

The retirement experiment subsequently passed both reset requests and golden
tokens at widths16 and32. Width16 retained17,784,946,688Metal bytes,512MiB less
than its previous build; width32 retained18,551,848,960bytes. Warm smoke TTFT
was13.560s/13.178s respectively, which does not establish a speedup between
these widths. Width64 still failed the workspace guard before inference.
All raw successes and failures remain in `evidence/native27b-retiredkv-study1`.
Width32 now has an explicit experimental worker option requiring the retirement
extension; coding and longer-request checks remain necessary before admission.

The width32 worker subsequently passed all8 original coding contracts, with no
truncation and zero native prompt/stream/HTTP accounting issues
(`evidence/native27b-prefill32-quality1`). This still does not qualify batching
or establish the performance target.

`GARNET_METAL_SHARED_PREFILL_SCRATCH` is a separate default-off native experiment
requiring the bounded wide-prefill build. It pools exclusively owned temporary
buffers by layout and per-graph lane. Inputs, checkpoint storage, hidden and
state outputs, aliases and read-before-write buffers are excluded. Only known
full-buffer initialization kernels can initialize pooled storage. Graphs retain
the pool, and both session and standalone execution lock it through GPU
completion; a graph cannot migrate to a second pool. Memory inventories include
pool-retained buffers even when another graph supplied them.

All10 CTests passed in `native-sharedscratch-build`. A four-layer mixed
linear/full-attention test matched pre-sharing outputs and states exactly with
padded/inactive batch4 rows. It also covered private sessions, reverse standalone
execution, foreign-owner rejection and standalone reuse after session close,
with43,904reused bytes in the small fixture. Evidence:
`native-sharedscratch-ctest.log`, `native-sharedscratch-mixed-session2.log`.
These storage checks do not yet qualify the learned27B route or simultaneous
host-call stress.

With scratch sharing, both widths32/64 subsequently passed the two-reset
learned27B smoke and golden token check. Width32 retained17,718,099,968Metal
bytes; width64 retained18,418,941,952bytes. Warm smoke TTFT was12.503s/13.856s.
The width64 working-set rejection from earlier builds is preserved; this new
build passes without raising the guard. Evidence:
`evidence/native27b-sharedscratch-study1`. This remains short-output correctness
evidence, not proof of coding quality or a performance improvement. The worker
admits an explicit experimental64 option only with both native memory features.

The width64 scratch-sharing worker subsequently passed all8 frozen sandbox
coding contracts without truncation, and the native/HTTP token audit found zero
issues (`evidence/native27b-prefill64-quality1`). Its worker exited before any
subsequent native build or GPU test.

`GARNET_METAL_CONTINUOUS_ROWS` is a default-off native row-ownership extension,
requiring request-row reset. `claim_request_row(request)` returns a stable
physical row and process-unique epoch; `release_request_row(row,epoch)` clears
and releases only that current owner. Duplicate/full claims reject without
replacing live ownership. Late releases cannot clear a replacement or a rebuilt
session. Once managed row mode is entered, execution rejects active unowned
rows before changing private state. A full reset cancels all leases and returns
to the existing unmanaged study contract.

All11 CTests pass in `native-leasedrows-build`. Independent mixed linear/full
batch4 tests with/without a prediction head cover A-to-E refill, B/C/D isolation,
stale release, unowned-row rejection and reset. Scratch sharing and legacy
rehydration still pass. Evidence: `native-leasedrows-ctest.log`,
`native-leasedrows-row-session.log`, `native-leasedrows-sharedscratch.log`.
This provides native state ownership for the scheduler adapter; it is not yet
an integrated continuous scheduler or learned27B batch admission. Existing
`ContinuousBatchScheduler` policy still needs its paged-cache plans bound to
these fixed Metal rows and hybrid state.

Width64's separate single-client HTTP timing admission completed both128-output
requests with zero accounting issues. Short260 input: TTFT23.201s,
decode6.434tokens/s, total42.947s. Long2054 input: TTFT122.139s,
decode4.961tokens/s, total147.750s. The same one-wave/no-warmup admission policy
was used as width16; the results do not show a clear benefit from wider chunks
and do not meet the target. Raw memory, streams and shutdown are retained in
`evidence/native27b-prefill64-http-admission1`. The process guard found no live
engine after collection, before building the scheduler portability check.

The existing scheduler now consumes a `KVSequenceAllocator` interface rather
than the concrete CUDA-implemented `PagedKVPool`. CUDA retains its concrete
storage API and supplies the default no-fixed-row placement; other backends
can supply a stable physical row/epoch in each scheduled item. A CPU-only
fixed-row fixture passed priority reordering, stable A/B/C/D placement, E queued
behind four occupied rows, refill after A completes, and cancellation. The
CMake test `native_portable_continuous_scheduler` passes without CUDA
(`portable-scheduler-ctest2.log`); the initial unreconfigured missing-target
attempt is retained separately. This tests scheduling and metadata placement,
not actual Metal tensor execution or CUDA GPU regression. The Metal allocator
callbacks and batch executor still need integration.

The isolated width64 operator diagnostic completed both reset requests. Across
them, 4,960 row64 MatMul dispatches accounted for23,795.217GPU ms, versus
850.762ms for160 chunk-attention dispatches. Raw per-op records and the hashed
summary are in `evidence/native27b-prefill64-op-profile1`. These submissions
disable some fusion and deliberately wait between operations; their timings
are diagnostic rather than normal serving latency. Q4 projection is the next
prefill target rather than further chunk-width changes.

`GARNET_METAL_AFFINE_TALL_TILES` is a default-off FP32 Q4 experiment. It reuses
one unpacked B tile across32/64 activation rows instead of16, retaining the
same per-output K accumulation order and existing rows1..4 decode kernels.
Pipelines check SIMD width, thread count and static threadgroup memory against
the device. All12 native tests pass in `native-talltiles-build`, split between
the Q4 test and the other11 tests. Q4 coverage now includes33/63/65/127-row
boundaries as well as actual5120/17408 inner dimensions and column tails
(`native-talltiles-affine-ctest.log`, `native-talltiles-other-ctest.log`). Learned
model correctness and performance remain separate required checks.

The first tall-tile learned diagnostic preserved both golden token sequences,
but its4,960 row64 MatMul dispatches took25,616.176GPU ms versus23,795.217ms
in the earlier diagnostic. It is not a speedup or an admitted optimization.
Evidence: `evidence/native27b-talltiles-op-profile1` and its hashed summary.
The next revision aliases activation and final-result threadgroup storage
across their disjoint phases. The final K-loop barrier precedes all result
stores. This reduces static scratch without changing arithmetic; all12 CTests
pass (`native-talltiles-reuse-ctest.log`). Learned timing still needs checking.

The portable scheduler now completes a one-token budget at prefill instead of
scheduling an unwanted second token. Its prefill completion API also accepts
an explicit finished flag for EOS, and negative output tokens reject before
changing request progress. The CPU regression covers all three cases along
with row refill (`portable-scheduler-budget-ctest.log`).

The shared-memory tall-tile revision preserved both golden sequences and took
19,328.246GPU ms for the same4,960 row64 MatMul dispatches, versus23,795.217ms
for the original kernel. This is about19% less serialized diagnostic GPU time,
not an end-to-end speedup. Raw evidence and hashed summary:
`evidence/native27b-talltiles-op-profile2`.

Native `run_leased` now checks the expected epoch of every scheduled row under
the same session lock held through encoding and GPU completion. Missing epochs
on active rows and late A execution after E owns A's old row reject before
private state mutation. Ordinary `run` remains the existing unmanaged study
entrypoint. All12 CTests pass in `native-leaseddispatch-build`; independent
batch4 mixed-state tests with/without a head prove the stale-dispatch and
missing-epoch rejections as well as correct current-lease execution. Scratch
sharing remains valid (`native-leaseddispatch-ctest.log`,
`native-leaseddispatch-row-session.log`, `native-leaseddispatch-sharedscratch.log`).
Scheduler-driven execution must use this leased entrypoint; policy metadata
alone cannot protect a row reassigned between planning and GPU submission.

The shared-memory tall-tile kernel subsequently passed all8 frozen coding
contracts without truncation, with zero token-accounting issues
(`evidence/native27b-talltiles-quality1`). Serving timing remains a separate gate.

The default-off continuous-row extension now exposes a native
`continuous_scheduler` bridge. Its allocator claims/releases actual Metal
session state, derives the row/context bounds from proved native KV layouts,
and returns physical masks and epochs for `run_leased`. It rejects duplicate
ticks while work is outstanding. Completion/cancellation retires finished
policy metadata; closing clears its owned rows without touching a replacement
with another epoch. The bridge retains its session owner through cleanup.

All12 CTests pass in `native-scheduler-build`. Its GPU integration test compares
combined mixed linear/full execution with separate per-request sessions through
A-to-E refill, priority reorder, decode plus incoming prefill ticks, budget/EOS,
queued/inflight cancellation, stale dispatch rejection, close and row reuse.
Tracked request metadata returns to zero. Evidence: `native-scheduler-ctest.log`,
`native-scheduler-gpu-integration2.log`. These are small synthetic layers with
one-token prompts. The learned27B batch profile, chunked-prompt integration,
serving wrapper and prefix reuse remain required; the local HTTP study worker
still advertises one active request.

The revised tall-tile HTTP admission completed with zero accounting issues:
short260 input TTFT20.518s, decode6.732tokens/s, total39.390s; long2054 input
TTFT101.057s, decode5.582tokens/s, total123.818s. This remains one wave per
cell, not a repeated qualification campaign or proof of the2x target.
Evidence: `evidence/native27b-talltiles-http-admission1`.

An explicit `metal_batch_study` factory now isolates an unqualified batch2 text
profile from normal batch1 admission. It requires the native scheduler and
guarded compressed-weight/workspace extensions. The study uses serial masked
prefill for different prompt lengths and genuinely combined decode. Two
distinct79/71-token prompts matched their frozen successful single-request
16-token prefixes across two repeats at capacity512, with no tracked scheduler
requests left. Metal allocation was about16.64GB
(`evidence/native27b-batch2-study2`). The initial harness failure from a missing
compile-timing counter is retained in study1; study3 repeats correctness with
staged cleanup observations. This is not a throughput benchmark, full coding
suite, combined prefill admission or general serving profile.

Those studies exposed1.22GB of retained Metal resources after close. A native
regression reproduced a16MiB temporary buffer retained by standalone row
validation (737,280bytes before versus17,514,496after). The validator read
Objective-C buffer properties without a local autorelease pool, unlike its
pooled read/write callers. Giving validation its own pool returns the test to
737,280bytes afterward. All12 CTests and the scheduler GPU integration still
pass (`row-validation-retention-before.log`,
`native-row-validation-pool-ctest.log`, `native-row-validation-pool-scheduler.log`).
The same learned batch study must confirm complete cleanup before admission.

Study4 confirms the fix on the learned checkpoint: both batch2 repeats still
match both16-token sequences, and complete close returns to720,896Metal bytes,
also unchanged after quiescence. Session close now releases its private state;
prefill graphs, decode graphs and the head release the remaining owners in
their staged snapshots (`evidence/native27b-batch2-study4`). This admits only
that finite correctness/cleanup study, not full coding quality, longer context,
batch4, combined prefill or HTTP serving.

The same batch2 study also passed at capacity2560 with2054/71-token inputs,
two repeats and both frozen16-token output prefixes. Metal allocation after
prefill compilation was18,526,502,912bytes; close/quiescence returned to720,896.
Evidence: `evidence/native27b-batch2-context2560-study1`. It remains a correctness
study with serial masked prefill, not a throughput measurement.

`GARNET_METAL_CHUNK_KV_INPLACE` is a new default-off memory experiment. It aliases
only complete row-preserving views of fresh chunk-cache outputs and schedules
attention's old-cache reads before committing KV updates. Native in-place
analysis must prove no later old-cache read. The kernel alias guard accepts
only identical cache views at the cache operand. Proven resident-unused graph
defaults retire at build completion; standalone execution rehydrates them.
The first candidate correctly hit the generic alias guard before its narrow
chunk-cache allowance was added; that failure is retained.

All12 CTests pass. New scalar and vector regressions pass with the feature off
and on, comparing append, backward overwrites, duplicate slots and inactive/
reactivated rows against sequential decode and standalone execution. The first
eight-channel fixture's invalid rotary sections were rejected on both builds;
the corrected fixture is retained separately. Scratch sharing and scheduler
GPU integration still pass. Evidence: `native-chunkcache-ctest2.log`,
`chunk-kv-inplace-control.log`, `chunk-kv-inplace-candidate2.log`,
`chunk-kv-vector-control2.log`, `chunk-kv-vector-candidate2.log`,
`chunk-kv-sharedscratch.log`, `chunk-kv-scheduler.log`. Learned-model memory and
coding gates remain necessary before profile admission.

The learned in-place-cache batch2 study passed both repeats and both frozen
16-token prefixes, asserted32 real in-place KV updates per prefill dispatch,
and returned to720,896Metal bytes after close. At context512, prefill-compiled
allocation fell from16,641,163,264to16,372,727,808bytes
(`evidence/native27b-chunkcache-batch2-study1`). Larger contexts remain a gate.

The explicit batch-study factory can now construct batch4 only when the native
chunk-cache extension is present. Normal CPU/Metal constructors still admit
only batches1/2, and normal27B admission remains batch1. The study subclass
defers unused legacy host state while the resident session is being built;
native state still initializes independently. A no-weight-open rejection test
confirms both default admission and extension gating
(`batch4-study-admission.log`). Learned batch4 is pending, with the working-set
guard unchanged.


The batch4 learned study passed at context512 with four distinct79/71/100/113-
token prompts,16 output tokens per row and two repeats. Every sequence matched
its frozen single-request native oracle; scheduler tracking drained to zero.
Metal allocation was18,306,334,720bytes and complete close/quiescence returned
to720,896bytes (`evidence/native27b-batch4-study1`). Prefill was serial and
masked; decode combined all four rows. These study timings are not a matched
benchmark or full coding-quality qualification. Reference and Garnet measured
campaigns must run one engine at a time, without concurrent builds or GPU work.

`GARNET_METAL_ROW_FINAL_HIDDEN` is a default-off native extension requiring
continuous row ownership. Each completed layer stack gathers each row's last
active token into a session-private hidden bank. Fully inactive rows retain
their earlier vector. `predict_rows(epochs)` projects that bank only after
validating current leases and completed hidden state under the session lock;
unselected rows return-1. Row release/replacement zeroes and invalidates the
bank; reset clears it. Inventory and row-reset byte accounting include it.
This allows variable-length prompts to prefill together without a shared final
head index or retaining full logits on the host.

All12 CTests, independent mixed-layer row tests with/without a head, scheduler
GPU integration and shared-scratch storage tests pass in `native-rowhidden-build`.
The dedicated bank regression compares different final indices, padding,
inactive retention and gapped reactivation with sequential token execution,
and rejects stale/replaced/uninitialized epochs and prediction after reset.
Its initial harness used slots beyond the fixture's capacity8 and was rejected;
the corrected test wraps fixture slots within capacity. Evidence:
`rowhidden-ctest.log`, `rowhidden-bank-test.log`, `rowhidden-bank-test2.log`,
`rowhidden-row-session.log`, `rowhidden-scheduler.log`,
`rowhidden-sharedscratch.log`. Learned parallel prefill remains a separate gate;
the study harness selects `prefill_mode=parallel` explicitly and keeps serial
mode as the default. HTTP integration, hybrid prefix snapshots and the2x
performance targets remain incomplete.


The learned parallel-prefill gate passed in
`evidence/native27b-parallel-batch4-study1`: the same four variable-length
prompts and16-token oracles matched in both repeats, with true combined
prefill and decode. The native head bank handled the shorter rows after they
became inactive. Scheduler state drained completely, Metal allocation was
18,306,416,640bytes, and close/quiescence returned to720,896bytes. The retained
artifact/token/resource audit found zero issues. Observed prefill times were
32.933s/20.335s and decode windows5.019s/5.289s. These are bounded correctness
study timings, not matched campaign results or evidence for the2x target.
The default-off feature is qualified only for this finite learned batch4 gate;
full coding output, serving and hybrid token-prefix integration remain required.


`GARNET_METAL_TOKEN_PREFIX_CACHE` connects the backend-neutral exact-token
prefix index to native hybrid row snapshots. It is default-off and requires
native row ownership/final-hidden state and prepared prefill families.
Configure a session before claiming rows with six identity strings (model,
execution profile, tokenizer, position policy, isolation scope, modality), an
entry limit and a host-memory budget (maximum512MiB). Initial admission is
plain text with `text_zero_based_3axis_v1` positions and proved KV capacity up
to4096; multimodal prefix reuse remains closed.

`run_tokens` receives token IDs alongside the existing embeddings/controls and
request epochs. Native code validates contiguous slots/positions, bounded
history and unchanged registered graph families. The runtime caller remains
responsible for embedding/token-ID consistency. Prepared prefill graphs must
share the original native weight buffers. Untracked execution invalidates the
affected row's history until reset/release; it cannot later label that state
with arbitrary token IDs.

`cache_token_prefix(row,epoch)` copies compact KV prefixes plus complete
convolution/recurrent rows and final hidden state into an immutable host
snapshot at the current tracked boundary. `restore_token_prefix(row,epoch,ids)`
looks up the longest stored token checkpoint in the configured identity and
requires a fresh current row. It clears that row before restore, leaving other
rows unchanged; exact hits can immediately use `predict_rows`. Changed or
appended suffixes continue from the returned token count. The cache survives
request release and full live-state reset; close clears it. These copies are
bounded host snapshots, not paged KV, zero-copy device leases or COW. The budget
accounts retained entries; one temporary candidate can coexist with retained
entries while capture/eviction completes. Normal admission remains unchanged.

All12 CTests and existing final-hidden, independent mixed-row, scheduler and
scratch regressions pass in `native-tokenprefix-build`. The dedicated mixed
linear/full test compares exact, appended and changed-prefix restoration with
fresh recurrent/KV/head execution, preserves B/C/D through A-to-E reuse, checks
misses/LRU/reset/clear, and rejects stale epochs, untracked history, wrong text
positions, mismatched graph families and closed owners. A too-small snapshot
budget declines capture without breaking prediction. Initial harness errors
(foreign scratch preparation and an overcounted expected hit total) are retained.
Evidence: `tokenprefix-ctest.log`, `tokenprefix-regressions.log`,
`tokenprefix-hybrid-test.log` through `tokenprefix-hybrid-test4.log`.
The learned checkpoint and HTTP integration remain separate gates.


The learned hybrid-prefix gate passed in
`evidence/native27b-tokenprefix-study1` at batch2/context512/chunk8. It captured
A at token64 and its full79-token prompt while B's distinct71-token prompt
shared execution. Original and exact-restored16-token outputs matched the
frozen successful single-request oracles. An appended first generated token
continued with the remaining15 oracle tokens. A changed token after64 restored
only that stored boundary, then matched four tokens from fresh execution of
the changed prompt. B's saved prediction stayed unchanged through A's release,
replacement, restoration and suffix execution.

The native index reported two entries, three hits and336,511,388accounted host
bytes under a384MiB retained-entry budget. Metal allocation before clear was
16,372,826,112bytes; close/quiescence returned to720,896bytes. Retained source,
artifact, token and resource audit found zero issues. This proves that finite
native learned prefix path; it does not admit HTTP prefix reuse, general
multimodal prefixes, paged KV, complete batched coding quality or performance
ratios. The serving wrapper still needs native scheduler/token-cache integration.


A separate research HTTP path now multiplexes multiple requests into one native
continuous scheduler (`local_coder_batch_http.py` / `local_coder_batch_worker.py`).
The bounded CPU transport routes interleaved native token events by request
identity, checks complete IDs/counts and exposes cached-token usage. Profile
rejection affects only that request; fatal execution/routing errors stop
admission. Native row allocation, prefix lookup and state ownership remain in
the runtime. The worker supplies tokenization, embeddings and graph dispatch.
A5ms idle coalescing window is included in client TTFT and frozen with the
profile. Prompt prefill shares chunk graphs, but whole-prompt policy ticks can
still stall existing decode during a new long prompt. HTTP disconnects drain
the bounded request; they do not yet expose native cancellation.

CPU-only fake-event transport and HTTP tests pass, covering interleaved client
routing, SSE final IDs/cache counts, isolated rejection and clean subprocess
exit. These tests are never performance evidence. The one-engine guard also
recognizes both new bridge/worker names. The common study HTTP handler adds
optional cached-token accounting and an opt-in request-error hook; existing
single-request/CUDA workers keep fatal errors as their default. Learned-model
HTTP admission and complete batched coding quality remain pending.


The learned native HTTP finite gate passed in
`evidence/native27b-batch-http-study1` with batch2/context512/chunk8 and prefix
reuse enabled. A and B began together; C arrived after A's first token, queued,
and reused A's physical row with a new epoch after A finished while B continued.
The raw log contains14 actual combined-decode plans. All8/24/8 output prefixes
matched frozen successful native quality-oracle tokens. Returning C produced
the same8 tokens with100 cached prompt tokens reported through SSE usage.
An oversized context request was rejected nonfatally and health stayed ready.
The native subprocess exited0 and final Metal allocation was720,896bytes.

The artifact/raw-event audit found zero issues, verifies native versus client
IDs/usage, row refill while B continues, exact prefix counts, working-set bounds
and clean close. Five CPU tamper checks reject missing combined execution,
reused epochs, corrupted outputs, false cache counts and retained allocations.
These are finite serving correctness gates, not complete coding quality or
matched performance measurements. Capacity512 and8/24-token outputs do not
qualify the4K/128-token comparator campaign; whole-prompt prefill interference
and HTTP cancellation remain limitations. Full batched coding output and the
larger context profile must be qualified before performance comparison.


The coding collector now accepts an explicit bounded `--parallel-tasks` option
(default1 preserves serial replay). Concurrent task clients retain separate
frozen inputs/raw SSE/results; the main thread records completion order and
canonical per-task status. A dedicated one-engine native batch-quality driver
freezes the worker/profile/library, collects all eight tasks and closes the
worker before separate sandbox evaluation. This remains coding qualification,
not a throughput campaign. The guarded batch2/context4096 profile is the next
admission attempt; unsupported memory admission will be retained as a failure.


The concurrent coding gate passed8/8 at batch2/context-capacity4096/chunk8,
prefix reuse disabled (`evidence/native27b-batch-quality1`). Eight clients queued
against two native rows; the raw log contains1,132combined decode plans, eight
distinct row epochs and repeated row refill. All complete generated sequences
were byte-identical in token IDs to the frozen successful single-request native
quality oracle (62..478output tokens). Native/frozen/HTTP accounting found zero
issues (`input-output-audit2.json`); five CPU tamper checks reject absent
combined plans, reused epochs, altered text, false cache counts and retained
close allocations. Native subprocess exit was0 and final Metal allocation was
720,896bytes. Ready allocation was17,793,024,000bytes versus the device's
19,069,665,280recommended working set.

Sandbox evaluation initially used Xcode's Python and failed infrastructure
startup before candidate execution (`evaluation.json`). The identical evaluator
and deny-default policy passed8/8 with the prepared interpreter
(`evaluation2.json`); sandbox permissions were not widened. This qualifies the
bounded coding suite with concurrent native serving at the allocated4K profile,
not every4K-length execution or general coding quality. Long-input performance
admission, repeated native campaigns and the2x speed/throughput targets remain
required. The frozen reference campaigns remain unchanged and run no engines
in parallel with Garnet.

The next prefill candidate is an explicit hardware-gated native Metal4 MPP
extension. Apple's [MPP programming guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
describes tensor operations targeting M5 GPU neural accelerators. The installed
MLX headers already contain a separate NAX quantized path using those APIs;
Garnet's current Q4 path still uses its own SIMD-group matrix kernels. This is
a concrete implementation difference to investigate, not a predicted speedup.
Start with non-relaxed FP32 input/accumulation semantics, separate allocation/
numerical gates and existing-hardware fallback. Any relaxed-precision experiment needs its own explicit profile, unchanged
numerical and coding gates, and preserved control results. Frozen comparators
must remain unchanged.


`GARNET_METAL_MPP_AFFINE_FP32` is a new default-off native prefill candidate.
On macOS26+ devices reporting Apple10 support, it compiles a separate Metal4
MPP pipeline for affine Q4 projections with more than four input rows. Other
devices retain the existing pipeline; decode rows1..4 are unchanged. Each
threadgroup dequantizes only a32x32 tile from the existing packed weights,
keeps inputs/weights/accumulators FP32 and explicitly disables relaxed precision.
The original xModel MatMul operations and packed checkpoint representation stay
unchanged. Dynamic output extents handle partial row/column tiles. Native
context/session dispatch counters distinguish an exercised MPP route from an
optional build merely containing it.

The M5 route is confirmed available and exercised. All24 projection shapes
(including model-width inner dimensions, tails and unchanged decode cases)
passed the existing independent FP64 contraction tolerances, ownership and
rejection checks. All12 CTests pass in `native-mpp-affine-build`; evidence:
`mpp-affine-numerics1.log`, `mpp-affine-ctest.log`,
`mpp-affine-ctest-detail1.log` (`device=Apple M5 available=1 non_relaxed=1`).
This is operator correctness only. Operator timing, learned-model coding and
end-to-end admission remain required; no MPP speedup is claimed.


The first MPP concurrent coding attempt is retained as incomplete in
`evidence/native27b-mpp-quality1`. Five completed tasks exactly matched their
frozen successful single-request token oracles. A queued task then hit the
transport's300s event deadline before its first token, propagating failure to
other active streams; the native worker was still producing tokens and exited0
after collection stopped. This is not full MPP coding qualification or evidence
that a speed target was met (`partial-diagnosis.json`).

The bridge now separates bounded queued first-token wait from active-stream
idle wait. Both the bridge and quality client accept explicit frozen observation
limits; serial quality defaults remain unchanged. A CPU regression accepts a
first-token wait longer than the active-idle limit while still rejecting an
actual active-stream stall. Existing routing/SSE/rejection tests still pass.
This changes observation bounds, not sampling, kernel math, reported timings or
the original performance targets. The failed attempt is not overwritten.


The FP32 MPP operator comparison completed three fresh processes per variant,
strictly sequential base/tall/MPP order,21 shapes, three warmups and nine
measured calls per shape (`evidence/affine-mpp-comparison1`). All binaries,
cache flags, source, raw GPU/wall samples and FP64 spot checks are retained.
The audit found zero issues; five tamper cases reject invalid timings, missing
shapes/cohorts, missing MPP execution and altered binaries. Aggregation takes the
median of within-cohort medians without discarding samples.

FP32 MPP is not adopted: at8/16 rows its GPU speed ratio versus the tall control
was0.65..0.72x (about1.4..1.5times slower). At64 rows it was0.95..1.08x versus
the tall control, depending on projection shape. These are synthetic projection
measurements, not TTFT, decode or throughput results. They do not establish any
2x goal and do not justify rerunning the incomplete FP32 MPP coding campaign.

`GARNET_METAL_MPP_AFFINE_BF16` is a separate default-off candidate, mutually
exclusive with the FP32 MPP profile. It explicitly rounds prefill projection
inputs and dequantized tile weights to BF16 and accumulates in FP32, with
relaxed precision still disabled. Packed weights, graph MatMul operations,
FP32 outputs and decode rows1..4 stay unchanged. Metadata identifies the operand
type, including per-cell fallback/decode paths; model study profiles must request
the correct MPP type explicitly. This is a precision-profile change, not an
exact FP32 replacement. Its numerical oracle independently rounds operands
before FP64 contraction rather than weakening tolerances. All24 projection
cases and12 CTests pass on the M5; the FP32 path also still passes after the
shared implementation refactor (`mpp-bf16-ctest.log`,
`mpp-bf16-ctest-detail1.log`, `mpp-fp32-refactor-numerics.log`). Learned coding
and end-to-end timing gates remain required before enabling it.


The explicit BF16-operand comparison also completed all three fresh sequential
cohorts per variant with the same21 shapes, three warmups and nine measurements
(`evidence/affine-mpp-bf16-comparison1`). Its audit and five tamper checks pass.
Per-cell metadata confirms FP32 decode rows1..4 and BF16 MPP operands for rows
above4; accumulators are FP32 throughout. All raw samples, binaries, flags and
CPU spot-check evidence are retained.

This32-reduction-tile BF16 candidate is also not admitted to serving: speed
ratios versus the tall FP32 control were0.77..0.91x at8/16 rows,1.01..1.18x at32
rows and1.10..1.27x at64 rows. Small packed profiles still regress. These are
operator-only ratios across explicit precision profiles and do not qualify
end-to-end speed or coding quality. Larger reduction tiles are the next
implementation experiment; do not rerun unchanged reference engines or claim
the2x target from these data. The qualified batch2/4K coding control remains
unchanged and both MPP candidates remain default-off.


MPP reduction tiling is now an explicit bounded build parameter:
`GARNET_METAL_MPP_AFFINE_TILE_K=32/64/128` (FP32 excludes128 because its staged
scratch exceeds the initial admitted bound). The BF16 variants retain coalesced
32-K unpack waves while reducing synchronization/MPP calls over larger K tiles.
Zero padding handles partial final K tiles. The native pipeline still checks
actual static threadgroup memory against the device limit. Backend/profile
metadata identifies the compiled K tile; a model profile must request that tile
explicitly rather than silently using a different candidate.

Both BF16 K64 and K128 passed all24 independent projection cases and all12
CTests, including K64 inputs padded inside K128, output tails and unchanged
small-row decode (`mpp-bf16-k64-ctest.log`, `mpp-bf16-k128-ctest.log` and retained
expanded test logs). Operator timing is next; no serving profile is enabled.
The reusable projection collector freezes binaries/flags/source and blocks any
other inference process before each strictly sequential fresh-process run.


The K-tile comparison completed all three sequential fresh-process cohorts for
base, tall, BF16 K32/K64/K128 across21 shapes
(`evidence/affine-mpp-ktile-comparison1`). Audits confirm dtype, actual MPP route
and compiled K tile. K64 gave1.15..1.38x GPU speed ratios versus tall at64 rows,
but only0.86..0.94x at8/16 rows. K128 regressed several larger shapes as well.
These results keep both wider-K strict candidates disabled in serving.

A separate `GARNET_METAL_MPP_AFFINE_RELAXED` experiment is now explicit and
default-off, allowed only with the BF16-operand candidate. The installed MLX
NAX implementation sets the tensor API's relaxed-precision descriptor flag;
this motivates testing the implementation difference, not assuming a speedup
or silently changing the qualified control. Native backend metadata, frozen
profiles and per-cell records identify this mode; mismatched modes reject
before model loading. Ordinary Metal compile math remains safe, operand types
remain explicit, and accumulator/output storage remains FP32.

The relaxed BF16 K64 candidate passed the same24 independent projection cases
and12 CTests without changing numerical tolerances
(`mpp-bf16-relaxed-k64-ctest.log` and expanded log). This is not learned coding
qualification. An isolated timing comparison is next. The audit also now checks
frozen collector hashes and rejects tampered tile/precision metadata; seven
CPU tamper cases pass against the archived K-tile comparison.


The explicit relaxed BF16 K64 operator campaign completed three cohorts and
21 shapes per variant (`evidence/affine-mpp-precision-comparison1`). Relaxed
arithmetic gave no material overall advantage; it remains default-off and has
no full learned coding qualification.

`GARNET_METAL_MPP_AFFINE_N_MAJOR` adds a separately identified, default-off
coalesced packed-K load layout, stages contiguous N-major weights and sets the
right tensor operand transpose explicitly. Strict BF16 K64/K128 both passed
24 independent numerical cases and all12 CTests. The frozen sequential
`evidence/affine-mpp-nmajor-comparison1` campaign passed all21-shape audits and
eight tamper checks. N-major K64 versus tall FP32 gave0.93..0.96x at8 rows,
0.94..1.01x at16,1.25..1.40x at32 and1.22..1.43x at64. K128 was less effective.
These explicit precision-profile operator ratios do not establish end-to-end
speed or coding quality; candidates remain disabled by default.

The experimental batch factory now permits bounded batch2 chunks16/32/64 only
with wide-prefill, in-place-cache and per-row-hidden native extensions. The
normal decoder still admits only batch1 wide prefill; batch4 stays chunk<=8.
A batch2/context4096/chunk32 parallel ragged learned admission is in progress,
using the already qualified FP32 control and frozen single-request tokens.
Native workspace guards remain enabled; neither allocation nor correctness is
assumed from profile construction.

The first batch2/context4096/chunk32 admission hit the native workspace guard
while constructing graphs (`native27b-batch2-wide32-context4096-study1`). It
exited1, retained diagnostics and released its resources. The limit was not
relaxed. Inspection found fully initialized layout, mask, linear-prepare,
chunk-convolution, chunk-delta and whole-width gated-normalization temporary
buffers excluded from existing cross-layer scratch sharing. Sharing now
includes those outputs under the existing alias/weight/input/output protection
and first-read/first-write proofs. Partial gated-normalization writes remain
excluded. All12 native component tests pass.

The identical wider profile then passes two exact16-output-token repeats for
79/71-token ragged prompts, with genuinely combined native decode and fresh
scheduler resets (`native27b-batch2-wide32-context4096-study2`). Allocated Metal
storage after graph construction is17,850,728,448 bytes and returns to720,896
bytes after close. The finite-study auditor independently matches frozen cases
to the pinned single-request oracle and verifies two repeats, process exit,
sampled admission limits and cleanup; five tamper checks pass. This establishes
neither execution to the4K limit nor full coding/performance qualification.

The matching batch2/context4096/chunk64 finite admission also passes two
exact16-token repeats, including masked final chunks
(`native27b-batch2-wide64-context4096-study1`). Graph-ready allocations are
18,259,361,792 bytes, below the unchanged device limit minus256MiB; close returns
to720,896 bytes. The independent audit and all five tamper cases pass. The full
eight-task concurrent coding collector is now running as
`native27b-batch-wide64-quality1`, prefix disabled with explicit1800s queue/client
observation limits. No quality pass or serving speedup is claimed before that
collection, independent sandbox evaluation and native accounting audit finish.
Coding collections now retain factory/decoder sources and CMake flags as well
as the native library and transport scripts.

Full wide64 concurrent coding qualification completed successfully:
`native27b-batch-wide64-quality1/evaluation.json` reports8/8 sandbox passes.
`input-output-audit.json` has zero issues, all eight full sequences exactly
match the frozen single-request control, and1037 combined decode plans are
observed. Epoch/refill, frozen inputs, uncached usage, native exit and720,896-byte
close baseline pass; all five existing accounting tamper checks pass. The auditor
now resolves exactly one hash-verified profile by its profile identity rather
than assuming the collector's input filename was `profile.json`. Original
collection artifacts remain unchanged. A six-cell, one-wave timing admission
is frozen in `native27b-batch-wide64-http-admission1`; it is not the required
three-cohort serving campaign.
