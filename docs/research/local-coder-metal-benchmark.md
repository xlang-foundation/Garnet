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

The latest batch2/chunk128 bounded32-bit kernel-profile admission completes the
two single-client cells with128 outputs each. Short/long overall throughput is
4.070/1.801 output tokens/s, TTFT13.050/49.951s and decode6.913/6.024 tokens/s.
These are single admission waves without warmups, not repeated serving
qualification. The preceding64-bit profile had decode5.305/4.930 tokens/s and
TTFT16.898/70.731s; the older batch1 tall-FP32 control had decode6.732/5.582
tokens/s and TTFT20.518/101.057s. Reference single-client medians are
vLLM5.761/3.382 overall tokens/s and SGLang5.502/3.275. The batch2/wide64 FP32
admission retains its four-client long-prompt600s timeout as a failed campaign.
All2x targets remain unmet. Reference results stay fixed and engines run
sequentially. Full chunk-selection and isolated kernel diagnostics are recorded
below; neither substitutes for matched serving performance.

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

The wide64 FP32 batch2 timing admission finished with a preserved failure
(`native27b-batch-wide64-http-admission1`). Short-prompt overall tokens/s were
2.142/3.444/3.212 for1/2/4 clients. Long-prompt one/two-client cells were
0.539/0.697; the four-client cell failed a queued client's600s observation
limit. Exact one-client long TTFT was208.290s and decode4.345 tokens/s. Thirteen
native completions and496 combined decode plans are retained; the incomplete
cell has no valid overall throughput. The failed accounting report retains all
seven issues, including incomplete final memory bracketing and a native stream
without completion. The collector stopped its owned group viaSIGTERM with no
forced kill; this timing run has no graceful native-close snapshot. Separate
quality cleanup evidence is not substituted for that missing timing evidence.
Five tamper checks each introduce a distinct additional detected issue, without
turning the already failed campaign into a passing one.

A frozen isolated-op probe of the same binary
(`native27b-batch-wide64-op-probe1`) passed two exact16-token ragged repeats and
cleanup. Across serialized diagnostic submissions, MatMul accounts for94.00%
of summed GPU time:19,561.975ms for1,984 prefill projections with128 physical
input rows and4,394.117ms for14,912 two-row decode projections. Total diagnostic
GPU time is25,485.225ms. Some fusions are disabled by this mode, so these sums
attribute cost and are not serving latency or speedup measurements. Existing
projection microbenchmarks stop at64 rows; the128-row packed case remains a
specific missing operator measurement.

Native optional chunked prefill is now implemented in the backend-neutral
scheduler and Metal row adapter. Each chunk retains its physical row/epoch,
commits partial logical length and requeues the remainder; decode plans run
between chunks. Early final-token completion, wrong chunk counts, duplicate
partial completion and mode changes while requests are tracked reject. Final
chunks require the first sampled token. The whole-prompt default remains in
place. All12 native CTests pass, including portable row-lease/tail/cancellation
and decode-interleaving cases. The experimental text worker can select
`prefill_schedule=chunked`; prefix reuse with this new schedule is explicitly
unadmitted until native restored-prefix progress is integrated. The existing
whole-prompt native prefix profile remains unchanged. Full learned chunked
coding/HTTP qualification is next; no serving improvement is yet established.

Native chunked64 concurrent coding completed:8/8 sandbox passes, every full
sequence exactly matching the frozen single-request control,1,058 combined
decode plans and9 observed decode advances between prefill chunks. Frozen
chunk progression, tail sizes, row epochs, native close and uncached usage pass;
all eight accounting tamper cases reject, including changed offsets, missing
chunks and removed interleavings (`native27b-batch-chunked64-quality1`). Serving
performance remains unmeasured for this schedule.

The projection collector now has explicit maximum packed rows64(default) or128,
so batch2/chunk64's actual shape can be measured. Metadata and shape audits
validate the frozen bound; old21-cell archives remain auditable without change,
and nine tamper checks pass. Native timing specifications can now request a
bounded client timeout and bridge-first graceful shutdown explicitly. A real
owned parent/child fixture verifies child cleanup before group disappearance;
legacy group termination stays the default for unchanged reference specs.

Next chunk selection study: compare8/16/32/64 on fixed prompts, context4K,
uncached policy and the same native FP32 projection build. Kernel operator
screening includes the actual16/32/64/128 packed row counts before expensive
model admissions. Smaller tails and interleaving tradeoffs must be measured;
no single chunk is assumed optimal for all prompt lengths or occupancy.

The actual128-row projection campaign completed three strictly sequential
cohorts for base/tall/strict-BF16/N-major-BF16 (`affine-packed128-comparison1`).
All24 cells and nine tamper checks pass. At128 rows, tall FP32 versus base is
1.30..1.35x and N-major BF16 versus tall is1.17..1.30x across the three major
projection shapes. These are operator ratios, not achieved model speedups.

The fixed tall-FP32 chunk-selection study is running as
`native27b-chunk-selection1`. All four short cases pass both exact16-token-per-row
repetitions and cleanup at4K allocation, using260/263-token prompts. First and
repeat loaded-native paired-prefill times (seconds) are:

| Chunk | First prefill | Repeat prefill | Metal bytes |
| --- | --- | --- | --- |
| 8 |41.648|25.289|17,544,396,800|
|16 |28.279|20.776|17,646,501,888|
|32 |26.595|20.308|17,850,728,448|
|64 |26.932|20.837|18,259,361,792|

The32/64 repeat difference is small with only two observations; do not promote
one as a statistically established universal winner. These native-prefill probes
exclude initialization/reset/HTTP and are not a serving-speed comparison. Long
2054/2057-token cases remain in progress. The sweep summary verifies that the
library, CMake flags, model factory, decoder and runner remain identical, and
that only chunk size varies within each prompt regime. Partial summaries remain
explicitly `in_progress` until all eight probes finish. No adaptive multi-shape
resident profile is admitted merely by completing separate fixed-shape probes.

The long8 chunk probe completed both exact-token repeats and bounded cleanup:
255.713/256.621s paired native prefill for2054/2057 tokens. Larger long chunks
remain running sequentially. Four sweep-auditor tamper cases reject negative
timing, re-signed changed CMake flags, changed planned context and missing
coverage while retaining `in_progress` for the unfinished sweep.

A separate default-off BF16 MPP register-cooperative-tensor candidate is being
prepared. It uses independent16x32x16 SIMD-group operations within64-row output
blocks, avoiding shared-memory operand staging and cross-SIMD barriers. The
outer reduction block and inner tensor extent are reported separately; native
profiles/audits identify register versus staged operands explicitly. All prior
operator archives remain auditable, and eleven metadata/timing tamper checks
pass. This candidate has not been built or numerically qualified yet: builds
and GPU tests must wait until the frozen chunk sweep exits. No physical NAX
instruction use, model correctness or speedup is inferred from the tensor API.

The chunk sweep is complete (`native27b-chunk-selection1/final-audit.json`):
all eight fixed-shape probes, frozen input/library/config invariants and four
tamper checks pass. Long paired-prefill first/repeat seconds are8:255.713/256.621,
16:215.738/200.624,32:199.919/192.026,64:203.627/197.312. All preserve exact16-token
outputs per row in both repeats. Chunk32 has the lowest observed repeat time in
both regimes and uses408,633,344 fewer Metal bytes than64, but two observations
are not a universal optimality or a repeated serving admission. Prefill remains
far below reference serving performance; no2x target is established.

The register tensor candidate's first shader attempt failed because `fragment`
is a Metal stage keyword. The original shader/library/cache and failed test log
are retained in `mpp-bf16-registers-attempt1`. Renaming that loop variable yields
all12 CTests passing, including24 independent affine cases through512 rows and
matrix/K/output tails (`mpp-bf16-registers-ctest2.log` plus expanded log). Backend
metadata identifies strict BF16 operands, FP32 accumulation, register storage,
outerK64 and tensorK16. Full-model coding and serving remain unqualified.
Sequential operator timing against unchanged tall/staged controls is next.

The strict BF16 register candidate completed three fresh-process operator
cohorts against tall FP32 and staged N-major BF16 controls
(`affine-registers-comparison1`):24 cells, all hashes/route/dtype/storage/extent
audits and eleven tamper checks pass. Versus tall FP32, register speed ratios
are1.53..1.86x at16 rows,1.33..1.58x at32,1.25..1.57x at64 and1.23..1.31x at128.
These are explicit-precision operator measurements, not serving or physical-NAX
proof. The next full-model coding run is
`native27b-mpp-registers-chunk16-quality1`: batch2/context4096/chunk16, native
chunk scheduling, prefixOFF, strict BF16 tensor operands and FP32 accumulator/
output. Its profile requires actual native MPP dispatches, outerK64, tensorK16
and register storage. No full-model quality or model speed result is assumed.

The register candidate's full coding run is now complete:
`native27b-mpp-registers-chunk16-quality1` passes8/8 sandboxed coding tasks.
All eight generated token sequences exactly match the FP32 single-request
reference. The raw-stream audit passes with978 combined decode plans and45
prefill/decode interleavings; all eight audit tamper cases are rejected. This
qualifies the finite coding suite, not a serving speedup or general precision
equivalence. The collected profile allocates17,646,501,888 Metal bytes.

`packed_token_plan.h` begins the proposed packed hybrid executor contract.
It packs actual scheduler segments without padding, orders decode first, and
rejects stale/repeated leases, invalid prompt/context ranges, unsupported stages,
position overflow, and total token-budget overflow. The portable scheduler test
passes these checks on real mixed scheduler output. This metadata helper is not
yet connected to a GPU executor and provides no measured speedup.

`native27b-registers-chunk16-http-admission1` completed all six single-wave,
zero-warmup timing cells. Aggregate output tokens/s for clients1/2/4 are
2.33488/3.84976/4.29588 at short input and0.65457/0.84110/0.85533 at long input.
The accounting audit passes all14 requests with432 combined decode plans and
clean native close/worker exit0; all five distinct campaign tamper checks pass.
Single-client TTFT/decode are25.826s/4.385tokens/s short and161.215s/3.702tokens/s
long. This is admission evidence, not three-cohort qualification, and all2x
performance targets remain unmet. Frozen reference engines were not rerun.

The packed native metadata integration builds and all13 CTests pass with
lifetime reuseOFF (`packed-contract-ctest1.log`). The real mixed-layer native
scheduler fixture also passes (`packed-native-scheduler1.log`), checking every
packed segment against source request/epoch/position metadata and comparing
per-request GPU execution. The opt-in masked mixed-forward worker still needs
its learned coding gate. Its default remainsOFF.

The separate default-OFF lifetime scratch candidate compiles with its flagON
and passes all13 CTests (`prefill-lifetime-ctest1.log`), including overlap-boundary
and deterministic interval-planner checks. The native test set does not prove
learned prefill alias correctness or actual memory savings; those remain
required before profile admission. Mixed-forward and lifetime experiments use
separate profiles/builds so their effects can be attributed independently.

`native27b-registers-mixed-chunk16-quality1` passes8/8 sandboxed coding tasks,
with all eight complete generated token sequences exactly matching the FP32
single-request reference. Its stream audit passes1127 combined decode plans,
47 actual masked mixed forwards and41 prefill/decode interleavings. All eleven
tamper checks pass, including missing/changed mixed-call evidence. This proves
the finite mixed coding gate, not a serving speedup or compact projections.

`GARNET_METAL_INDEXED_AFFINE` is a new default-OFF native primitive, requiring
strict BF16 register MPP. It projects only explicit unique token indices,
loads/stores existing physical activation rows, and zeros inactive outputs via
an ordered blit. Even one selected token retains BF16 operands/FP32 accumulation;
it does not silently take the FP32 small-row decode path. It has no graph/model
routing yet. All13 native tests pass (`indexed-affine-ctest1.log` and expanded
log), including41 indexed mask cases across physical rows1/17/32/65/128/512,
K64/128/5120/17408, column tails, reversed/noncontiguous selections, empty/full
masks, reused output storage, independent BF16/FP64 oracles and rejected duplicate/
out-of-range selections. Existing24 affine numerical cases also pass. No indexed
operator timing or learned-model result has been measured.

`native27b-lifetime-chunk16-quality1` passes8/8 sandboxed tasks and all eight
complete token sequences exactly match the FP32 reference. Its stream audit
passes1005 combined decode plans and all eight tamper checks. Ready Metal
allocation is17,642,569,728bytes versus17,646,501,888 for the same chunk16
register control: only3,932,160bytes saved. This is a limited saving and does
not establish that512-token prefill will fit. Larger workspace owners still
need inspection; no performance target is satisfied by this result.

The separate default-OFF `GARNET_METAL_INDEXED_TOKEN_PROJECTIONS` graph route
builds and passes13 native tests (`indexed-token-ctest1.log`). A lowering pass
proves token order from a six-input chunk graph's hidden input through known
layout, row-local, projection and chunk attention/recurrent operations,
resolving aliases. Unknown/state layouts remain on the original path. Shape
coincidence alone never enables indexing. The resident session derives indices
from validated active flags and native leases; the head and ordinary decode
retain their original route. Full-active chunks avoid indexed zeroing overhead.
The frozen worker profile requires an exact indexed-policy backend match and
emits actual sparse dispatch counts before head prediction can reset counters.
The previous mixed coding archive still passes all eleven audit tamper checks.
Learned indexed model qualification remains pending.

`native27b-indexed-mixed-chunk16-quality1` passes8/8 sandboxed tasks, all eight
full token sequences exactly matching the FP32 single-request reference. The
stream audit passes1061 combined decode plans,51 mixed forwards,45 interleavings
and56 indexed-prefill diagnostic records. Sparse records demonstrate496 actual
indexed projection dispatches; full-active chunks keep the regular route.
All13 tamper checks pass, including missing/invalid indexed-route records.
The captured memory inventory declares17,645,764,256 unique resource bytes:
14,425,303,040 weights,1,391,460,384 session state,23,884,288 shared scratch,
1,368,436,736 other prefill storage and436,679,808 other decode/head storage.
A hash-linked inspection report identifies594,542,592bytes with no resident
read/write references; this diagnostic alone is not a proof to free them.

`GARNET_METAL_RETIRE_NONRESIDENT_PACKS` is a separate default-OFF candidate.
It retires only uniquely owned, default-layout, complete concat outputs with
one writer, no earlier reader, no resident storage references, and no input,
public output-part or weight ownership. Existing standalone materialization
restores these defaults before packed execution. The candidate builds and
passes13 native tests (`retired-packs-ctest1.log`). Initial restoration checks
used a decode fixture with no extra retireable pack, then failed to re-retire
already-restored buffers between cycles; both failure logs remain. The corrected
prefill fixture (`retired-packs-native-restoration3.log`) exercises an actual
retired pack in both linear and full attention, three cycles each, checking
standalone/private-session parity, rehydration and two-session/host-input
isolation. Native candidate code was unchanged between these fixture repairs.
Learned-model cleanup, memory saving and quality remain unqualified.

The next indexed serving admission is frozen at
`native27b-indexed-mixed-chunk16-http-admission1`: only short/long single-client
cells, one cold measured wave each, no warmups,128 maximum generated tokens.
This isolates latency feedback before larger-batch changes; it does not replace
the full repeated serving campaign or qualify aggregate throughput. It uses the
already-qualified frozen indexed library/worker and graceful native shutdown.

The indexed single-client admission completed both frozen cells with clean
native close/worker exit0. Short/long TTFT is22.1556/112.4734s, decode is
4.5233/3.8069tokens/s, and end-to-end time is50.2590/145.8518s for128 generated
tokens each. The earlier batch2/chunk16 control measured25.8263/161.2150s TTFT
and4.3850/3.7023decode tokens/s. These are individual cold samples, not repeated
speedup qualification; the earlier batch1 tall control's long TTFT101.057s is
still lower. All requested2x targets remain unmet against frozen references.
The first campaign audit falsely required combined decode in this predeclared
single-client-only manifest. It is preserved as `accounting-audit1.json`.
The corrected auditor derives the requirement from frozen request-body counts,
not observed completions. `accounting-audit2.json` passes both requests; all
four applicable tamper checks pass. The existing six-cell concurrency campaign
still passes all five checks including removal of combined decode.

Larger token budgets have another concrete guard: the current signed-char
attention token index path and explicit127-token cap cannot admit256-token
chunks. A future large-chunk extension must change representation and scratch
accounting together, then verify high token indices, old-cache selection,
causal overwrite and inactive queries against the independent oracle. Raising
only the public chunk limit would be unsafe. The installed MLX primary source
`mlx/backend/metal/kernels/quantized_nax.h` also shows64x64 block tiling with
cooperatively dequantized shared weights and per-SIMD register tiles. This is a
specific projection-reuse reference for larger-M work, not evidence that our
current register kernel or scheduler budget already matches its performance.

`native27b-retired-packs-chunk16-quality1` passes8/8 sandboxed coding tasks;
all eight full generated sequences exactly match the FP32 reference. The stream
audit passes1034 combined decode plans, clean close and all eight tamper cases.
Ready Metal allocation is17,051,959,296bytes,594,542,592bytes below the otherwise
matching register chunk16 control. This is a device-resource saving, not a claim
about peak system RAM. It qualifies the finite chunk16 memory/coding gate.

The default-OFF large-prefill extension now compiles in separate control and
indexed builds. Each passes13 native tests (`large-prefill-ctest1.log`,
`large-indexed-ctest1.log` plus expanded logs). At4K capacity the independent
attention oracle passes128-token contiguous and256-token overwritten-slot cases
with2-byte signed indices (max errors1.35e-8/2.68e-8); batch2/chunk256 is also
covered at512 capacity. Projection cases now include256x5120 and512x17408 rows/
inner dimensions (max errors4.14e-6/8.13e-6) without tolerance changes. The indexed
build retains all41 indexed mask cases. Prefill standalone restoration and native
scheduler GPU reference checks also pass in that build. These are component
results, not learned large-profile or serving acceptance.

The xModel batch study exposes128/256 only for batch2 with the explicit large,
16-bit-index and retired-pack extensions; base shapes and batch4 limits remain
unchanged. The worker requires exact large/indexed/retirement and optional
contiguous-slot policy identities. The first learned candidate is
`native27b-large-indexed-mixed-chunk128-quality1`: context4096,batch2,chunk128,
strict BF16 register operands, indexed/mixed execution, pack retirement and the
GPU-verified contiguous slot fast path. It uses one fixed graph family and the
existing recommended-working-set-minus256MiB allocation guard. Chunk256 is not
assumed to fit or qualify. No learned result or speedup is assumed for128 yet.

The chunk128 coding run has now completed:8/8 sandboxed tasks pass, all eight
full token sequences exactly match the single-request reference, the stream
audit records1,063 combined decode plans and9 mixed forwards, and all13
corruption checks pass. Ready Metal allocation is18,188,271,616bytes and close
returns to720,896bytes. Its timing admission is separately frozen at
`native27b-large-indexed-mixed-chunk128-http-admission1`: the same qualified
library and worker, short/long single-client cells, one cold wave each, no
warmups. This does not qualify repeated performance or concurrent throughput.

### Further direct vLLM/MLX inspection and cooperative projection experiments

`evidence/vllm-source-inspection2` freezes and hashes the installed vLLM0.32.0
core scheduler/cache manager, pinned vLLM-Metal runner/linear wrapper, and
MLX0.32.4 quantized headers. Core `schedule()` budgets outstanding token
positions using `num_computed_tokens`, `num_tokens_with_spec` and remaining
shared input capacity; it is not a fixed per-request chunk loop. Core
`get_computed_blocks()` delegates longest block-hash prefix lookup to the cache
coordinator. The actual baseline disables prefix caching, so cache reuse cannot
explain its uncached speed. The Metal runner packs valid decode/prompt tokens
into one matrix, while each request retains independent convolution/recurrent
state. Garnet's indexed projections skip inactive rows, but other graph
operators and storage remain padded. These are distinct scheduling, kernel and
layout gaps; matching the512-token budget alone does not fix them.

The first default-OFF `GARNET_METAL_MPP_AFFINE_REUSE64` prototype stages64x64
BF16 A/B tiles, reuses dequantized weights across64 queries and uses an MPP
operation spanning four SIMD groups. Independent native references and all13
CTests pass without tolerance changes (`reuse64-ctest1.log`). The frozen
`affine-reuse64-comparison1` measures30 shapes over three sequential cohorts
with3 warmups and9 samples. Larger256/512-row synthetic projections improve,
but some64/128-row cases regress. The eight actual checkpoint projection
dimensions are separately measured in `affine-reuse64-qwen-shapes1` (80 cells
per variant, three cohorts). This includes48-output gates and1024-output K/V
projections; a5120x5120 synthetic shape is not representative of all layers.
Both campaigns pass route/precision/hash/count audits. Initial sources/library
are retained in `affine-reuse64-sources1` before changing the prototype.

A direct isolated MLX large-prefill probe uses the same deterministic affine
INT4/group64/BF16 operands, three fresh shape traversals and3+9 warmup/measured
calls (`mlx-affine-shapes-probe2.json`,96 cells). It excludes loading,
compilation and transfers, measures synchronized wall time, and verifies three
FP64 samples per shape plus all-output finiteness. MLX stores BF16 output;
Garnet stores FP32, so this is a diagnostic comparison with different output
precision. The first probe incorrectly applied the matrix oracle to MLX's
small-row grouped quantized-matvec arithmetic and failed; its partial archive
and traceback are preserved. The corrected probe explicitly covers only
64/128/256/512-row matrix work and passes the unchanged matrix-oracle bound.
No serving reference is rerun or replaced by this operator probe.

MLX's source combines cooperative padded weight staging with per-SIMD register
matrix operations. A second Garnet prototype follows that structure using a
64x64 weight tile padded to72 BF16 elements per row, two16x32 FP32 accumulators
per SIMD group and mapped active-row loads/stores. It keeps strict BF16 operand
and FP32 accumulator/output semantics. All13 native tests pass
(`reuse64-registers-ctest2.log`). `affine-reuse64-qwen-shapes2` passes80-cell
three-cohort accounting; ordinary decode still uses its original kernel.
Illustrative synchronized operator wall medians in milliseconds:

| Valid rows; output x inner | Existing registers | First cooperative prototype | Second cooperative prototype | MLX BF16 output |
| --- | ---: | ---: | ---: | ---: |
|128;17408x5120|10.401|7.791|8.203|2.099|
|128;5120x17408|10.605|10.346|8.566|2.428|
|256;17408x5120|22.931|15.942|16.012|3.942|

The existing-register column is from the first actual-shape campaign; its raw
control is remeasured in the second campaign and retained, rather than assuming
identical controls across runs. These are isolated operators, not model token
rates or a speedup qualification. Narrow projections still regress under the
new prototype. Both prototypes remain default-OFF with full learned coding and
serving gates pending; all2x goals remain unmet.

The chunk128 single-client admission then completed both cells with a passing
accounting audit, all four applicable corruption checks, native close and
worker exit0. Short/long TTFT is22.4204/104.6731s; decode is4.2286/3.5956tokens/s;
full request time is52.4915/140.0375s for128 output tokens each. Relative to the
previous indexed chunk16 cold samples, long TTFT improves6.94%, but short TTFT
and both decode rates regress. The older batch1 tall control's101.057s long
TTFT remains lower. These are cold admissions, not repeated speedup evidence.
Frozen vLLM-Metal's corresponding long single-client TTFT is15.228s.

Based on actual-shape operator results, the next default-OFF cooperative
dispatch keeps narrow output projections on the existing kernel: at least64
valid rows **and**5120 outputs are required. The backend/worker profile exposes
the threshold explicitly. New independent route/oracle cases cover63/64/65
rows, large unaligned output widths and sparse/reversed128-row maps. Earlier
unconditional prototype timings remain archived and cannot stand in for this
gated policy's measurements or learned-model qualification.

The gated build passes all13 native tests including the new route boundaries
(`reuse64-gated-ctest3.log`). Its frozen sequential operator campaign,
`affine-reuse64-gated-qwen-shapes3`, passes80-cell accounting and16 independent
corruption checks. GPU-time speed ratios versus the matched strict register
control for projections with at least5120 outputs are1.035..1.159x at64 valid
rows,1.128..1.178x at128,1.163..1.238x at256 and1.244..1.503x at512. Narrow
projections retain the original route. These component ratios are not token
rates, a repeated model campaign or evidence that the2x target is reached.
The next learned gate uses the explicit `require_reuse64` and
`reuse64_min_outputs:5120` profile identities on batch2/4K/chunk128.

`native27b-reuse64-gated-chunk128-quality1` now passes8/8 sandboxed coding
tasks and all eight complete reference token sequences. Its audit records1,128
combined decode plans,7 mixed forwards,9 indexed/reuse64 prefill calls and a
clean close; all15 corruption checks pass. The qualified library is frozen
for `native27b-reuse64-gated-chunk128-http-admission1`. Both single-client
cells pass serving accounting and four corruption checks, with128 outputs each
and graceful native close/worker exit0. Short/long TTFT is17.4112/82.1074s,
decode4.5663/3.8018tokens/s and total request time45.2549/115.5511s. Compared
with the prior chunk128 cold samples, TTFT is about22% lower. These individual
cold admissions do not establish repeated performance or attribute changes in
the unchanged decode kernel to prefill optimization. Frozen vLLM-Metal/SGLang
references remain unchanged, and all2x targets remain unmet.

### Active request rows for FP32 decode

The existing batch2 graph still projects both physical rows when only one
request is decoding. Unlike indexed BF16 prefill, this requires preserving
ordinary decode's unrounded FP32 operands and reduction order. The new default-OFF
`GARNET_METAL_INDEXED_DECODE` primitive specializes the selected row count1..4,
uses validated indices for loads/stores, zeros inactive outputs, rejects invalid
or duplicate rows before encoding and retains command ownership checks. Five
native shapes exercise46 mask variants, including reversed maps, reused buffers,
real inner widths5120/17408 and large unaligned output dimensions. Selected
outputs must equal ordinary decode bit-for-bit as well as satisfy the existing
independent FP64 bound. The first test run failed because its uninitialized-input
check expected `invalid_argument` instead of the intended `logic_error`; the
corrected fixture passes all13 CTests (`indexed-fp32-decode-ctest2.log`). The
native candidate and numerical tolerances were unchanged by that fixture repair.

`GARNET_METAL_ACTIVE_DECODE_PROJECTIONS` separately enables graph/session routing,
requiring the primitive, native continuous rows and final hidden rows. Lowering
proves request-row identity from2D decoder/head hidden inputs through supported
operations and stateful controls; unrelated equal-shape matrices do not suffice.
The session derives sparse rows from validated flags/leases and prediction
epochs, while full-active projections retain their existing route. Prefill
keeps its explicitly identified BF16 policy. The new routed build passes all13
native tests (`active-fp32-decode-ctest1.log`), but full learned concurrency and
serving qualification remain pending.

The one-request preflight first failed its coverage assertion because a native
decode command includes the vocabulary head: its counter must cover496 body
projections **plus**1 head, not496 alone. That failed archive and cleanup remain
at `native27b-active-fp32-decode-preflight1`. The worker now records actual
coverage before checking and distinguishes combined decode commands from
standalone head prediction. Without changing native code, preflight2 passes:
the frozen71-token stable-unique prompt produces all62 reference output tokens
exactly,61 decode commands each report497 indexed FP32 projections, the initial
head reports1, and native close returns to720,896bytes with worker exit0. This
is one learned active-row preflight, not concurrent coding or speed acceptance.

The concurrent active-row gate, `native27b-active-fp32-decode-chunk128-quality1`,
passes8/8 sandboxed tasks and all eight complete reference token sequences.
It records1,139 combined decode plans,9 mixed forwards and1,165 FP32 projection
records:1,139 full-active decode commands use the original route,16 sparse
commands report497 indexed projections, and three sparse standalone heads
report1. Seven full-active standalone heads report0. All20 corruption checks
pass; cleanup returns to720,896 Metal bytes.

The matching frozen single-client admission completes both128-output cells.
Short/long TTFT is20.4280/81.6551s, decode5.1582/4.2548tokens/s and total time
45.0788/111.5435s. Decode is higher than the preceding cooperative-prefill cold
samples4.5663/3.8018, but short TTFT regresses and these are individual cold
admissions. The older batch1 tall control still has higher decode rates than
this batch2 profile. All2x targets against frozen references remain unmet.
`accounting-audit3.json` additionally matches every pure decode command to its
native plan, subtracting actual mixed executions, and verifies exact FP32
projection coverage. Eight corruption checks pass, including missing commands,
missing sparse head coverage and altered projection counts/stages. CPU edge
cases preserve valid first-token-only EOS and full-active-only campaigns.

The cooperative SIMD tensor-policy experiment is separately identified by
default-OFF `GARNET_METAL_MPP_REUSE64_RELAXED`; ordinary prefill/decode policy is
unchanged. Its source reference is the installed MLX0.32.4 NAX descriptor in
`steel/gemm/nax.h`, frozen in `reuse64-tensor-policy-reference1`. All13 native
tests and the unchanged FP64 bounds pass. The80-cell, three-cohort operator
comparison `affine-reuse64-relaxed-qwen-shapes1` passes precision/route accounting
and17 corruption checks, but shows no useful speed advantage: median strict/
relaxed GPU ratios for large-output shapes at64/128/256/512 rows are
0.985/0.972/0.969/0.973. Keep the strict serving candidate. The relaxed candidate
has no learned-model or serving admission.

### Bounded prefill hidden-input retirement

The qualified chunk128 memory inventory identifies335,544,320bytes in64 hidden
input defaults, plus the same amount in hidden outputs. This is a resource
inventory, not permission to free arbitrary outputs. The new default-OFF
`GARNET_METAL_RETIRE_PREFILL_INPUTS` targets only uniquely owned, default-layout,
read-only hidden inputs of explicit six-input chunk graphs. Checkpoint owners,
aliases and public outputs remain excluded. The first layer retains its freshly
uploaded input; later layers borrow the predecessor's hidden output and may
retire their unused defaults. Standalone execution uses the existing lazy
materialization path. Backend/profile identities and per-layer planned-slot
checks must agree before loading this research profile.

The candidate passes13 native tests and the linear/full-attention standalone,
three-cycle restoration, two-session/host-input isolation and native scheduler
GPU fixtures (`retired-prefill-inputs-ctest1.log`, `...-restoration1.log`,
`...-scheduler1.log`). `native27b-retired-prefill-inputs-chunk128-quality1` also
passes all eight sandboxed coding tasks with all eight full token sequences
equal to the single-native reference,1,019 combined decode plans, seven mixed
forwards and20 corruption checks. Ready allocation is17,852,727,296bytes,
335,544,320bytes (320MiB) below the prior profile. Completed-request samples
are17,858,068,480bytes; these are snapshots, not peak measurements. Native
close returns to720,896bytes and the worker exits0. Larger chunk admission
remains pending. Retiring an input does not prove that the corresponding output
can be pooled or that chunk256 fits this Mac.

### Compiling bounded kernel profiles

A profile specifies **upper bounds**, not mandatory actual input lengths or
occupancy. For example, a512-row operator profile accepts1,2,4,8 through512
actual rows without requiring padding to512. A serving profile additionally
caps request slots, context and scheduler chunk budget. Model dimensions,
quantized weight layout and buffer strides can be compiler constants while
actual token counts, positions, selected-row maps and lease epochs remain
runtime data. Compiling a kernel profile is distinct from merely allocating a
fixed-size buffer or rejecting all shapes except one exact input length.

The default-OFF `GARNET_METAL_AFFINE_PROFILE` is a first native implementation
of that contract for affine projections. Preparation compiles K,N and both
storage strides into Metal function constants for regular decode, register
prefill, indexed FP32 decode, indexed BF16 prefill and cooperative64 routes.
It prepares every possible1..4 decode occupancy within the bound; prefill row
counts stay dynamic. Pipeline entries record the prepared row bound and their
cache is capped at512 entries. Encoding performs lookup only, reports actual
specialized dispatches and misses, and uses the original pipeline on a miss.
The learned worker requires exact body/head coverage and zero misses.

Both control and candidate pass13 native tests. A dedicated case prepares
an up-to128 profile from a one-row input, accepts1,2,4,5,63,127,128 actual rows,
uses fallback for129, and accepts129 after widening the prepared bound to512.
It checks idempotence, rejects0/513 and compilation during an open command,
and proves that encoding does not grow the pipeline cache. Existing FP64
bounds, inactive-row and indexed-decode exact-parity tests remain unchanged.

`affine-profile-qwen-shapes1` freezes80 Qwen body-projection cells, three
fresh-process cohorts per variant, three warmups and nine measured calls.
The only build difference is bounded profile specialization; tile and precision
policies are unchanged. All80 cells pass route/precision accounting and21
corruption checks. Median control/profile GPU latency ratios across eight
shapes are1.154 at2 rows,1.122 at128,1.084 at256 and1.092 at512. One-row
results regress in several shapes: median ratio0.964, with a weighted body
projection sum ratio0.875. Thus this is not a uniform gain. The body sum is an
operator diagnostic excluding the head, state, activations and serving costs.
These results do not establish tokens/s, a vLLM speedup, or any2x target.

Native sources and hashes are frozen in `affine-profile-sources1` and the
coverage-instrumented `affine-profile-sources2`. The first eight-task run,
`native27b-bounded-affine-profile-chunk128-quality1`, passes all eight sandboxed
tasks with all full sequences equal to the single-native reference. Its1,221
profile commands comprise1,201 combined body/head decode commands, ten
prefills and ten standalone heads, with zero misses. However, its original
audit fails the generic-engine720,896-byte close-allocation requirement:
the candidate closes at1,507,328bytes with worker exit0. Preserve that failed
audit; this run is not the qualified serving candidate.

A dedicated lifecycle investigation preserves an initial failed assumption
that all Metal driver allocation returns to the cold-device baseline. The
owner-instrumented fixture then checks all Garnet matrix, affine weight, BF16
weight and pipeline-set owners reach zero. Three complete context/profile
create-destroy cycles return to the same737,280-byte warmed device allocation,
without growth or retained Garnet owners. This supports bounded driver
allocation; it does not identify a particular private driver allocation type.
All14 native tests, including that lifecycle test, pass.

`affine-profile-sources3` records native ownership instrumentation. The fresh
`native27b-bounded-affine-profile-chunk128-quality2` profile declares a2MiB
post-close allocation budget before loading the model and additionally requires
all four Garnet ownership counters to be present and zero. The generic control
keeps its original close requirement. This fresh gate passes all eight sandboxed
tasks, all eight exact full token sequences and28 corruption checks. Its1,176
profile commands comprise1,156 decode, ten prefill and ten standalone head
commands, with zero misses;1,138 decode plans combine requests and nine
forwards mix decode/prefill. Ready allocation is17,853,775,872bytes with497
affine weight owners,2,737 matrix owners and one pipeline set. Native close
reports all four owner counts zero,1,507,328 Metal bytes and worker exit0.
The matching `native27b-bounded-affine-profile-chunk128-http-admission1` uses
the exact qualified library and worker. Both C1 cells complete128 outputs;
the accounting audit and11 corruption checks pass. Its276 profile commands
include254 decode,20 prefill and two standalone head calls. Close reports
zero Garnet owners,1,507,328 Metal bytes and worker exit0.

| One cold C1 admission | Actual prompt tokens | TTFT (s) | Decode (tokens/s) | Total (s) | Overall output tokens/s |
|---|---:|---:|---:|---:|---:|
| Short |260|16.898080|5.304505|40.898939|3.129666|
| Long |2054|70.730545|4.929651|96.531900|1.325987|

The preceding active-FP32 batch2 candidate had short/long TTFT20.428/81.655s
and decode5.158/4.255 tokens/s in separate cold admissions. The latest profile
also includes hidden-input retirement, so these end-to-end differences cannot
be attributed to kernel specialization alone. The older batch1 tall control
still decodes faster. Frozen vLLM-Metal single-client TTFT is2.450/15.228s and
decode6.417/6.278 tokens/s; SGLang is2.389/14.633s and6.094/5.843 tokens/s.
Those references are repeated cohort medians, unlike this admission. All2x
targets remain unmet. The experiment does not yet constitute a complete
compiled engine memory arena, tile autotuner or portable production xModel
profile; a full matched repeated Garnet campaign is still pending.

### Compiler threadgroup bound: negative experiment

Default-OFF `GARNET_METAL_AFFINE_PROFILE_TGP128` sets the specialized pipeline
descriptor's maximum threads to128, matching the unchanged dispatch. It leaves
threadgroup-multiple assumptions, precision, dimensions and tiles unchanged.
The control reports an actual pipeline maximum of1024 and the candidate128.
Both pass14 native tests, including upper-bound and owner-lifecycle cases.

`affine-profile-tgp128-qwen-shapes1` passes80-cell/three-cohort accounting and23
corruption checks. Median control/candidate GPU ratios across shapes at
2/4/64/128/256/512 rows are1.002/1.001/1.000/0.998/0.985/0.993. One-row
weighted body sum ratio is0.984 and larger-row body sums likewise show no
useful improvement. Keep the compiler-bound option disabled. This candidate
has no learned-model or serving admission. Source hashes and build commands
are in `affine-profile-tgp128-sources1` and its configuration evidence.

### Proved bounded32-bit projection indexes

Default-OFF `GARNET_METAL_AFFINE_PROFILE_INDEX32` uses profile bounds to select
uint32 element indexes in specialized affine kernels. Generic pipelines retain
64-bit indexes; device pointers, KV/state layouts, packed parameters, precision,
tile geometry and arithmetic order are unchanged. Preparation rejects a
specialization unless the padded input/output row spans and packed-weight
indexes fit uint32, with additional room for masked tile coordinates and loop
increments. Division-based host checks avoid overflow in the proof itself.

The portable range test covers72 Qwen row/shape profiles including the untied
head, exact packed/stride boundaries and ten invalid cases. Both64-bit control
and32-bit candidate pass all15 native tests with the original FP64 bounds,
selected-row parity, upper-bound input and ownership-lifecycle requirements.
Source hashes and commands are in `affine-profile-index32-sources1` and its
configuration evidence.

`affine-profile-index32-qwen-shapes1` passes all80 cells across three cohorts
per variant and25 corruption checks. Median control/candidate GPU latency
ratios across shapes at1/2/4/64/128/256/512 rows are
1.095/1.232/1.318/1.666/1.681/1.660/1.658. Weighted body-projection sum ratios
at2/128 rows are1.266/1.673. These sums exclude the head, state, activations,
embedding, scheduler and transport; they are not tokens/s or whole-model
speedups. The compiler128-thread option remains disabled, so index width is
the only build-policy change in this study.

`native27b-bounded-index32-chunk128-quality1` passes all eight sandboxed coding
tasks with all eight full sequences matching the single-native reference and30
corruption checks. All1,176 projection commands report32-bit indexes and zero
misses:1,156 decode, ten prefill and ten standalone head commands. The trace
contains1,138 combined decode plans, nine mixed forwards and three prefill/
decode interleavings. Ready allocation remains17,853,775,872bytes. Close
reports zero matrix/affine/BF16/pipeline-set owners,1,507,328 Metal bytes and
worker exit0.

The exact qualified library/worker completes the two C1 cells in
`native27b-bounded-index32-chunk128-http-admission1`. Its accounting audit and12
corruption checks pass, with276 specialized commands and clean native close.

| One C1 admission wave | Actual prompt tokens | TTFT (s) | Decode (tokens/s) | Total (s) | Overall output tokens/s |
|---|---:|---:|---:|---:|---:|
| Short |260|13.050460|6.912587|31.450907|4.069835|
| Long |2054|49.950831|6.024020|71.067979|1.801092|

These separate cold-admission samples improve on the preceding64-bit profile,
but they are not repeated performance qualification. Short decode exceeds the
frozen vLLM median in this sample; long decode is below it, and both TTFTs and
overall output throughput remain substantially worse. All2x targets remain
unmet. The four-cell C2/C4 admission
`native27b-bounded-index32-chunk128-batch-admission1` also passes accounting and12
corruption checks:12 requests,495 actual combined decode plans and833
specialized projection commands. Four-client workloads queue through two
physical request rows. Natural EOS is retained and aggregate rates use actual
output counts, not an assumed128 tokens per request. Native close releases all
tracked owners, returns to1,507,328 Metal bytes and exits0.

| Prompt target | Clients | Garnet admission overall output tokens/s | Frozen vLLM-Metal | Frozen SGLang |
|---|---:|---:|---:|---:|
|256|1|4.070|5.761|5.502|
|256|2|5.581|7.352|6.834|
|256|4|7.881|9.935|7.546|
|2048|1|1.801|3.382|3.275|
|2048|2|1.701|3.525|3.281|
|2048|4|1.945|3.479|3.789|

Garnet entries are one wave per cell without warmups, from the two admissions;
references are medians across three repeated cohorts. The table does not
establish a repeated speedup. No reference engine was rerun. Long-prompt
prefill and queueing still dominate the batched overall rates, so further
stage attribution is needed; all2x targets remain unmet.

### Native stage attribution after bounded32-bit admission

The optional, default-false `trace_stage_latency` observer reads existing
native prepare/encode/wait/GPU timers, measures embedding and scheduler wall
time, and measures standalone head wall time. It deliberately omits head GPU
attribution because the session's aggregate timer belongs to the preceding
body execution. GPU duration is nested in wait time; those columns must not
be added together. No kernels, projection precision or scheduler policy change.

`native27b-bounded-index32-stage-diagnostic1` uses the exact qualified native
library with the explicit observer worker. Both complete128-token C1 sequences
match the preceding admission byte-for-byte. Native accounting and12 corruption
checks pass. `stage-summary1.json` verifies finite nonnegative durations,
native component/wall consistency,276 execution/head records and the library/
output identities. This is diagnostic attribution, not performance acceptance.

| Prefill stage sum | Short260 tokens | Long2054 tokens |
|---|---:|---:|
| Embedding wall time |0.014s|0.044s|
| Native preparation |0.003s|0.010s|
| Native encoding |0.023s|0.057s|
| Native wait, including GPU |14.982s|47.382s|
| GPU duration within wait |14.975s|47.366s|

The first short-prefill GPU chunk is12.476s, followed by1.808s for another128
active tokens and0.690s for the four-token tail. This first-use outlier does
not identify a private driver, paging, clock or compilation cause. It reinforces
that single cold admissions cannot substitute for the reference warmup/cohort
protocol. In the long prompt, full128-token chunk GPU duration grows from1.810s
to4.141s as the visible context grows; the six-token tail takes0.837s. CPU
embedding, preparation and encoding do not explain that growth.

Source inspection identifies a concrete attention candidate: `attention_chunk`
stores unnormalized exponentials and repeats `scratch[s]/denominator` inside
each output-dimension loop. The same denominator/probability pair is therefore
divided repeatedly. A future candidate can normalize each probability once,
synchronize the threadgroup and keep the original ordered value accumulation.
The candidate was subsequently implemented and rejected in the experiment
below. The trace alone does not establish a complete per-operator attribution.

### Shared attention probability normalization: negative experiment

Default-OFF `GARNET_METAL_CHUNK_PROBABILITY_REUSE` specializes a function
constant in `attention_chunk`: after the existing softmax denominator reduction,
normalize each probability once in shared memory, synchronize, then use the
original ordered value accumulation. A control pipeline retains the original
division expression. No extra scratch allocation, cache format, projection or
scheduler change is involved. Native coverage counters and an explicit worker
policy prevent silently measuring the control as the candidate.

The initial stricter bitwise gate fails: for batch2/chunk8/capacity64/dim8,
423 values differ, with maximum absolute delta1.86265e-9. The original main
shader library uses default compile options, unlike the explicitly safe affine
libraries; the installed Metal SDK documents default fast-math behavior.
The prototype's changed rounding therefore cannot be advertised as bitwise
equivalent. Initial sources/failures remain in `chunk-probability-sources1`,
`chunk-probability-strict-sources2` and the original CTest/parity logs.

The unchanged standard FP64 error bounds pass all15 native tests, including
real27B heads24/KVheads4/dim256 at4K capacity, causal overwrites and inactive
queries. The optional `--strict-bitwise` diagnostic remains failed and is not
converted into a pass. The numerical test also compares every candidate output
with control under the original bounds. Neither numerical acceptance nor the
kernel benchmark would substitute for exact frozen coding sequences and
sandboxed model correctness before serving admission.

`chunk-probability-kernel-comparison1` measures12 cases across three fresh
process cohorts, alternating control/candidate order within each cell, with
three warmups and nine measured calls per variant. Physical batch2/chunk128,
24/4 heads, dimension256,4K FP32 KV and gated outputs match the text profile;
one/two active rows, prefixes0/128/1024/1920/3968 and4/6-token tails are covered.
All collections and eight corruption checks pass. Maximum candidate/control
delta is1.11759e-8 and strict bitwise equivalence is explicitly false.
Control/candidate GPU latency ratios range0.989..0.997: the candidate is slightly
slower in every cell. Keep it disabled; it has no learned-model or serving
admission and does not meet any2x target. Source-level repeated division alone
did not establish an executed GPU cost that this rewrite improves.

Attention still uses64-bit index arithmetic in its hot loops, while the affine
profile's successful narrowing does not cover those kernels. Proved bounded
attention indexes and compiled geometry are the next candidates to evaluate.

### Bounded attention indexes and profile semantics

A compiled profile specifies upper bounds. A512-token input bound accepts
shorter inputs; it does not require512 real input tokens. Layout, resource and
tile choices can be compiled from that envelope while actual row counts,
positions and masks remain runtime values. Chunk capacity and total context
capacity are separate bounds. Sparse indexed projections skip unused token
rows; the fixed chunk graph still masks its inactive attention queries.

Default-OFF `GARNET_METAL_CHUNK_ATTENTION_INDEX32` narrows only the chunk
attention element-index arithmetic after a host proof checks all16 ABI fields,
batch/chunk/head products, logical widths and padded stride spans against
UINT32_MAX. Pointers and the16-ulong host parameter ABI retain native width.
Batch1..64, chunk1..256 and context up to4096 are bounded independently of the
number of active tokens. Unsafe spans are rejected before dispatch. The
original64-bit pipeline is compiled separately before ready as the A/B control.
Probability reuse remains OFF; CMake rejects enabling both experiments.

All16 native tests pass, including35 portable Qwen geometry profiles, nine
padded-stride boundary pairs, every wide ABI field, invalid geometry and the
original FP64 attention gates. The separate strict bitwise diagnostic also
passes every attention test case. No numerical tolerance was changed.

`chunk-attention-index32-kernel-comparison1` repeats the previous isolated
attention protocol: three fresh sequential process cohorts, three warmups,
nine measured calls per variant and alternating order within each cell. All
12 cells are bitwise equal with maximum control delta0; archive audit and
eight corruption checks pass. Times below are medians of cohort medians in
milliseconds, for physical batch2/chunk128, heads24/KVheads4/dim256, FP32 KV
capacity4096. Ratio is original64-bit time divided by candidate32-bit time.

| Active rows | Actual tokens per row | Cached prefix | Control GPU ms | Candidate GPU ms | Ratio |
|---:|---:|---:|---:|---:|---:|
|1|4|256|0.652|0.491|1.328|
|1|6|2048|8.107|7.395|1.096|
|1|128|0|3.795|2.943|1.290|
|1|128|128|9.845|8.239|1.195|
|1|128|1024|70.131|62.455|1.123|
|1|128|1920|151.266|138.079|1.096|
|1|128|3968|308.573|281.717|1.095|
|2|128|0|7.505|5.829|1.288|
|2|128|128|19.460|15.950|1.220|
|2|128|1024|143.204|127.967|1.119|
|2|128|1920|300.510|274.687|1.094|
|2|128|3968|614.772|560.415|1.097|

These are operator measurements, not token throughput or evidence of a2x
model gain.

`native27b-chunk-attention-index32-chunk128-quality1` subsequently passes all
eight sandbox coding tasks and all eight complete output-token sequences
exactly match the original single-native reference. Accounting and33 corruption
checks pass. The study records1026 combined decode plans, nine mixed forwards,
three prefill/decode interleavings and1278 checked attention-policy commands.
Each prefill command uses all16 full-attention layers; ordinary decode uses
zero chunk-attention dispatches. Ready allocation remains17,853,775,872bytes;
close allocation is1,507,328bytes with all four Garnet owner counts zero and
worker exit0. Serving performance remains a separate gate.

The exact qualified library/worker then completes
`native27b-chunk-attention-index32-chunk128-http-admission1`: one cold C1 wave
per cell, no warmups,128 outputs each. Both full output sequences match the
preceding affine-index32 admission. Native accounting and15 corruption checks
pass;274 checked attention-policy commands cover both requests.

| Actual prompt tokens | First token s | Decode tokens/s | Total s | Overall output tokens/s |
|---:|---:|---:|---:|---:|
|260|14.376|6.926|32.741|3.909|
|2054|47.738|6.032|68.830|1.860|

The preceding single cold admission recorded13.050/49.951s first-token times
and4.070/1.801 overall output tokens/s. The new short result is slower and the
long result is faster; these individual cold observations cannot establish a
repeatable model gain. Ordinary decode is outside this kernel change. The
frozen vLLM/SGLang warm/cohort references are unchanged. No2x target or full
matched campaign is satisfied. Retain the candidate as optional and keep
both attention experiment flags OFF by default.

### Profile-prepared FP32 cached/live matrix attention

Default-OFF `GARNET_METAL_CHUNK_ATTENTION_MPS` replaces the scalar chunk
attention contraction with four FP32 MPS matrix products. Cached and live
keys produce separate score matrices. The causal softmax selects the latest
active live token at each slot, sends its probability to the live operand,
and zeros the corresponding cache probability. Two value products and a
gated merge produce the output. Earlier queries retain old cache values when
a later token overwrites the same slot; cache commits still follow attention.
No BF16 conversion, cache-format change or speculative tokens are involved.
The qualified index32 kernel is the comparison control.

Matrix dimensions and seven scratch buffers are prepared during graph
compilation. Identical profiles share a device pipeline-owned plan across
graphs and the executing session. A native lease protects its mutable scratch
through partial submissions and GPU completion; preparation is forbidden while
encoding. The process cache is weak, so the final context releases all owners.
For batch2/chunk128/24 query heads/4 KV heads/dim256/capacity4096, the single
plan owns exactly121,634,816bytes (116MiB), independent of the number of layers.

All17 native tests pass the original FP64 and control-comparison bounds,
including causal overwrites, duplicate/backward slots, inactive rows, gated
and ungated outputs, chunk256 and real27B geometry. The new lifecycle test runs
two contexts against the same scratch through24 partial-submission runs in
each of two cycles, verifies separate outputs against control, rejects lazy
preparation and closes every Garnet resource owner. Device allocation after
each test cycle is737,280bytes. The explicit strict bitwise diagnostic fails
(420 values, maximum delta2.56114e-9 in its first small case); matrix reduction
order differs and bitwise kernel equivalence is not claimed.

`chunk-attention-mps-kernel-comparison2` freezes the shared-plan implementation
and uses three sequential fresh process cohorts, three warmups and nine
alternating-order measurements per variant. All12 cells pass numerical checks;
maximum control delta is2.60770e-8. Archive audit and eight corruption checks
pass. The initial single-context implementation/study remains archived under
`chunk-attention-mps-sources1` and `chunk-attention-mps-kernel-comparison1`.

| Active rows | Actual tokens per row | Cached prefix | Index32 control GPU ms | Matrix candidate GPU ms | Ratio |
|---:|---:|---:|---:|---:|---:|
|1|4|256|0.633|9.941|0.064|
|1|6|2048|8.317|10.397|0.800|
|1|128|0|3.800|10.999|0.345|
|1|128|128|8.064|8.802|0.916|
|1|128|1024|62.200|9.139|6.806|
|1|128|1920|137.799|9.235|14.922|
|1|128|3968|277.688|9.752|28.475|
|2|128|0|5.822|8.952|0.650|
|2|128|128|15.709|9.028|1.740|
|2|128|1024|134.666|9.575|14.064|
|2|128|1920|274.651|10.076|27.259|
|2|128|3968|559.074|11.200|49.916|

These are isolated attention milliseconds, not model tokens/s. Full-bound
matrix work explains the structural tradeoff: this prototype still computes
the physical batch/chunk/context envelope, including inactive queries and
unused cache positions. It accepts shorter inputs but does extra work for
short prefixes and small tails. Keep those regressions visible; dynamic
context buckets or a sparse-tail route require a separate measured candidate.
`native27b-chunk-attention-mps-chunk128-quality1` passes all eight sandboxed
coding tasks and all eight complete output-token sequences exactly match the
original single-native reference. Accounting and39 corruption checks pass.
It records1036 combined decode plans, nine mixed forwards, three interleavings
and1268 checked attention commands. Every prefill command uses all16 MPS
full-attention dispatches; ordinary decode has zero, and the index32 control
dispatch count remains zero. Both routes report native coverage separately.
The same one-plan/121,634,816-byte scratch inventory remains fixed throughout.
Ready allocation is17,975,410,688bytes, below19,069,665,280 recommended bytes.
Close records1,769,472bytes, all four Garnet owner counts zero and worker exit0,
inside the profile's unchanged2MiB close limit. Serving timing remains separate.

The exact qualified library/worker completes
`native27b-chunk-attention-mps-chunk128-http-admission1`: two cold C1 cells,
one wave each, no warmups,128 output tokens each. Both full output sequences
match the preceding attention-index32 admission; accounting and19 corruption
checks pass, with274 checked MPS command events.

| Actual prompt tokens | First token s | Decode tokens/s | Total s | Overall output tokens/s |
|---:|---:|---:|---:|---:|
|260|16.444|6.817|35.121|3.645|
|2054|43.153|5.905|64.695|1.979|

The preceding attention-index32 cold samples were14.376/47.738s first-token
time and3.909/1.860 overall output tokens/s. The matrix candidate's short case
is slower, while its long case improves modestly. Individual cold samples are
not repeated performance qualification. The large isolated attention gain does
not translate into a comparable whole-model gain here. Decode kernels are
unchanged, but the measured decode samples are slightly slower. Keep the
candidate optional, all failures/regressions visible and every2x target unmet.

`native27b-chunk-attention-mps-stage-diagnostic1` then uses the same qualified
library with the timing observer enabled. All two128-token sequences match
the admission exactly; stage/accounting checks pass. Long-prefill preparation
and encoding sum0.113s, versus50.322s GPU duration nested in50.332s wait.
Full128-token chunks are approximately flat at2.906..3.100s instead of growing
with context; the six-token tail takes2.361s. Short-prefill GPU sum is15.748s,
including a12.903s first-use chunk. These diagnostics identify GPU execution
as the dominant interval; they do not identify its internal cause.

An immediate sequential remeasurement of the exact earlier affine-index32
control (which retains64-bit attention),
`native27b-bounded-index32-stage-diagnostic2`, preserves both full output
sequences and passes accounting. Its long-prefill GPU sum is53.059s, versus
47.366s in its earlier diagnostic; short-prefill sums are13.657/14.975s.
Long ordinary-decode GPU sums are21.266s for the remeasured control and26.180s
for the matrix diagnostic. Control drift is observed, but it does not explain
every candidate difference or prove a thermal, paging or private-driver cause.
The Mac reports AC power and no recorded thermal/performance warning.
No repeatable model-level speed gain is accepted. A subsequent candidate
should select precompiled context buckets and a sparse-tail path from actual
validated controls while retaining the maximum profile's memory envelope.

### Actual-control context buckets and scalar tails

Default-OFF `GARNET_METAL_CHUNK_ATTENTION_BUCKETS` requires the MPS/index32
study flags and prepares128/256/512/1024/2048/4096 context variants for the4K
profile. Physical matrix descriptors, row/batch strides and the seven-buffer
116MiB scratch envelope remain fixed. GEMM dimensions select leading context
submatrices; function constant31 specializes softmax extent. Execution only
looks up prepared variants. Smaller inputs are valid within the profile.

Native code reads the small slot/active buffers only when their shared Metal
storage is completed, host-readable and not pending GPU output. It validates
flags/integer slots, counts real tokens and selects a context bucket containing
every active query's visible prefix. At most eight real tokens use the original
index32 scalar kernel. Unknown/pending GPU controls retain the full prepared
context without readback. Python observes counters and verifies decisions;
the routing decision lives in the native inference runtime.

The first reduced-descriptor prototype fails the unchanged numerical gate
at the256-token bucket, with maximum control delta0.00149938. Sources and
`chunk-attention-buckets-ctest1.log` remain archived. Retaining full physical
descriptors while limiting GEMM submatrix dimensions repairs the failure;
the installed SDK explicitly permits larger operand/result matrices than
the requested GEMM dimensions. No private-driver cause is established.

All18 native tests pass after repair and the explicit host-readability guard.
Coverage includes18 context boundaries,13 active-token counts, padded control
strides, invalid flags/slots, pending GPU controls, concurrent shared-plan
partial submissions and FP64 attention checks at all six real27B bucket sizes.
The strict bitwise diagnostic still fails for the FP32 matrix route; its first
small case has the previously observed2.56114e-9 maximum delta. Original error
limits are unchanged. Scalar tails remain bitwise equal to control.

`chunk-attention-buckets-kernel-comparison2` freezes the final native library
components and runs three fresh sequential cohorts, three warmups and nine
alternating-order calls per variant. Geometry, precision and12 cases match
the preceding studies. Numerical checks and13 corruption checks pass;
maximum control delta is2.60770e-8. Rows below are medians of cohort medians.

| Active rows | Actual tokens per row | Cached prefix | Index32 control GPU ms | Bucket/tail GPU ms | Ratio |
|---:|---:|---:|---:|---:|---:|
|1|4|256|0.490|0.491|0.997|
|1|6|2048|7.404|7.402|1.000|
|1|128|0|2.951|0.874|3.378|
|1|128|128|8.307|1.121|7.413|
|1|128|1024|64.859|4.642|13.972|
|1|128|1920|137.802|4.782|28.817|
|1|128|3968|281.328|9.734|28.902|
|2|128|0|5.830|1.028|5.674|
|2|128|128|16.006|1.297|12.338|
|2|128|1024|132.145|4.953|26.680|
|2|128|1920|273.430|5.202|52.561|
|2|128|3968|559.393|11.164|50.105|

The earlier full-bound prototype's short-context regression is removed in
these isolated operator cases; scalar-tail samples are approximately neutral.
These are attention milliseconds, not model token throughput or2x model gains.

`native27b-chunk-attention-buckets-chunk128-quality1` passes all eight sandbox
coding tasks, with all eight full token sequences exactly matching the original
single-native reference. Accounting and46 corruption checks pass, including
joining bucket decisions to native scheduler slots/token counts rather than
trusting a worker flag. The run records1050 combined decode plans, nine mixed
forwards, three interleavings and1254 checked bucket commands. It exercises
128/256/512/1024 context buckets and two scalar-tail commands, with zero unknown
controls. All six buckets are prepared before ready; their count and the
121,634,816-byte scratch inventory remain fixed. Ready allocation remains
17,975,410,688bytes. Close is1,507,328bytes, all four owner counts zero, worker0,
inside the unchanged2MiB profile limit.

The exact qualified library/worker completes the two-cell cold C1 admission
`native27b-chunk-attention-buckets-chunk128-http-admission1`, one wave each,
no warmups. Both complete128-token sequences match the preceding index32
admission. Accounting and24 corruption checks pass.

| Actual prompt tokens | First token s | Decode tokens/s | Total s | Overall output tokens/s |
|---:|---:|---:|---:|---:|
|260|14.603|6.754|33.439|3.828|
|2054|36.738|5.729|58.942|2.172|

The preceding full-bound matrix cold samples were16.444/43.153s first-token
time; the attention-index32 samples were14.376/47.738s. This bucket candidate's
long cold sample is lower, while decode shows no speed gain. The reference
vLLM/SGLang campaigns remain frozen and use warmups/cohorts; these admissions
do not establish a matched repeated comparison. Every model-level2x target
remains unmet. Keep the experimental flags OFF by default.

`native27b-chunk-attention-buckets-chunk128-batch-admission1` completes four
C2/C4 cells, one wave each, no explicit warmups, through physical batch2.
All12 full output sequences match the preceding affine-index32 batch archive;
accounting and24 corruption checks pass. It records495 combined decode plans
and822 checked bucket commands. Natural EOS is retained; output counts are
190 tokens for each C2 wave and446 for each C4 wave.

| Target input tokens | Clients | Overall output tokens/s |
|---:|---:|---:|
|256|2|5.454|
|256|4|7.568|
|2048|2|2.338|
|2048|4|2.674|

Overall throughput includes prefill and queueing. The preceding single-wave
affine-index32 batch observations were5.581/7.881/1.701/1.945 in this order.
Short observations are slightly slower and long observations faster; these
are still admissions, not repeated performance acceptance.

The strengthened plan test separately exercises capacity64 and4096 with two
concurrent contexts and24 partial-submission runs per context pair/cycle.
In the4K case, completed controls select128 while pending GPU-produced controls
retain4096; outputs match the original numerical limits and all owners close.
Driver allocation after the two different-capacity cycles is737,280 and
1,785,856bytes. The exact qualified model library remains unchanged by this
test-only rebuild.

`run_local_coder_native_campaign.py` prepares the exact frozen phase1 manifest
for three fresh native processes, six cells, two warmups and five measured
waves per cell. It checks coding/full-token qualification, prior serving
admission, runtime xModel identity and every pinned checkpoint byte before
launch. Per-cohort profiles differ only in inventory output paths; logs and
restart identities are independently audited. Native request epochs are
scoped per process. A CPU-only synthetic namespace fixture and26 corruption
checks validate this audit plumbing; the temporary fixture is not GPU or
performance evidence. Actual matched collection is now complete; the repeated results and retained limitations follow.


The full `native27b-chunk-attention-buckets-phase1-campaign1` completes three
fresh processes,90 measured waves/210 measured requests (294 including warmups).
Accounting passes with no issues;27 distinct corruption checks pass, including
cohort argv and process identity. All three native workers exit0 with no forced
kill; all Garnet owner counters return to0 and final Metal allocation is
1,507,328bytes. The final token review independently joins all210 measured
requests to native completions, SSE token IDs and frozen prompts; identical
requests have no full-token divergence across the collected repetitions. This
is text/accounting/trace qualification, not completion of the broader27B goal.

The archived library remains SHA256
`0145d729e96b335fcf9ed4b9ceaf738b2fcd14d8c2ee54a0a9a40f7197cefe6a`.
The prior eight-task coding/full-output-token gate is unchanged. Ready allocation
is17,975,410,688bytes. The two reference campaigns were reused without rerunning.

Overall output tokens/s below include prefill and queueing. Each value is the
median of three cohort medians, each using all five measured waves. Actual
prompts are260..263 and2054..2057tokens;256/2048 are workload targets, not required
exact input lengths. Native physical batch2 queues clients4. Natural EOS and
all paging/variance are retained.

| Target input tokens | Clients | Garnet | vLLM-Metal | SGLang Apple |
|---:|---:|---:|---:|---:|
|256|1|3.843|5.761|5.502|
|256|2|4.883|7.352|6.834|
|256|4|5.391|9.935|7.546|
|2048|1|1.479|3.382|3.275|
|2048|2|1.811|3.525|3.281|
|2048|4|2.073|3.479|3.789|

Single-request results use the same aggregation. TTFT is seconds and decode
is tokens/s, excluding prefill.

| Actual prompt tokens | Garnet TTFT | Garnet decode | vLLM TTFT | vLLM decode | SGLang TTFT | SGLang decode |
|---:|---:|---:|---:|---:|---:|---:|
|260|6.131|4.675|2.450|6.417|2.389|6.094|
|2054|55.469|4.093|15.228|6.278|14.633|5.843|

Every2x decode/overall-throughput target and every half-TTFT target remains
unmet. Preserve the vLLM cohort1 final longC4 missing trailing memory sample,
automatic vLLM cache/active-limit differences, and SGLang active2/512MiB cache
configuration. These limitations do not justify selecting favorable native
cohorts or dropping slower waves. The full machine-readable comparison is
`evidence/native27b-chunk-attention-buckets-phase1-comparison1.json`; accounting
is in the campaign's `accounting-audit1.json`, and the final measured-token
join is `evidence/native27b-chunk-attention-buckets-phase1-final-token-review1.json`.

Read-only inventory review identifies623MiB of unpooled prefill storage with
multiple same-storage graph views (320MiB hidden-shaped temporaries,288MiB
recurrent state-shaped temporaries,15MiB convolution-state temporaries). These
are candidates, not proven safe-free or achieved model savings. The separate
default-OFF alias-group implementation stages the pool and all bindings before
a nonthrowing commit. It preserves protected owners and requires shared scratch
plus lifetime reuse; the qualified reference build has lifetime reuse OFF.

The normal candidate passes19 CTests, including alias-group lifetime planning.
An independent K128/V16-multiple row reference passes with and without a head;
mixed four-layer execution preserves exact pre/post-sharing outputs and state,
separate sessions, foreign-owner rejection, inactive rows, standalone calls and
pool lifetime. A separate default-OFF fault build explicitly verifies its backend
identity, injects rejection after staging but before commit, and preserves the
seeded pool's resident storage aliases, both private states and exact standalone
output before successful retry. The older small-state reference also passes.
The generic row helper now derives all state and packed dimensions from the
fixture; the two prior hard-coded-helper failures are retained. No numerical
tolerance was changed. Source and normal/fault library hashes are archived in
`evidence/prefill-alias-groups-source1/manifest.json`.

The default-OFF compatibility build also passes19 CTests. The composite
alias-group/lifetime-reuse candidate subsequently passes all8 frozen27B coding
tasks, with all8 complete generated-token sequences equal to the original
single-native control. Its independent accounting audit and53 corruption checks
pass. These are bounded regression contracts, not a coding leaderboard score.
The frozen archive is `evidence/native27b-prefill-alias-chunk128-quality1`.

At batch2/context4096/chunk128, preparation and the quality worker both report
17,297,309,696 allocated Metal bytes,678,100,992 bytes (646.6875MiB) below the
previous qualified candidate. This comparison includes enabling lifetime reuse,
which was OFF in that reference; it does not isolate alias grouping alone.
The prepared64-layer layout contains160 alias groups and320 graph views and is
checked exactly before worker readiness. The largest sampled quality completion
allocation is17,302,650,880 bytes; samples are not a measured peak. Model close
returns all4 Garnet owner counters to zero and Metal allocation to1,507,328
bytes. The native worker exits0. Serving admission and performance qualification
remain separate requirements; these results establish no speedup.

The subsequent C1 serving admission passes native/HTTP accounting and31 distinct
corruption checks. Both128-token outputs exactly equal the previous qualified
candidate's full token sequences. At actual prompt lengths260 and2054, the cold
waves report respectively13.126s/32.251s TTFT,6.902/6.026 decode tokens/s and
4.057/2.398 overall output tokens/s. These are single unrepeated waves without
warmups; they cannot replace the preceding matched three-cohort campaign or
establish a speedup. All native owners close to zero, allocation returns to
1,507,328 bytes and the native worker exits0. Evidence is
`evidence/native27b-prefill-alias-chunk128-http-admission1`.

The separate four-cell C2/C4 serving admission also passes:12 complete output
token sequences exactly match the preceding candidate,498 combined decode plans
and822 attention-bucket commands are joined, and31 distinct corruption checks
pass. Short C2/C4 overall output rates are5.764/8.255 tokens/s; long C2/C4 rates
are2.356/2.664. Each cell has one wave with no warmup. These observations retain
queueing and natural EOS and establish no repeated throughput gain. The largest
sampled allocation is17,302,650,880 bytes, all4 owner counters close to zero and
the native worker exits0. Evidence is
`evidence/native27b-prefill-alias-chunk128-batch-admission1`. The upgraded auditor
also accepts the prior alias-disabled C1 archive without rerunning inference.

A source-backed planning estimate is retained in
`evidence/27b-m5-decode-bandwidth-analysis1.json`:497 native affine projections
contain14,412,349,440 logical payload/coefficient bytes. Apple's published M5
153GB/s implies about10.616tokens/s for an idealized full-weight pass per token,
below the12.834tokens/s shortC1 target against vLLM. This is not measured DRAM
traffic or an absolute hardware impossibility. It motivates weight-pass reuse
across verified tokens; MTP/verification/state rollback remain unqualified.
Published hardware source: https://support.apple.com/en-ph/125405 .

### Parent MTP asset discovery

The converted pinned artifact contains no MTP parameters, but the original
`Qwen/Qwen3.8-27B` parent at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` publishes15 MTP tensors in its final
shard. Exact bounded HTTP Range header reads identify eight BF16 matrices and
seven normalization vectors, totaling849,398,784 payload bytes. This establishes
available auxiliary weights, not conversion lineage or runtime support.

`fetch_qwen_parent_mtp.py` extracts exactly that contiguous auxiliary range,
requires206/range/length agreement, preserves raw BF16 bits, checks every value
for finiteness and records source/package hashes. The resulting MTP-only file
is849,400,560bytes, SHA256
`dec6630788d1ba4557230ca02b5df2db239f2b2bbcfff1c6742776e231e7540b`.
Its manifest status is explicitly raw/not runtime-qualified. The fixed generator
artifact is unchanged. A derived INT4 auxiliary package, target/head sharing,
Metal MTP lowering, verification and hybrid-state commit/rollback all remain
unimplemented/unqualified. Existing pinned vLLM `qwen3_5_mtp.py` and Garnet's
CUDA MTP/verifier are relevant implementation references, not Metal support.
