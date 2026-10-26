# Qwen3.8-27B on one RTX 5090: implementation task

Status: native tokenizer whitespace/Unicode and declared NFC input normalization
are corrected. Unchanged955frozen/edge prompts,1112064single-scalar normalizations,
30012normalization sequences and5027full encodes pass actual private native/Rust
CPU differential checks. Original failures remain preserved. Native Windows
rebuild on5090 is closed/replayed on both hosts: new1a1d DLL,49CPU regressions,
955actual full-DLL C API matches, all32CUDA objects bitwise unchanged. Old8ae and
five runtime dependencies retained. Fresh4B4/8B1 model corpus now runs, with24CPU
gates and additional every-captured-state/raw-vector comparison against8ae.
New model/HTTP/quality/speed remain unqualified until actual completion/replay.
Original HTTP attempt1 remains failed after10accepted replies and request11
input-ID mismatch; no timing ran. Restart only as a fresh proof-bound study.
Matched4K/tuned/MTP references, INT4 W4A16, matched4B and repetitions stay open;
original2x decode/concurrency and half-TTFT objectives remain unmet.

The pinned 27B checkpoint uses Qwen3.5 architecture. Garnet supports its mixed
NVFP4/FP8 projections, BF16 vision/MTP, host embedding and joined MLP. Latest
measured short-prompt register128 HTTP C1 decode is72.54text/70.70image tok/s;
C8aggregate208.53/167.86 versus faster ordinary references445.77/439.86. All
cohorts, regressions and sampler gaps remain recorded. These are single pilots
on the prior qualified DLL, not results for the new reference DLL or causal
estimates.4096cache capacity is not an actual4K-input benchmark.

The packed BF16 integration on01a3...DLL passed its full default-off model check,
but its enabled B4/T32 case failed retained FP32 token IDs at text26/image8.
Ordinary/MTP state and hidden agree for that failed case. Controlled32-step
teacher-forced FP32/T4, BF16/T4 and BF16/T32 diagnostics are closed/replayed;
BF16/T4 image first differs at19, BF16/T32 at text26/image8, with substantial
normalized-vector differences. Preserve every failure; broad BF16 quality is
unqualified. These correlated short controls are not a broad-quality benchmark.

The independent full-cache reference rounds currentK/V toBF16 and uses unchanged
legacy dense FP32 attention math, independent FP32 caches and no packed offsets
or compact commit helper. Its42GPU cases,442880selected independent FP64 scalar
witnesses and readonly/alias/graph/cache controls pass; maximum absolute error
7.38e-8. Separate wrapper CPU metadata contracts pass. The default-off B4/T4/T32
model path now builds as97c364...DLL, with old sources/DLL and allfive runtime
libraries retained. Original remote build ran49CPUtests versus promised50 because
one older fixture lacked an existing batch4 test. The preserved50-test completion
and corrected replay require both immutable archives; both machines agree.
Six separate state-byte/reference-routing CPU controls also pass. No speed or
full-model quality claim follows from component/CPU/build checks.

The five-profile model diagnostic is closed/replayed on both machines, with
identical replay reports (archive5ab8f15a...,320raw hidden/normalized arrays).
Packed BF16 and independent BF16-in-FP32 reference match all32teacher predictions,
all raw vectors and complete logical state hash witnesses under BOTH T4 and T32
prefill routing. The fresh FP32/T4 control reproduces retained vectors/state.
The independent reference reproduces the old-FP32 token differences as well;
this supports packed-cache correctness on these cases, not broad quality or
serving admission. Full raw caches are not archived. Every FP32-reference cache
word passed the zero-low16-bit BF16-roundtrip check. No DLL/source drift.

Localization: FP32/T4 versus BF16/T4 first differs at K cache layer3 after exact
GDN layers0-2 (first V cache agrees). Reference T4 versus T32 first differs in
GDN layer1 before the first attention cache. Thus the observed precision and
prefill effects are not explained by packed-cache corruption on these fixtures;
this does not identify every upstream operator or prove harmless rounding.

The independent ordinary-only B4/T32 reference is now closed/replayed on both
machines (archive2d086ee0...,8raw final hidden/normalized arrays). Allfour cases
(two groups repeated twice) have exact repeated IDs/state/vectors. Descriptive
text/image rows generate128tokens; red/green rows stop after2tokens under their
original32token limits. Red/green IDs match the old FP32 outputs; primary
128token IDs still differ, preserving the old precision-change observation.
This is a prospective same-precision arithmetic oracle, not broad quality.

The full packed ordinary/MTP comparison is now closed/replayed on both machines
(archive009de883...,126files/20raw hidden+normalized arrays). Allfour cases have
exact independent-reference IDs and complete logical prefill/final state witnesses
and raw final vectors; packed ordinary/MTP physical states also agree exactly.
Historical FP32 producer/failure remain unchanged. Its old8856metadata/16negative/
3zero-prefix block is retained verbatim after newline normalization and passes.
Eight CPU controls pass on both machines; no source/DLL/runtime drift or rebuild.
This qualifies this arithmetic corpus only; broad quality remains unqualified.

Original bf16-padded-lifecycle-attempt1 closed exit1 after10cases, failing on
free variable base in a nested lambda/comprehension test guard. Archivefa5acbd6...
(187files/72rawvectors) and both matching failure replays preserve it as FAILED.
Sources/DLL unchanged; prefix/retirement/full14case qualification remain open.
A CPU-only native-runtime reproducer fails on the original closure and passes
direct/default-bound calls; normal CPython passes allthree. Prepared scope-fixed
probe retains all14cases, six identical guard calls/inputs and assertions, adding
interim prefix witnesses. Scope-fixed attempt2 is now running after seven
acknowledged uploads, idle/source/DLL preflight and11remote CPU checks. The
original supervised attempt2 handle remains authoritative. External rotary/cache
checker correction is now uploaded only AFTER original terminal/archive; archived
51451a54...checker preserved, current0d5e52f7... used for failure replay.

Standalone parallel-batch-top1-correctness-attempt1 closed exit0 on unchanged
97c364DLL, which it never loaded. Archive4675c545... (94files/483857B) passes both
independent replays:330cases/1000raw observations,2captures,12invalid calls,
CPU/legacy/candidate raw IDs+bits, readonly/canaries/nullable outputs all exact.
Data reports agree except declared Python/NumPy versions; no mismatch or speed
result. Actual compiler13.2.78/VS14.44 was observed and forced before build; all
19files/60inherited sources/3tools matched, nine uploads acknowledged. Failed
model parent remains unqualified. Default-off native integration is prepared
locally, with19CPU tests and private host/CUDA compilation passing. Require full
model/protocol correctness including the scope-fixed14case corpus
on the chosen binary before any matched timing. Correctness memory is not serving
memory. B8/broad quality/actual4K/reference/4B/repetition requirements stay open.

Offline retained-trace analysis identifies BF16 top1 selection at70.24/70.14ms,
about4.05percent of kernel union in its two marked windows. The standalone
parallel batch-top1 candidate now passes330CPU/legacy/candidate GPU comparisons,
tie/signed-zero/nonfinite, capture/readonly/canary checks and independent replay.
Local production dispatch is implemented and compiled; remote deployment, full
model/protocol gates and matched timing are pending.
This small subset cannot explain2x by itself; dominant matrix/batching/low-batch
work stays in scope.
Raw-observation probe and seven CPU controls now pass on the actual5090, with
all330predeclared fixture outputs and1000raw observations independently replayed.
Existing sampler/header files and observed compiler sources were pinned exactly.
No remote production integration or new speed result; do not promote the failed
parent. Configure GARNET_PARALLEL_BATCH_TOP1=1 before the first sampler or
memory_json call to opt in. Only batch top1 with1..128rows andvocab>=8192
uses the candidate; allother valid shapes retain the legacy route. The process
freezes its mode. memory_json reports mode/bounds and separate host dispatch
attempt counters; CUDA graph replays do not increment those host counters.
Four-process production-API controls (0/1/unset/01) are prepared but GPU-unrun.
Each requires352cases/1066raw observations, original330fixture preservation,
row129 fallback,2capture profiles and12invalid calls. Candidate kernels remain
byte-identical to the qualified standalone operator.

Broader quality preparation now includes a pinned public MMMU validation source
audit (revision876ce5cb..., October2026 corrections included). All30Parquet blobs/
341234748B hash-verify and900rows are CPU-inspected:805single-embedded-image MC
candidates across30subjects;95excluded source rows with explicit reasons. This
is structural selection before outputs, not a quality score or full official
MMMU evaluation. All805requests are now CPU-prepared and source-replayed: exact
aspect-preserving white256px letterbox JPEGs,64image tokens,112..627prompt tokens
plus32output reserve, pinned no-thinking template and strict single-letter scoring.
All95source exclusions remain explicit; no additional rows were dropped. Archive
ac3d9dd5... is58097753B; source JPEG transforms, gold labels and independent
Jinja/native prompt IDs replay exactly. Three synthetic CPU protocol tests pass.
Full source payloads remain in fixed private cache; Git retains compact metadata.
This low-resolution adapted screen is not official900question/high-resolution
qualification. No model responses, runtime preprocessing/ID checks, quality margin
or quality promotion is admitted. Actual inference/reference comparison is open.

Actual4K request fixtures are CPU-prepared/replayed using the pinned tokenizer,
exact chat template and64image slots:124requests,3968input tokens each plus
128output reserve. The immutable193369B corpus is on the research branch.
Actual server IDs, boundary behavior, quality and matched timing remain pending.
Unrun B8/128MiB drafts remain outside the repository; legacy256MiB/B4 and other
profile guards remain unchanged. Wider batching cannot establish C1 decode gain.

Next gates, in order:

1. DONE: close/replay scope-fixed attempt2 on97c; retain failed attempt1.
   Full14case scope-fixed parent now closes exit0 and both independent replays
   agree exactly:102raw vectors,224files, archived5650c97..., unchanged97cDLL. Source install
   observer/factory/replay and actual prerequisite refusal guards are ready;
   expected-exit2 checks leave runtime preparations absent while parent is live.
   DONE after B1 operator: preserving source install, four production API GPU
   processes, actual SDK native build and both independent replays. Next require
   complete14case/102vector parity in separate disabled/enabled processes on5ac,
   with actual dispatch counters, then protocol controls before timing. Retain
   original failed harness/FP32 comparisons and require broad-quality evidence;
   no operator or partial-case result substitutes for complete model/serving gates.
2. After full parent closure/replay and before source installation, qualify
   the separate B1 attention candidate:44predeclared FP64 cases/119040selected
   scalars, fullT32first/last boundaries,3graph/104negative/40cache controls.
   DONE: actual5090compile/CPU33/10Python controls and44GPU cases pass both
   independent FP64 replays (archive2eeb8aa8...,119040scalars,maxabs1.87e-7).
   Actual pre-GPU frozen FP64 bytes also pass unchanged tolerance; fresh
   oracle-vector hash/error last-bit differences are preserved. Separate B1
   commit/layout code now privately compiles;5CPU oracle/legacy isolation tests
   and63CPU shape checks pass.20raw-case/3graph/66negative GPU gate remains
   UNRUN while the original full-model sampler job owns the GPU. Standalone
   runner/checker now pass17CPU controls; actual missing-parent refusal creates
   no runtime preparation. Separate B1 metadata validator passes7820malformed
   layouts and12cross-dialect rejections while old B4/B8 parser still accepts
   its four profiles. After actual model closure/both replays, collect/replay
   isolated B1 operator; a failed model parent remains failed. Require typed
   commit/helper/plugin/lowering and complete model/lifecycle/4K/quality/protocol
   admission for C1. Existing B4/B8 guards remain unchanged. Canonical427294720B
   for B1 versus1709178880B for physical B4 is a layout calculation, not total
   model VRAM fit or speed evidence. Continue measured dominant C1 work.
3. Qualify a distinct B8 MTP/workspace profile and actual M32 projection behavior,
   allocation fit, text/image quality and protocol correctness. Preserve legacy
   B4 admission guards. Wider batching does not establish C1 decode speed.
4. Run actual4K prompts and broad text/image quality controls against pinned
   references; then matched HTTP cohorts and clean repetitions. Tune feasible
   vLLM/SGLang MTP references, preserve INT4 quality differences and complete
   matched4B reference comparisons. Count only real emitted tokens, retain raw
   methods/results/failures, and assess the original target without weakening it.

The preserved source baseline was `5b0dc7ff8e5c1a77807138db310605bcd12aa390`
plus inherited edits captured at2355fc4b5. Preserve unrelated local edits and
existing deployment/model assets; reuse the fixed remote study/build/cache paths.
Detailed methods, exact hashes, negative results and historical observations are
in `docs/research/qwen38-5090-development-log.md` and the evidence archives.

## Objective and acceptance

Implement native Windows Garnet inference for `Qwen/Qwen3.8-27B` on one RTX
5090 32GB. Prefer NVIDIA's mixed NVFP4/FP8 checkpoint on Blackwell. Compare a
separately identified INT4 W4A16 implementation and Qwen3.5-4B. Start with batch
1 and 4096 total context capacity, including image tokens and output reserve.
Target >=2x the faster of vLLM and SGLang for both decode tokens/s and aggregate
output tokens/s at each matched concurrency. Report TTFT independently; seek
half TTFT as an additional optimization objective. No successful-target claim
until both independent baselines have valid matching results. Retain failures,
OOMs, quality regressions, and below-target results.

## Verified checkpoints

Pinned configs, revisions, URLs and SHA-256 values are in
`test2026/cuda/qwen38_5090/evidence/checkpoints.json`.

* Qwen/Qwen3.8-27B: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
* nvidia/Qwen3.8-27B-NVFP4: `482ca0f3832238542f8f5295dde86b5f22711d80`.
* Qwen/Qwen3.5-4B: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
* INT4 candidate pearsonkyle/Qwen3.8-27B-GPTQ-W4A16:
  `4e2c2d8f592d5641beefa6a75357f43cf3756392`.

The INT4 candidate config uses compressed-tensors pack-quantized, symmetric
group128 INT4 weights and unquantized activations (W4A16). Actual index/header
inspection proves all496 text projections, including linear attention, are packed;
lm_head, embedding, vision and MTP remain unquantized. The earlier config-only
inference about ignored linear attention was incorrect. Installed compressed-tensors
0.18.0 passes the CPU packing fixture: eight offset-binary nibbles per int32,
low lane first, decoded value nibble minus8. Actual-weight numerical checks,
image inference, SM120 kernel dispatch and Garnet compatibility remain unverified. Keep it a candidate until
these pass. Do not compare quantization recipes as if identical.

The 27B config uses `Qwen3_5ForConditionalGeneration`, model_type `qwen3_5`.
It has 64 dense decoder layers: 48 Gated DeltaNet and 16 full-attention layers.
Hidden width 5120; MLP width 17408; Q/KV heads 24/4, head dimension 256;
linear QK/V heads 16/48 with dimension 128. Embeddings are untied.
Vision differs from 4B: depth 27, hidden width 1152, intermediate width 4304,
16 heads, output width 5120. Preserve patch16, merge2, temporal-patch2 and
multimodal RoPE behavior. Architectural reuse does not prove checkpoint parity.

The NVIDIA config declares ModelOpt MIXED_PRECISION: NVFP4 block16 MLP/head,
FP8 attention projections, and an ignored MTP set. Its card recommends Linux,
lists both engines and Blackwell, and reports testing on GB300, not RTX 5090.
The card says ModelOpt 0.48.0; the pinned config producer reports
0.47.0.dev80+g913f5e224. Preserve that discrepancy in provenance.
Do not relabel this checkpoint all-NVFP4, MXFP4, GPTQ, or W4A16.

## Current Garnet constraints

* `runtime/cuda_serving.py:Model` admits only Qwen3.5-2B and 4B. Its 8192
  capacity ceiling allows a 4K profile but does not validate the new model.
* `xModel/qwen35/vl_4b/model.py` validates 4B-specific decoder and vision shapes.
  Add a dedicated 27B contract and factory; never alias to the 4B factory.
* Shared external projections require BF16 matrices in
  `xModel/qwen35/vl_0_8b/cuda_decoder.py` and
  `src/backends/tensorrt/cuda_projection_io.cpp`.
* `trt_builder.cpp` has captured-weight `int4_fp16` block64 symmetric
  quantization/dequantization. It is not a ModelOpt loader, calibrated GPTQ/AWQ,
  or evidence of packed shared-projection support. No NVFP4 implementation was
  located in the audited serving/projection path. GPT-OSS MXFP4 is unrelated.
* Existing Garnet/vLLM Qwen3.5 benchmark scripts constrain context to 384/512
  and model names to 2B/4B. Do not merely change labels on their old results.
* CMake defaults to SM80/86/89. Explicitly configure SM120 on RTX5090.
* Reuse current Qwen3.5 CUDA lowering and state kernels below the backend-neutral
  tensor-program boundary. No Python/PyTorch model fallback presented as Garnet.

## Target machine and safe setup

SSH target `CD-RTX5090` / `192.0.2.1` reports Windows hostname
`zsdx-5090`, one RTX5090, 32607 MiB, driver 617.14, CUDA toolkit12.8.
`D:\CantorAI\Garnet` exists but contains deployment assets (`out`, `xModel/qwen3`),
not a buildable Git checkout. `C:\CantorAI` contains deployment output too.
Initial checks found no WSL installation, Visual Studio build tools or CMake
at their standard locations. PATH alone is not a full installed-software audit.

Preserve these deployments. Stage current source in a fresh sibling study
directory, with adjacent current XLang3 and separately versioned TensorRT SDK.
Record per-file hashes including dirty files and dependencies. Build native
Windows Release with SM120. Preserve a control executable/DLL before changes.
Use WSL only for Linux-only reference dependencies as needed; user authorized
installation. Do not reboot automatically. Never save SSH credentials in code,
reports, Git config, model manifests, or the source archive.

## Ordered implementation

1. Environment: inventory compiler, CMake/Ninja, CUDA, TensorRT and XLang3;
   resolve SM120 compiler/runtime compatibility. Pin exact versions and build
   commands. Compile current Garnet unchanged, then run tensor capture/native
   API, Qwen3.5 serving/state/vision correctness checks on the 5090.
2. Checkpoint loader: download only pinned metadata first. Inspect safetensors
   headers/index, naming, tensor shapes, tied/untied embeddings, scale formats,
   FP8 exclusions and total actual bytes. Make unsupported formats fail before
   allocation. Integrate dedicated 27B factory, serving admission and lifecycle.
3. Correctness first: tiny layer/subgraph numerical tests against independent
   Transformers references, including full/linear attention, conv/state reset,
   multimodal positions, vision merger, logits and greedy tokens. Use one layer
   at a time or an offloaded reference if BF16 model does not fit. No BF16 full
   model load assumption on 32GB. Exercise real images, text, repeated requests,
   cancellation, EOS and context overflow. Keep MTP/speculation disabled initially.
4. Quantization: implement packed NVFP4 block scales, per-tensor scales and FP8
   projections compatible with the pinned ModelOpt format and actual SM120
   kernels. Keep stable accumulation/state precision; document unquantized
   vision/norm/embedding tensors. Separately implement/admit INT4 W4A16 with
   declared group size, zero-point convention, scales and calibration dataset.
   Existing block64 symmetric path can be a separate experimental arm, never
   labeled calibrated GPTQ/AWQ. Verify kernel dispatch does not expand weights
   into a persistent full BF16 copy. Pin any external INT4 checkpoint only after
   config, tensor-layout and image-path inspection; the pinned candidate above
   is not admitted yet.
5. Fit test: load B1 context4096, reserve256 output tokens. Track build/load and
   inference VRAM separately. Warm text, then actual image. Fail explicitly on
   OOM; don't silently offload, split across GPUs, lower context, or reduce image
   pixels. Scale concurrency1/2/4/8 only as memory allows, recording failures.
6. Reference engines: use isolated pinned environments; run vLLM then SGLang
   sequentially with TP1, context4096, max sequences1 initially, the same local
   checkpoint, image budget and deterministic generation. The NVIDIA card's
   TP4/262K command is not suitable for this study. Verify help/flags and actual
   selected ModelOpt/FP8/NVFP4 kernels on the installed releases. Start 4B BF16
   baselines before the 27B quantized ones. Cache and recurrent-state dtypes
   must be recorded; separate unmatched precision arms from matched comparisons.
7. Optimization experiments after valid control: profile dispatch/host gaps,
   projection kernels, decode memory bandwidth, vision/prefill and state copies.
   Test packed-weight reuse, fused gate/up and attention projections, CUDA graph
   replay, in-place KV/state updates and chunked prefill independently. Benchmark
   each candidate against the preserved current control in alternating order.
   Add speculation/MTP only as a separately quality-validated experiment after
   serial decode parity; apply equivalent baseline optimizations where supported.

## Measurement contract

Separate text and image cohorts. Use fixed UTF-8 prompts and hashed image bytes;
record decoded dimensions, image grid, processor revision, chat template hash,
thinking mode (disabled initially), rendered prompt IDs and visual-token counts.
Use short, ~1K and near-3840-token input cases, always input+output<=4096.
Use matching semantic prompts for cross-model quality; exact token-count matches
are required for speed ratios. Set temperature0, seed0, output cap256, and
record actual generated tokens/EOS. A length-forced throughput arm is separate.

For each engine/model/precision/cohort: one cold run, >=3 warmups excluded,
>=20 measured requests (repeat full cohort >=3 times). Report median/p95 and
sample counts. Disable prefix, image/processor and answer caches for primary
tests; execute vision afresh. Report cached scenarios separately. Establish
single-request measurements before concurrency sweep1/2/4/8. Concurrency is
simultaneous clients; static model batch is a distinct field. Count queuing.

* End-to-end TTFT: monotonic submission timestamp to first generated token;
  includes tokenization, image decode/vision and queueing. Also record internal
  GPU prefill/vision and first visible text latency separately, especially if
  reasoning parsers buffer tokens. SSE chunks are not tokens.
* Per-request decode tok/s: (generated_tokens-1)/(last_token_time-first_token_time).
  Fewer than2 tokens => undefined, never infinity or a fabricated value.
* Throughput: total actual output tokens / cohort wall seconds from first
  submission to final completion. Also report requests/s, errors and OOMs.
* VRAM: poll whole-device used memory with timestamps and UUID; preserve idle
  baseline, absolute peak and incremental peak. Record sampling interval and
  that sampling can miss spikes. Add framework allocator high-water and native
  allocation instrumentation. Report compile/load peak separately. Windows WDDM
  process memory may be N/A; never substitute host RSS or reserved cache as
  measured GPU VRAM. No unrelated GPU work during timed runs.

Save raw per-request timings, token IDs/output, prompt/image hashes, failures,
logs, GPU temperature/clocks/power, driver, build flags, commit+dirty-file hashes,
quantization/calibration provenance, engine versions and launch commands.
No benchmark result files are pre-populated by the preparation task.

At4K, the contract checker estimates canonical FP32 full-attention KV at512 MiB
and recurrent matrices at144 MiB per request, plus convolution history. These
are shape calculations, not measured VRAM; duplicate buffers, weights, graphs,
vision and workspace can dominate. Do not infer fit from 27B*0.5 bytes.

## Completed safe changes and tests

Added offline checkpoint-contract preflight and pinned config evidence under
`test2026/cuda/qwen38_5090`. It diagnoses architecture/cache sizes and always
reports 27B runtime integration pending. It does not register unsupported model
names, mutate existing serving code, launch GPU workloads or invent results.
Added metric arithmetic and a two-engine 2x gate. Its test records are synthetic
unit fixtures, never benchmark output. The caller still must establish workload,
precision, hardware and quality equivalence before using that gate.

Run from Garnet using a Python3 interpreter:

```text
python -m unittest discover -s test2026/cuda/qwen38_5090 -p test_*.py -v
python test2026/cuda/qwen38_5090/preflight.py test2026/cuda/qwen38_5090/evidence/base-config.json
```

## Primary evidence

* https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json
* https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/blob/482ca0f3832238542f8f5295dde86b5f22711d80/config.json
* https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4
* https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json
* https://docs.vllm.ai/en/latest/features/quantization/modelopt/
* https://docs.sglang.io/docs/advanced_features/quantization

Published engine support is a candidate baseline, not evidence that a particular
Windows/WSL package and consumer SM120 GPU have passed this workload.

## Current native-build path

The optional CMake wrapper in `test2026/cuda/qwen38_5090` accepts preserved
XLang3 Release binaries and a matching public SDK. From a VS x64 developer
shell, configure it with `XLANG3_RUNTIME_DIR`, `XLANG3_ROOT`,
`GARNET_TENSORRT_ROOT`, `CMAKE_CUDA_COMPILER` and an explicit architecture.
Use89 for the recorded local RTX4080 correctness build; use120 for the target
RTX5090 build, which has passed native build, component correctness and the full64-layer text smoke. Full VLM executes with four semantic color/spatial answers, but strict parity and broad quality remain incomplete. Build target `garnet`. No VM sources
are modified by this wrapper. Record executable/DLL/import-library and SDK
hashes, as well as the complete compiler and dependency versions.

`windows-sdk-requirements.txt` pins the whole native CUDA/TensorRT package set.
Install it as one set with `--no-deps`; unconstrained transitive dependencies
were observed selecting newer CUDA components. The study uses one SDK folder
under the fixed remote root. Do not use the mixed-version first attempt as a
validated performance environment.


## Current integration checkpoint and next gates

Native SM120 packed NVFP4 and FP8 projection/plugin checks passed, including GPU
activation quantization, serialized engines and actual CUDA graph replay. The
embedded-weight/refit fix preserves BF16 behavior. Actual layer0 MLP matches5120
CPU-oracle outputs exactly. Actual decoder layers0 (GDN) and3 (full attention)
passed two stateful tokens with all hidden values and sampled states checked.
Grouped-query matrix multiplication removed explicit sixfold K/V expansion for
the verified27B geometry and allowed the full-attention graph to build at64MiB;
the old graph had failed at both64/128MiB. Native output-head W4A4 mode passed
synthetic batches1/3/8 and11 actual sparse checkpoint rows, while preserving the
separate W4A16 mode. Owned-buffer accounting includes untied heads/native workspace.

Mixed plans embed compressed weights after TensorRT rejected stripped opaque
INT8 constants. Do not retain separate full-model prefill/decode plans until
shared packed bindings or measured memory admission proves they fit. FP4 projection
rows are bounded to1024; use explicit tiling or validate a larger bound for wide
prefill. Initial strict CUDA factories are batch1/capacity32..4096; general serving
admission remains closed. Do not silently truncate context or lower the comparison
capacity to fit larger batches.

1. Full64-layer decoder smoke passed with4096 capacity, the saved reference26-token
   prompt and exact[19,248046] output. This is not broad quality or a benchmark.
   The checkpoint JPEG preprocessing gate passed; four matched JPEG VLM cases
   give4/4 semantic colors and3/4 exact token sequences on the completed fused16
   run (earlier serial run2/4). Preserve the strict failure and broaden quality
   checks. Real serial6/fused4/fused16 layer, state, inactive mask and replay gates
   pass with the corrected native-order FP8 CPU oracle. The exploratory fused16
   direct-call text pilot measured99.8ms TTFT and36.5 decode tokens/s; it excludes
   HTTP and used32 output tokens, so it is not matched to the new reference run.
2. Establish text/image quality beyond basic smoke and record the precision
   differences. Current Garnet state/cache arithmetic is F32; initial references
   use BF16 KV/conv and F32 SSM. Close this matching gate before any2x claim.
3. Native HTTP B1 validation and128-token pilots completed. Original per-layer
   text TTFT104.85ms/decode36.10tokens/s; packed full64 text123.04ms/34.68tokens/s.
   Packed lowers sampled peak from27.81GB to24.51GB but does not improve speed.
   Both retain4/4 native VLM-ID/prompt-ID gates and3/4 exact SGLang sequences;
   broad quality/precision remain open. Image pilots are exploratory: the old
   single-pixel JPEG nonce collided after compression. Rerun all engines with the
   tested binary-grid nonce and precomputed request bodies. The bridge streams
   actual native tokens, has63 CPU study checks, and remains a single queued slot;
   it does not establish batch2/4/8. Validate native concurrency and actual4K
   admission without duplicating packed weights or silently lowering capacity.
4. Establish tuned vLLM/SGLang profiles, including feasible CUDA graph/kernel
   options. Both initial profiles deliberately used eager startup for feasibility;
   they are not evidence of the fastest attainable reference. A completed
   SGLang full-decode-graph candidate captured1/2/4/8 and replayed during the
   seed3/JPEG/image-first128-token benchmark: text/c1 decode65.1tokens/s and
   TTFT141.9ms; c8 output423.6tokens/s text and351.1tokens/s images. Preserve
   early image EOS and actual lengths. Autotuning/prefill graphs remain disabled
   in that candidate; broader reference tuning is still required. Compiler/linker
   failures and SGLang's first memory exhaustion are retained in the research log.
5. INT4 is now partially integrated through a distinct symmetric G128 W4A16
   loader, direct packed CUDA kernel and serialized zero-workspace plugin.
   Generic TensorRT INT4-DQ graphs failed in three retained configurations;
   the native route passed an actual projection and stateful layers0/3 against
   independent CPU oracles, including graph replay. The guarded factory requires
   all496 text projections to remain packed and preserves the BF16 head/sidecars
   through an explicit index. Full64-layer INT4 VLM, chunk16 quality, memory,
   HTTP timing and the downloaded Qwen3.5-4B comparison remain unexecuted.
   Do not label the mixed NVIDIA checkpoint as INT4 or promote component gates
   to full-model support. The opt-in MTP worker and16-sequence greedy gate are
   implemented but have no GPU correctness/performance result yet.
6. Collect repeated text/image TTFT, decode, aggregate throughput and memory with
   fixed4K total capacity and matched request hashes, quality, cache behavior,
   output lengths and concurrency. Both the initial32-token/eight-request SGLang
   data and the new128-token graph candidate are exploratory until matched native
   quality/arithmetic/transport gates close. Preserve repeated longer runs,
   distributions and raw failures. The NVML
   sampler requested20ms but measured about31ms typical gaps and one7.787s gap;
   investigate timing/observer repeatability and report sampled peaks honestly.
7. Optimize from measured bottlenecks, then test against twice the faster valid
   reference for decode and throughput; seek half TTFT separately. Publish unmet
   targets as measured. No automatic success follows from primitive correctness,
   server process duration, a reduced workload or an untuned reference.


## Current execution dependencies (2026-10-23 continuation)

The vLLM graph attempt failed at final FlashInfer SM120 linking, not benchmarking.
Existing CUDA-library aliases were repaired and the retained objects successfully
linked with the original WSL driver-library path. No post-repair full-server result
exists. Preserve the original failure/source snapshot, use a fresh attempt label,
record resolved cache capacity, and require readiness/VLM/token protocol gates
before the corrected grid workload. SGLang's graph pilot above remains a candidate,
not a fastest-attainable claim; old image cache uniqueness is unproven.

User authorized stopping Cantor to release GPU memory; verified Cantor/WorldSense
processes were stopped with identities retained. After4B exited, GPU use was464MiB.
A later Cantor desktop restart was observed; recheck actual memory before each engine.
Keep the fixed study directory. Do not manufacture results, lower4K capacity, change
arithmetic or omit a stronger reference to fit. The linker-repaired vLLM graph retry
is now live under separate logs/output/metadata with source1e19927bc. Do not restart
it for quiet JIT output or run another GPU model concurrently. Next, run strict
MTP/self-greedy gates; run full INT4 and4B
correctness before timing; repeat native/SGLang/vLLM matched grid cohorts and actual
concurrency admission; compare the faster valid reference with raw-event replay.

Attention scheduling optimization is a separate, unwired experiment. The first
per-query compact core passes all seven CPU/KV/graph gates and improves short-prefix
component timing, but regresses full prefixes. It is not a serving default or a
model speedup. The output-tiled variant also regresses full prefixes. A split-key candidate
passes the seven numerical/KV/graph profiles and improves tested4K-capacity warm
component timings with extra scratch memory. It remains unwired, with model-level
qualification pending. Retain all variants/failures and raw CUDA-event samples.
Do not extrapolate a component ratio to HTTP TTFT, decode, throughput or the2x goal.

The goal remains active. Completion requires the requested full-model comparisons,
quality/precision/cache/transport matching, repeated metrics, actual concurrency
and honest evidence for any claimed speedup; the2x objective is currently unmet.


The attention experiments now include a split-key candidate. All seven component
qualification profiles pass; warm-core4K cases improve versus legacy with extra
scratch memory. It remains unwired and requires opt-in plugin/model validation.
Small-capacity regression is retained. See P067 and verified attention-split evidence;
no component result establishes the requested2x end-to-end reference comparison.


4B current status: initial full32-layer/vision load and text4 passed; image failed
because default min_pixels1024 produced14x14 while preloaded grid was16x16.
Initial failure archive is preserved and verified. Native image-only probe proves
explicit min=max65536 gives16x16. Source1e19927bc adds opt-in imageMinPixels and
separate retry output paths. The corrected image/repeat/3968-input qualification
subsequently passed (see current checkpoint below). Model token callbacks passed
actual GPU smoke;4B HTTP transport remains GPU-unqualified. Do not infer4B speed
or broad quality from these checks.

User explicitly requested replacing local D:/CantorAI/Garnet with our source.
Original dirty/untracked changes are retained in a pinned Git backup ref and
D:/CantorAI/backups/garnet-before-qwen38-5090 replacement record. Primary checkout
now uses a detached study commit, preserving qwen38-5090 ownership in the dedicated
worktree.9992 tracked files matched byte-for-byte immediately after replacement.
Ignored environments/build products were preserved, so source replacement alone
is not proof those old binaries were rebuilt. Remote study DLL identity remains
separately pinned and verified. Do not silently use original installed binaries.


## Current execution checkpoint (P096)

Completed corrected-grid vLLM candidate cf140177a4f24bf3ad845ad32f8cb4c0: actual
decode graphs1/2/4/8, four JPEG token/color gates, eight measured cohorts and raw
event replay verified. B1 text/image decode both about68.5tok/s; text/image TTFT
152.4/166.5ms. Output-head tuning coverage remains incomplete; MTP reference modes,
repetition, full4K concurrent admission and native/reference precision/quality
equivalence remain required before a fastest-valid-reference or2x claim.

Corrected full4B native text/image/repeat/token-event/3968-input smoke now passes.
It generated78 sampled outputs on the long input before EOS. Independent4B
reference/quality and real HTTP/concurrency measurements remain pending.

Native MTP3 exact16-sequence self-greedy gate plus actual HTTP pilot passed;
text/image decode72.1/71.7tok/s and TTFT111.2/269.9ms. B1 request bytes and counts
match vLLM, but native speed is only about1.05x and image TTFT is worse. MTP7
passes the same gates but regresses to69.0/64.0tok/s; preserve this negative result.
Neither measured profile establishes2x or real native GPU batched concurrency.

Optional resident vision input now passes six full raw output equality pairs,
actual graph replay0->4 and wrong-shape rejection, using1.5MiB extra input storage.
It is allocation staging, with fresh image preprocessing on every call. No
end-to-end TTFT benefit has yet been measured. Initial B2 packed backend qualification failed full B1-ID equality, with first
divergences at text token0 and image token8; preserve the strict gate and raw IDs.
Fixed-B2 prefill isolation now passes ten captures/eight full-row checks, and
head-only replay of identical normalized vectors agrees for B1/B2 and native/
dequantized heads. Both raw evidence archives were independently verified.
Fresh B1 retain0 configuration control is live on sourcef8587f5ee, isolating the
saved MTP3 target's retain1 setting before assigning an upstream numerical cause.
Original cross-B1 full-ID gate remains failed; no actual native GPU batch serving
or concurrency throughput is qualified.

Next GPU sequence, preserving serial GPU ownership and raw evidence:
1. Finish/archive the same-configuration B1 diagnostic; use actual hidden/ID
   differences to choose the next batch correctness investigation. Do not weaken
   the original gate or treat within-B2 repeat equality as B1 equivalence.
2. Qualify opt-in MTP3 draftT4 against unchanged targetT16 using all16 complete
   self-greedy sequences, retained VLM checks and actual stream/count gates, then
   benchmark the corrected workload only on success. Default draftT16 stays.
3. Separately repeat MTP3 image/text timing with component-qualified resident
   vision input. Keep this separate from draftT4 initially to attribute effects.
4. Establish feasible stronger reference MTP modes. vLLM MTP3 launcher and a
   same-engine ordinary/speculative16-sequence full-ID capture/check tool are
   prepared; actual mixed-checkpoint runtime/quality remains pending. Repeat
   corrected-grid SGLang ordinary and tuned feasible MTP before final comparison.
5. After batch correctness, implement the real native batch worker, qualify
   HTTP/early-EOS/ragged/state isolation, then measure actual C2/4/8 where memory
   permits. CPU grouped-transport fixtures alone do not establish GPU throughput.
6. Complete full INT4 model/precision/VLM/HTTP and4B reference/quality/HTTP runs,
   repeated measured cohorts, actual long-context/concurrent admission and memory
   accounting. Preserve all failed candidates and negative results.

Actual worker depth/chunk/vision profile can now be required before native MTP
quality gates/timing. Keep source on the dedicated branch and pin runtime DLL
separately.2x remains unmet; no claimed final fastest-reference baseline yet.


P091 update: fresh B1retain0 matches all four saved B1 sequences/repeats, so retained-output setting alone does not explain B2 mismatch. Truncated prefix1 matches B1/B2 byteexact; prefix4 differs. Isolate prefixes2/3 next, preserving full-model failed gate. Opt-in MTP3draftT4 passed all16 self-greedy/VLM/stream gates but measured71.3text/71.9image tok/s, no clear throughput gain versus defaultT16; negative result archived. Resident vision end-to-end, stronger reference MTP/tuning, actual native batching, full INT4/4B comparisons and repeated final evidence remain required.2x unmet.

P096 update: complete chunk/state evidence confirms tiny first-layer hidden differences despite exact first-layer GDN states; later states amplify differences. Substitute-input norm/MLP and residual-barrier experiments are exact B1/B2 and do not reproduce the real-layer issue. No barrier/default fix is justified. All three raw archives now locally integrity/array verified. Actual layer-intermediate isolation remains needed for batch admission; stronger referenceMTP, resident vision end-to-end and full INT4/4B/final repeated comparisons stay in scope.2x unmet.


## Structural TTFT candidate: shared compressed weights across static profiles

Original constraint: the legacy mixed packed factory embeds weights and requires one
shared prefill/decode chunk profile. At T16,41-token text uses3 target calls and
104-token image uses7. This is a call-count fact, not a measured attribution of
latency. The native NVFP4 lowering itself notes external packed bindings are
needed to share compressed constants across prefill/decode engines. The earlier
external weight bindings admitted BF16 only, which cannot safely stand in for this
mixed checkpoint within the32GB target budget.

Implementation sequence (opt-in code now present; qualification remains staged):
1. Add an IO-owned readonly packed matrix/scale pool retaining exact checkpoint
   bytes and validated logical dimensions/scalars; repack NVFP4 block scales once.
   Keep FP8 and NVFP4 recipes separate and reject BF16 expansion/fallback.
2. Add guarded external compressed weight bindings to TensorRT lowering. Validate
   declared input types, physical shapes, calibration metadata and single-owner
   lifetime. Record external compressed bytes separately from embedded weights
   and existing externalBF16 statistics; do not double-count shared storage.
3. Build static wider-prefill and T4-decode programs against one pool/canonical
   state/head, initially B1/4096 with bounded prefillT128. Reuse existing MTP wide
   prefill normalization/slicing rather than adding a second draft weight copy.
4. Qualify exact raw packed storage and same-profile component outputs versus
   embedded lowering for both recipes; prove two engines share GPU allocations,
   close/lifetime rules and no silent precision substitution. Then qualify real
   prompt/state/full-token/VLM behavior across profiles before any performance.
5. Only after these gates, repeat matched text/image HTTP TTFT/decode/VRAM against
   the original T16 candidate and valid tuned references. Preserve any numerical
   or performance regression. Wider prefill does not establish a2x model speedup.

This is a backend optimization candidate motivated by concrete current-source
constraints; it neither replaces unresolved B2 correctness work nor shrinks the
remaining reference, INT4,4B,quality,concurrency and repetition requirements.

Current status, 2026-10-24: exact packed/tiled bytes and retained-owner lifetime
passed for actual NVFP4 and FP8 matrices. Ten synthetic plugin cases and the
actual MLP at T4/T128 passed embedded-versus-external full-output checks. Both
static full64-layer shared engines now construct with 400 matrices in one pool.
Twelve untimed full-model runs passed repeat determinism and four VLM semantic
smokes. Text/image long IDs differ from the old T16 profile; this is recorded,
and full numerical equivalence is not established by those smokes.

Full same-profile T128 arithmetic now matches embedded lowering across every
packed hidden/state chunk for text/image and three repeats. MTP3/draft4 matches
its own target on20pairs, including16full128-token sequences. Six pinned retained
shared-profile goldens and real SSE/ID checks passed. The old T16 VLM gate rejected
bottom-blue because old output was blue and shared output Blue; preserve that
rejection and explicit versioned-golden policy, not a silent relaxation.

First matched shared B1 pilot: text/image TTFT71.27/105.08ms, decode69.96/71.89tok/s,
aggregate67.10/66.15tok/s. Raw SSE and sampled memory were independently replayed.
Relative to the faster supplied ordinary reference, decode1.022x/1.049x and
TTFT0.468x/0.672x. Text meets half ordinary-reference TTFT in this one short-prompt
trial; image and2xdecode/concurrency do not. No fastest-tuned or broad-quality claim.

Head-cache follow-up passed strict correctness and exercised reuse in all timed
requests, but its separate single HTTP pilot was slower:67.65/69.56tok/s and
TTFT82.74/112.51ms. No benefit demonstrated; default stays off and causal repeat
controls remain necessary. Instrumented stages point to target verification as
the main cost; nested graph/state/head profiling is running before kernel changes.

Next: profile decode/verification costs and improve actual decode and native
batching, not just first-token delay. Preserve all regressions and repeat trials.
Separate construction memory from serving memory. Continue reference MTP/tuning and
precision/kernel audits, including native F32 KV storage versus reference BF16
KV, broader quality, actual long-context admission, INT4/4B measurements, native
GPU concurrency and repeated trials. Full per-layer INT4 now passes12repeat/VLM
smoke runs with496native G128 projections; HTTP/reference measurements and the
optional whole-graph comparison remain pending. Installed-source audit identifies
a small merged96-row INT4 projection risk for SGLang; confirm actual startup
before declaring support or failure. The overall2xdecode/concurrency and half-TTFT objectives remain
unproven; no goal completion is claimed.


### Joined NVFP4 MLP implementation plan (implemented and bounded full-model qualified)

The raw calibration audit verifies all64 gate/up pairs have identical input_scale and weight_scale_2. Original output width17408 is divisible by128, so concatenating packed row matrices and corresponding128-row scale tiles is a plausible lossless storage transformation. This is an implementation hypothesis, not measured kernel compatibility or numerical equivalence.

1. Add an explicit default-off joined-MLP study option. Validate every original scalar, dtype, shape and tile boundary before allocation; reject unmatched metadata. Preserve the existing400-matrix path.
2. Load one34816x5120 NVFP4 gate/up matrix per layer directly into a readonly pool, with correct concatenated tiled scales. Avoid retaining duplicate original gate/up GPU buffers. A complete joined candidate should contain128 NVFP4 and208 FP8 projection matrices (336 total), versus192+208=400 original; expose actual counts and owned bytes. Synthetic joined metadata must reference both pinned originals and exact calibration provenance.
3. Lower one existing NVFP4 GEMM followed by correctly ordered gate/up slices, SiLU and product. Preserve every downstream rounding/materialization boundary initially. Both T4 andT128 graphs must borrow the same joined pool; bound bindings and output storage exactly. Do not concatenate the full weight tensors on every inference.
4. Before performance, qualify original-versus-joined packed/scales, standalone actual MLP atT4/T128, then full text/image teacher sequences and normalized/state comparisons, repeats and all VLM gates. Joined output shape can change cuBLASLt tactics, so identical input/scales alone do not prove identical arithmetic. Archive any failure and retain default OFF.
5. Only after correctness, measure cold-weight component timings (avoid a single matrix fitting entirely in L2), then repeat matched unprofiled HTTP pilots and concurrency1/2/4/8 where qualified. Record algorithms, actual profile, per-request outputs, GPU telemetry and sampled VRAM. Compare against faster matched reference; retain negative results. Layer-count or kernel-launch reduction is not an end-to-end speedup claim.

Current first-use eager profiler has cold initialization artifacts and generic plugin names; it does not establish a hot-kernel bottleneck. A warm diagnostic trigger or external profiler remains useful before promoting this candidate. Packed INT4 correctness has completed and independently matched all twelve per-layer token/vector goldens. Joined pool/binder and explicit MLP component are now GPU qualified: pinned packed/tiled-byte oracle, ownership/lifetime assertions and six full raw T4/T128 output vectors matching the original MLP. These use deterministic synthetic inputs and actual layer0 weights. This describes the earlier component stage. Subsequent explicit joined full-program bindings, linear/full-attention lowering, bounded complete state/token gates and HTTP pilot are complete; see the current result below. The original path remains available and joined mode is default OFF. Full remaining scope includes reference tuning/MTP, native batching correctness, INT4/reference benchmarks, Qwen3.5-4B comparisons, long-context and broad quality gates. 2x target remains unachieved.


### Joined MLP current result and next performance step

Complete bounded full-model state/token/normalized gates and20 MTP self-target pairs passed; matched B1 pilot produced72.259 text/74.157 image decode tok/s,TTFT64.719/103.715 ms,aggregate69.184/67.902 tok/s. This is about3.3%/3.2% above the prior original-MLP pilot,not a controlled stable gain or2x result. Native concurrency has not been qualified. Keep the option explicit/default OFF. Evidence and raw event verification are in research-log P143.

Next compare original/joined on the same qualified DLL with repeated ABBA trials and optional GPU clock/thermal telemetry; then measure supported cuBLASLt algorithms/workspaces on real shapes with streaming/cold-weight behavior. Do not turn a hot-L2 microbenchmark into a full-model claim or assume arbitrary FP4 layouts are supported. Correctness gates precede each full-model performance candidate. Resume the remaining INT4/4B/reference/concurrency/long-context/quality requirements as stated above;the target stays2x the faster feasible matched reference.


### Current active comparisons and recent negative kernel experiment

The standalone operand-order NVFP4 experiment passed structured synthetic BF16 sum/scale/transpose oracles on RTX5090, including projection dimensions at B1/4/128. ABBA component medians showed only about4–5% gate/up improvement at B1, no down improvement, and T128 regressions. Do not promote this candidate or infer model tok/s; actual checkpoint/general numerical coverage and end-to-end gains are unproven. Raw archive and independent CPU verification are recorded in P145.

INT4 whole-packed correctness is complete (all12 retained per-layer full token/vector goldens). The explicit B1/T16/cap4096 HTTP worker was deployed from5194b73b7 with qualified2d902723 DLL retained. That server/workflow later completed the12 exact HTTP cases,2 SSE gates and measured pilot; both are terminal, as recorded below. Never restart a live run solely on an observation timeout. Fixed folders: results/native-int4-worker, native-int4-http-pilot and corresponding ledgers. Source was frozen while the worker lived. Its terminal evidence was independently verified in P152.

Pinned INT4 vLLM/SGLang launch candidates preserve the existing4K/seq8/cache/dtype policies and pass shell syntax checks. Actual startup resolved vLLM INT4 to MarlinLinearKernel with padding; stock SGLang0.5.22 failed its96-wide repack against tile64 and yielded no admitted timing. Preserve this specific launch failure; it is not a universal SGLang support claim. After terminal native completion, identity-guarded shutdown and evidence verification, test one reference engine at a time, preserve startup failures and check actual resolved quantization before quality/timing. No different precision/checkpoint substitution. Then complete4B serving/reference comparisons, repeated same-binary original/joined trials, tuned MTP reference investigation, native batch correctness/concurrency, long-context and broader quality. The requested2x faster-baseline decode/concurrency target remains unmet.


### Latest INT4 measurement status

The whole-packed INT4 B1/T16/cap4096 pilot is terminal and independently verified after12 exact HTTP goldens and2 SSE gates. Text/image decode4.4556/4.4495 tok/s; TTFT697.010/1641.609ms; aggregate4.3815/4.2406tok/s; sampled peak21786648576/21792940032bytes. This target-only compatibility path is much slower than the measured mixed NVFP4 MTP3 configuration; configurations differ in quantization and speculation, and no precision-only causal gain is claimed. Raw evidence/source/metric verification recordedP152. The ordinary vLLM INT4 pilot and later cache-reset exploratory pilot are terminal and archived; see latest qualification limits below. Keep SGLang INT4,4B/reference, repeated original/joined, native batching/concurrency and broader/full-context requirements intact. Do not substitute this cross-precision comparison for the2x faster-reference target.


### vLLM INT4 reference update and cache gate

The first ordinary vLLM INT4 trial completed4 VLM and16 full128-token quality captures plus all8 B1/2/4/8 text/image cohorts. Raw-event verification records about77.37/77.33tok/s B1 text/image decode and535.72/501.22tok/s C8 aggregate throughput. Current native INT4 implementation is much slower. All16 prompt-ID pairs match; only5 full output-ID pairs match, so broad/cross-engine quality equivalence remains unproven. Nonzero multimodal cache hits after reused quality fixtures disqualify matched image TTFT from this first trial. Retain it with explicit cache flag; verify installed reset routes, clear BOTH processor and encoder caches while idle, and repeat with a new preserved result path and cache evidence. SGLang INT4 original load terminated9 at its96-wide/tile64 Marlin repack; startup evidence is independently verified, with no timings admitted. Remaining optimization/comparison scope and2x target are unchanged. SeeP153 for complete raw numbers/hashes/limitations.


### Current cache-reset qualification and fixed NVFP4 candidate gates

The first strict vLLM INT4 cache-reset workflow stopped before timing because previous-process full128-token captures matched15/16, although all16 prompts/semantic requests matched. A further same-live-process repeat matched16/16 exactly. A separately labelled exploratory pilot then completed8 cohorts after both processor and encoder reset acknowledgements; raw counters show62 queries and zero new hits, process unchanged. All559 archive file hashes,64 raw measured SSEs,8 sampled VRAM cohorts and current C1 texts were independently verified. Text/image C1 TTFT86.3078/124.8349ms,decode77.4505/77.4228tok/s; C8 aggregate531.6427/490.3604tok/s. Retain unexplained cross-process drift; do not label this2x-eligible or broadly quality-equivalent. Earlier cached image TTFT stays excluded. Both servers are terminal;GPU39MiB observed after shutdown. Research logP159.

Repeated synthetic NVFP4 screening identifies a fixed down-projection candidate: N5120/K17408, algorithm70,tile20,stages37,cluster5,inner0,swizzle0,custom0,splitK-2,reduction2,64MiB workspace allowance. Component gains~1.26(B1/B4) and1.22(B128) are best-of-many synthetic screening, not end-model gains. Candidate remains unpromoted. Next gate uses one actual pinned full down-projection and signed synthetic inputs with full baseline/candidate raw BF16 outputs and a bounded independent decoded CPU oracle. Do not assume exact production activation coverage.

For runtime integration, preserve the legacy plugin/workspace contract. An opt-in candidate must serialize its workspace/selection policy, retain old-engine deserialization with the legacy policy and prevent environment changes from changing a cached engine's scratch-buffer requirement. Test clone/serialize/reload, fixed configuration matching, supported batches1/4/128, missing-candidate/error behavior and all retained full-model target/MTP/HTTP gates before timing. Keep the original path available for comparison. Measure full-model performance after correctness; do not extrapolate the component ratio.4B serving/reference, native batching/concurrency, repeated/full4K-context and broader quality remain open;2x faster-reference target remains unmet.

### Current runtime candidate qualification: failed prefill option excluded

The default-off rebuilt native DLL e82279af/source5d6759336 passed12 target/vector goldens and both complete option-off original/joined T128 state controls against the retained teacher. The all-batch opt-in v2/policy1 candidate FAILED image state equality:21 chunks and307084 retained hidden values differed, max absolute15.375, affecting active rows44..103. Same-binary controls isolate this to candidate configuration; internal cause and quality impact remain unproven. It is not admitted for VLM serving or end-to-end benchmarking. Failure/control archives and independent verification are recorded inP164; no new model-speed claim.

Next test a separately serialized decode-only policy that preserves baselineT128 prefill; retain v1/v2 engine semantics, add a new version/tag and rerun workspace/clone/deserialization plus the SAME full-model state/target/MTP/HTTP gates. Decode correctness and speed are unproven. This changes the implementation candidate, not the required full VLM/4K/concurrency/4B/reference/quality scope or2x target.
### V3 decode-only implementation status

V3/tag2 now preserves baselineT128 prefill and routes only qualified1/4 activation rows to the fixed candidate. V1 and historical failedV2 retain their serialized semantics; contracts/component output checks and native build passed on5090, source754cfa424/DLLba67dc9d. The originalC2660 test compile failure and corrected run are retained and independently verified (P165). Full same-binary state/target/vector qualification completed and was independently verified. Subsequent HTTP20 target/MTP pairs match each other and six retained goldens, but five full outputs differ from the prior qualified20 set and repeat exactly on the same live worker. A new same-DLL option-off HTTP20 control restores all20 prior full outputs. V3 option-on is therefore rejected for serving/timing; no SSE or benchmark was admitted. Both HTTP workers are terminal and all raw failure/control evidence independently verified (P166/P167). P164 failure remains valid, and no kernel ratio is a model gain. Full VLM/4K/concurrency/4B/reference/repeat/quality scope and2x target remain unchanged.

### Next optimization decision: warm full-model CUDA evidence

The installed Nsight Systems2024.6.2 passed a bounded actual CUDA trace smoke on the loaded runtime:5 normalization kernels, CUDA API/graph records, graph capture/replay and exact producer output checks; raw trace/SQLite and all13 archive file digests independently verified. Cold module-loading times are not serving bottleneck evidence. The first interactive warm-mtp-cuda-trace-attempt1 passed12 warm cases but exited0xC0000005 at collection start, completing zero captured cases; raw SQLite has no CUDA kernel/API data and the failure archive is independently verified. The startup-instrumented attempt also exited0xC0000005 after12 exact warm-up cases and before either capture case; its partial CUDA trace has no valid diagnostic markers and is not a warm ranking. Both failure archives are independently verified. The identical unprofiled14-case control completed0 and was independently verified: all12 warm and2 follow-up full outputs match, same probe bytes, same DLL and launcher differing only output directory. The failure did not reproduce in this one control; internal cause and universal stability remain unproven. A side-by-side signed NVIDIA2026.5.1 profiler extraction into the fixed study tools folder is now running; official release notes explicitly support loadedCUDA13.4. No driver/runtime/application-source replacement or restart. Revalidate actual tool identity and full trace coverage; a version upgrade alone admits no profiling or speed result. Unique35892/35908byte HtoD markers must independently delimit the warm window; full startup statistics contain cold/build work and are not a hot ranking. It must pass twelve warm exact text/image/VLM cases before collection, then two full128-token text/image cases inside a bounded window; launch/start/stop/stats and controller have distinct raw ledgers. Verify actual warm CUDA rows, full token IDs and linked source/binary evidence before choosing the next kernel/host-control optimization. Instrumented timing is diagnostic and cannot replace uninstrumented matched HTTP benchmarks.

The implementation priority after that trace is an evidence-selected change to the dominant target/MTP/head/state/control work, with full original correctness gates and repeated same-binary A/B performance trials. For native concurrency, return to actual layer-intermediate B1/B2 diagnosis; synthetic norm/MLP/residual substitutes did not reproduce the real discrepancy. A serial B1 queue is only an offered-load service diagnostic and never substitutes for true GPU batching or the2x concurrency requirement. Complete Qwen3.5-4B matched HTTP/reference measurements and long-context/broad-quality coverage, preserve INT4/reference/cache limitations, and tune feasible reference/MTP configurations. The objective remains at least2x the faster matched feasible baseline in decode and true concurrency throughput; half TTFT is additional. No candidate or chart is promoted from a component ratio or weaker substitute gate.


### Current4B comparison and profiler decision

Native Qwen3.5-4B BF16 target-only B1/T16/cap4096 HTTP pilot is complete and independently verified (P171): text/image decode85.8324/85.7470tok/s,TTFT63.2354/119.1346ms,aggregate82.6198/79.6275tok/s,sampled adapter usage11335159808/11337256960bytes. Gates include four retained old smoke captures,four VLM semantics,16 full128 repeats,two real SSE gates; all16 measured streams and worker IDs replay exactly. Same semantic requests/images and full prompt IDs as the prior27B pilot are verified; architectures,precision,speculation,chunk profiles and binaries differ. This descriptive4B comparison does not establish model-quality parity, causal model-size/precision gain or the27B2x objective. Native4B vLLM/SGLang references, repeated confidence, GPU concurrency and broad/full-context checks remain open.

Old Nsight2024 full traces failed0xC0000005 after12 warm cases, while the identical unprofiled14-case control passed. The newer signed Nsight2026.5.1 Windows payload is now extracted side-by-side into fixed tools/n26 after preserving MSI/path/guard failures. Bounded actual CUDA smoke and all49 evidence hashes/SQLite records are independently verified. A new original full-model trace handle warm-mtp-cuda-trace-2026-attempt1 is live on the5090, app754cfa424/DLLba67dc9d/candidate0 frozen. Same producer as unprofiled control;12 warm plus2 capture full-output gates and unique ordered HtoD markers precede any hot ranking. The full trace is not yet a success or a serving-speed result. After it closes, select the next optimization from verified hot operations, rerun original correctness gates and refresh current27B HTTP same-binary baseline/candidate with repeated trials. Keep native batching diagnosis,4B references, full-context/broad quality and fastest feasible tuned reference requirements intact.2x decode and trueGPUconcurrency remain unmet.

P172: full2026 warm CUDA trace independently verified (24-file archive6d8662718341ffb9e5580ff41a263458aae0e7780e9577a596ce211d87f81dec). See development log for scoped kernel/API ranks. Next correctness experiment is existing resident_controls=True (inline request updates/chunk gathers), first transport48trials then14 full-model cases, unchanged754/ba67/defaultNVFP4. No speed/default admission before full exact-output and actual SSE/HTTP gates; uninstrumented same-binary A/B required. Entire2x and trueGPUbatching target remains unchanged.

P173: optional resident request-control HTTP wiring prepared, defaultOFF and restricted to joined shared MTP3. Transport producer48trials passed; full-model14-case process remains live. Do not deploy worker edits while it runs. After terminal, archive/independently verify before HTTP20pair+20oldID+2SSE gates and B1 pilot. BaselineABBA/trueCUDA concurrency and original2x requirements remain open.

P173 follow-up: resident-controls14case archive independently verified SHA802c8ab7dec50edb7984c19f0a4cd9d1ebf9af276e7fd00f43d4adaf9e565ba4. Guarded two-file script-only deployment keeps DLLba67 and original backups. Original native-resident-controls-http-server(95279) and workflow(48577) launched; source frozen. Next poll original handles, independently reaccount outputs/SSE/VRAM after terminal; performance comparison still needs repeated same-binary A/B. No2x admission.

P174: independent resident-control HTTP archive6d8c4e7049a9bafe7df3c4d91963f96a02915860ac5d2b0554e046666589f708 verified199files,20retained pairs,16measured streams,VRAM. Decode73.92196190968554/76.12551135754535; TTFT73.81760000134818/104.93154999858234ms. No2x. Original48577 native-controls-abba-attempt1 launched for four fresh OFF/ON/ON/OFF legs (prior pilot excluded); same source/runtime/checkpoint/profile, perleg20pair+20oldID+2SSE gates. Next poll original controller; do not sync app source or launch competing GPU job. Independent perleg reaccounting and variation/telemetry review required before causal claims. TrueCUDA batching and original broader scope remain unfulfilled.

P175: a1OFF/b1ON verified199files each, archiveSHA da0dbf3d2307d6f1e4ccefccbc35b208ea0d58f40688dad38d3301c460133f65 /36c42f4c5bc4255c1a30ad7d947e08894671d1f239f4379b358bab0bb36a4870. Pauses retained in both modes; no causal claim. Original48577ABBA stilllive/b2running; source/runtime frozen. Parent4leg verifier prepared. Separate actual layer0 stage diagnostic and guards/probe prepared; CPU /Zs syntax0 only, no binary build/deployment/GPU. See layer0-stage-protocol. Native full64batch mismatch and original2x/broader scope remain unresolved.

P176/P177: four freshABBA legs independently verified(parentSHAa09e3f0587b7cf252b2f60bf26df6d3e52642504f09f1496dbf629877b6d7245). Observed ON decode means+6.7%text/+10.4%image; text aggregate worse, pauses retained, defaultOFF/no2x. Actual deployed ba67 CUDAresource inspection verifies IOkernelsLOCAL0/STACK0, rejects spill hypothesis. Guarded new native34cb221458064a4b7264586412be5b3c7acca888cd9e74681ac2a41d259413da build0; oldbinary/source preserved, no serving admission. First stage probe1 failed test graph replay and input construction; immutable61file failurearchive65cb85f337c943a1025fe67f6201bb5a150087fea15a68b06d4723d153cccfc1. Corrected fresh prompt input+fenced warmups installed on samebinary; original48577 layer0-real-stages-attempt2 is live. Do not change source/binary during liveGPU job. Next close/archive/replay input/primary/all raw stages before attribution; default fullmodel and full64B2/trueGPUbatching and broader originalscope still pending.


P178: actual stage attempt2 archive d05aa555feac3183d54dd7c1975741fc3bb525739eccff26a98166b5db761a99 independently replayed locally and remotely (actual checkpoint embedding rows checked). Primary/ordinary/historical/repeats/duplicate rows all exact. First B1/B2 difference at FP8 attention_output:9/81920words,maxabs3.0517578125e-5; gated GDN input identical. See development log. Next use real checkpoint/input projection fixture and require native helper control reproduces both M16/M32 outputs before candidates. Not unique algorithm-cause, full-model qualification, batch admission or speed proof. Original2x and remaining baselines/context/quality/concurrency scope unchanged.


P179: rebuilt34cb default full27B14-case old-ID regression independently passes (six128-token text/image plus eight2-token semantic cases), archive4175e4a45c34f1e57cb3cc8b0932b9301a2dca470889a553cae51fc1f71c785a. Actual projection same-executable control and checked shared-algorithm intervention independently pass, archive2d1061813fd425d6a01c3c59836d06e01d09b9e487723c6ffb03f3678294b364: native default M16/M32 reproduce old outputs, common M16 algorithm35 makes both M32 duplicate rows match canonical M16 twice, cast/quantized inputs unchanged. This is one real projection fixture, not full-model batching or speed proof. DefaultOFF GARNET_FP8_GDN_M32_CANONICAL candidate is restricted to physicalM32/N5120/K6144,SM12.0/cuBLASLt130400/4MiB and checked canonical heuristic; plugin default16-byte wire preserved with optional policy metadata. Isolated source build/policy/real-matrix gates precede app installation, then actual prefix1/2 stages and complete64-layer B2/old-ID/HTTP/SSE correctness before any concurrency/performance claim. Initial isolated test compile error retained and lifecycle-only correction under explicit source receipt. See development log; original2x/fullscope unchanged.


P180: actual TensorRT control/candidate layer0 first16-token witness independently passes on both remote/local CPython:202file observations/57losslessly deduplicated objects/162raw arrays, archivebf0788e7c4fdd95f50c16a525aeb9acafd9ed5f6b6e0043cb5b2af8be10f97a7. Default preserves old per-profile outputs; candidate all stages and complete primary hidden/state match canonicalB1 across both profiles/rows/repeats. Only this zero-state first chunk qualified. Full-attention o_proj shares GDN output geometry, so restrict lowerer by .linear_attn.out_proj.weight name before deeper tests (all broader model goals retained). Next unchanged historical producer with prefixes1/2/3/all41tokens/allchunks/state and oldB1 witnesses; full64B2/oldIDs/HTTP/SSE and performance still pending. See P180 development log for explicit source/worker/build/evidence records.

### Current full-depth batching investigation (2026-10-24, P181-P186)

The default-off canonical FP8 intervention passes actual layer0 raw stages and all41-token prefix1/2/3 states. Full64 ragged B2 generation still fails preserved B1 IDs (text first difference14, image8), so no batch serving/throughput admission. Independent prefix4 replay passes all8 raw final-hidden arrays for both GDN-only1ed and geometry-wide8568 binaries, retaining historical B1 outputs. Expanding to full-attention output showed no difference on this fixture. The original full64 duplicate-text prefill diagnostic is now live; keep source/runtime frozen until terminal, then archive and independently replay4 raw arrays. If prefill diverges, localize deeper layers/projections; if exact, investigate ragged/decode/head profiles against saved per-step witnesses. Do not count this prefill test as full generation, VLM or speed qualification.

Source inspection identifies an additional required shared-pool fix before external B2 qualification: ProjectModelOptFP8 checks weight->getName() for the GDN suffix, but external packed inputs are named compressed_packed_<index>. BindExternalModelOptWeight already receives the validated logical checkpoint name; carry that name in NativeFP8Weight metadata for both embedded and external registrations, then select family from that metadata. Existing defaultOFF behavior must remain unchanged. Test both binding forms, GDN vs same-shape attention, invalid policy and serialization/clone/profile guards before model gates; no fix/build deployed during current worker.

Code/checkpoints/results remain in fixedD study; only temporary engine files use the fixed owned C NVMe slot. Owner/reserve/path/reparse guards and closed-run hash receipts precede retirement of new private cache files. This addresses observed large-file I/O investigation and disk hygiene; it is not warm TTFT/decode evidence. Current baseline comparison remains the admitted B1 pilots, with repeated ABBA showing modest/noisy controls benefit and no2x. Complete true batching, fastest feasible tuned references including MTP,4B references, actual4K prompts, broad quality and repeated confidence measurements before final target assessment.

### First-attention projection localization (P187-P192)

Current GDN-only policy passes final prefix4 hidden but every-token capture reveals earlier-token differences; expanding to attention output changes13minor words and leaves dominant token4/23/33 differences. Default-off full-attention stage support and logical-name metadata fix are built on5090. Actual layer3 ordinary/diagnostic complete hidden+KV preserve old witnesses, all171raw arrays and six negative profiles independently verified. The first observable difference is query/gate FP8 projection (16words), with input normalization and K/V projections exact. Bounded actual N12288/K5120,M16/M32 native helper reproduction is now running.

After exact control reproduction, test a checked common-M16 algorithm with the same executable; do not assume an algorithm ID. Only if the real fixture passes, introduce bounded/default-off full-attention policy using validated logical checkpoint names, actual geometry/version/workspace/AlgoCheck guards and unchanged default wire behavior. Re-run every-token actual attention stages and primary parity, then deeper state/hidden and full64 ragged text/image B2 IDs before serving/concurrency timing. Negative candidates remain archived. Shared-weight logical-name path still needs its own actual binding qualification; embedded stage success is insufficient. Existing measured B1 pilots remain the only admitted comparison;2x decode/concurrency, tuned references,4B references, actual4K inputs and broader quality/repeats remain open.

Current next execution gates (2026-10-25 UTC): private-padding12case state and13group/40request streaming both passed and independently replayed on both machines. Start the prepared padded-mtp-http-pilot-attempt1 only after method hashes/CPU gates match; preserve every cohort/failure and replay raw HTTP/native/reference pairs before interpreting speed. Then run kv-delta-decode-cuda-trace-attempt1 separately using pinned Nsight2026, qualifying the unchanged eight-case math first and selecting only the two marked warm decode calls. Use measured kernel costs to choose the next native optimization. The2x matched decode/concurrency, half-TTFT, actual4K workloads, tuned/MTP references, broad quality, clean repeats and matched4B references remain open.

Padded HTTP attempt1 now closed, archived(e1d32d11...) and independently replayed on both machines. C1decode68.63text/67.28image tok/s; C8aggregate195.57/149.97tok/s vs faster references445.77/439.86;2x and halfTTFT unmet. Charts include every cohort and regression. Separate kv-delta-decode-cuda-trace-attempt1 is running, with10CPU tests and strict8case/16rawhidden qualification before raw marker-based kernel analysis. Use its evidence to prioritize native output-head allocation/graph handling or actual dominant operator; no bottleneck attribution before replay. All original objective requirements remain intact.

Current evidence: marked CUDA profile ceb182e0... closed and independently replayed on both machines,8cases/16rawhidden exact and two decode-only marker windows. Matrix kernels dominate recorded work; raw host waits mostly overlap GPU execution. Full2x/halfTTFT remain unmet. Original register128-padded-state-attempt1 is now live on the same34bea DLL using an existing default-off backend specialization;12full/partial cases must match retained38rawhidden and allstate/counters before stream/HTTP gates. Preserve negative results. Revalidate existing resident-controls behavior for current B4 host-mapped/compact profile rather than repeating the previously negative head-reuse experiment. Larger batch/KV-precision and wider-prefill work needs explicit new contracts and quality/memory tests; no assumption of2x gain. All long-context/reference/quality/repetition/4B requirements remain intact.

Updated gates: register128 state12cases/38rawhidden and allphysical states match register-off, archived6d4718fd... and independently replayed on both machines. Register128 stream attempt1 is running with the same13groups/40request wire. Only after its terminal archive and both replays, start the prepared register128 HTTP pilot; observe its original handle only during timing and retain every cohort including regressions. Do not infer a speedup from correctness. BF16 KV B4/B8/T4/T32 layout has six CPU checks and separate native commit/readonly-overlay sources compile for sm_120, but no GPU qualification, plugin/helper wiring, model integration or VRAM fit yet. Qualify operators (including current-chunk BF16 rounding, cache boundaries511/512/513/4095, all64layer offsets, accepted prefixes, inactive rows, readonly inputs, noalias/failure handling) before creating an explicit opt-in BF16 model profile. Preserve existing B4FP32 admission/guards. Then validate ordinary/MTP BF16 correctness, broad quality against pinned references, B8/T32 memory fit and matched HTTP before judging progress toward2x.

Latest execution: register128 HTTP archivef3e7a67f... closed and replayed independently on both machines. C1decode72.54text/70.70image; C8aggregate208.53/167.86, below445.77/439.86ordinary refs. Preserve long sampler gaps/textC2,imageC1/C2 regressions; clean repetitions needed before causal claims. Allfive charts and portable report generated with every cohort. Original standalone BF16 commit operator attempt1 now running after all19hashes matched; requires34full-state byte controls/62negative calls and independent archive replay. Overlay numerical/public wiring/model/quality/B8-T32fit/performance remain open, along with all original2x/halfTTFT/actual4K/tuned-MTP-reference/4B/repetition requirements.

BF16 commit operator attempt1 is now closed/replayed on both machines: archive70f0446c..., four B4/B8,T4/T32 profiles,34full-state byte controls and62rejected calls pass; readonly sources preserved. This qualifies the standalone finite-value accepted-prefix commit operator only. No model/runtime admission or27B memory-fit/performance claim. Next implement and run independent FP64 BF16-overlay numerical/mask/readonly/noalias/boundary tests on the dedicated5090; qualify a separate serialized plugin and native helper before model integration. Preserve the existing FP32 B4 path and all original goals. No GPU worker remains live after the commit probe terminal/idle cleanup.


### Current BF16 integration gate (2026-10-25 UTC)

Standalone BF16 accepted-prefix commit(34state/62negative controls), dense attention(70cases,1172480FP64 witnesses,200negative calls,62cache-equivalence controls/four graph replays) and separate plugin CPU contracts(64configurations/roundtrips/clones,3328wire/1984descriptor/1344format rejections) now pass and are independently archived/replayed on both machines. Maximum attention absolute error7.50e-8. Installed TensorRT runtime and headers report101601; actual remote DLL is pinned and preserved. No model/runtime admission or serving speed claim follows from these bounded gates.

1. Add a separately named BF16 native commit helper that strictly validates the new complete64layer metadata contract and typed B4/B8,T4/T32 profiles. Preserve readonly input leases, write ownership, bounds/overlap rejection and unknown-completion poisoning. Keep the legacy FP32 helper/guards unchanged.
2. Add explicit default-off BF16 lowering and packed-decoder/replay profiles using the separate plugin identity. Q/K normalization and RoPE precede current-K BF16 rounding; V enters attention/commit at the same BF16 precision. Preserve FP32 GDN states. T32 prefill excludes replay payload; T4decode retains it and shares only a correctly sized owned output pool. Qualify real TensorRT enqueue/engine serialization/lifetime before full-model serving; the CPU wrapper test never enqueues.
3. Introduce B8 MTP/workspace admission as a distinct qualified profile, without relaxing existing B4 guards. Test actual logical-M32 projection behavior, both ordinary and verified decoding, partial/reset/retired rows and cache boundaries against an explicit BF16 ordinary oracle. Do not compare packed BF16 carrier words numerically as FP32 KV or demand physical equality to the old FP32 cache.
4. Preserve the known34bea DLL before any rebuild/switch; require no live GPU worker and hash-bound source/build logs. Prove default-off old-model parity on the new DLL, then new BF16 full-model quality, source ownership, allocation fit and protocol correctness. Operator-only4.55GB memory is not27B model fit.
5. Only then measure B8/T32 text/image HTTP against matched references and previous Garnet. Keep every cohort/failure and investigate sampler gaps separately; require clean repetitions for causal gains. Actual4K prompts, fastest feasible tuned/MTP references, INT4 quality comparability, matched4B references and broader model quality remain required. Full2x decode/concurrency and half-TTFT objectives stay intact and unmet.


BF16 integration status update: helper/package, strict64layer metadata, default-off lowering and B4/B8,T4/T32 decoder/replay profile now compile in production /MT native DLL01a3d784..., build archivec8d0d802... independently replayed on both machines;33CPU contracts pass. Earlier standalone plugin /MD was not production ABI qualification; correction retained. Original bf16-model-default-off-attempt1 is live on the new DLL, with byte-identical retained full-model probe and exact IDs/state/hidden controls. Prepare actual BF16 model gate locally while it runs; do not change deployed sources/runtime. Require both closed default-off replays, then actual helper metadata/negative/zero-prefix controls and real plugin/model enqueue with strict same-BF16 ordinary/MTP state and fixed-corpus token IDs. Wider-prefill/B8 memory fit, serialization/lifetimes, partial/reset/retired rows, true4K, broad-quality and protocol controls remain open before timing. Keep C1 decode optimization distinct from batching; qualify a direct low-batch compact profile or other measured dominant-kernel improvement, rather than treating B8 throughput as C1 proof. All original2x/halfTTFT and reference/INT4/4B/repetition requirements remain open.


Original bf16-model-default-off-attempt1 now terminal exit0 with private-cache cleanup complete. Actual new /MT DLL retains exact four-group/repeat token IDs, ordinary/MTP full-state hash witnesses, allfour real-row duplicate-pair checks and all eight raw last-hidden arrays versus the retained original default-off model. Immutable870224B/68file archive8d87b81661cf02b003f3b16d6e4c1b88439957b950e6a01abe626aeeb884b46c independently replayed on both machines; current/archived verifierf24a58bdaf8b9e52054d5e2df582700429fbe0ce18bd23e8eabda5673baa91f0 match. Correctness adapter peak33727782912B is not serving memory or a BF16-fit/performance result. Candidate23sources pinned before launch and unchanged after; extra inherited18source files match retained old pins on post-process readback. This last readback is explicitly not retroactive prelaunch evidence. BF16 enabled path remains unqualified.
Prepared hash-bound BF16 model probe/runner and independent archive/replay methods. Preparation488dd08013862de61160991250db3188efeb60ef993d87eddafc213e25e95ffb binds11methods and10retained oracle/report inputs, checkpoint config71fa01bb..., metadata fixture7481110b... (996367B). Runner now pins the complete inherited-plus-candidate source set before/after launch, gates on both verified default-off/build archives and byte-derives the new probe from the retained full-model probe. Four runtime metadata profiles/8856invalid metadata records,16malformed commit calls andthreezero-prefix modes are planned, not yet passed; then B4/T32prefill/T4decode fixed text/image groups repeated with exact retained token IDs and exact same-BF16 ordinary/MTP state/rawhidden. No broad quality, actual4K, B8, protocol or2x timing claim. No restart of a silent process; original default-off handle completed before any of the new BF16 method uploads.


2026-10-25: Separate B1 overlay plugin preparation (runtime UNRUN).
Original integrated-batch-top1-model-lifecycle-attempt1 handle re-polled LIVE;
read-only native disabled report advanced to11/14cases with prefix phase passed,
no recorded error, neither complete mode closed/replayed. No remote mutation.

Prepared separate GarnetQwen38BF16KVLowBatchOverlay type/version1 with strict
batch1, T4/T32, all16full-attention layers and unchanged72byte nine-int64 wire.
Literal old-wrapper derivations preserve all shape/format/workspace/offset
guards and forward to the already GPU-qualified B1 attention candidate. Old
B4/B8 plugin sources remain unchanged; new files are isolated under test2026,
not linked or registered in Garnet. CPU-only workspace adapter extracts exact
host validity/workspace formulas; Run symbols refuse and count any call.

Private local /MT C++17 compile/link against matching remote TensorRT10.16.1
headers/import library passed. EXE4326d674a1793f63d7fd2675e8aeb1f550fa81f81d052c95ee9f9f2469b5f7c8
imports only nvinfer_10.dll/KERNEL32.dll; no CUDA or Garnet dependency. Executable
NOT RUN: local runtime DLL differs from remote. Actual build ledger and unchanged
source hashes retained; literal wrapper/probe/adapter derivations replay passed.
Initial preflight wrong fixture path failed before compilation; original method
and explicitly summarized observed failure retained. Corrected generated fixture
include/path only. No C++/kernel/control-count/tolerance change for that correction.

Probe plans32roundtrips/clones,1632bad wires,992bad descriptors,672bad formats,
96cross-dialect rejections and64legacy-positive roundtrips, zero Run calls.
Those runtime checks are UNRUN. Future remote registry/ABI and valid GPU enqueue,
B1 commit/helper/lowering/model, quality, actual4K and HTTP remain unqualified.
Archive/JSON in outputs Garnet-B1-Overlay-Plugin-Preparation; this is preparation,
not numerical/performance admission. Original2x/halfTTFT and all comparisons remain.


2026-10-25: B1 wrapper GPU forwarding probe prepared; GPU UNRUN.
Same44case/119040scalar source fixture is pinned byte-for-byte to closed
2eeb8aa8...operator archive, with unchanged candidate CU/header. The only
positive-dispatch change sends87host calls through new B1 plugin descriptors,
workspace validation and enqueue:44ordinary,3capture construction,40cached
prefix calls. Original3graph replays, full canonical readonly/canaries and
104direct-kernel negatives remain. Negatives are explicitly not wrapper checks.
Separate32profile CPU registry/serialization/negative wrapper probe is required.

Private local /MT sm120 compile/link passes; EXE94c6f9b3a20bf95890812c94ddc47c2d9924242802b657827c13c0ba978aef7b,
imports nvinfer_10.dll/KERNEL32.dll (CUDA static runtime linked). Actual NVCC
inherited-overload warnings retained unsuppressed. No local or remote GPU run.
Strict future replay requires all90observation records (except declared final
wrapper fields), all476160selected raw bytes and exact87dispatch count against
qualified same-kernel parent. Whole-model/quality/performance are never admitted.
Six actual CPU synthetic mutation controls pass: raw byte corruption/truncation,
stream order/incompletion, GPU/control/boolean types, dispatch/admission counters,
and wrong parent corpus. Both original independent FP64 proofs remain the parent;
no synthetic transformed stream is counted as a new GPU result. Build/method/
mutation ledgers and derivations archived. Original model handle re-polled LIVE;
deployment unchanged. Next: close/replay original model, collect standalone B1
commit and CPU/plugin forwarding checks, then helper/lowering/full-model gates.
Original2x/halfTTFT, broad quality, actual4K, tuned/INT4/4B and repetitions open.


2026-10-25: B1 plugin controller/factory/archive replay prepared, runtime UNRUN.
Added actual-toolchain build recipe for two isolated executables in fixed
tools/bf16-kv-native. Both link one B1 wrapper object; CPU contracts separately
link exact host workspace/refusal stubs and old B4/B8 wrapper, while GPU forwarder
links the qualified B1 CUDA candidate. No deployment source/DLL/SDK installation.
Preserve all unknown/existing target outputs; new distinct owner marker only.

Factory requires actual terminal model cleanup and equal independent replay
pair, actual successful20case B1commit proof pair, and both44case B1attention
FP64 passes with declared fresh-FP64 last-bit/environment differences retained.
It verifies parent archives, unchanged source/runtime lineage, source recipes,
compiled-source pins, local/remote header+import-library correspondence and
actual remote TensorRT DLL versus closed-model runtime. Positive path UNEXERCISED.
Actual current guard returned2 with six missing real files and created neither
preparation nor upload plan; raw expected nonzero ledger preserved. AllPython
methods parse, six synthetic forwarding stream mutation tests pass. Remote
build/controller/archive checker UNRUN; no test data is treated as actualGPU.

Future collector enforces idleGPU/no study executable and one sequential CPU
contract then GPU forwarding run; hashes sources/tools/SDK/runtime before/after,
keeps executable/import/compiler/resource logs and archives failures in place.
Independent checker replays literal wrappers/probes/refusal adapter, verifies
actual32profile CPU result and intended87host forwarding calls, all90observations
and476160selected bytes against actual closed same-kernel GPU archive. Failed
model stays failed; plugin qualification cannot promote model/quality/HTTP.
Current integrated full-model original handle remains LIVE; no remote upload,
source/build/runtime mutation or competing GPU worker. B1 helper/lowering/full
model, broad quality, actual4K, tuned/INT4/4B, repetitions and2x goals stay open.


2026-10-25: Disabled mode closes, enabled mode lives; isolated B1 helper compile.
Original integrated-batch-top1-model-lifecycle-attempt1 controller remains LIVE.
Disabled mode now exits0, controller exact-parent comparison passes14cases and
102raw vectors; final reset completes and eight owned private engine files are
hashed/retired by original method. Enabled mode starts in a separate process.
Same retained native hashlib warnings remain visible; SHA256 continues working.

Single fixed evidence-only snapshot outside live job directory is4,869,668bytes/
105payload files, SHA f0269fb8e0a797465df20a8c5199f68edff99a17c496f430e08dc36d2b532692.
Local independent CPU replay verifies every snapshot/224parent manifest member,
all14case report/ID/state/callback/prefix/reset/retirement fields and102raw vectors
(8,355,840bytes); equals original controller parity. Actual sampler host dispatch
counters0->4078legacy and0->0parallel, correct for disabled mode. Counters exclude
GPU graph replays and are not speed. Snapshot is explicitly whole-run UNFINISHED:
no final source/runtime-after closure, enabled completion or final model proof.
It cannot unblock B1 runtime factories or serving. Transfer/auth-prompt/bulk-input
issues and preserved partial copies are recorded separately; no GPU restart,
deployed source/DLL/SDK mutation or competing GPU work occurred.

Prepared CUDABF16LowBatchKVDeltaCommit as a distinct unregistered native package.
Literal old helper with new B1-only metadata dialect/class and B1 commit call;
all tensor/runtime ownership, descriptors/types/strides/device/writable/byte-size/
alignment, overflow/alias, strict integer span, lease/Finish, mutex and poison/
close/rebuild guards remain. Original B4/B8 helper/parser sources unchanged.
Private local /MT compilation of B1 and unchanged legacy helper succeeds against
main public SDK header cffd02c2...matching actual remote native build; current
remote compatibility header852e5c3c...also matches. Objects c8943217.../ec1848fe...,
never linked, loaded, registered or executed. Initial wrong JSON-header pin
path failed before compilation; original method/observed-failure summary retained,
only pin path corrected. No C++/kernel/control/tolerance change for correction.
Actual build/source/hash methods and preparation ZIP/JSON preserved. Native
ownership/lifetime/GPU helper behavior, plugin/lowering/full-model, broad quality,
actual4K, HTTP, tuned/INT4/4B and repetitions remain open; original2x/halfTTFT unmet.


2026-10-25: True physical B1 route now wired LOCALLY, default OFF, runtime UNQUALIFIED.
Added bf16_kv_low_batch_state keyword to packed program/decoder and separate
create_cuda_bf16_low_batch_study_decoder factory. It pins B1/cap4096/T32 prefill/
T4 decode, retained/deferred/shared/host/register128/joined mixed projections;
verifies128NVFP4+208FP8 pool, native NVFP4 W4A4 head/host BF16 embedding and graph
binding counts. Strict type/architecture/mode gates precede allocation; missing
matching native helper raises before tensor I/O. Existing default factories and
old B4/B8/reference validators remain separate. Capacity4096 is not actual4K.

Distinct B1 layout/module, native class/package, serialized plugin type and
canonical_bf16_low_batch_kv_overlay lowering keyword connect the path. Existing
transaction machinery handles compact BF16 state; no private B4 padding in this
physicalB1 profile. Native lowering checks exact B1 geometry/offsets/modes and
pinned RoPE theta/epsilon, selects B1 plugin; old B4/B8 plugin/metadata/kernel
sources unchanged. CUDA candidates relocate by include-path changes only, not
new arithmetic. CMake existing globs collect new native units on future build.
No active DLL was built/loaded/installed and no remote source/runtime changed.

Actual49CPU tests pass: seven new strict metadata profile tests, twelve new
graph/factory/default-off/unsupported-profile/native-absence/pool/head/batch/
cleanup tests, plus30existing BF16/reference/joined/graph/transaction controls.
AST construction traces preserve63FP32 barriers,48GDN/16full layers, exact B1
offsets/new marker, no dense FP32 cache views, and T4-only replay. Trace buffers
and fake factory metadata are CPU controls, never model/GPU evidence.
Six selected native units compile privately /MT C++17/sm120: helper, plugin,
actual TRT builder and native entry, both CUDA units. Full DLL link/ABI/native
helper/lowering behavior UNRUN. Actual ledgers, source/object hashes and19-file
derivations against83f5ad146 retained. Review restored unrelated mixed line
endings without changing normalized code; initial recipes/build receipt remain.
Final compilation repeats against corrected bytes and the extra B1-only theta/
epsilon gate. No dropped assertion, control or tolerance to obtain a pass.

Original integrated sampler full-model handle still LIVE: disabled14case/
102vector comparison closes and local observer agrees; enabled7partial case
records, no recorded error. Newly wired local branch is ahead of deployed5ac
source/DLL and cannot be deployed or built on that host while worker lives.
Next: terminal full archive/both replays, actual B1commit then CPU/plugin forward
gates, preserve/install source and native build, native helper/ownership checks,
old B4 regression and full B1 ordinary/MTP/text/image/state/prefix/reset quality,
actual4K/protocol checks before timing. B1M4/M32 projections use existing admitted
row ranges; M32 policy/precision and full-model numerics still need measurement.
Original2x/halfTTFT, broad quality, actual4K, tuned refs, INT4,4B,B8/concurrency and
clean repetitions remain open. No benchmark chart or claimed gain is updated.


2026-10-25: Adapted805MMMU HTTP quality collector/replayer prepared, real inference UNRUN.
Added standalone stdlib client and independent raw-file replay for already-ready
native/vLLM/SGLang servers, with no server launch or GPU work. Caller must first
establish exclusive GPU ownership/native-model correctness and externally bind
server/checkpoint/runtime identity. Native and SGLang choice-level actual IDs,
vLLM root prompt/choice output IDs follow installed-study reference adapters.
Only ID-return flags and declared NVFP4/INT4/4B alias may change canonical wire.

Every frozen805request/prompt/JPEG/gold hash and protocol-source hash is checked.
Actual CPU replay checks7245wire adaptations (3engines x3aliases x805requests),
semantic payload unchanged; alternate-model tokenizer compatibility is unproven.
Sequential no-retry HTTP collection preserves gzipped exact wire, raw success/
error reply, IDs/usage/finish metadata, errors, source-before/after hashes, UTC
and file manifests in one explicit attempt directory; existing attempts cannot
be overwritten. No latency/tok/s numbers are collected by this quality client.

Independent replay recomputes all raw parsing, frozen prompt IDs/64image tokens,
context/output/usage counts, wire adaptation, strict letter scoring and complete
805cohort. Completed non-letter answers remain incorrect. Failed/interrupted
cohorts retain raw prefix/failure and any unadmitted score artifact, no verified
complete accuracy. No token inference from text or hidden retry/drop selection.
Provided server identity is explicitly not attested by client. Actual pixel/grid
preprocessing remains unverified even when image-token count agrees; external
runtime pixel/grid evidence is still required. This adapted single-image256px
MC screen is not full900MMMU, broad/high-resolution quality or speed promotion.

Actual17CPU tests pass, including synthetic complete805 raw replay, failure
prefix/no-retry/timeout/unsupported schema and prompt mutation despite refreshed
file hashes. All transport responses are synthetic temporary CPU fixtures;
actualHTTP requests/model responses/GPU executions remain0. Initial14test run
failed because the synthetic fixture omitted its manifest; original raw failure
and bytes kept. Added explicit synthetic manifest to reach unchanged frozen
hash rejection, no assertion/protocol/threshold relaxation. Methods/logs/pins and
preparation ZIP/JSON preserved. No actual quality score is claimed or published.

Original integrated full-model handle remains LIVE: disabled14/102closed and
locally observed; enabled11partial records, prefix checks report passed, no
recorded error. Remote sources/DLL/SDK unchanged. B1 component/run/native/model
gates, quality/preprocessing, actual4K, tuned/INT4/4B/repetitions and2x remain open.


### 2026-10-25: closed integrated model and B1 cache-update correctness

Original supervised integrated-batch-top1-model-lifecycle-attempt1 closed with
exit 0, cleanup complete and source/runtime unchanged. The immutable 391-file
archive independently replays on the 5090 and local CPU with identical proofs.
Disabled and enabled processes each match all 14 cases and 102 raw vectors
against the retained parent. Actual host dispatch counters are 4078 legacy/0
parallel when disabled and 0 legacy/4078 parallel when enabled. Counters include
capture construction and exclude graph replay; they establish routing only.
The first local wrapper failed before verifier execution because a terminal-wrap
checksum transcription had 65 characters. Original method preserved; corrected
64-character digest agrees with archive and machine remote proof. No data,
verifier, source, assertion or tolerance change. Actual successful replay ledger
and correction receipt retained alongside both proofs.

After that closure, the B1 runtime factory positive path ran; launch preflight
reported 109 MiB adapter memory and no source/runtime/tool drift or occupied
outputs. The 18 missing inputs were uploaded once, each acknowledged. Original
bf16-low-batch-commit-native-attempt1 then closed exit 0. Actual remote compiler
passed 17 CPU tests and 63 shape controls. All 20 B1/T4,T32 full-state cases,
three graph profiles with three replays each, 66 rejected calls, readonly inputs
and canaries pass. Independent remote/local replays agree exactly on all 104704
frozen witness bytes and all 205 archived files. The Garnet DLL was not loaded
or changed; only standalone private compiler outputs were created.

These are correctness results. Neither run measures HTTP TTFT, decode speed,
concurrent throughput or serving VRAM. Sampled adapter memory is retained only
as correctness-run provenance. B1 plugin/helper/model qualification, broad
quality, actual 4K, B8, tuned/MTP references, INT4, matched 4B, clean repeated
HTTP comparisons and the original 2x objective remain open.


### 2026-10-25: actual B1 plugin forwarding and native DLL build

The original bf16-low-batch-plugin-native-attempt1 closed exit0 and independently
replays on both machines with identical proofs. Actual TensorRT10.16.1 matches
its pinned DLL/header:32profiles/roundtrips/clones,1632bad wires,992bad descriptors,
672bad formats,96cross-dialect rejects,64legacy roundtrips and zero CPU operator
calls. The44case GPU forwarding stream matches all119040selected FP32 scalars
and complete records from the already qualified B1 kernel, with87host enqueues,
three graph replays and40cache-prefix controls. The104negative GPU calls remain
direct-kernel tests, not wrapper negative-enqueue coverage. No model/speed claim.

Native install preflight correctly rejected an older shared CPU trace fixture;
downloaded bytes match the qualified parent's exact hash. The branch fixture's
existing default-false B1 option is now included explicitly in the source bundle.
Initial factory/preparation/verifier/failed receipt and original fixture retained;
no product code, assertion or tolerance change to bypass a failure. Revised
preflight passes.19product files plus1test fixture installed, all9original files
archived, old5ac DLL retained, existing build tree and SDK reused.

Actual bf16-low-batch-native-build-attempt1 closes exit0:49remote CPU controls
pass, both new CUDA units/helper/plugin/builder/entry link successfully into
8ae63a41b532f6e2d67cf01ca60b7684b86451105711dbb3573f2e325f63521b.
Both independent build replays agree across199archived files. All non-Garnet
runtime dependencies and build SDK remain unchanged; native DLL was not loaded
during the build. Next prepared model collection first checks retained B4
text/image row0, then true B1 four prompts/two repeats and ordinary/MTP/prefix
0/1/2/4/reset/callback behavior. It records B1-vs-B4 differences explicitly.
Prepared work is not an execution/pass. Current8ae full-model, broad quality,
actual4K, HTTP, serving memory, tuned references and2x target remain unqualified.


### 2026-10-25: true B1 HTTP worker prepared locally during model collection

The original bf16-low-batch-model-consistency-attempt1 handle is running. Read-only
observation sees B4 decoder metadata and the complete empty-state witness, no
completed case or recorded error yet. Remote sources/runtime remain unchanged.
Native hash-library warnings match retained runtime behavior; SHA256 works.

Local-only native_bf16_low_batch_stream_worker.py prepares ordinary or verified
MTP3 with physicalB1, T32/T4, cap4096 and128MiB draft workspace. It requires the
actual closed/replayed current-DLL model proof before initializing. Existing
single-slot HTTP bridge can queue concurrent clients; no GPU batching is implied.
Its request validator is AST-identical to the established bridge policy, copied
locally so the embedded model runtime need not import an HTTP server. Seven
actual CPU controls pass: text/image/invalid request admission, single-row and
ordinal guards, and exact one-event handling of prefill-first/MTP-duplicate-first
notifications. Actual output retained as an exec-tool transcript, not presented
as a run_logged command. All805frozen adapted-MMMU canonical request bodies also
pass extracted CPU policy after full frozen corpus hash validation, with a real
separate command ledger. No native worker import/upload, HTTP requests, model
responses, image pixel/grid qualification, quality score or benchmark occurred.
Actual B1 model completion/replay, HTTP event/ID correctness and quality/4K gates
still precede admitted comparisons. Original2x goal remains unmet.


### 2026-10-25: model attempt1 typed-comparator failure and guarded correction

Original bf16-low-batch-model-consistency-attempt1 is terminal exit1, cleanup
complete, source/runtime unchanged. Actual first B4 text ordinary/MTP/callback
checks passed internally, then the harness incorrectly called same_logical_row
with two packed witnesses. That predicate deliberately requires its second
input to be the independently BF16-rounded FP32 reference. It rejected the real
packed metadata before comparing values. Original221-file archive and methods
are immutable; remote/local failure replays agree. B1 never executed in attempt1.

Independent CPU replay of the actual saved first case checks the complete
selected canonical/logical row, all128component hashes and both full raw hidden/
normalized rows against the retained5ac parent: all exact, including token IDs.
This is one archived B4 case, not a completed model qualification. Original
failure stays failed. New bf16_packed_state_gate validates both real packed
representations, full128component geometry/hash metadata and selected physical/
logical/component/slot hashes. Original packed-versus-reference predicate stays
unchanged; no fake reference flag, product math or tolerance change. Seven new
and six unchanged CPU controls pass (13actual tests; raw tool output retained).

Four fixed model methods are literal derivations from failed archived sources;
new factory binds the original failure proof and actual partial-data reanalysis.
Fresh attempt2 retains all4B4 and8B1 cases and adds18remote CPU controls before
GPU execution. Same8ae DLL, old libraries and shared scratch/cache retained;
no rebuild or runtime replacement. Read-only preflight reports no source/runtime
drift, competing study process or target conflict. Prepared attempt2 is not a
pass. Broad quality, actual4K, HTTP metrics and original2x objective remain open.


### 2026-10-25: fresh model attempt2 active; next actual HTTP stage prepared

Original supervised attempt2 is live. All18CPU controls passed on the5090;
read-only report now records the first fresh B4 text case with its retained
complete canonical/logical row0 and raw hidden/normalized vectors exact, no
recorded error. This is partial, not a closed model pass; remote source/runtime
remain frozen until the original controller is terminal.

Local HTTP worker now binds the corrected attempt2 proof/verifier instead of
the permanently failed first attempt. Seven CPU policy/notification tests and
all805frozen request-admission checks pass on these actual new bytes, with real
two-command ledger. Prepared launcher/server probe remain local and unrun. The
probe first requires a real positive current-model proof, then actual queued
nonstream/stream token-ID controls, complete805screen and two exact3968-input/
128-output-reserve boundary controls. MTP also requires actual completed ordinary
controls and exact ordinary/MTP IDs across all805screens and both long-context
controls. Optional short41/104-token exploratory pilot remains explicitly
different from a full4Kbenchmark or2x qualification. Corpus paths and both frozen
boundary wire hashes/3968prompt counts were checked on CPU; no runtime4Krequest
was sent. A guarded outer ownership/archive/scratch controller and actual corpus
installation still precede execution. No new HTTP/model-quality/benchmark score
or2x admission is claimed.


### 2026-10-25: NFC HTTP methods prepared; B4 model comparison passes

Original supervisor48577 confirms tokenizer-nfc-model-consistency-attempt1
running. Read-only workflow observation records all4B4cases complete, exit0:
18reported states/36raw vectors and full canonical metadata, IDs and callbacks
exactly match retained8ae. B1 remains active; observation sees3completed cases.
This is partial correctness evidence, not final model/quality/speed admission.

Seven fresh HTTP methods are literal-derived from the immutable failed HTTP1
archive, binding the new model proof/verifier and saved tokenizer DLL alias.
An initial module-stem substitution was caught before testing and added to the
derivation recipe; final bytes passed29actual CPU tests in8.642s. Original
805screen,12protocol,2boundary,124full4K and124short requests per mode, strict
ID/usage matching, ordinary/MTP pairing and failure preservation remain intact.
The factory replays all seven derivations and binds the original failed proof.
Actual negative run returns2 on missing new model proof and creates neither a
preparation nor upload plan. Methods, receipts and real command ledgers retained.
No deployed method/runtime/cache mutation or HTTP execution occurred.

Supplemental all-vector auditor is prepared to compare every persisted .f32
file, including transient prefix snapshots absent from report state objects,
against retained8ae after actual model closure and both independent replays.
It has not yet run on a completed new archive. B4 report comparisons cover all
reported vectors; do not broaden that claim to all transient files before audit.
Original2x, full4K/quality, strongest reference/tuned/MTP, INT4/4B and repetition
requirements remain open. Single-slot HTTP queues do not imply GPU batching.


### 2026-10-25: reference4K repeated-image cache boundary addressed

Reference4K nonstream token controls reuse every later timed image. The prepared
client now performs the established idle-only vLLM dual reset after all124ID
controls and before NVML/timing, records both endpoint acknowledgements/raw
counters, and rejects timing completion unless processor queries increase with
zero new hits and unchanged server process. Timed corpus JPEGs are unique;
encoder reset acknowledgement alone is not a universal cache-residency proof.
Existing API schemas/cache-negative controls pass10actual CPU tests (0.631s).
No vLLM server/model run occurred; config source was read without GPU imports.

New SGLang ordinary/MTP3 fixed4K launchers are exact current launcher derivatives
changing only dedicated log paths and explicit --mm-preprocess-cache-size-mb0.
Actual installed mm.py from5027ba7f... archive defines0as disabling this CPU
cache. Prefix-MM/global embedding caches remain default off and actual
server_info must verify these values before timing. Both Bash syntax checks and
reference Python compilation pass. Source recipe/actual CPU ledger retained.
These methods are local only while original NFC model runs; no deployment,
runtime compatibility,805quality/4K metric or2x admission follows.


### 2026-10-25: rebuilt tokenizer model closes and independently replays

Original tokenizer-nfc-model-consistency-attempt1 exited0, preserved all393
source/input/method hashes and six runtime hashes, and finished owned cleanup.
Immutable archive90098fe581655c8c7eb3bdf3e43d6baed2b899b41388be601f2fe840b6a7b65c
contains583files/9134964bytes. Actual remote/local independent replays match;
both use cb9dcac1... verifier identical to archived method.24CPU/4B4/8B1cases,
ordinary/MTP exact IDs/callbacks, full reported canonical states/raw vectors,
repeats/resets/prefix0/1/2/4 and dialect/pending negative guards pass.

Supplemental independent all-file auditor also passes: all44B4 and124B1 .f32
files (168total), including transient prefix snapshots absent from report state
objects, match retained8ae file sets and every raw byte. B1 versus oldB4 text/
image token differences remain documented, exactly as retained8ae behavior;
do not claim external/batch precision equivalence. Correctness monitor maxima
B4=27035906048B,B1=23010648064B are not serving memory measurements.

Only after actual closure and identical proof pair did fresh HTTP factory pass,
materializing27method preparation49d72442... and29target upload plan. Earlier
negative absent-proof guard remains preserved. Read-only target/source/runtime/
parent/cache preflight and uploading only absent new files precede one HTTP job.
Actual805score/4K/HTTP/timings and original2x remain unmeasured at this entry.


### 2026-10-25: fresh proof-bound native HTTP study starts

Read-only tokenizer-nfc-http-study-attempt1 preflight passed: no392pinned-file or
six-runtime drift, no target conflicts/occupied result, GPU125MiB used before
launch and750900301824B free in owned C scratch.20of29targets matched and were
skipped, including the existing805corpus. Nine absent new methods/recipe/prep
files received individual acknowledged uploads. No product source/DLL changed.
Actual preflight/input/upload plan/method and explicit tool observation retained.

Only original supervisor48577 started tokenizer-nfc-http-study-attempt1 and
confirms it running with CPU-test progress. No completed HTTP/quality/4K or new
timing result is claimed. One controller owns29CPU gates,12protocol/805screen/
2boundary/124full4K/124short requests for ordinary then MTP with actual paired
IDs and immutable failure archive/owned cleanup. Only that original handle will
be polled while timing may run; no auxiliary remote observers/uploads/jobs.
Original2x and matched/tuned/MTP references, INT4/4B, repetitions remain open.


### 2026-10-25: reference4K owner/driver and strict MTP controls prepared

Native tokenizer-nfc-http-study-attempt1 original48577 remains running. Only its
original handle is observed remotely. No deployed method/source/runtime changes,
reference server, extra GPU job or remote probe occurs while it lives.

Prepared benchmark reference client now requires explicit ordinary/MTP mode.
MTP compares all124actual nonstream prompt/completion IDs and text to completed
same-engine ordinary controls, with identical request bytes, before any timing.
Changed27th output token with identical decoded text is rejected. Fifteen actual
CPU schema/cache/short-MTP/new4Kmutation tests pass in1.819s; no GPU imports.

Prepared probe_reference_fixed4k requires closed successful native HTTP proof,
its actual archive digest/cleanup and separately verified outer server metadata,
PID/source/launcher/cache policy before requests. Four VLM/sixteen short greedy
controls and805raw screening precede124full-ID controls/124frozen4Kstreams.
MTP requires same-engine ordinary full captures, all805completion-ID equality
and all124control-ID/text equality. Two actual local absent-proof CLI controls
return2with no output/network requests. These are explicitly CPU fixtures, not
remote state or native HTTP completion observations. Probe syntax passes.

Installed archived vLLM router641fbd7b... registers reset/server_info routes only
when VLLM_SERVER_DEV_MODE is enabled. New ordinary fixed4K launcher sets it on
literal loopback, retains existing graph/cache/precision/scheduling flags, binds
port before load and requires fresh controller-owned LinuxPID/log paths. Both
SGLang fixed4K launchers gain the same PID/log ownership guards; CPU preprocess
cache0and default prefix-MM/global-MM off remain. All three Bash syntax checks
pass. Recipe source bytes independently match recorded Git parent; methods and
actual CPU ledgers/absent-proof results retained. No launcher deployed/executed.

Next reference execution must wait for native HTTP terminal/replay, confirm idle
GPU and unchanged checkpoint/installed-source pins, launch exactly one owned
server and verify real server_info, model alias/context/cache/SSM/graph settings,
PID and source after startup. Only then declare metadata guards true and run the
probe. Stop verified PID and observe original server exit, archive/replay raw
results before ratios. SGLang ordinary/MTP3 and vLLM ordinary are candidates;
retained vLLM MTP divergence remains failed, fastest feasible tuning and actual
4K/805/INT4/4B/repetitions remain open. No benchmark/quality/2x claim follows.


### 2026-10-25: existing B8 route identified for real concurrency optimization

Read current Python BF16 profile/layout and C++ layout parser: they already
admit B4/B8, separate from the B1-only route. B8 requires actual whole-model
build/memory/numerical/ordinary-MTP/prefix/stream admission, not a blanket guard
relaxation. Existing128MiB MTP option has no batch-specific restriction in these
constructors; actual B8 tactic/build success remains unknown.256MiB option stays
explicitly B4/T4/cap4096. Preserve an actual128MiB failure before considering a
separate bounded B8 option; defaults/known profiles remain intact.

Two named state buffers alone add2344615936B fromB4toB8. This is only canonical
BF16 carrier plus one linear replay buffer arithmetic, excluding target/output,
activation, vision/MTP, graph/workspace and allocator overhead. It is neither
measured VRAM nor a fit/speed prediction. Source hashes/arithmetic/method retained.
After current native HTTP and matched baselines, attempt isolated existing B8
memory/correctness admission and actual HTTP batch coalescing, then profile B1/
B8 dominant GPU work. Current single-slot B1 queued concurrency cannot be labeled
GPU batching. No B8 runtime or2x measurement has occurred.


### 2026-10-25: native natural-EOS trial closes failed; fixed-length regression active

Original tokenizer-nfc-http-study-attempt1 exited1 during ordinary imageC2r0.
Immutable archive2ef0d7c0036ceb8e578ea64673131464d45191e0cebafaa68824583e396406c4
contains3140files/96542443bytes; source/runtime unchanged and cleanup finished.
Actual remote/local failure replays agree with archived d155aaf2... checker;
expected child1preserves the failed status. No MTP process or short pilot ran.

Actual stream has74IDs, all exactly matching final IDs and native worker IDs;
prompt3968, usage4042, stop=<|im_end|>/248046, finish_reason=stop. Frozen request
had max_tokens128but no ignore_eos. This is valid natural stopping, not lost
tokens. It violates the predeclared exact128output workload, so keep the entire
trial failed. Independent raw replay validates all805screen replies and computes
502correct/805=0.6236024844720497, zero unparseable responses. This is the adapted
256pixel screen, not official900MMMU or broad quality promotion. Five completed
partial cohorts replay actual event accounting/arithmetic; they remain excluded
from admitted comparisons/charts. ImageC1sample gap2.6454064s is retained with
sampled memory maxima. No2x result follows.

Initial independent replay completed comparisons but requested an events digest
after its ZIP context closed. Initial method a6d3abfa... and actual failed ledger
retained. Corrected only digest lifetime/report construction; final d4dd7e34...
reruns the unchanged archive and passes. No comparisons/thresholds were removed.

Created separate explicit fixed-length corpus596991dd.../manifest25808fff...:
add ignore_eos:true to all124request bodies, leaving every prompt-ID byte,
message/image/nonce/cohort unchanged. Both actual installed reference protocols
support this field. Natural-stop corpus/trial remain untouched. New named native
worker/bridge preserve default stopping and implement per-request ignored EOS,
restore the normal stop set on the next request and retain/count EOS token events.
No product math/native DLL/SDK change or rebuild.32initial worker/quality/schema
CPU controls pass; final client suite43tests passes in5.756s. Initial client
43test fixture incorrectly assumed an empty final delta; static real-bridge
inspection caught it before deployment. Retained that source/recipe/ledger,
corrected the fixture to actual delta.content and only require choice0/length,
keeping exact128IDs/usage. Final43CPU source/ledger retained separately.

Focused fixed-length-eos-consistency-attempt1 requires actual normal74/forced128/
normal74restore plus forced128SSE on the same old image, ordinary thenMTP. Prefix
74must remain exact (including EOS at73), full128MTP IDs match ordinary, and all
stream/native IDs/usage/length must agree. A real-model control precedes new
performance runs. Passed preflight sees no source/runtime drift, GPU125MiB used,
750910033920B owned scratch free. A146248byte verified bundle installs17absent
methods/corpus/receipts and verifies all30targets, skips13matching files and
reuses the existing805corpus. Original48577 supervises the new correctness job;
43remote CPU tests passed in7.418s. It is running; no new model-policy or timing
qualification yet. Original cache-retirement function AST is unchanged.

Local reference metadata preparation pins74vLLM/197SGLang installed named files
and8checkpoint metadata files from retained archives. Historical actual server
schema plus synthetic cache0/resolved-MTP mutation tests:4CPU passes. Installed
SGLang NEXTN resolves to EAGLE; identity checks use the primary-source resolved
value. Linux helper checks/copies named installed sources and owned PID/argv/
starttime; actual ready-server config must pass before metadata guards become
true. This source/config tooling is unrun remotely. Reference client now requires
the future successful fixed-length HTTP proof and uses its exact new corpus.
Old failed proof cannot be promoted. Full strongest references/tuning/MTP,
INT4/4B, real batching/profile optimization and repetitions remain required.


### 2026-10-25: real EOS regression closes; full fixed-length HTTP starts

Original fixed-length-eos-consistency-attempt1 exited0 and finished owned cleanup.
Immutable archivea18679e271eb5eb65fb2749fcdb8b17bdfb53deac01cfa0f64b4288d646cb136
contains457files/6017911bytes. Both independent actual replays match using
separately versioned checker cbd5ffb2... (Git80836a43d); the checker was not
injected into the original runtime archive.43remote CPU tests pass in7.418s.
All8real requests verify natural74/forced128/natural74restore/forced128SSE,
old74token prefix including EOS at73, complete stream/native IDs/usage and
ordinary/MTP equivalence. Native1a1dDLL/source/runtime stay unchanged. Correctness
only; these request timings are not serving benchmark or2x evidence.

Three fresh full-workflow methods derive exactly from immutable failed2ef0d7c0...
sources. New factory requires identical actual native-model/EOS proof pairs and
same checked EOS-policy sources.43CPU gates,805natural-stop adapted screening,
12protocol/2explicit fixed-length boundary controls, complete124request
3968+128actual-token cohorts and separate124request short pilot run ordinary
thenMTP. Old native-model protocol oracle remains unchanged; verifier additionally
checks actual per-request ignore_eos flags and every frozen4Krequest body, full
IDs/usage and MTP pairing. Original natural-stop trial remains failed.

Actual factory materializes32method preparation e0a433d4... and34target plan.
Passed idle/source/runtime preflight; a33619byte verified bundle installs only
six absent workflow/receipt files and verifies all34targets, skipping28matching
files including both corpora. No product/DLL/SDK rebuild or model/data recopy.
Original48577started fixed-length-http-study-attempt1 and confirms it running;
43actual remote CPU tests pass in6.807s. Only its original handle will be polled
while timing is possible. No current full fixed-length metric or2x result yet.

Future reference runs must use the exact same explicit-ignore_eos corpus and
match the whole-probe plus cohort NVML instrumentation used here; retain actual
sample gaps. Do not discard a faster reference merely by selecting an easier
ordinary configuration: preserve MTP numerical divergences/quality evidence,
assess strongest feasible settings and report limitations before any2x claim.
Actual references/tuning, INT4/4B, real batching/kernel work and clean repetitions
remain part of the original objective, not replaced by this correctness fix.


### 2026-10-25: isolated attention score tiling candidate compiles; GPU unrun

Read current B1 attention source: scores launches512position tiles per query at
capacity4096, eight positions per256thread block. Inactive T4rows still enter
the launch and return; this is a source work-assignment observation, not a
measured bottleneck or causal explanation for whole-model38tok/s.

Prepared separate unwired bf16_low_batch_scores_tiled CUDA/header under study
tests. Each warp processes16positions, score gridX32instead of512, query vector
reused in eight registers. Every score intends identical eight ordered fmaf
operations and warp reduction. Exact reverse transformation proves production
source, normalization/context, workspace/profile/alias guards unchanged. No new
shape/default admission or Garnet target integration. Sixteenfold block-count
reduction is not a measured speedup. Native benchmark DLL/source remain frozen.

Prepared44case GPU probe inherits all original case specs/FP64 witnesses and
104negative calls, additionally rejects each malformed call through candidate,
compares every full output and score/softmax workspace byte versus baseline,
captures/replays candidate graph and checks full cached-prefix output. GPU probe
is not executed. Actual private CPU-only CUDA13.2.78/VS14.44/SM120 compilation
and link pass; all input/tool/header/fixture hashes unchanged after build.
Compiler emits39baseline versus40candidate score registers, zero stack/spills;
normalization/context resources unchanged. These are compile facts, not model
parity/speed. Outputs stay in a fixed private work folder, CUDA_VISIBLE_DEVICES=-1;
no local GPU/runtime/SDK/existing-build or remote deployment mutation occurred.

Current full fixed-length HTTP original48577 remains live; only its handle is
polled remotely. First ordinary4Ktext cohort console reports38.4995decode tok/s,
4948.35ms TTFT and15.3692aggregate tok/s, pending full archive/raw replay. Sample
gap1.6437291s retained. Do not admit partial measurements or claim2x. After
baseline closure, strongest feasible matched references and source-based
profiling, qualify/timestamp this candidate and actual B8batching; keep full
INT4/4B/quality/repetition scope. No speed benefit has been demonstrated here.


### 2026-10-25: owned SGLang fixed4K controller prepared; runtime unrun

Added run_sglang_fixed4k.py and prepare_sglang_fixed4k_run.py to carry the
matched reference through real server startup, ready configuration/PID/source
checks, fourVLM/16greedy/805screen and124IDcontrols/124frozen4Kstreams, verified
owned-parent SIGTERM, actual terminal/idle checks and immutable raw archive.
Reuse the existing WSL environment/checkpoint/cache and fixed study folders.
Require the completed full native proof pair before deriving deployment pins;
MTP additionally requires the closed independently replayed ordinary attempt.
Pin eight checkpoint metadata files before/after; no weight/model copies.

Actual26CPU identity/owner/fixed-output/cache/MTP controls pass in2.112s;
two real missing-proof CLI attempts return2 without preparation/run output or
server/GPU startup. Retain both executed ledgers and exact methods. These do
not qualify actual server startup, model correctness, timing or2x. Controller
preserves readiness failures, observes the original owned wrapper, and never
interprets a timeout as process termination or permission to restart. Whole
client monitor includes quality; only cohort phase samples are serving memory.
Reference archive replay still needs implementation/execution against real data.

Current native fixed-length HTTP original48577 remains live; ordinary4Kcohorts
have emitted console records and separate short pilot is ongoing. No remote
source replacement or extra observers/transfer during possible timing. Await
terminal/archive/raw replay; then execute strongest feasible SGLang/vLLM
profiles and GPU-qualify the attention candidate/real B8batching. Full matched
INT4/4B/repetitions remain open; no new benchmark/2x admission in this change.


### 2026-10-25: opt-in B8 fixed-length dispatcher CPU path ready; GPU unrun

Existing model/layout/StackRows/head sources admit8rows, but the current study
batch bridge only admits2/4. Retained its original bytes and changed only the
capacity lookup to a class declaration, keeping default2/4 and the ready-message
capacity check. New native_fixed_length_batch_http_bridge reuses that bounded
dispatcher with explicit4/8 and the strict fixed-length EOS HTTP handler.

Actual final14CPU tests pass in1.508s, including unchanged8legacy bridge tests,
8client child-process token routing, deterministic8ready-ticket group with no
ninth wait, actual HTTP IDs/count/health and EOS boolean/malformed400. Earlier
13test pass retained too. Fixtures are explicitly CPU-only; none qualify real
GPU batching or speed. No B8model worker/deployment/runtime default added.
Next B8gate is actual full-model/vision/4Kmemory-fit, row-isolation/reset/state
and token/quality checks, then real HTTP coalescing and measured profiles.

Native fixed-length job original48577 still running. Ordinary4Kand short pilot
emitted all8cohorts; last short-imageC8 sampler maxgap23.660419899999397s retained
as unexplained pending raw replay, no causal/performance promotion. Only that
original supervisor is polled remotely. Full references/INT4/4B/repetitions and
2x remain unmet; do not substitute CPU routing tests for GPU qualification.


### 2026-10-25: SGLang independent fixed4K archive replay prepared; real run pending

Added verify_sglang_fixed4k.py before freezing reference deployment. It checks
all raw archive/source pins, completed native parent, actual ready config,
Linux PID/argv/starttime/source copies and stop receipt, closed lifecycle,
4VLM/16greedy/805screen, all124nonstreamIDcontrols and124frozen request streams,
raw cohort arithmetic and phase-memory samples/gaps. MTP also replays ordinary
pairing. Reference streaming IDs remain unobserved; only counts/text and
separate actual nonstreaming IDs are claimed. Whole client/quality peak is
labelled separately from cohort serving peaks; no process-memory attribution.

Actual final18CPU controls pass in0.017s. Five new tests are synthetic in-memory
mutations, checking timestamp/request/text/cache/stop-length tampering; none
are measurements. Initial18test ledger also retained before adding whole-archive
greedy/VLM/control metadata checks. Full archive verifier execution remains
pending real SGLang data; passing unit controls does not qualify that run.

Native original48577 still reports running after ordinary4Kand short pilot
output. No other remote commands, transfers, source changes or observers were
issued during possible timing. Next actual milestone remains native terminal
and raw replay, then real matched references and GPU optimization qualification.
The2x target, stronger-reference tuning, INT4/4B and repetitions remain unmet.


### 2026-10-25: B8 whole-model admission probe prepared; GPU unrun

Prepared probe_bf16_batch8_admission.py using the current existing B8 BF16-cache
T32/T4 constructor and unchanged256MiB target-program builder limit. No MTP
workspace/profile changes. It will build model+vision, exercise8short mixed
rows and8actual3968+128 text/image rows from the frozen corpus, each in original,
rotated and reset-repeat order. Nine actual GPU cases remain unrun. Complete
state-component/physical/logical hashes and raw hidden/normalized vectors must
match after inverse row permutation. Full snapshots include unused cache slots;
only hashes, not multi-GB raw caches, are retained. All raw vectors are saved.

New pure B8 gate selects complete rows through the unchanged B1/B4 representation
validator and validates every8row component before comparing, so malformed later
metadata cannot hide behind an earlier mismatch. Actual20CPU tests pass in0.105s;
literal constructor/current pure Python profile/layout and frozen input hashes
check against the actual archived checkpoint config, and Python syntax compiles
without importing/executing the model. These are no B8GPU/memory-fit/quality/speed
proof. Buffer byte figures in the offline receipt are logical allocations only.

Native HTTP original48577 now emits MTP4Kcohorts, still live; no remote mutation,
extra observations or transfers during possible timing. After its actual terminal
and cleanup, isolated correctness work can use otherwise idle GPU during native
archive transfer, with no official timing overlap. The full reference/matched
2x/INT4/4B/repetition scope remains active and unmet.


### 2026-10-25: compiled score-tiling GPU correctness launch prepared; unrun

Prepared isolated run/prepare/verify_bf16_scores_tiled methods bound to the actual
495104byte privately compiled SM120 executable, unchanged source/header/fixture
hashes and actual NFC model proof pair.24upload targets are prepared; existing
matching files should be reused. No executable is added to the Garnet runtime.
Actual10CPU profile/FP64-oracle controls pass in0.081s; preparation executed and
retained. Original HTTP supervisor must first be observed terminal; runner also
checks its exact completed archive SHA/workflow/cleanup and idle GPU before any
probe. This prerequisite is no permission to infer termination from silence.

Standalone probe intends44realGPU cases: full forward output/workspace bytes,
3candidate graph-output replays,40full cached-prefix output checks, readonly/
canaries and104malformed input cases rejected by both baseline and candidate
(208actual reject calls).119040selected scalars require independent FP64 replay.
Closed raw archives preserve failures and sampled correctness memory. No model
DLL loads/replacements, timing, B8, MTP, broad quality or2x qualification occur
in this isolated experiment. Numerical correctness alone will not prove speed.

Current full native HTTP remains live with MTP4Ktext/image cohort observations.
Matched reference speed/tuning/quality, target2x, INT4/4B and repetitions stay
open. No new graph/kernel/model performance number is admitted by this change.


### 2026-10-25: full native HTTP closed/replayed; tiled-score GPU math passes

Original48577 full native job terminal0. Archive6fbf308714e90bf68046c8080258e6ba9981931f18f902ce1408cf735f5b7320,
132951412bytes/6860files. Both805natural-stop screens score502/805 with exact
ordinary/MTP IDs; all12protocol/2boundary/124full4K/124short requests per mode
replay.16measured4Kcohorts and16short cohorts are native-only single trial.
No source/runtime changes, and owned cache cleanup finished. Native dashboard
and4metric figures rendered only from independently replayed measurements.

Original memory checker failed because4final phase samples arrived after the
saved summaries. Actual raw diagnosis checks all32cohorts: every saved snapshot
exactly equals its raw phase prefix and brackets all measured requests; all4
extra samples are after completion. Original checker/failure log/archive stay
unchanged. Separate verify_fixed_length_http_snapshot plus strict prefix/end
bracketing helper retains full-phase and in-request-window statistics. All
other gates reverse-derive byte-exact to original9ff06c4c...checker. Remote and
local snapshot proofs are identical.5CPU memory-mutation controls pass; current
SGLang owner31CPU controls pass in2.134s and reuses unchanged sampler overhead.

Then actual RTX5090 standalone tiled-score probe terminal0: archiveb790522e63d93c4c9b0eb7e1a8ee58f608a464692ecda0d67c2f46818846b0ff,
683942bytes/40files.44forward output/workspace comparisons,3candidate graph-output
replays,40prefix-output checks and104malformed cases through both implementations
pass, readonly/canaries intact. Both independent119040scalar FP64 replays pass
unchanged atol3e-5/rtol2e-5; maximum absolute error about1.868e-7. Exact proof JSON
equality initially failed on62derived CPU error floats; all other fields/raw
archive match and both pass. Retain every difference and actual Python/numpy/
BLAS/CPU metadata. No rounding of reports or tolerance relaxation. This proves
isolated math, not a whole-model integration or speed benefit.

SGLang33target source/native-proof/runtime preflight passed;17absent files installed
from64490byte bundle09b238574aea762aeb5473fc33f80d9bab9dc0dbc9a350594152f345fe495d59,
16matching targets skipped. No checkpoint/corpus/runtime duplication. Next run
ordinary fixed4K then strongest feasible MTP/vLLM profiles;2x/halfTTFT, B8GPUfit,
tiled whole-model speed, INT4/4B and repetitions remain open. Only the owned
reference supervisor may be polled while official timing is possible.


### 2026-10-25: SGLang actual two-slot context reserve; preserve failed attempt

Ordinary attempt1 terminal1, owned server stopped/idle, source/runtime unchanged.
Archive551e50256cb6efc417d06c454e893fd9a29ff5da84191a9aca45a18b48674a74,
99941597bytes/3158files. Independent base failure proofs are identical. Separate
partial-control checker replays actual ready-server/config/source/PID/stop,
four image smoke/16greedy controls, all805 raw adapted screen responses and the
first frozen4K response. Both partial replay proofs are identical:487/805screen,
not broad/official quality; actual3968promptIDs+126completionIDs/4094usage,
requested128/ignore_eos true, finish length. Strict128 gate still rejects this.
No timed streams; no SGLang latency/throughput admitted.

Primary installed source archive7b58470f0436466b1488ccb359416de5b6c581f9158457d7a9bc71e4ed37e92f
contains19 named files. tp_worker.get_worker_info bounds max_req_len to
min(context_len-1,kv_capacity-1); scheduler.init_req_max_new_tokens then bounds
new tokens to max_req_len-input_len-1. Configured4096 therefore permits126after
3968inputs. Separately named launch/metadata/client/controller/preparer/replayer
use explicit configured-engine4098/pool32784 with unchanged actual3968+128/4096
workload; engine settings and workload capacity are separately labelled. No
installed SGLang code, precision, cache/graph controls or frozen requests changed.
The new attempt must pass early text+image full-ID boundaries before805screen.
Original attempt1 methods/preparation/archive remain intact.

Actual37CPU controls pass1.761s; six new reserve/metadata/frozen-ID/126negative
controls. Eleven Python files parse; controller differs only by named candidate
paths after CRLF normalization. Initial byte-exact derivation assertion failed
only on generated CRLF-vs-LF; normalized text is exact. Offline QA caught CRLF
in proposed bash scripts before preflight/install; corrected to LF and retained
unused original preparation/plan and explicit correction receipt. No server
ran those CRLF scripts. Remote preflight passes existing433source/pin/runtime
checks and64targets:32absent/32matching, no conflict, about591.7GiB free. Only
absent methods/primary-source evidence are bundled296277bytes, SHA
861935c9c28ba68ee3ac80c023295956b61f55d820fcac22559f3a006c6abd80.

New reference attempt, strongest feasible MTP/tuning/vLLM, broad quality,
repetitions, actual B8/model-level tiled optimization, INT4/4B and2x/halfTTFT
remain open. This receipt is preparation history, not a new performance result.


### 2026-10-25: B8 outer owner/raw replayer prepared while SGLang runs

No remote activity beyond original48577 SGLang attempt2 polls. Actual remote37
CPU reference controls passed1.800s; readiness/boundary/screen/timing not yet
observed at this offline preparation. Existing B8 probe remains unchanged.
New outer owner requires a named reference archive with actual closed workflow,
terminal server code and source/runtime unchanged, then idle GPU/process guard.
It reuses existing bounded owned SSD engine cache, logs every new retired engine
file hash/size and preserves older/unknown files. No new model/runtime/SDK build.

B8 independent raw replayer binds unchanged frozen inputs/images and all9
eight-row short/full3968+128 text/image cases, complete logical/component hashes,
finite hidden/normalized bytes, inverse row mapping, reset-repeat and sampled
construction/correctness memory. This is internal consistency only: it does not
prove external B1/B4 model-math equivalence or broad quality/serving/speed.
Actual28CPU controls pass3.099s, including4new synthetic mutation replay cases;
initial24ownership/state controls passed0.222s. Missing/active/drifted reference,
unowned/unknown cache files, last-row state/input/generated IDs, changed finite
vectors with consistent hashes and nonfinite vectors remain rejected.

Five Python sources parse; new launcher reverse-derives exactly to existing
model launcher with only probe path/argument replacement. Preparation freezes
444files and20upload targets against both actual closed native proof pairs.
Remote deployment, GPUfit/model admission and independent raw replay are unrun;
no GPU or timing evidence claimed. Next deploy only after official reference
supervisor is authoritatively terminal.2x/halfTTFT and full scope remain open.


### 2026-10-25: actual worker-time breakdown; prefill16 CPU candidate; SGLang2 partial failure

All128 measured native full4K streams from immutable6fbf3087...archive reconcile
by request ID, actual3968prompt/128completion IDs and raw worker token sequence.
Only durations within each clock are subtracted: never equate absolute native
and HTTP clock epochs. C1 median client-wall minus worker-wall is about22-37ms;
worker wall is7.4-8.2s. Worker time outside first-to-last emitted token is about
4.94s ordinary and5.96-5.98s MTP, including pre-first processing and finalization,
not an isolated GPU/prefill measurement. C8 median outside-worker time is about
25.8-29.0s, including queuing/transport/handling. Additive mean-wall figures are
from existing evidence only; initial legend overlap was caught visually and
corrected. Neither kernel cost nor a new benchmark is inferred.

Actual retained MTP full4K counter shows993draft-prefill calls. New explicit
CUDAMTPPrefill16StudySession keeps the existing B1/T32target/T4decode, depth3,
full-head/prefix-replay/128MiB policies and shared explicit draft KV/last hidden,
but uses a second T16draft engine only during target-prefill hooks. A finally
restores every narrow decode binding after success/failure/reentrance. Existing
factories, validators, target/speculative source and native DLL remain unchanged.
Extra engine duplicates resident draft weights; additional buffers/engine memory
are included in session reporting and require actual full4KVRAM measurement.
Proposed249prefill calls are derived from loop bounds, not a GPU measurement or
speedup. Actual11CPU policy/lifecycle/binding/extra-resource controls pass0.013s.
The unchanged eight-case real B1 target ordinary/MTP/prefix/reset/full-state/raw
vector probe wrapper is prepared but unrun. Actual short/4Kmodel,805quality,
VRAMfit and matched repeated performance gates remain mandatory.

SGLang ordinary attempt2 terminal1, source/runtime unchanged and owned cleanup
finished. Archivee17f3af00fd9502a8708115f194dc4464706320baa399426855b8bf1533c0b9d,
100968830bytes/3498files. Two early full4K boundaries,4VLM/16greedy128ID controls,
805adapted screen487/805 and124serial3968+128ID controls actually completed.
One textC1 timed cohort completed: raw arithmetic310.8568ms TTFT,64.0726decode
tok/s,55.7582total outputtok/s, sampled adapter peak29281132544bytes. The first
C2warmup then failed serial-control text equality: both streams still return
3968+128/4096usage/length; text first differs at character290and225. Twelve saved
streams replay10serial-text-exact/twochanged. Cause and harmlessness unproven.
Original failed manifest/archive/helper remain intact. Separate partial checker
replays all startup/source/PID/stop/early boundaries/VLM/greedy/805screen/124IDs,
all12SSE bodies/counts/times/text and completed C1 arithmetic/memory without
changing the failed manifest status or inventing seven missing cohorts. Both
independent base-failure and partial raw proof pairs are identical. This is
partial failed-profile evidence, not a complete/strongest/tuned reference.

B8preflight initially refused a copied65-character archive identifier before
GPU execution. Preserve method/log; separately named v2 derives the real64hex
SHA from saved remote proof. Corrected preflight passes425existing source files,
20targets and runtime with no drift/conflict;11absent methods installed from
39408byte bundle47af7bb265b9664457368357a9fedc34e00c82bf65a3dfcd5b9832e581b6082e,
nine matching files skipped. No runtime/model/SDK duplication. Original48577
acknowledged bf16-batch8-model-admission-attempt1 start only after actual SGLang
terminal cleanup/archive observation. Untimed construction/9internal8row cases,
full4K text/image/state/raw vectors and correctness VRAM are still unobserved.

Next: finish/replay B8 and classify source/config/VRAM/row findings before any
real batched serving/performance. Characterize reference output variation with
serial repeats, concurrent nonstream IDs, same-byte streams and mixed-color
isolation/reordered rows atC1/2/4/8. Preserve all variations; do not infer benign
batch arithmetic or a bug from text alone, or replace fastest ordinary reference
with a weaker deterministic setting. Any new full-throughput protocol must keep
strict actual token budgets/quality controls and explicitly retain batch-related
output variation; keep strict failed experiments immutable. Complete strongest
SGLang MTP/tuning and vLLM, repetitions, independent B1/B4-vs-B8 math/805quality,
MTP16short/4Kmodel and memory gates, and current B1full4K kernel profiling before
selecting whole-model optimizations.2x/halfTTFT, INT4/4B and full scope remain open.


### 2026-10-25: prefill16 model owner; reference output/isolation diagnosis prepared

Original48577 B8 model admission remains authoritatively running. Actual saved
report observations progress through all3short-mixed and all3full3968+128text
rows/permutations/repeats, then first full4Kimage case, each internal inverse
mapping exact. Decoder construction/vision/empty state had completed. These are
live partial observations, not a closed attempt, independent replay, external
B1/B4 equivalence, broad quality, serving or performance proof. Known embedded
hashlib warnings did not establish terminal failure and no restart was issued.
Correctness-only read of existing NVML sample showed28355711808adapter bytes used
with5833805824free at that moment; not an instantaneous/serving peak or final
memory-fit claim. A separate read-only instantaneous utilization query returned
0percent/22.20W between test work; not a power/performance benchmark or causal
kernel diagnosis. Existing runtime, SDK, model source and probe are unchanged.

Prepared explicit prefill16 whole short8B1 ordinary/MTP/text/image/color/prefix/
reset owner/preparer/independent raw replayer. Both the actual closed B8 archive
and closed SGLang reference archive plus idle GPU/process guard are required;
no automatic restart after observation timeout. Reuse bounded owned SSD cache,
archive every failure/source/runtime/raw target-state/vector/counter/sample and
compare to actual retained1a1d B1 snapshots. Extra T16 draft engine/buffers are
counted, primary/decode/verifier source unchanged. Actual41CPU controls pass
0.244s. Initial package import failed before GPU; preserved initial owner/log,
then package/script sibling import corrected. A following doc-only edit fixes
inherited no-MTP wording. Reversing both changes exactly restores original owner
derivation bytes. First unused preparation/plan also retained. Current preparation
pins575files/21uploads; no remote install, GPU model or new speed result.

Prepared separate ordinary SGLang output/isolation diagnosis and owned server
lifecycle/preparer/raw archive replayer.16actual frozen4K inputs and8instances
of four retained color/spatial cases load/validate from immutable prior evidence.
The future288requests comprise192actual nonstream-ID controls and96streams,
with135observed-overlap waves: serial before/after, C1/2/4/8, rotated inputs,
64full3968+128streams and32short color streams.48nonstream plus24stream mixed
color concurrent controls must remain semantically correct. Every actual output
ID/text variation is retained, with no inferred cause or harmlessness. Short
color replies may arrive all at once; no decode rate is measured. Full4K budgets,
actual IDs/nonstream usage, raw SSE totals/lengths and complete request/row/wave
coverage remain strict. These source changes never alter the old failed strict
benchmark/helper/archive, and do not choose a weaker deterministic reference.

Actual33CPU suite passes10.169s, including a synthetic local HTTP288request/
135wave integration and independent raw replay with wrong counts, wrong colors,
false overlap and missing rows rejected. Earlier actual16/18CPU stages and raw
logs are retained, including the intermediate synthetic256request nonstream-only
color coverage before streamed-color coverage was added. All fixture data are
explicitly synthetic; no SGLang/GPU/benchmark/quality result is fabricated.
Current reference preparation pins509files/12uploads, engine4098/two reserved
slots/pool32784 and actual4096workload capacity, BF16KV/F32SSM/cache/graph settings
unchanged. Model server run and real independent archive replay are unrun.

After B8 closes, replay/retain its full attempt before admission. Next use real
reference diagnosis to distinguish serial repeat variation, concurrent ID/text
variation, stream differences and request isolation, without assuming harmless
roundoff or a bug. Complete stronger tuned/MTP SGLang and vLLM full4K timings;
qualify prefill16short then full4K/805quality/memory and matched repeated HTTP.
Current primary prefill invokes T32 target chunks124times for3968tokens; inspect
actual CPU input/tokenizer versus GPU prefill/kernel costs before choosing wider
primary-prefill/tiling or packing changes. Do not infer a kernel bottleneck from
wall-time subtraction alone. Existing tiled-score44case correctness is not model
speed. B8external numerical/quality/padded-worker/MTP/serving, INT4/4B and2x/
halfTTFT remain open. No stage here establishes the requested speed target.


### 2026-10-25 — physical B8 model admission closed and independently replayed

Original supervisor48577 reported bf16-batch8-model-admission-attempt1 terminal0.
Actual unchanged1a1d runtime completed all9cases:8short mixed rows and3968prompt+
128output text/image rows, identity/rotate3/identity-repeat. Full canonical state
component hashes, finite hidden/normalized raw vectors and output IDs match after
inverse permutation; every reset including final reset matches empty state.
Archive bdd7978b9969c26fbe931da134ec66b7bd73fdf5247c0033f9b8fd98100aa01f,
71041774bytes/527files, was fully transferred and independently replayed with the
unchanged verifier on both machines. Proof JSONs are identical. Actual28CPU tests
passed remotely. Workflow cleanup, original source/runtime identity and adapter
idle guard passed. Sampled correctness peak adapter use30854819840bytes (~28.74GiB)
is a construction/correctness observation, not serving peak or throughput.

This establishes full physical8row internal permutation/reset execution at4K.
Independent B1/B4 numerical equivalence,805quality, padded HTTP worker/streaming,
MTP and performance remain open. No2x or halfTTFT claim is admitted. Next qualify
the explicit wider MTP draft-prefill candidate against retained B1 target state,
then full4K/quality/memory and repeated matched timings; diagnose actual SGLang
concurrent output variation without rewriting its preserved strict failure.


### 2026-10-25 — wider MTP prefill actual run started; reference deployment prepared

After B8 terminal0 and independent identical proofs, absent-only installation of
11explicit prefill16module/method/evidence files passed555existing-input/runtime
hash checks and21target checks (10already matched). Bundle52b8176bcbcd3c93f66ae6acba8944f78aab4e1fa9b5c392789a57bf2327cb8f,
37902bytes; existing qualified source/runtime/defaults/checkpoint/SDK unchanged.
Original supervisor48577 started mtp-prefill16-model-consistency-attempt1 with
actual immutable SGLang2 and B8 archive hashes. Actual remote41CPU tests passed
0.201s; raw log retained. Last authoritative poll is running; last read-only
snapshot showed model starting,0completed cases,0constructed sessions,no error.
Embedded Python again printed previously observed unsupported optional-hash
warnings; the SHA256 evidence path remains unchanged. Do not infer failure or
restart from a quiet observation. Target-state/ID/extra-memory/model/performance
results require terminal cleanup/archive and independent replays.

Local next SGLang output/isolation deployment is syntactically prepared, unrun:
12targets/498existing inputs. In addition to successful closed B8 proof and prior
native/failed SGLang closures, preflight now requires the actual closed MTP
archive/workflow and idle adapter/process. No simultaneous GPU jobs. The real
288request/135overlap-wave study must retain raw variants and mixed-color/spatial
controls before changing future benchmark policy. Current strict failed reference
archive/helpers are preserved. Strongest matched refs, full4Koptimized quality/
repeated timings, externalB8/HTTP/MTP, INT4/4B and2x/halfTTFT remain open.


### 2026-10-25 — current B1 full4K host-stage diagnosis prepared and CPU checked

Prepared a separate explicit diagnostic copy of the qualified native fixed-length
worker. Removing tracing imports/setup/event wrapper/phase contexts reproduces
the entire original AST. Existing model math, input construction, policy, MTP
configuration and qualified runtime/defaults remain unchanged. Flat host phases:
reset, text/image input processing, embedding/row preparation, target/MTP prefill,
generation. Actual same-process token events mark first/last observations; MTP
may emit its first target token before draft prefill finishes. Phase duration is
not TTFT, GPU-only or kernel time. Runtime clock implementation/resolution is
recorded. No cross-process timestamp epoch comparisons or HTTP timing admitted.

Four actual parent C1 requests (two text/two distinct JPEG nonces) are selected
before execution,3968+128 each. Their raw request bytes/corpus hashes and both
actual ordinary/MTP prompt/output IDs are bound through replayed immutable native
HTTP evidence. Future24requests use original/rotated/reset-repeat orders for each
mode, including first-use observations; no synthetic IDs stand in for real data.

Actual23CPU tests pass0.207s including synthetic temp-only24request/two-owned-child
success/failure, raw/count/ID/image/chronology mutations, exact whole-worker AST and
unchanged wire controls. Final first-token phase labels/scope pass8CPU0.165s.
Earlier21CPU stages and failure ledgers retained. Initial old sibling-import test
invocation failed1/17; corrected process-only test search path, no old source edit.
Initial synthetic controller failed1/23 because local coarse monotonic intervals
could be zero for a CPU stub. Fixed fixture clock injection only; real strict4K
positive durations unchanged, exact failed source hashes recovered/preserved.

No remote deployment, GPU profile or performance result for this diagnostic yet.
Original wider-MTP qualification supervisor remains the sole GPU job; do not
mutate its frozen sources or restart from silence. After its actual closure,
replay/retain it, execute prepared SGLang output/isolation diagnosis and strongest
matched references. Run current B1 host/GPU-stage profiling before choosing wider
target prefill, input packing or tiled-kernel integration. Full optimized4K/
805quality/repeated timings, externalB8/HTTP/MTP, INT4/4B and2x remain open.


### 2026-10-25 — wider MTP short-model correctness closed and replayed twice

Original supervisor48577 observed mtp-prefill16-model-consistency-attempt1 terminal0.
All8actual B1 short text/image/red/green/reset/prefix/rejection cases completed.
The complete captured target logical state and raw hidden/normalized vectors,
output IDs and callbacks match retained1a1d B1. Both machines independently replay
the immutable68759719byte/715file archive; proof JSONs are identical. Actual41CPU
tests passed0.201s on remote. Workflow cleanup, qualified source/runtime identity
and owned-cache retirement passed. Actual counters:41token text12→4draft-prefill
calls,104token image27→8,93token colors25→7, across both repeats. These are call
counts, not speedups. Extra T16 engine weights are resident separately; captured
graph/buffer/memory details retained. Sampled correctness adapter peak23527370752
bytes (~21.91GiB), not full4K construction/serving peak or performance admission.

Actual4Ktarget state/memory,805quality, HTTP/repeated speed and strongest matched
vLLM/SGLang remain unqualified. No2x or halfTTFT claim. Current native default
factories/source/runtime remain unchanged; candidate remains explicit study opt-in.
Next execute prepared SGLang output/isolation diagnosis after its closed-job/source/
idle preflight, then qualify wider MTP at4K and run host/GPU stage diagnostics.

First SGLang diagnostic preflight failed before any receipt/install/model startup:
manual wrapped-terminal archive-hash copy duplicated one character (65hex). Failure
and original script retained. Corrected invocation reads all three closed archive
hashes directly from saved proof JSONs and requires64lowercasehex. No qualified
source/runtime/old failure archive or metric rewritten.


### 2026-10-25 — actual SGLang output/isolation study deployed and running

After current MTP supervisor terminal0 and identical independent proofs, corrected
preflight read archive hashes from proof JSONs and passed498existing-input checks,
all runtime hashes and12target checks. Absent-only bundle installed8new methods/
evidence files with4existing matches,41958bytes,
SHA2200ffa6b76222a5b708a51001405211e001c9a1785f6f84371740b578aa773d.
Existing native/checkpoint/SDK/defaults unchanged; strict prior reference failure
remains preserved. Original supervisor48577 started sglang-output-characterization-
attempt1; actual33remote CPU tests pass11.691s. Last process poll is running.

No actual client success or final diagnosis observed yet. Owned server config,
PID/installed sources, raw288requests/135client-overlap waves, actual full4K budgets,
nonstream IDs, streamed and nonstream mixed-color/spatial isolation, variants and
cleanup must pass independent archive replay before interpretation. No throughput/
TTFT summary, physical GPUbatch, numerical cause/harmlessness or2x admitted.
While this HTTP study is active, use only its original supervisor remotely;
perform independent source preparation and CPU checks locally. Next full4K wider
MTP/quality/memory, host/GPU profiling and stronger matched reference timings.


### 2026-10-25 — full4K wider-MTP target-state admission prepared, CPU qualified

Actual short-model T16 qualification is closed/replayed; next owner/probe/replayer
is prepared for actual3968+128 full4K. Use four retained native C1 inputs (two text,
two distinct JPEG nonces), then first-text/image reset repeats. Six cases compare
ordinary versus explicit candidate target canonical hashes and finite hidden/
normalized raw vectors at prefill/final, all actual prompt/output/callback IDs,
empty/final reset and actual249draft-prefill calls. Fixed128outputs explicitly
ignore EOS; target slots must be3968after prefill and4095at final prediction.
No805quality, HTTP, throughput or speed admission from this correctness study.

Actual29CPU tests pass0.406s. Complete owner suite50CPU pass0.775s, including
synthetic six-case/32state raw/count/ID/image/reset/callback/slot fault rejection,
closed-reference source/runtime/server/proof lifecycle and unchanged target/MTP
guards. CPU fixture bytes/hashes are explicitly synthetic, not model evidence.
Before deployment, source audit corrected an unrun replayer metadata expectation:
CUDACompiledGraph output storage is absent before first execution (0bytes) and
33882112bytes after T16 execution. Initial verifier/preparation/plan preserved;
model, raw-state/count gates and admitted results unchanged. Final checker parses;
CPU-tested modules unchanged by this checker-only metadata correction.

Final preparation pins586files/12targets; absent-only deployment preflight is
prepared with575existing inputs and explicit actual reference-characterization
closure/proof plus previous HTTP/B8/short-MTP closure and idle guards. No remote
installation or GPU execution yet. Active SGLang characterization remains the
sole remote HTTP/GPU study; only its original supervisor is polled. After actual
closure, archive/replay all variants and color/isolation results, then select the
next matched reference benchmark policy from evidence. Qualify candidate4K,
805quality and current host/GPU bottlenecks before repeated speed comparison.
Stronger tuned/MTP vLLM/SGLang, externalB8/HTTP/MTP, INT4/4B and2x remain open.


### 2026-10-25 — SGLang raw output/isolation diagnosis closed and replayed twice

Original supervisor48577 observed sglang-output-characterization-attempt1 terminal0,
owned Linux parent271 stopped, wrapper terminal and adapter/process idle. Actual
33CPU tests passed11.691s. Source/runtime unchanged. Complete70670636byte/1612file
archive transferred and independently replayed on both machines; proofs identical.
All288requests/135observed client-overlap waves meet strict full4K/nonstream-ID/
raw-SSE/count/semantic-control coverage and ownership/config/source cleanup gates.

All24serial after IDs/texts equal their serial-before controls. Actual concurrent
nonstream outputs differ95times, streamed text differs47times versus serial
controls; all96mixed-color/spatial semantic controls (72concurrent) pass. This is
not a numerical-cause/harmlessness, global request-isolation or long-output
semantic-equivalence proof. Rotated concurrent IDs and later streams also vary
relative to earlier same-concurrency controls. Actual physical GPUbatch was not
observed. Detailed per-phase raw-derived counts/variants retained. No timing,
accuracy score, broad quality or2x admitted from this correctness study.

The previous strict baseline/helper/archive remain immutable. A separate successor
must retain3968+128counts/prompt IDs/precision/cache/source/GPU checks, raw actual
nonstream IDs, all observed stream-text variants, semantic controls and quality
screening rather than excluding the faster normal reference for serial-text
inequality. Complete C1/2/4/8text/image timings, strongest tuned/MTP SGLang/vLLM and
repetitions before any2x claim. Do not infer unobserved streamed IDs or harmlessness.

First independent replay CLI failed at existing-scratch-directory assertion before
reading archive. Same checker succeeded using existing results scratch location;
failure observation retained, no guard/data change. Next wider MTP actual full4K
correctness owner/preflight pins qualified native/short-model proofs and requires
this actual closed/replayed reference archive plus idle guards before execution.


### 2026-10-25 — wider MTP full4K admission deployed and running; diagnostic figure

After actual SGLang diagnostic closure and identical independent proofs, new
preflight verifies all575existing inputs/runtime, prior native/B8/short-MTP and
actual characterization archive/workflow/proof, and idle adapter/process. All12
targets absent. Bundle6521c8625dbc4498400ee409e5f58ef3346aa55b3b9a4892e93bcec72f13bf3d,
49265bytes, installed only reviewed new methods/evidence. Existing native runtime,
qualified model/defaults/SDK/checkpoint remain unchanged; fixed root/cache reused.
Original supervisor48577 started mtp-prefill16-full4k-admission-attempt1. Actual
50remote CPU tests pass0.675s; last handle poll running, no4Kcase success yet.
Embedded optional-hash warnings again observed; no SHA256 evidence-path change.
Do not infer failure/restart from quiet construction; await actual closed archive
and independently replay full target-state/raw-vector/token/reset/memory results.

Four-panel raw-derived SGLang output-stability PNG/SVG visually inspected. Actual
text/image nonstream token-ID and streamed-text exact matches versus serial:
C1=8/8, C2=0/16IDs and0/8texts, C4=0/16IDs and0/8texts, C8=2/16IDs and1/8texts
per group. Nonstream C2/4/8 combines original and rotated observations; streaming
IDs are unobserved. All96color/spatial semantic checks pass;24serial repeats match.
This figure describes output stability, not speed or an accuracy/cause verdict.
Full raw variants/source/lifecycle retained. Stronger ordinary/tuned/MTP reference
timings must preserve these observations without dropping the faster normal engine
or rewriting the prior strict failure. Candidate805quality/HTTP/repeated speed,
host/GPU bottleneck work, externalB8/HTTP/MTP, INT4/4B and2x remain open.


### 2026-10-25 — wider MTP actual full4K correctness closed and replayed twice

Original supervisor48577 observed mtp-prefill16-full4k-admission-attempt1 terminal0.
Six3968+128 cases (four retained text/image inputs plus reset repeats) match actual
ordinary target IDs, callbacks and complete canonical state/raw hidden/normalized
vectors at prefill/final. All32captured target states replay. Actual50remote CPU
tests pass0.675s. Qualified1a1d source/runtime and cleanup unchanged. Immutable
67594006byte/666file archive fully transferred/replayed on both machines; properly
named proof JSONs identical. Actual sampled correctness adapter peak23688986624
bytes (~22.06GiB), construction/full4K fit observed, not serving peak or speed.

Actual candidate draft-prefill counter249in every case. The same four prior native
MTP HTTP worker records have993calls, with exact matching prompt/output IDs. These
are invocation counts, not speedups.805quality, HTTP, repeated matched timing and
strongest tuned/MTP references are still required before performance admission.

Initial checker verified all data but wrote results/memory_after-verified.json:
memory-loop variable shadowed the original job-name variable. Preserve that initial
proof/source. Separate named checker changes only the literal output expression;
2CPU whole-AST/filename regressions pass0.006s. Replayed on both machines again,
all proof fields except executing checker SHA equal the original successful proof.
No model run, archive, numeric/state guard or metric changed for this correction.

Next exclusive ordinary SGLang successor retains all lengths/precision/cache/
source/PID/805screen guards and raw output variants with comparison bookkeeping
outside timing. Keep old strict failures immutable; measure complete C1/2/4/8
text/image cohorts. Native candidate805quality/HTTP, host/GPU bottlenecks and real
GPUbatching/INT4/4B/stronger tuned/MTP vLLM/SGLang and2x remain open.


### 2026-10-25 — ordinary observed-output reference successor CPU qualified

Prepared separate ordinary SGLang benchmark/controller/probe/independent verifier.
Old strict helper/archive remain unchanged. Require actual replayed same-profile
288request diagnosis and closed native full4K proof before startup. Actual serial
prompt/completion IDs,3968+128/4096usage, metadata precision/cache/context/source/
owned-PID/stop, two early full4K boundaries, fourVLM/16short and fresh805adapted
screening remain mandatory. All124controls and124raw streams remain frozen.
Every stream-text variant is recorded; no hidden streamed IDs or long-output
semantic/numeric equivalence inferred. Source helper protocol ASTs unchanged.

Comparison hashes and sidecar file writing are deferred until all timing and
serving-memory sampling closes, avoiding added CPU work while peer streams are
measured. Actual60owner CPU tests pass11.180s; final36protocol/HTTP/metrics tests
pass10.080s. Explicit ephemeral synthetic HTTP integration sends124ID controls+
124streams, completes8cohorts/124sidecars and independently replays raw results.
It proves126-token clamping still fails before a completed cohort, original failure
retained, sampler closes, false sidecar assertions rejected. Fake CPU sampler/
responses are not model metrics. Earlier23/36stages and failed36case ledger/source
are retained; finalizer no longer masks the primary invalid-budget error.

Read-only actual retained-data policy check binds current diagnosis metadata and
all12old SGLang2raw streams to their real serial IDs:10text matches/2variants,
strict lengths/hash/usage preserved. The old failed manifest stays failed; no
retrospective cohort or benchmark promotion. Source derivation/import and initial
preparer package-path assertion corrections preserved; no model/GPU run involved.

Preparation pins525files/15targets, expected60CPU controls. Absent-only deployment
preflight prepared with511existing inputs plus actual closed native/characterization
proofs and idle adapter/process guards. No new reference server/timing yet. Initial
successor is ordinary same-profile reference; stronger tuned/MTP SGLang and vLLM,
repeated matched native candidate timings/805quality, host/GPU optimization,
real GPUbatching, INT4/4B and2x remain open. Once HTTP timing starts, only its
original supervisor may be polled remotely until terminal closure.


### 2026-10-25 — full ordinary SGLang successor started; host diagnostic owner prepared

Actual absent-only deployment passed:511existing pinned inputs/runtime hashes,
15absent methods installed, bundle4c49c4558726cb32dee91c6e5146a931c00a81ca2fca3ca2c151ebd4e5fab634.
Commit126e7428d is pushed to qwen38-5090. Actual original48577 supervisor started
sglang-observed-fixed4k-ordinary-attempt1 using the actual saved closed wider-MTP
full4K archive/proof; remote60CPU tests passed8.847s. Last observed running, no
terminal timing result yet. Preserve original strict failures. During this job,
only its original supervisor may be polled remotely; no transfers/other observers.

Prepared exclusive next native-host-stage-full4k-attempt1 owner/preparer/verifier,
never run before current reference terminal closure and independent replay. Pins
599files/13upload targets,587existing inputs. Actual29CPU controls pass0.922s:
whole-worker AST equivalence, two owned synthetic children/24requests, raw stage/
event/ID/order mutations and closed-reference namespace/hash/proof/lifecycle guards.
Independent read-only rebinding also verifies four actual3968+128 request bodies,
distinct image bytes and eight actual ordinary/MTP completion controls directly
against the retained native HTTP archive. No synthetic timings enter model metrics.

Initial preparer rejected a nonexistent test_fixed_length_policy import before
writing a preparation; preserve initial source, use existing tested
test_fixed_length_worker_contract instead. CPU child explicitly resolves the
unchanged sibling native_worker_io import without global environment mutation.
The host diagnostic will retain reset/input+vision/embedding+rows/prefill/generation
host-wall phases for three request orders per mode. MTP first token occurs inside
prefill before draft work may finish; full prefill duration is notTTFT. No isolated
GPU time, causal kernel bottleneck, HTTP speed improvement or2x claim.

User chart report combines historical short-prompt B1 wins and C1/2/4/8 throughput
with actual4K native ordinary/MTP and the clearly hatched one-cohort SGLang failed
attempt2 observation. No zero for missing vLLM4K, no complete/strongest SGLang
baseline claimed, no new benchmark generation. Source and input hashes retained.
Next: close/replay full SGLang successor, then actual native host-stage profiling,
follow evidence to prefill/batching/kernel optimizations, qualify805quality/HTTP
before speed. Tuned/MTP vLLM/SGLang, repeats, INT4/4B and2x remain open.


### 2026-10-25 — wider-MTP HTTP worker/probe CPU qualified, model unrun

Separate explicit native_prefill16_stream_worker and probe_prefill16_http use the
already full4K-model-qualified CUDAMTPPrefill16StudySession; original workers,
target/defaults/runtime/SDK unchanged. Actual closed full4K archive/proof,
source/runtime/DLL and candidate source hashes required before construction.
Ordinary mode rejected by new entry. Ready identity declares16draft-prefill/4draft-
decode/extra resident draft weights and no HTTP quality/performance admission.

Actual22CPU tests pass0.166s. After removing only the extra proof/ready fields and
explicit class substitution, whole worker AST equals qualified source46c2565a.
Whole HTTP probe AST equals old probe after entry/identity assertions removed:
all12protocol cases,805quality+actual ordinary-token comparisons, two full4K
boundaries and124frozen streams are retained. Launcher changes worker entry only.
Synthetic fake-file tests reject wrong mode/unqualified proof/archive/source/
runtime drift and unfinished workflow before any real model import. Initial AST
normalizer empty-dictionary error and22test failure source/ledger retained; fixed
test utility, production candidate source unchanged.

GPU/HTTP worker unrun; exclusive owner/preparation/independent HTTP archive replay
must precede805quality and speed.249vs993 actual draft-prefill calls are workload
counts only, not a speedup. Current SGLang observed-output job remains live under
its original supervisor; no remote competing observer/transfer. Actual4VLM and
16short128token controls passed after ready identity verification. No timing or
2x result yet. Host-stage diagnostic remains the next exclusive profiling job.


### 2026-10-25 — wider-MTP full HTTP owner/verifier CPU qualified, timing unrun

Prepared exclusive prefill16-fixed4k-http-attempt1 owner/preparation/independent
verifier. Requires actual terminal independently replayed host-stage job and actual
full4K wider-MTP target-state/source/runtime proof, pinned closed native ordinary
HTTP archive.16targets/601files/586existing inputs,68CPU controls. Separate explicit
worker only, target/T4decode/defaults/SDK/DLL unchanged. Actual worker exit now
recorded/requires integer0; unknown/nonzero/boolean exits preserved and rejected.
Retain prior probe source/derivation before exit-receipt change.

Before any timing:12protocol cases, fresh805natural-stop adapted screening with
every prompt/completion ID exact to actual closed native ordinary, two3968+128
boundaries. Then124frozen streams/eight text/image C1/2/4/8 cohorts, actual ordinary
IDs exact; physical batch1 remains queued. No short-prompt pilot. Raw stream/worker
IDs/usage/corpus/timing/sample-memory/gap arithmetic retained from snapshot checker.
Independent verifier binds closed native/full4K/host proofs/archives, executing
helper hashes and clean worker exit. No invented combined archive. CPU AST controls
prove whole benchmark replay arithmetic unchanged after ordinary-archive routing;
whole candidate worker/probe model/quality/stream operations remain as documented.

Actual68CPU tests pass7.403s. Initial68 failure1 due generatedCRLF vs recordedLF
byte derivation; retain failed ledger/checker/test/derivation. Generator explicitly
writesLF and records source newline normalization; final exact-byte derivation and
all guards pass. No actual data tolerance/math/precision guard relaxed.

Additional read-only actual-data check executes the new benchmark replay helper
against all124previously closed nativeT4MTP streams. All eight raw timing/memory
cohorts and ordinary output IDs match the existing independently replayed result
exactly. This validates helper arithmetic only: oldT4run is unchanged, no newT16
model/quality/speed result. Absent-only deployment helpers prepared locally, not
uploaded during active SGLang timing. Original48577 supervisor now reports full4K
timing has begun and first textC1 cohort emitted; wait terminal archive and both
independent replays before charts/metrics promotion. Profiling remains next onGPU;
strongest tuned/MTP references/repeats, INT4/4B and overall2x remain open.


### 2026-10-25 — complete ordinary SGLang actual4K baseline closed/replayed twice

Original48577 supervisor reports sglang-observed-fixed4k-ordinary-attempt1 terminal0.
Archive63461ebf577a6bb0c4beacc7ce30ceffbb41b13a8b2f07dba2fd2f7f67cb1900,
101948863bytes/3988files. Remote independent replay terminal0; actual full SFTP
acknowledged before local replay. Local terminal0; proof JSONs identical, SHA
d635f14aa14dd51cfaeff7a0ab7970281e2ff788e29a9afd82471ce240d14b0a.
Actual60CPU pass8.847s,197installed+19context sources/eight checkpoint metadata,
owned LinuxPID294 verified; actual SIGTERM ownership/stop0/wrapper0/idle/cleanup
and unchanged source/runtime pass. Same ordinary profile; engine4098 reserves2
slots for actual3968+128/4096usage, BF16KV/F32SSM, all cachesoff, full decodegraphs
1/2/4/8, prefillgraphs/autotune disabled. Strongest tuning/MTP not established.

Two early full4K boundaries/fourVLM/16short controls and fresh805screen pass:
487/805 adapted score; all124actual serial-ID controls and124streams/eight measured
text/image C1/2/4/8 cohorts independently replay.124post-timing comparison sidecars:
26serial-text exact/98variants retained, no hidden streamed IDs, numerical cause,
harmlessness or long-output semantic equivalence inferred. Serving sampled peak
29589413888bytes; whole quality/client peak29655699456bytes. Maximum serving gap
0.050200600002426654s. Samples adapter-wide incl background; no process/instantaneous
VRAM claim. Old strict failed attempts remain unchanged and are not promoted.

New complete8panel chart/report compares actual frozen4K nativeordinary/MTP with
ordinary SGLang. Single-client text MTP75.14vs64.07decode, TTFT5.98vs0.307s,
total output16.61vs55.80tok/s. Image87.20vs60.68decode, TTFT5.96vs0.334s,
17.16vs52.62total. At8clients: native MTP text75.09vs44.39decode but16.60vs151.79
throughput; image90.40vs42.61decode(2.12x this one ordinary-reference point),
17.26vs146.54throughput. Overall2x/halfTTFT target remains unmet. Native physical
batch1 queues clients; SGLang up to8, nativeWindows vsWSL. Native screen502/805;
cross-engine broad quality parity/unrepeated uncertainty and vLLM4K remain open.

After all transfers/replays and idle/source preflight,13absent host diagnostic
methods installed: bundle74268f9a38ba437ad3a1e19c24262c025278842d1670d1326a10832adc2ca12e.
Original48577 started native-host-stage-full4k-attempt1 with actual closed SGLang
and nativefull4K archive identities. Last running, actual27remoteCPU controls pass
0.626s. Initial preparation expected29: local audit finds the old worker-I/O test
module omitted from file pins; four diagnostic/contract modules are pinned. Actual
remote IO source not yet observed. Preserve discrepancy, original preparation,
verifier and GPU run. No proof admitted yet and no model restart. A unique new
test_native_host_stage_worker_io copy pins all five local IO tests; all29local CPU
controls pass0.819s with source hashes. After original job closes, retain original
strict verifier count-check failure and explicitly pin/run supplemental CPU tests
before diagnostic admission. No numerical/state/timing guard or tolerance changed.
Only original supervisor polled remotely during the live host diagnostic.


### 2026-10-25 — actual native host phases closed and replayed twice, wider-MTP HTTP started

Original native-host-stage-full4k-attempt1 terminal0,24actual retained3968+128
requests in two modes/threeorders. Archive59c529869f970f40935ee463c62e7c06eaa06bbf6d7171090360dee5e2a97dd4,
66604875bytes/618files. Existing1a1d DLL/model source/runtime unchanged; owned
children closed and caches retired. Adapter-wide construction/correctness peak
22808047616bytes, not serving peak. Entire diagnostic worker AST remains qualified
worker operations after trace-only nodes removed; every actual prompt/output ID,
image byte/control and raw same-process timing interval independently replayed.

Preserve original preparation expected29 vs actual27CPU count failure, local and
remote strict checker logs. Post-run remote old IO file has three tests, hash
bbe22691c974b15149b22fe0970ca65d62c8311aedd67d69ed9032d837447796; it was omitted
from original file pins. Post-run observation is not historical certification.
Separate CPU-only supplement installs8absent files,6already exact/14targets.
All five test modules/29IDs explicitly pinned including a new unique5IOtest copy.
Actual remote29pass0.575s; local34controls pass0.817s including five archive/
mutation/full-verifier-AST tests. Original599source/runtime hashes checked before/
after CPU supplement. Supplement archivec04d6e591cfab740c9ea67f72585a4fe7de36c342224dd31b921ebaf70469031,
118472bytes. Original model/archive unchanged; no GPU rerun. Separate verifier
8685afae2eb86ac6d71270526d9675f9ecce7a44c4d69b1886ba279000f7f26a requires the
actual supplement and keeps all original raw trace/token/case/memory arithmetic.
Remote/local independent replays pass with identical proof JSONs, SHA
fa9e8c05f10038f961b056baab7e6a85292ff91bcf87f34a38edad524ca5284d.

Phase medians: ordinary target prefill4.775text/4.771image seconds, generation
3.289/3.288; oldMTP prefill5.804/5.814 and generation1.581/1.509. Embeddings/rows
about0.15s; input processing about0.01text/0.02image. These are host-wall call
intervals, not isolated GPU time or HTTP TTFT. MTP first token may precede the
end of prefill. All24requests retained. Late-repeat stalls remain unexplained:
ordinary-r2-text-c1-r0 whole22.681s; mtp-r2-image-c1-r0 whole14.123s. Plot uses
means including delays and source data retains phase medians/allindividualrows;
no trimming, causal graph/kernel assertion or performance comparison from profile.
Initial plot clipping corrected by automatic axis range; final visually checked.

Future wider-MTP HTTP preparation also lacked an explicit memory-test module pin.
Preserve initial unrun601file/16target preparation/plan/input/source. Freeze a
unique copied memory-test module; preparer and owner now require every test-module
file in source pins. Initial local cross-drive rename error left work files intact;
verified backup copies/same-drive renames used. Initial64CPU run with missing-module
error retained. Final68pass6.939s; new preparationc08ec706c955a67dce42cecfc035fe6759b5760d88f22e1e14d4a9f8bff997e6
pins602files/17targets. Actual closed-parent/source/runtime/idle preflight passes,
15absent files installed/2exact shared helpers skipped, bundle
6f28d434fe823044f413ff62230344c6716ba772a6654b7a73589005e174ca62.
Original48577 supervisor started prefill16-fixed4k-http-attempt1; actual68remoteCPU
controls pass7.462s. It must pass
12protocol/805natural-stop ordinary-ID parity/two4Kboundaries before124frozen4K
streams; no short pilot. No serving result yet. Only original supervisor polling
permitted remotely while this job may time; all transfers acknowledged beforehand.

Next optimization path follows actual dominant prefill call interval. Collect a
current B1/full4K CUDA graph/kernel/API trace with verified same-process markers,
investigate late-repeat stalls and inspect actual capture/replay/kernel coverage.
Then explicit target-prefill128 study with compactBF16KV and unchangedT4decode:
new native metadata/commit/plugin contract and numerical admission are required;
current Python/native profile guards remain4/32. M32 FP8 policy applies specific
projections at rows32, so wider arithmetic/prefix/state equivalence must be proven.
CPU metadata/invalid-input, GPU BF16 score/PV/commit reference cases and complete
model state/raw-vector/128ID/reset/prefix checks precede805quality/HTTP timing.
For concurrency, B8MTP workspace128MiB is allowed by current MTP program but actual
memory fit, external math/quality, partial-row serving and throughput remain
unqualified; qualify before any real batching claim. Strongest tuned/MTP SGLang/
vLLM4K references/repetitions, INT4/4B and overall2x/halfTTFT remain open.


### 2026-10-25 — current B1/full4K CUDA phase diagnosis prepared, not executed

Previous chart-report turn was a verified wait: original48577 supervisor confirmed
prefill16-fixed4k-http-attempt1 live. Current turn continues polling the same owned
job only; no remote command/upload/probe competes with its official HTTP timing.
Actual text C1/2/4/8 and image C1 cohorts appeared on the live supervisor; their
numbers remain provisional until terminal closure and independent raw replay.
Earlier complete comparison charts remain the admitted result. Overall2x and
halfTTFT targets remain unmet/unproven.

Committed/pushed1ac05f7f7 records actual host24request closure, explicit original
27vs29CPU source supplement and widerHTTP68test source pins. Source/model/runtime
were not changed in this follow-up. New explicit CUDA-phase diagnostic source is
prepared against actual current1a1d B1/full4K model. Original ordinary/T4MTP host
worker operations remain identical after trace-only AST removal; generation,
token/image/context/retained-case checks are unchanged. Full worker/launcher/probe
derivations retained. Existing native/SDK/default/checkpoint/corpus untouched.

Nsight2026 node collection runs from worker launch, avoiding the previously failed
interactive capture method. Twelve unique HtoD copies bracket request and five
flat phases,144markers each mode/288total. All24actual3968+128 requests in the
three original orders remain. Raw SQLite must contain exactly one matching HtoD
copy per marker, same process/device/context/stream and ordered GPU timestamps.
No Python/Nsight epoch conversion. Real kernel/graph-node records required in
prefill/generation; separate API/kernel/copy/memset activity is recomputed, crossing
records clipped/count retained, overlaps combined by interval union. Activity gaps
are not attributed to CPU/GIL/scheduling/thermals/capture. Profiler/marker overhead
prevents uninstrumented model-speed claims. No HTTP timing in this diagnosis.

Separate witness is written only after native session/vision/decoder close calls
return. Nsight wrapper exit is not assumed to prove native model cleanup. Initial
unrun preparation is retained before adding that witness. Initial9CPU and full42
CPU pass; final43CPU pass0.979s with all test IDs and seven module sources pinned.
Direct probe CLI import-path smoke passes without loading Garnet/GPU. New
preparation27ed5e53806dbc0b68bef8b9d388713e88486411688a8c7b6bdf7d4fd0dedec3 pins
630files/14uploadtargets;617existing inputs. Read-only profiler executable SHA
e597347ef6cb45456612c0c9593401bb6d1b85497bb18534968f309955831291 and original
signed-tool setup proof required. Prior short trace CPU schema audit confirms
runtime thread/process encoding,0copy-correlation PID mismatches; it is not a new
full4K trace or bottleneck result. Final all630source/runtime/profiler checks and
43actualremote CPU IDs precede model construction. Reserve96GiB engine scratch/
32GiB study disk; stream large trace archives rather than buffer them all in RAM.

Deployment has not run while HTTP timing is live. After its closure/archive/replay,
preflight/install only absent new methods, then profile current B1/full4K and replay
on both machines. Investigate actual dominant prefill operations and late-repeat
stalls without trimming cases. Wider target-prefill128 needs explicit native
metadata/commit/plugin and math/state/ID validation. Throughput still requires
actual B8MTP fit/externalmath/quality/padding/HTTP qualification. Strongest feasible
tuned/MTP SGLang/vLLM4K/repetitions and INT4/4B comparisons remain open.


### 2026-10-25 — wider-MTP HTTP closed/replayed; verified charts; outer-profiled driver installed

Original48577 prefill16-fixed4k-http-attempt1 terminal0, source/runtime/1a1d unchanged,
owned nativeworker exit0 and cache retirement complete. Archive
d64adb0cdba9d7c3446f15034b34d095025394815d28c4dae463b0056377e748,
99391572bytes/3446files downloaded with explicit SFTP ACK. Remote independent replay
terminal0 and local replay exit0; proof JSONs identical SHA
b43f4adfd3d04989312b3e1f637bb6eed61e3bf1706151660e8fec7be5ee7c27.
All805screen cases,12protocol/two4Kboundaries/124fixed4Kstreams and943actual worker
completions bound to raw events; all actual native ordinary output IDs exact.
Adapted502/805 matches old native, not officialMMMU/broad cross-engine quality parity.

Actual3968+128, eight measured requests per group, C1/2/4/8. New text C1 TTFT
5327.85ms/decode75.69tok/s/total18.19tok/s; image C1 5324.77ms/88.26/17.66.
C8text29858.89ms/73.75/18.23; C8image28624.57ms/91.45/19.08. Compared to old
T4draft-prefill single trial, first-token medians about9-11%lower and text total
throughput about9-10%higher; imageC2total15.45vsold17.20 is a regression retained
without exclusions. These are independent single trials with no uncertainty or
causal optimization-speed admission. T16 extra draft weights raise sampled memory
to23831457792bytes=22.195GiB; maximum imageC1sample gap3.67318s preserved. All
raw samples/summary-prefix/trailing accounting independently replayed; no instant
or process-attributed memory peak claim.

New C8image per-request decode91.45vsSGLangordinary42.61=2.1465x this one ordinary
reference point. Total19.08vs146.54 and TTFT28.62vs1.604s remain much worse.
Native physicalbatch1queue vsSGLangmax8. No overall2x/halfTTFT claim; strongest
tuned/MTP and matchedvLLM4K/repetitions/INT4/4B/broadquality remain required.
New eight-panel4K PNG/SVG, standalone HTML and all32cohort data generated only
after identical two-machine proofs. All curves/delays/regressions retained; old
historical charts unchanged. Final new figure visually inspected.

Actual CPU Nsight IPC qualification caught a pre-model transport issue: two
90sreadiness smoke attempts, unchanged and explicitshow-outputtrue, receive no
live forwarded ready event. Both terminal1, no model/GPU constructed. Do not
launch the model worker behind that direct live Nsight pipe. Reused successful
prior process-tree architecture: outer Nsight profiles the request driver while
its normal nativeworker pipes stay internal. Actual CPU driver/child Unicode
echo and cleanexit checks pass, observed Nsight wrapper mirrors requested driver
exits0and7. Raw three attempts/scripts/logs/tinyreports archived separately:
0846e405501a70ba5dadd4e7444b55ae9cf534563f2dda29c6685f0ae6591d36,
33103bytes/22files; local rawmanifest replay passes. No GPU model or benchmark.

Final driver retains each ordinary/MTP actual worker exit, marker/nativeclose
witness and all12retained-ID requests, merges mode reports with explicit negative
guards. Nsight collection surrounds each full driver; export/replay follows its
closure, without live Nsight stdout dependency. Entire worker model AST still
equals qualified ordinary source after trace-only removal; source derivations and
earlier unrun pipe-launch preparation retained. Local46CPU pass1.493s, all eight
module files and46testIDs pinned. New preparation
8d96a7783b95012c14215bcfa30258f50a6550c90bea8a0bb01ad661682f087e,
633files/17targets. Actual read-only closed-parent/source/runtime/profiler/idle/disk
preflight terminal0; all617existing inputs exact, no conflicts. Absent-only bundle
f47dfa4cd78060d6aca57ae077ed95a02ec67d2aaa4a369b8214560c0c96a4c1,
58871bytes installed17new/all17verified, no baseline/SDK/model change. GPU trace
has not executed in this preparation snapshot. Next original supervisor owns
native-cuda-phase-full4k-attempt1: actual46remoteCPU IDs before model,24full4K
requests/288markers, source/runtime/cleanup/rawarchive and two-machine replay.
Use collected kernel/API evidence for target-prefill and realB8MTP work; no causal
kernel bottleneck or speedup is inferred from host phases or offline preparation.


Actual native-cuda-phase-full4k-attempt1 started on original48577 after every setup
transfer ACK. Actual source-pinned46remote CPU checks pass1.297s. Original handle
confirmed live, outer profile driver now proceeds toward native model readiness.
No collected GPU trace, kernel bottleneck or new performance result yet. Only
original-handle polling remotely while this exclusive diagnosis runs.


### 2026-10-25 — failed CUDA profile preserved; direct native bootstrap passes; T128 CPU contract

Original48577 native-cuda-phase-full4k-attempt1 terminal1 after actual46CPU pass,
before native ready/any request. Archive a698a95d3f2b8a3e7f7fa60794260cb4451ae2712785be6ecff9266e0283297f,
68326635bytes/663files, remote/local independent failure-preserving replay exit1,
identical proof8a269e39ab2410ce3c097c2054f624d9b830b472f122c53aef921e103083d703.
No request GPU timings, kernel bottleneck attribution or performance admission.
Postrun Nsight stats/ProcessStreams expose the driver's exited-before-ready error
and a finally assertion masking it; native worker stdout/stderr remain empty.
Derived SQLite/stream export is separate from original immutable raw archive.

Logged unprofiled successor preserves source633 and runtime1a1d; ten new helpers,
643source pins. Actual13remoteCPU tests/.130s and embedded-Python trace class/
intentional-exception fake-IO compatibility pass. One actual3968+128text control
then completes with original retained IDs,12marker requests, workerexit0 and
native cleanup witness. Archive8104499a0eace60c00e9af1add17640b5d29e0810894eee8e1b264164b31a9b7,
151303bytes/33files linked to failed parent. Both independent replays exit0,
identical proof9d5b3a3f4f0fad830d53c66be2b64dea5f5ec34c88ad5ec197412d3b747e57107.
This proves the logged worker operates without Nsight; original profiled failure
cause remains unresolved. Host intervals are not isolated GPU or benchmark time.

Separate opt-in targetT128/B1/BF16 layout/commit source scaffold keeps canonical
106823680words and linear39714816words unchanged; target delta44564480words/
178257920logical bytes. Actual11PythonCPU pass/.010s; private CPU/MT metadata
compile, import inspection without GPU/Garnet dependencies, native execution pass
2241malformed metadata and2827profile tuples; oldT4/T32 still reject new layout.
CUDA source is unbuilt/unexecuted, no runtime integration, memory-fit or speed
claim. Current-chunk and prefix KV both round BF16 per actual admitted source.

Next: whole-model trial of the already GPU-qualified tiled score kernel. Existing
B1/T4,T32 attention launches one warp per key; tiled16 retains eight-FMA and warp
sum order while reducing score-grid blocks16x. CTA counts are not speed evidence.
Keep baseline1a1d/source and SDK preserved; build a separate candidate, then exact
state/token/quality gates before matched HTTP timing. Real B8MTP/serving admission
is also needed: source inspection finds Row allocates/copies GPU data, so old B8
probe retains embeddings, copied rows and stacked rows concurrently. Existing
PackPrompt returns views but caps each packed allocation256MiB; full B8/3968x5120
cannot use one such allocation. A chunked view source needs exact prompt/state
qualification, including images/unequal lengths, before memory/speed claims.
Matched vLLM4K, strongest MTP references, repeated trials, INT4/4B and overall2x/
halfTTFT targets remain open; scope is unchanged.


### 2026-10-25 — opt-in tiled attention integrated/built; native API qualified; actual model trial started

The exact44-case/FP64-qualified tiled score kernel body is now an explicit opt-in
in src/cuda/qwen38_bf16_kv_low_batch_overlay.cu. Exact environment value1 selects
GARNET_QWEN38_BF16_LOW_BATCH_TILED_SCORES, frozen on first native query/enqueue.
Default path, B1/T4,T32 geometry, BF16 rounding, normalize/context and workspace
checks are unchanged. Exported selector and Garnet's read-only native API query
use one selector. Existing model/precision/cache/default settings are unchanged.

Native build attempt1 terminal0, only intended CUDA object changed. Actual49CPU
(.248s), ten fresh native-export selector subprocesses and955native tokenizer
cases passed. Separate90e2825ddd96b4029eff110805692fe5cbfdb276c9a32b27fdb936bccb01579c
DLL retained. Source643/runtime/SDK restored to baseline1a1d. Raw archive
2ebe8fef4a952cd8b16bc5cc3beaa8d83bb6b65e49eabd7b0d868a02c674f0b3,
2691896bytes/44files; independent local/remote replay passes with identical proofs.

Actual first model admission stopped before model construction: embedded Python
ctypes cannot invoke the exported query because libffi is unavailable. Actual ten
CPU replay faults passed/.831s first. Failure archive
027f2f5c544320c0610de15a82b27f1590320157969dfa045e8f765194c4b45b,
92747bytes/17files, both independent replays preserve exit1 and zero model routes.
No correctness, model timing, memory fit or speed claim. Baseline653/source/RTE
restored, actual child exited1. Earlier failure/helper files remain immutable.

Safe correction exposes bf16_low_batch_tiled_scores_enabled through GarnetAPI;
no ctypes, libffi, SDK or runtime dependency change. Native API build terminal0:
same tiled CUDA source plus minimal entry header, only intended CUDA object may
change; actual49CPU/.231s, ten external selector, ten actual embedded native API
fresh processes and955tokenizer cases all pass. Default/invalid/exact1 values and
post-query environment flips checked. Candidate
2533ad6867bb33137420272c975d33a26fb3637c1229fa58f5ad5dfaf1657ff7.
Archive3dd4c289fe7a26c6aec98e92066ce19c8a9e8c6e005182f4534d5f3b4bfe046c,
2712172bytes/68files, local/remote independent replay passes identical proofs.
Both candidate source files and baseline1a1d DLL always restored, SDK unchanged.

Successor bf16-scores-tiled-api-model-admission-attempt1 started on original48577
after all six absent-target uploads received ACK. Preparation
94e0f475712baa1d08cdc5ef67c41a2ea17decbc0c2003ce6ab96dd0da606fb3,
662source pins, five new helpers. Two fresh processes with candidate2533 DLL:
default/tiled, each six actual3968+128text/image cases and ordinary/T16draft MTP.
Each compares32complete state witnesses and64raw finite hidden/normalized vectors
to retained1a1d full4K archive92eef543, plus exact128IDs/callbacks/reset repeats.
Mutual ordinary/MTP agreement alone cannot pass; CPU fault controls reject shared
state/vector drift. Current trial results are pending in this start snapshot.
Observe resource memory only for construction/correctness, not HTTP serving peak.
No new benchmark or speed claim until whole-model and broad805/protocol gates,
then matched actual4K HTTP with source/runtime pinning and independent replay.

Uncompiled T128 CUDA/header/metadata scaffold moved byte-exact to study directory
test2026/cuda/qwen38_5090, outside production src/cuda glob. Isolation receipt maps
old coordinates from earlier CPU evidence. No accidental default compilation or
runtime qualification. Further target prefill and realB8MTP/memory/HTTP work,
matched vLLM4K/tuned MTP references, repeated trials, INT4/4B comparisons and
overall2xdecode+throughput/halfTTFT remain required. Goal stays active.


### 2026-10-25 — bounded B8 prompt source CPU controls; native qualification pending

Added study-only PagedPromptRows, no model/serving default change. Uses existing
native embedding, storage_view, packed slice and PackPrompt APIs. CPU IDs retained;
materialize128tokens per page and retain two pages. TargetT32 slices and MTP's
shifted single-token lookups keep ordinary model interfaces. Visual64x5120 tensors
come from the pinned caller-owned encoder, stay unchanged/alive until close;
cross-page intervals use exact FP32 slices. Unequal rows pad token0 under existing
active/length controls. Failure never installs a partial page or returns stale data.

Actual six CPU fixture tests pass/.014s, covering B8/3968 whole target/MTP access,
image boundaries, different lengths, cache eviction, malformed metadata, failures,
negative/slice bounds and releasing refs.31native-pack call requests in the CPU
access fixture; not observed native calls or performance. Two B8/128 packed pages
have41943040logical bytes, excluding visuals/transient embedding/slices/fences and
outstanding caller views. Native/GPU/whole-model qualification unrun, no allocator
or serving-VRAM saving, speed or2x claim. Class remains outside serving code.

Read exact retained B8 model report (not nested historical source reports): final
pool adapter-wide snapshot28968419328bytes, TRT allocator live/owned totals null.
Keep this unknown accounting explicit. Next actual native component comparison
to independent checkpoint BF16 embeddings and deterministic visual-row oracle;
then same B8 ordinary complete-state/raw-vector/ID permutation/reset controls and
real B8MTP construction, prefix commits,805quality, padded HTTP and measured peak.
Current tiled B1 model trial remains owned by original48577; no concurrent remote
GPU work or other probes while live. Overall goal and pending reference scope intact.


### 2026-10-25 — native B8 input-byte gate prepared; default-route4K model cases passed in process

Original48577 bf16-scores-tiled-api-model-admission-attempt1 remains running.
Actual six default-route text/image3968+128/reset-repeat cases emitted PASS and
its resource report saved; second opt-in tiled process started. No terminal
archive/independent replay yet, so no promotion or new benchmark. One read-only
operational check confirmed native process35548, cases_running report and active
GPU; output silence was not treated as termination. No restart/input/DLL/SDK edits.

Prepared PagedPromptRows native component byte gate, not installed/executed.
Eight matrix cases B1/2/4/8, page32/64/128, variable lengths/zero-ID padding,
64x5120channel-distinct dyadic visuals crossing page boundaries, new visuals and
rotated rows; long B8/3968 case. TargetT32 slices and shifted MTP reads plus held
views after iteration/close total11274observations. Actual first-page eviction is
recorded separately; some short cases have only two pages. Old source already
uses two-page cache; returned native views can outlive its eviction/close.

Retain every actual native row byte string once by SHA256 content filename and
ordered event row references. This is lossless deduplication, not a hash-only
capture: independent replay reconstructs every original read and compares all
FP32 bytes to checkpoint BF16 integer-word conversion or deterministic visual
oracle. Source/read count/order/shape/lifetime and missing/corrupt/stale bytes
checked. Tampering with data plus its saved digest still fails the oracle.

Actual12CPUcontrols pass, final.540s after oracle/eviction metadata changes.
Earlier source-pinned CPU runs retained in append-only ledger. Actual retained
15unique embedding rows/[248320,5120] also replayed on CPU from archive
7a7a7effcd1af201f103b5df5f7a74ef4b8aca67eda5c2c219eea86e57d75801;
raw fixture SHA b2a77e1a661fb958b186dee18b32ae24a570c67944879b1b76bf151506521bd9.
No native execution or observed memory reduction claimed. Eight new Python/batch
method sources syntax-inspected. Owner requires same twelve tests in actual
embedded runtime too, before checkpoint preparation or GPU calls. Fresh CPU
fixture must match all31retained fixture files. Explicit parent archive/proof and
source/RTE/SDK/idle guards prevent overlapping the live model or using its private
candidate DLL; this I/O gate requires baseline1a1d. Next preparation refuses while
actual closed two-machine parent proof is missing, as observed before GPU work.

After native byte admission, integrate source in B8 ordinary model and compare
retained complete states/raw hidden vectors/IDs across all nine old groups/orders;
then real B8MTP memory/prefix commits/quality/HTTP. Component memory sampling is
untimed construction/correctness only. No serving-memory saving or2x claim;
matched vLLM4K/tuned refs, repeated trials, INT4/4B and full objective remain open.


### 2026-10-25 — tiled whole-model and paged native bytes closed/replayed; B8 model preservation started

Original48577 bf16-scores-tiled-api-model-admission-attempt1 terminal0. Both
default/tiled candidate2533 processes complete six3968+128text/image/reset cases:
64complete canonical-state witnesses and128raw hidden/normalized vectors total,
all exact to retained1a1d baseline including all token IDs. Both workers exit0,
source662/RTE/SDK unchanged and baseline1a1d restored/cache retirement complete.
Archivecb4effcb8021b4a8320b304646e49f70e17f4161b2447a9acf1c006231f276c2,
2577890bytes/153files. Download ACK; independent remote/local replay exit0,
identical proof80f974e9aca3adad82c13eed8c812f15e81a727914f5fad62a5d5c5d947cd8f2.
Construction/correctness resource samples freshly recomputed: default peak
23868383232bytes/maxgap9.4791534s; tiled23682560000bytes/maxgap16.6060927s.
Retain gaps; these are adapter-wide sampled correctness peaks, not serving memory
or evidence of savings. No new benchmark or kernel-speed claim from this trial.

Native paged input gate then actual terminal0:12CPython tests/.461s, same12 in
real embedded runtime, fresh independent BF16 fixture exactly matches all31old
fixture files, eight actual GPU I/O cases. All11274read events reconstruct
1703301120observed bytes exactly to independent BF16 integer-word/visual oracles;
held views survive iteration/eviction/close. Raw rows retained losslessly once per
content digest; archive deda4bac52c095871633d2ffc615ef9b0fa6d6e197ea3d4c72e4d6408e4357ee,
6934610bytes/975files. Both independent replays exit0 with identical proof files.
Baseline1a1d/source672/SDK/runtime unchanged, no decoder or model constructed.
This admits input bytes/lifetimes only, not B8 model/MTP/serving-memory savings.

Next actual B8 model uses the admitted128two-page input source in unchanged
baseline1a1d B8/T32prefill/T4decode/BF16-cache model. Existing decoder, vision,
generation/reset/permutation operations retained; input source closes after
prefill. Compare all nine short/full4Ktext/image/order/reset cases against actual
old B8 baselinebdd7978b:29complete states/58raw vectors and all inputs/tokens.
Mutual permutation agreement alone cannot pass shared numerical drift. Actual
26CPUcontrols pass8.403s plus actual retained old B8 report replays on CPU.
Initial preparation caught four older B8 helper/test files absent from current
lineage inventory; bound all four to actual old archive hashes and existing local
bytes before any GPU/deployment. A preparation variable-shadow error was fixed
before inputs were installed. No existing source file replaced. Absent-only
bundle8c8f5291 installs nine new targets;676existing/all684source pins/runtime/SDK
verified. Preparation15d9e2b98f62e327604f19968bd47b00c440526b897812de7a2d75a4b0da46f3.
Original48577 paged-batch8-model-preservation-attempt1 started; terminal result
pending. Existing B8 external B1/B4 math/MTP/805quality/HTTP remain unqualified.

Tiled HTTP admission/worker/probe/launcher prepared but GPU-unrun: new candidate
DLL proof distinguished from old baseline1a1d NFC/EOS/ordinary controls. Actual
native API must select tiled route before model construction. Entire generation
and request body is literal unchanged; full protocol/805/two-boundary/frozen124
HTTP probe AST differs only in worker filename and explicit baseline-vs-candidate
library checks. Five CPU guard/AST/launcher controls pass/.012s and actual closed
model proof schema passes new gate on CPU. Owner/independent HTTP replay and real
805/124request execution still required. No new timing/plot or2x claim. Preserve
full scope: real B8MTP/math/quality/HTTP, matched vLLM4K/tuned references, repeated
trials, INT4/4B, and overall2xdecode+throughput/halfTTFT remain open.


### 2026-10-25 — tiled HTTP owner/checker prepared; GPU run not started

Committed prior tiled whole-model and paged native-byte closure as `07d887d7c` and pushed the dedicated branch. The original `paged-batch8-model-preservation-attempt1` supervisor is still authoritatively live. A read-only observation confirmed native PID31364, report status `constructing`, zero completed cases, and active native allocations. Do not start a second GPU job or infer a hang/performance result from silence.

Prepared an explicit candidate2533 tiled-score MTP3/T16 draft-prefill HTTP owner, independent raw checker and absent-only deployment package. Its fixed workload remains12protocol controls,805adapted natural-stop quality cases, two actual3968+128 boundaries, then all124frozen streams in eight text/image C1/2/4/8cohorts. Complete generation/request body and probe AST are unchanged except explicit candidate/baseline admission and worker entry names. Every raw benchmark/SSE/native request matching function remains AST-identical to the qualified previous checker. No short pilot, batch/profile relaxation or model arithmetic change.

Both local runs passed all77CPU controls (11.399s initial,6.510s after replay-helper/self-identity pins). Source/proof/SDK/DLL/selector restoration and wrong-route/shared-state failures are covered. The new checker also admitted the actual two-machine closed tiled-model proof and checked its662source pins. Preparation pins696files and six runtime dependencies plus15SDK/environment files; bundle contains13new targets and does not overwrite existing inputs. The old1a1d NFC/EOS/ordinary/full4K proof identities remain distinct from candidate2533. Owner temporarily selects candidate DLL/exact flag1, records runtime before restoration, restores baseline DLL and caller selector, and retires owned cache only after the child closes.

CPU-only preparation initially omitted live-vs-archived pins for the two new replay helpers; corrected before deployment, preserved the initial unrun manifests/checker/derivation, and reran the full77controls. See `evidence/tiled-scores-http-preparation-closure.json`, append-only CPU ledgers and exact derivations/install methods. No remote installation, new HTTP/GPU measurement, chart update, speedup or2x claim is supported by this preparation. Once the live B8 owner closes and its terminal source/runtime/cleanup are checked, install absent-only methods and execute/replay the complete HTTP trial. Keep the broader matched vLLM4K/strongest MTP baselines, B8 math/MTP/serving, repeat statistics, INT4/4B and overall2x goal open.


### 2026-10-25 — B8 MTP correctness owner/probe/checker prepared offline

While the original paged B8 ordinary supervisor remained live, added an explicit existing-session B8 MTP3/T4/128MiB/prefix-replay correctness and construction-memory study. This is the existing `CUDASpeculativeSession`, not a new backend or a relaxation of its256MiB B4-only guard. Target/vision/paged input/state-witness/permutation/reset code is retained by exact recorded derivation. Empty stop list retains the ordinary fixed-count8short/128full4K oracle semantics; natural-stop/805quality remains untested.

New owner, probe, actual-two-machine-parent preparer and independent raw checker require all nine mixed text/image/original/rotated/reset cases,29complete target-state witnesses,58raw hidden/normalized vectors and every output ID to match actual closed paged ordinary B8. MTP counters must cover exactly `limit-1` target commits per physical row, with bounded per-iteration accepted lengths and original target/draft geometry. Source/runtime/SDK/canonical-state/finite-vector/output-ID/counter/stop-list/parent-closure faults reject promotion. The raw resource checker recomputes sample count, adapter peak and median/max sample gaps; no memory-fit estimate or correctness-run memory snapshot becomes a serving-speed claim.

Actual local30CPU controls passed in12.167s. A separate CPU checker exercise replayed the original actual B8 ordinary raw bytes with explicitly synthetic MTP metadata; this is only a checker test, not an MTP execution/result. No native/GPU MTP construction, memory fit, natural-stop quality, HTTP or benchmark occurred. See `evidence/paged-batch8-mtp-offline-closure.json`, raw append-only CPU ledger and exact probe/owner derivations.

The original supervisor `paged-batch8-model-preservation-attempt1` was confirmed running; a partial report had two cases and no error. Native PID31364 CPU time increased from570.4375 to720.140625seconds between read-only observations. These establish continued activity only, not final case admission or timings. No new MTP preparation manifest/deployment was created: the preparer requires the actual terminal successful paged-B8 proof, identical two-machine replay, immutable archive and unchanged source/runtime/SDK before it can pin a GPU trial. Do not run a competing GPU job or overwrite existing study inputs while the original owner is live. HTTP candidate preparation is pushed in `79049c0ab` and remains separately unrun. Overall2x/halfTTFT, full matched vLLM4K/strongest tuned MTP baselines, B8 external math/serving/quality, repetitions and INT4/4B remain open.


### 2026-10-25 — matched ordinary vLLM4K pipeline prepared; live B8 has five exact cases

Prepared an explicit ordinary vLLM0.32.0 WSL trial using the already existing shared fixed-length reference HTTP client and unchanged decode-graph launcher. Same frozen actual3968+128 requests, eight text/image C1/2/4/8cohorts, dual idle processor/encoder resets before timing and zero new cache hits. Two early full4K boundaries, four actual VLM/16short ID controls, fresh805adapted screen and all124serial4KID controls precede all124warmup/timed streams. Common streamed individual IDs remain unobserved; no reconstruction from text. Strict same-engine text mismatches remain failures requiring diagnosis; never omit/cherry-pick those requests or claim quality parity.

Read actual installed vLLM config/router source declarations through CPU-only WSL commands, without model/torch/GPU imports or source edits. Structured `/server_info?config_format=json` is supported by the exact installed router. Saved1809byte schema/hash record `vllm-fixed4k-config-schema-attempt1.json` SHAa8b9431b47317d00341945e79ad5110ae4e95afba77735a6b32839e6f56a4229; five installed source hashes cover model/cache/scheduler/compilation/router. One CPU inspection reported a systemd root user-session warning but returned0 and source data; no systemd repair, reference server or package change was made. Startup metadata collector separately requires actual JSON/version, pinned checkpoint/modelopt_mixed/BF16KV/F32SSM/TP1/graph settings, reported resolved capacity for at least eight actual full4Krequests, exact named installed sources and owned PID. A configured4GiB budget alone is not admitted as memory fit. Existing historical short-prompt cache-capacity observations remain historical, not a new matched4Kbenchmark.

Final stable43CPU controls passed in1.807s. Every new boundary/probe/owner/raw-checker/owner-test file has an exact recorded derivation; stream timing/count/text replay arithmetic is unchanged. Independent checker adds actual config/source/version/boundary and raw cache-reset/counter replay. It retains all protocol, actual quality/ID/rawSSE/count/usage/metrics/cohort memory/gap and verified owned PID shutdown checks. Preparation pins782files and the actual replayed native/SGLang ordinary parents. Absent-only bundle contains14new targets; existing launcher/corpus/checkpoint/runtime/source/SDK are not overwritten. See `evidence/vllm-fixed4k-preparation-closure.json`, CPU ledgers and exact install/package/derivation methods.

Two predeployment corrections are preserved: initial expected SGLang proof-status spelling differed from the actual immutable schema; corrected to the exact actual status. A local copy/preparation overlap produced a superseded unrun preparation; cancelled only the identity-checked owned local copy process, observed its tool handle terminal, retained initial source/preparation/plan/append-only CPU logs, then regenerated and tested sequentially. No GPU, remote installation or reference server occurred under that preparation. Final preparation and bundle hashes in closure, not the superseded manifest, are authoritative.

The original `paged-batch8-model-preservation-attempt1` supervisor was re-polled live. Latest partial report contains five cases with `inverse_permutation_exact=true`: all three short-mixed orders and first two full4Ktext orders; no reported error. Probe leaves its status string `constructing` until final completion, so partial progress is read from cases/actual process, not that string. These are partial ordinary-model preservation results, not terminal admission, serving memory savings or timings. Do not install/upload/start another GPU/reference job until the original supervisor is terminal and cleanup/source/runtime are checked. Tiled HTTP and B8 MTP pipelines remain separately prepared/unrun. Matched full vLLM baseline execution/replay, strongest MTP/tuning, B8 external math/MTP/805quality/HTTP, repetitions, INT4/4B and overall2x/halfTTFT remain open.


### 2026-10-25 — isolated T128 commit compiles offline for SM120; independent B8 reference selection guarded

The existing isolated B1/T128 final-prefill commit CUDA source compiled successfully to a private SM120 relocatable COFF object using local nvcc13.2.78/MSVC19.29, explicit `-arch=sm_120`, and CPU compiler tools only. Dry-run stage commands retained; no executable was produced or run, no Garnet runtime/GPU query/context was loaded, and no SDK or production build tree was modified. All three input source hashes and eight selected SDK tool/header hashes were checked before/after. Existing source isolation from the default CUDA glob remains unchanged. Object SHA4bd87f1c7f474a81e1ff26817308942c57a6b627b1584c1b541609e51082b7af. Compiler reports18registers for KV scatter and14for linear copy, with no spill; these are compiler metadata, not GPU timing or occupancy measurements. CUDA header C4819encoding warnings are preserved without editing vendor files. Remote runtime uses CUDA13.0 and must compile/qualify separately. No T128 component execution, full forward-operator implementation, model math/memory fit, HTTP or speed admission exists.

Retained a bounded archive containing the object/PTX/cubin/fatbin, exact recipe/dry-run/object headers/command ledgers/result and per-file hashes; vendor preprocessed headers and broad compiler intermediates stay private and are not committed. See `evidence/prefill128-offline-and-b8-reference-selection-closure.json` and `prefill128-commit-offline-sm120-object-evidence.zip`. This advances compilation readiness only; fewer prefill calls are not a measured speedup.

Added a pure CPU B8-row selector for the existing strict independent B4 FP32-cache/BF16-value reference validator. It validates every B8 row and all128components before selecting four distinct physical rows, preserves actual packed106823680/F32-reference173932544per-row geometry, and checks complete raw finite hidden/normalized vectors. Duplicate/boolean/out-of-range mappings, unselected-row malformed metadata, slot/component/global hash changes, changed raw tails/NaNs and fake reference layouts reject. The original B4 validator/native profile guards remain unchanged. Nine synthetic old/new gate controls passed; a durable source-pinned ledger and CPU exercise validate29actual retained B8 state witnesses/58four-row projections/116raw-vector projection checks. Those reference sides are subsets of the same B8 data, not independent model outputs. Fresh same-input B4 full-cache reference execution is still required; no B8-vs-B4 numerical equivalence is claimed.

Original `paged-batch8-model-preservation-attempt1` supervisor remained live. Latest partial observation showed six cases with exact inverse permutations and no reported error: all three short-mixed orders and all three full4Ktext orders; image orders remain pending. A read-only sample reported adapter usage28313071616bytes during correctness, not serving peak or memory-saving evidence. Do not run a competing GPU job/upload or restart on silence. Prepared tiled HTTP, B8 MTP and matched vLLM4K trials remain unrun. Overall2x/halfTTFT, actual strongest matched reference baselines, external B8 math/805quality/HTTP, repetitions and INT4/4B remain open.


### 2026-10-26 — same-input full-cache B4 reference producer/checker prepared offline

Added a fresh ordinary B4/T32/T4 full-FP32-cache/BF16-value producer for actual closed paged B8 inputs. It retains the existing native profile guard and constructor (`retain_linear_inputs`, deferred commit, shared output and mapped embedding enabled; `bf16_kv_float_reference=true`), explicitly requires no compact commit/delta layout and actual173932544word physical canonical rows, and uses the existing full-cache roundtrip witness. No native/CUDA/default profile is changed. Same B8 actual request/image/prompt bytes split by physical rows0..3/4..7 produce18fresh B4cases covering original/rotated/reset text/image inputs and fixed8short/128full4K output semantics. Explicit two-half empty/final-reset captures plus18x3per-case captures yield58complete reference states and116raw hidden/normalized vectors. Every output ID, complete expanded state component, logical hash, slot and finite raw vector must match the retained B8 row selection exactly. The comparison is specifically independent cache-layout arithmetic within Garnet; target projections/linear operators remain shared, so no independent-framework or broad model-quality claim.

Thirteen synthetic producer-AST/old-new gate/malformed metadata/shared-drift/vector/row-map/boolean/count controls passed in7.561s under a durable source-pinned CPU ledger. An additional CPU checker exercise used actual retained B8 bytes with explicitly synthetic F32-reference metadata; no new B4 model output was generated and no numerical equivalence is admitted. Actual preparation/owner/GPU run and two-machine archive replay still require current terminal/replayed B8 closure; the producer refuses to construct without that preparation/parent. See `evidence/b8-external-reference-probe-cpu-closure.json` and raw CPU logs. The old real full-cache B4 corpus proof retains its actual historical library97c36465 identity; it is not relabeled as current1a1d. New producer must run fresh on current baseline.

Latest read-only original B8 observation contained all nine cases with exact inverse-permutation checks and no model error. At that observation, the original supervisor remained live, workflow status running, exit code absent, cleanup false and no archive yet. Final reset/complete replay/cleanup must still finish. Last image-order2 reset/prefill/final raw vectors were observed as163840bytes each; those files do not establish owner completion. Do not restart or run another model based on partial reports. Once original terminal/source/runtime/cleanup are observed, run the existing strict paged-B8 archive verifier on both machines, then start the prepared matched ordinary vLLM4K baseline. Tiled HTTP/B8 MTP, fresh B4 reference, wider-prefill GPU correctness, strongest tuned/MTP baselines,805quality/HTTP/repetitions/INT4/4B and overall2x/halfTTFT remain open.


### 2026-10-25 — paged B8 model preservation closed; missing vLLM launcher fixed, fresh attempt2 started

Original paged-batch8-model-preservation-attempt1 supervisor terminal0. All nine
actual short/full3968+128 text/image and row-order cases preserve the retained
ordinary B8 baseline: 29 complete canonical states, 58 raw finite hidden/normalized
vectors, all input/output IDs/text and inverse permutations, plus final reset.
6336 generated IDs total. Remote 26 CPU controls pass (exact raw elapsed retained).
684 source pins, six runtime dependencies and 15 SDK/environment entries unchanged;
owned worker gone/cache retired. Archive3bbec8c568cfcf5d7a1fdf7405b152b14593c2117cd4b4307094ddf5def00f76,
7002650bytes/92files. Independent local/remote original-checker replay exit0;
identical proof e87316f391f85419648842045f245495d2ea871e26614a3478c073e7b5aa5306.
Freshly recomputed 95302 resource samples: peak30854819840bytes (28.73579GiB),
median gap.031640600005630404s, max9.206485100003192s, requested interval.02s.
This equals the old B8 whole-study sampled peak: no serving-memory saving or speed
claim. Bounded input now preserves actual model arithmetic; B1/B4 independent math,
MTP/805quality/HTTP performance still require fresh execution and admission.

Matched vLLM ordinary attempt1 terminal1 was a setup failure: 43 CPU controls
pass1.903s, WSL bash exits127 because serve_vllm_fixed4k.sh was absent from the
deployed copy AND omitted from the prepared file manifest. No model ready, quality
or benchmark requests. This was our deployment dependency omission, not evidence
of model incompatibility. Archive43e1d905ebecf8ae4f56cbde3e5a928b9854c860c009d7ba9110c004616ded86,
70565639bytes/788files. Independent original checker exits1 as expected on both
machines; identical failure proof699d13a9c8caa08400172ebebc36e0058a18cc68e2b9885849d19822abfba8ab.
Cleanup/source/runtime preserved. Original manifest and checker remain byte-exact.

Corrected preparer explicitly requires launcher plus all named runtime/test-fixture
dependencies and valid hashes before admission. Added missing-launcher regression,
every-dependency/hash faults, LF recipe and exact unchanged replay-AST tests.
Derived separate immutable launcher-completed checker, keeping all raw protocol,
805quality/actual ID/SSE/cache/count/timing/VRAM/cleanup arithmetic unchanged.
Initial CPU-only completed manifest was superseded to pin the original preparation
fixture exercised by tests, before deployment; both versions and append-only logs
retained. Final47CPU controls pass1.854s; final preparation pins787files, SHA
dffbeaab7ef5885589246ec7d0392a71de1f14f5e0b3d148917d457bc4d5df74.
Delta bundle a0fa1ceadf901ace3b1586032e11db7228d643e2a007cf7252e5910c91ed252d,
45157bytes, six targets, exactly one hash-checked source replacement with backup;
13 existing targets preserved. Installer terminal0, all787source pins verified,
runtime/SDK unchanged. Actual WSL bash -n sees the launcher and exits0. Download
receipt ACKed before new GPU work. No benchmark during installation/syntax check.

Original48577 vllm-fixed4k-ordinary-attempt2 started under the completed preparation;
actual remote47CPUcontrols pass1.901s and original supervisor observed running.
Only this owner runs on the GPU; source/transfer freeze maintained while live.
Terminal startup/805quality/full124stream results and independent replay pending.
See paged-b8-and-vllm-launcher-closure.json and exact methods/raw ledgers/receipts.
No new performance figure from these correctness/setup results. Matched vLLM4K,
strongest tuned/MTP references, B8 math/MTP/serving, repeated trials, INT4/4B matched
comparisons and overall2xdecode+throughput/halfTTFT remain open.


### 2026-10-25 — actual closed B8 parent now bound to MTP trial; GPU execution pending

The unchanged MTP preparer now consumes the actual two-machine closed paged B8
ordinary parent e87316f3/archive3bbec8c5, not a hypothetical/synthetic parent.
All immutable archive members verified; 693source pins and actual parent/runtime/
SDK identities bound. Fresh30CPUcontrols pass12.566s. Preparation
d87e3d2bd123d449802b2be71fde229aee138c016e495b81aac725742413240b.
Absent-only ten-target methods packagef249f76305a0066668d7aa793ed6c662d11080ba31251cef4ef4a05e8a571b09
(48326bytes), installer6d89538cd39c35769e8a3385ee324621eac52718e702db05a1cd65aba3c3cc01.
Installer is an exact seven-substitution derivative of the previously executed
ordinary B8 installer, with parent/status/preparation/package/receipt and count
changed; same all-source/runtime/SDK and resolved-path/absent-target checks.
No remote upload/install/start while the exclusive vLLM reference owner is live.

Future actual B8 MTP trial uses existing depth3/T4/prefix-replay session with
128MiB, bounded128/two-page input and unchanged ordinary target arithmetic.
All nine cases, 29complete target states/58raw vectors and all IDs must match the
actual closed ordinary run; actual model/draft/vision fit is required. No natural
stop/805quality/HTTP or performance admission is implied by this preparation.
The independent B4 full-cache reference producer remains prepared separately;
shared operator numerical drift still requires that fresh reference gate.


### 2026-10-25 — complete fresh B4 cache-reference pipeline prepared; vLLM loading actual model

Prepared the exclusive owner/launcher/independent checker for the already committed
fresh same-input B4 full-FP32-cache/BF16-value reference producer. It preserves the
old B4 guards and ordinary target projections/linear operations, splitting every
actual B8 case into ordered halves:18fresh cases,58complete captures/116raw finite
vectors, all input/output IDs/text and reset/prefill/final state components required.
Only batch/cache representation is an independent reference here; shared operators
and B1/framework/official broad quality/performance remain unqualified.

Parent admission requires actual closed B8 archive/proof/state/vector/source/runtime/
SDK and cleanup identities. Eleven malformed parent transitions, falsely promoted
prior flags, exact lifecycle/checker/preparer/launcher derivation and mandatory new
method/test fixture dependencies are exercised. Fresh43CPUcontrols pass17.813s;
these include previous full state/raw-vector/input/row-map/padding/page faults.
Preparer verifies695source pins and actual closed two-machine parent, producing
fafaf5a3fa360bc65f8fcb60e259be6e1915b30db6e100f2766edca8191eb4a4.
See bf16-batch8-external-reference-bundle-preparation.json for exact12target absent-
only package and installer hashes. Derived owner retains exclusive idle guards,
actual raw ledgers, memory sampling gaps, full source/runtime/SDK checks, cache
retirement and failure archive. Independent checker replays actual raw vectors,
128components and real parent bytes; synthetic fixtures never enter result replay.
No B4 model executed or math/performance admission from CPU preparation.

The original vllm-fixed4k-ordinary-attempt2 supervisor remains live after47CPU
controls. Actual startup source reports vLLM0.32.0/Qwen3_5ForConditionalGeneration,
4096max length, BF16KV/F32SSM, TP1, eight max sequences, decodegraphs1/2/4/8;
selected native NVFP4 CUTLASS/FP8 FlashInfer kernels and GDN prefill. Checkpoint
loading is in progress from existing shared9P directory, with no extra copy.
FlashInfer missing precompiled-kernel warning is retained; no package/kernel-cache
mutation or restart while this owner is live. Startup selection is not benchmark
evidence. No remote transfers/installations/second model during the reference run.
Matched timing/quality/replay and all remaining scope including2x remain open.


### 2026-10-25 — vLLM ready but metadata endpoint failed; offline pip dependency repaired, attempt3 started

Original vllm-fixed4k-ordinary-attempt2 terminal1. Actual47CPUcontrols pass1.901s;
all weights load, kernel tuning/decodegraphs complete and health returns200.
Before any quality/client/timed requests, /server_info?config_format=json returns
HTTP500. Actual server traceback is vLLM collect_env.get_pip_packages: missing
pip falls back to uv in a uv-created venv, but uv executable is absent from the
server PATH; run_and_read_all returnsNone and out.splitlines raises AttributeError.
This is an environment-collection dependency failure, not model inference failure.
Owned SIGTERM/shutdown observed wrapper0, source787/runtime unchanged, cleanup true.
Archive7008eb5e5f33fb6c03fe5332ae66c7ebbe5817633ceb608ab2818835058742ac,
70678418bytes/800files. Independent local and remote unchanged v2 checker exit1
as expected with identical proof72901a6925c48ad8cfb8c186f892c53d9fb3e286382039409c79d2ac9d7423f4.
First remote replay invocation omitted --job and correctly exited2 without a proof;
preserved the original model run and repeated only the CPU replay with explicit job.
First read-only WSL source inspection timed out; authoritative Windows process
inventory showed no remaining wsl.exe command before a fresh CPU source read.
Neither event restarted a model. Raw server trace and all ledgers retained.

Repair uses existing Python bundled ensurepip wheel offline, not a framework edit
or network install. Actual CPU-only repair terminal0 adds pip24.0 only. All198
existing distribution versions unchanged; five named config/router sources plus
collect_env.py and Python binary SHA256 unchanged. Raw before/after inventories,
commands/stdout/stderr and installer source retained in five-file archive
22fb323439ab993fd0ae67070421c2122ff304d3aae98a46da64e12210b513ac,
11161bytes. Downloaded archive metadata and all members independently verified.
An initial local hash assertion used an extra8 transcribed from wrapped PTY output;
it failed before any operation. Saved/downloaded authoritative remote metadata
instead and verified its JSON digest against the actual archive; no archive edited.
No GPU/model execution or modification of existing packages/runtime/SDK in repair.

After all transfer ACKs and replay/repair terminal observations, fresh ordinary
attempt3 starts under the exact same launcher/preparation/source787 corpus and
805quality/124stream protocol. Remote47CPUcontrols pass1.755s; original48577 owner
observed live. Actual configuration endpoint, cache-capacity/ID/quality/timing and
cleanup/replay remain pending. No benchmark or chart addition from setup results.

Prepared explicit native pre-import diagnostic wrapper: durable exclusive entry
file before importing Garnet/model, exact retained instrumented source and original
exception/traceback/SystemExit code, launcher stderr entry/exit markers. Four calls
around the two native import statements are the only changes to the original model
AST; ordinary worker remains byte-exact and __file__/argv/IPC preserved. Eight
CPU-only fault/AST/bootstrap controls pass.101s, source hashes/raw ledgers retained.
Fixtures intercept native imports, never execute actual local Garnet/GPU/model.
Actual embedded compatibility and profiler owner/probe/checker wiring/execution
remain open; original profiler root cause is not yet established by these tests.

All batch8 independent-cache/MTP/805quality/HTTP, wider-prefill GPU qualification,
strongest tuned/MTP references, full matched vLLM4K, repetitions, INT4/4B matched
comparisons and overall2xdecode+throughput/halfTTFT scope remain open.


### 2026-10-25 — explicit entry-logged profiler pipeline prepared; repaired vLLM reaches request controls

vllm-fixed4k-ordinary-attempt3 original48577 supervisor observed live. Repaired
actual config endpoint now succeeds; installed74reference source files/eight
checkpoint configs plus five named config/router source hashes and ownedPID350
verified. Actual resolved cache37774tokens/9.22full4Krequests, configured4096total
context, all explicit BF16KV/F32SSM/TP1/decodegraph settings pass strict admission.
All four VLM color/layout controls pass. All16short greedy text/image captures
generate128actual IDs. Workflow client_running; quality screening and subsequent
124serial4KID controls/124warmup-timed streams remain pending. No timings inferred
from startup/control duration, and no extra HTTP request sent outside the owner.

The pre-import logger is now wired into an explicit exclusive Nsight diagnostic
pipeline. Same actual retained ordinary3968+128text control and normal inner Worker
pipes, exact old request/model/generation/cleanup operations; only launcher/logging
and declared profiled scope differ. Actual old unprofiled control archive/proof
8104499a/9d5b3a3f and643source pins required; new654source manifest binds11new
methods plus preparation. Existing core/runtime/SDK/defaults are not replaced.
New embedded compatibility gate first runs old fake-IO trace fixture plus a fake
native-import exception under actual embedded Python; only then may collector
launch pinned Nsight2026.5 on the real control. Actual raw .nsys-rep/SQLite, twelve
unique HtoD markers, owned SM120 graph-node kernels/API/copies and complete original
IDs/host stages/model close/entry-source logs must replay independently. Source/
runtime hashes and owned cache retirement remain mandatory. A failure is retained
with original import/launcher/profiler logs; no diagnostic success from mere process
exit. Collected activity remains instrumented, not serving-speed/TTFT evidence.

Fresh21CPUcontrols pass.178s: existing10phase replay/3bootstrap controls,5entry
import/exception/exit/IPC/AST controls and3raw-entry/parent/derivation fault families.
Initial21CPU preparation/.252s did not reject boolFalse as entry ordinal0; before
any deployment or GPU run, preserved old verifier/test/manifest/plan, added strict
integer ordinal gate and boolean fault, reran full21controls and regenerated.
An attempted cross-volume Path.rename correctly failed WinError17 before changing
the source; verified the already copied snapshot hash and used native Move-Item
for that exact owned preparation file. Both unrun manifests and old source bytes
are retained. Final preparation
efcaf1f2fa19f7d4285798fec9769ad69e6d2d1535beb33ac93eca13680f549f.
Twelve-target absent-only bundle63e74e977fae7bcf31bcfe2e3eab7ba45b7dbe0905f1efa41f48594376dba793,
49577bytes, installerf6ed03293e09b3bd37f2fda61508886f13049d4b1052c4846f1547ef85940806.
No remote transfer/install or second GPU model while vLLM owner is live. Actual
embedded/profiler execution, independent closure and full text/image ordinary/MTP
phase diagnosis remain open. This is a prerequisite diagnostic, not a substituted
benchmark or completion of the broader optimization/2x objective.


### 2026-10-25 — vLLM4K strict concurrent-text failure closed; native profiled control started

Actual vllm-fixed4k-ordinary-attempt3 owner terminal1, cleanup/source787/runtime6
preserved and owned wrapper exit0. This trial passed ready configuration/source/
PID/context admission, both early4K boundaries, four image controls,16short greedy
captures,805adapted screening and all124actual nonstream3968+128ID controls.
Strict benchmark collected textC1 then rejected the first textC2warmup because
stream text differs from its serial control. Original strict checker/rules remain
unchanged; preserve failed archive, do not publish one passing cohort as a full
comparison. Archiveabeebf29f6e7527bc7f3f652c6f0850a2ea309308c79419be88e2d9318130cb7,
102606666bytes/3655files. Both original independent checkers exit1 as expected,
identical failure proofc9fa4e365329a5a3ad7387b854e252a23cd57121c63fb75839a6f8d7cff2cadb.

Separate source-pinned CPU postprocessor independently replays actual ready JSON
settings, all124serial prompt/completion IDs, every12saved raw SSE/request/result
stream, typed continuous usage/full128tokens and the retained first cohort's raw
arithmetic. All12streams have valid3968+128length completions; two C2warmup texts
differ. Streamed individual IDs remain unobserved. Both machines exit0 with
identical partial proofc10239956b52f1749cf91b14638a03b45bfcee5a002b67d40f5d495c0c41aa1e.
This neither explains the numerical cause nor proves harmlessness/equivalence,
and never promotes a partial benchmark. Postprocessor SHA/source and logs retained.

Actual805screen raw response/wire/prompt-ID replay independently completes on both
machines,488/805correct; identical proofea52ee42fb5e5a54bd2fad9e1295de2e7c1a4adc91a12dff6c5c0e12824ac633.
This is the adapted805question/single256pixel image screen, not official MMMU or
broad quality parity. Initial direct-file remote CLI invocation failed relative
package import before creating an output; reran only the CPU verifier with -m
from the owned repository. Full original model trial was never restarted for a
replay. The new partial checker was first uploaded as an exact results snapshot,
then copied absent-only to its intended methods sibling location so imported/helper
hash paths resolve. No existing source overwritten. An initial local derivation
command used the repository cwd for a chat-work relative path and correctly failed
before creating code; reran by absolute path before tests. All later commands use
explicit absolute paths for chat work and repository targets.

Prepared engine-specific vLLM288request/135wave output/isolation client and raw
replayer by exact recorded substitutions from the already qualified SGLang methods:
actual vLLM ID JSON schema,18080/4096config/metadata-version and actual failed3parent.
All serial-before/after, C1/2/4/8, rotated nonstream IDs, full4K counts and concurrent
color/spatial isolation remain. Six synthetic CPU tests pass11.111s, including all
288local HTTP fixture requests and missing-row/overlap/count/color faults; no real
vLLM characterization requests executed. Owner/preparer/closed verifier and actual
two-machine characterization remain to implement/run. No observed-output benchmark
policy admitted from this failure or from SGLang results. Never silently relax
the original exact-text gate or infer unobserved streamed IDs.

Prepared combined actual4K chart renderer for five real series (native ordinary,
MTPprefill4/16, ordinary vLLM/SGLang). It requires a complete8cohort/124stream
two-machine proof and raw archive hashes before importing plotting or emitting
any output. Actual failed vLLM2proof is correctly refused in source-pinned CPU
control, exit1; previous published chart unchanged. No placeholder/synthetic chart
or partial vLLM3timings inserted. An observed-output figure, if later qualified,
must explicitly retain its distinct policy and output-equivalence limitations.

Native entry-logged profiler package installed absent-only after vLLM3terminal:
643existing/654total source pins and runtime verified; twelve new targets, no core/
runtime/SDK/model replacement. All transfers ACKed, remote/local CPU replays closed,
then original48577 native-cuda-entry-profiled-bootstrap-attempt1 started. Remote
21CPUcontrols pass.226s; actual embedded Python fake-import/fake-IO compatibility
passes. Known optional hashlib algorithm warnings retained; sha256 works, no SDK
repair attempted. Owner is proceeding to the pinned Nsight/native control; actual
CUDA records, terminal cleanup and raw replay are pending. No new performance
result or2xclaim from this correctness/setup/partial data.

Next: close the real profiler diagnostic and inspect actual GPU phases/failure
location; finish/run vLLM output-isolation diagnosis before a fully retained
observed-output baseline; execute fresh B4-cache/B8-MTP admission, then batched
HTTP with complete805quality/full4Kprotocol checks. Wider-prefill128 still needs
actual component/forward/runtime/model qualification. Strongest tuned/MTP
references, repeated statistics, matched INT4/4B and overall2xdecode+throughput/
halfTTFT remain open. The objective and branch scope are unchanged.


### 2026-10-25 — profiler ready-event transport failure located; paired CPU diagnosis and isolated writer

Actual native-cuda-entry-profiled-bootstrap-attempt1 closed exit1 with cleanup,
654 source pins/six runtime dependencies/baseline library preserved. Native imports,
decoder/vision construction and trace binding succeed; native_ready_observed_before_pipe_emit
is recorded. The first ready-event emit fails in original native_worker_io.py
os.write(1,raw), OSError('write failed'). No model request ran. Retain original
traceback, entry-instrumented source, construction-only Nsight report and raw logs.
Archive 762a1b928d964969c164baec301d2094c34038bd04194fb3ff04973f51a031ae,
68287144 bytes/684 files. Both unchanged checkers exit1 as expected; identical proof
a72e26c27277036a59d5480a37738f2e7832bafbeba071dfe984c0bc4d353426.
Construction CUDA activity does not establish request-phase bottlenecks or timings.

Actual embedded CPU-only paired experiment uses ordinary and pinned outer Nsight
process trees, identical stdin/frame probes and raw pipe captures. Ordinary: all
seven operations succeed. Profiled: raw descriptor stdout/stderr and fdopen1 fail
with errno9; sys.stdout/sys.__stdout__ writes reach the captured pipe and stdin
reads the exact JSON. Runtime6, source2 and profiler hashes preserved, both child
processes and owner exit0. Archive 58c2d74880076137e9dba4e2bc55898aaca611c319b0ba3fc32c965b1f4400d8,
24186 bytes/20 files, every raw member and observation independently replayed
locally. This is actual CPU evidence, not a model run or a two-machine equal-proof
claim. End-of-process text capture does not establish prompt live delivery.

Added diagnostic-only native_profiled_worker_io: original request parser, JSON
ASCII serialization, text normalization and partial-write checks retained; an
explicit callback uses sys.stdout.write/flush. The new model worker differs only
in that transport import; original production native_worker_io/model/runtime/SDK
remain unchanged. Same retained ordinary3968+128 request is preloaded then stdin
closed; all130 ready/token/completed events, exact3968prompt/128output IDs, original
12 CUDA markers/model-close and pinned raw Nsight SQLite must validate. No arrival
timing or HTTP performance admitted. This bypasses possible profiler text buffering
only for diagnostic collection and does not repair or change the serving path.

Initial27CPU/.229s interactive preparation was never installed or run. Preserved
all initial methods/manifest/plan before switching to a preloaded control and
adding the seventh protocol fault test. Final28CPU/.216s pass; all initial and
final ledgers/recipes/source derivations retained. Final preparation
5eb1ab9180c010f3097a78c599feb6595149bde674f8394892ec84db3d328363;
bundle 9ecb251b6fd14250346cedf6b913d97c5886f968b80c33244a0674e990cf4a14,
65629 bytes/12 absent-only targets. Actual install exit0 confirms654existing/
665total pins and runtime preserved. Installation receipt downloaded and verified.

Original48577 supervisor started native-cuda-stdio-profiled-bootstrap-attempt1
after prior jobs and transfers closed. Actual remote28CPU pass .194s and embedded
fake-import/fake-IO compatibility succeeds; optional hashlib warnings retained.
Owner observed running; source frozen, no second GPU model or transfer. Actual
request/GPU records, terminal cleanup and independent raw replay remain pending.
No benchmark/chart changed. A read of a nonexistent retention helper filename
failed harmlessly; existing helper was found before creating this distinct recipe.

Next: close/replay profiler data and use actual GPU phases to choose the native
prefill/throughput change; finish vLLM288request isolation diagnosis before a
distinct fully retained observed-output benchmark. B4-cache/B8-MTP math/model/HTTP,
wider-prefill128 actual GPU/forward integration, tuned/MTP references, repeated
statistics and matched INT4/4B remain open. Overall2x decode+throughput and halfTTFT
unmet/unproven; no reduced completion objective.


### 2026-10-25 — first real CUDA request-phase trace closed on both machines; vLLM output diagnosis

Actual native-cuda-stdio-profiled-bootstrap-attempt1 completed exit0. All665source
pins/runtime6/baseline1a1d preserved, cache retired and owned worker/collector/owner
closed. Remote28CPU/.194s and actual embedded compatibility passed. The preloaded
ordinary text request has exact3968prompt/128completion IDs, all130protocol events,
twelve unique HtoD markers, original host-stage/native close/entry source and real
owned SM120 CUDA graph-node kernels/API/copies. Archive
b9ce1450d7034abb266d44b41766ff4879c313be064c9f063756326108c85f30,
109282384bytes/700files; identical two-machine replay proof
d313d2c5c07a400a8e8794a13fb0791b8133f6220f86e592e9faeb63fe5a75f6.
The original ready-stdout failure and paired CPU observations remain immutable.

Instrumented marker windows: prefill5098.506808ms, observed owned CUDA union
4471.493237ms and234490kernel records; generation3471.777583ms, observed union
2863.552330ms and236855kernel records. Whole request8865.800824ms/7404.137426ms
CUDA union. Prefill kernel clipped sums include NVFP4 GEMM1054.070955ms,
BF16 attention scores966.525308ms/context526.422004ms and GDN register-state
483.142945ms. Decode includes NVFP4 GEMM1208.725643ms and FP8 GEMM569.280279ms.
These are one profiled ordinary text control, not uninstrumented HTTP TTFT,
decode tok/s or a code-change speedup. Kernel/API durations can overlap; use
interval unions where additive duration is needed. Gaps without observed owned
CUDA are unattributed, not evidence of a CPU/GIL/scheduling cause. Construction/
correctness sampled adapter memory is not a serving peak or memory-saving claim.
Raw source/entry/host/protocol/stats/ledgers/manifests and all failures retained;
full .nsys-rep/SQLite/VRAM samples remain in hash-pinned output archive.

Actual trace prioritizes wider target prefill128 and attention scores/context
qualification, plus batched native/MTP to address concurrency throughput. This
does not prove a particular optimization wins. Existing T128 is an offline
component only; actual GPU canaries/oracle, full forward/runtime/state/model and
HTTP remain required. Existing tiled-attention model correctness does not supply
serving metrics. Full MTP/image phase controls and repeated trials remain open.

Prepared separate vLLM output/isolation owner from qualified SGLang ownership,
using the actual vLLM launcher/ready metadata/source/config/PID/stop machinery.
Full checker derives from strict vLLM checker, replacing benchmark/805screen block
only with raw288request/135overlap-wave diagnosis:192actual nonstream IDs,
64full3968+128streams and32mixed-color/spatial streams, serial-before/after and
rotated C1/2/4/8 controls. All variants retained; streamed IDs remain unobserved.
Actual strict trial3 failed proof/partial12stream+124IDproof/805screenproof and
archive are required and independently equal on both machines. The diagnosis
does not weaken strict benchmark gates or admit an observed-output policy.

Source review before any diagnosis requests found two inherited dependency issues.
First, the strict checker's linked-reference loop overwrote the native parent_raw
binding used by later metadata admission. New diagnostic checker alone now keeps
linked_parent_raw separate; actual-loop CPU fixture uses distinct proof bytes.
Original strict checker/failure proofs unchanged. Preserved original unrun57test
preparation before this fix;58tests passed11.270s. Second, inherited SGLang proof
was in parents but absent from archived source list. Before requests, preserved
installed801pin/58test preparation and added explicit proof/archive pair admission
and pinning, with missing/drift regression controls. Final59tests pass10.693s,
802pins; preparation d7680e2eacab12eb1cdb8b941b1ca2615545db52cb63f0cd9a870a6af3b93c18.
Initial57tests11.143s and all revision snapshots/ledgers/recipes retained.

Initial absent-only vLLM diagnostic install added11targets and verified unchanged
partial checker,801pins/runtime. Narrow dependency completion backed up all three
owned unrun files before exact old-hash/new-hash replacement; original preparation
and install receipt preserved, all802pins/runtime verified, GPU unrun on install.
Final delta7236e11f7ffbcbb61c7ff253dc3c65f3800504003c2ff95fe4ef6f0c6eaf9697.
Only after profiler transfer ACK, local/remote replay closure and all install
receipts, original48577 supervisor started vllm-output-characterization-attempt1.
Owner observed live; no other GPU model/transfer/replay scheduled concurrently.
Actual diagnostic requests/terminal/source/runtime cleanup and two-machine replay
remain pending. No new HTTP benchmark, comparison chart or2xclaim.

Two read-only source searches used nonexistent guessed paths and a Windows glob;
they failed without changes, then rg --files supplied actual paths. No local GPU
or SDK/production build mutation. Private external-prefix probe remains untouched.
Next actual diagnosis must close/replay before a separate fully qualified vLLM
observed-output baseline. Native wider prefill/batched correctness and HTTP,
strongest tuned/MTP references, repeats, matched INT4/4B and overall2x remain open.


### 2026-10-25 — vLLM output/isolation diagnosis independently closed; T128 commit full-state GPU component

Actual vllm-output-characterization-attempt1 completed exit0; ownedPID372 stopped,
wrapper exit0,802source pins/runtime6 preserved, cleanup finished. All59CPU controls
pass12.687s. Actual ready metadata verifies74installed vLLM sources/eight checkpoint
configs and five config/router files; resolved37774cachetokens/4096context, same
ordinary settings. Actual288requests/135overlap waves collect192nonstream actual
IDs,64full3968+128streams and32mixed-color/spatial streams. Preserve105nonstream
output variants and52stream text variants. Every24serial-after text/image/color
control restores its serial-before IDs/text exactly; all48concurrent nonstream/
24concurrent streamed color/spatial observations pass. This does not establish
numerical cause/harmlessness or cross-engine quality equivalence; individual
streamed IDs remain unobserved. No timings promoted from this diagnosis.

Archive2cf6104c43b63f7770edb4329edc8d198dee0bb875a3bc1ccbacf5d313e3223e,
72278003bytes/1768members. Both raw checkers exit0 with identical proof
4be4865b75a2b75f020f4371f072ef31153327bb72fef27d3c70aec6ca389f3b.
All archive/method/config/PID/parent/stream/ID/color/overlap/source/runtime/cleanup
evidence replayed. Strict trial3 failure remains immutable. A separate declared
vLLM observed-output benchmark can now be prepared using this exact profile,
fresh805screen/124serial4Kcontrols/124rawstreams and retained variant sidecars;
its complete eight-cohort proof is still required before changing comparison charts.

Added isolated T128 final-state commit GPU harness because the actual ordinary
prefill trace identifies substantial GPU prefill work. It invokes only the already
unwired commit component, not a widened old runtime/parser guard. Full canonical
106823680word body plus32word guards at each end, full44564480word target plus
guards, all48FP32 GDN states and16packed-BF16 KV layers. Sixteen valid cases include
empty/partial/full spans, head/page/end boundaries and three replays of one captured
two-node graph. Thirty-five malformed pointer/alignment/overlap/overflow/profile/
span/unused-row cases must reject without writes. Input fixture includes signed
zero, subnormal/tie-to-even BF16 cases and varying token/head/layer data. C++ CPU
oracle compares every actual canonical word and target readonly per call. Full
canonical outputs and input/final targets are stored in a lossless repeated32word
block codec, not hashes/samples. Independent NumPy replay must reconstruct and
compare all51complete canonical states; per-call target readonly checks are
reported by source-pinned C++, with independent complete input/final snapshots.

Eight CPU layout/codec/BF16/untouched-slot tests pass.025s. Local explicitSM120
offline probe object compilation succeeds, object
c4b99a04335ab6234d092df9f53eb52774d7476124818eefc487210660e8b918.
No executable linked or run locally, no GPU query/context/model load, no SDK/
production build changes. Known code-page936 CUDA-header warnings retained;
eight SDK file hashes unchanged. Final component preparation adds closed-reference
and complete-case controls:10tests pass.148s, then.089s after explicit profiled
parent archive/proof binding. First preparation assumed namespace-package init
files existed and stopped before tests/output; removed those nonexistent files.
Preserve the initial unrun manifest/owner/checker/plan before parent-binding update.
Local/remote component ledgers use only explicit owned input inventories and
exclude unrelated untracked probes. No private external-prefix probe edited/staged.

Toolchain correction: earlier offline descriptions called the remote toolkit13.0
without a current version observation. Actual fixed remote nvcc reports13.2.78;
its pinned executable hash matches the local13.2.78 tool. Preserve original raw
compile result/descriptions as historical data, not as current version evidence.
Current owned observation retains exact version output and630754566144bytes free
on study drive. Remote component owner records its actual compiler/MSVC version
again and hashes SDK/compiler/runtime before/after; it does not infer compatibility
from directory names. Local and remote MSVC versions still differ and actual
remote rebuild/link/GPU qualification remains required.

Component24source-pin/10test preparation
8a08df7bb386465b7729a716367bae4837befa38045945a67512bb2d9c052fa9;
22target bundle515132ec707d7ea6a853debfe49cd698b9aa29a36fbe3ae1280059a157a0a932,
70071bytes. Actual absent-only install adds15targets and verifies seven existing
methods plus24source pins/runtime6/SDK15 unchanged, no GPU execution on install.
After all vLLM transfer ACKs and both CPU replays close, original48577 supervisor
starts prefill128-commit-component-attempt1 bound to the actual closed reference
archive. Owner observed live; build/link, actual51GPU cases and full raw replay
are pending. No second GPU model/transfer/replay while this owner runs. Fixed
private tools/bf16-kv-native build reused; existing binaries/objects are never
overwritten. Full T128 forward operators/runtime integration/model/805quality/
HTTP, B8 math/MTP/HTTP, strongest tuned references/repetitions, INT4/4B and overall
2xdecode+throughput/halfTTFT remain open. No chart or benchmark changed.


### 2026-10-25 — T128 commit actual GPU cases and remote raw full-state replay passed; vLLM observed client ready

Actual prefill128-commit-component-attempt1 owner exit0 and cleanup completed.
Remote10CPUcontrols pass.094s; actual compiler13.2.78/MSVC14.44.35207 builds/links
standalone SM120 executable. Actual GPU probe completes16valid,35invalid cases
and3replays of one captured two-node graph; every complete canonical state,
canary and per-call target readonly comparison passes C++ CPU oracle. Source24,
runtime6, inheritedSDK15 and eight directly observed SDK files remain unchanged.
Garnet library/runtime was not loaded or replaced. Archive
2f28fd531f0d01ffd8c267be355d8a9102cd626f8c7aa9f369b78ebc820d0eea,224386502bytes/90members.

Remote independent NumPy replay completes exit0: all51actual canonical outputs
reconstructed losslessly,5448010944uint32words checked against declarative layout
and integer BF16 round-to-even oracle. Full target input/final snapshots exactly
match the independent fixture;51per-call readonly checks remain source-pinned
C++ observations. All16valid/35invalid/3graph cases, guards, archived methods and
parent/source/runtime/SDK/owned cleanup pass. Remote proofa892c87a48b64fb1a82d7491b8d3078e2197f7efbaa7a14b93ff1c94225943d9.
Correctness-only sampled adapter peak1631006720bytes,367samples, median gap
.031640549990697764s/max.035658600012538955s. This is a standalone component,
not model/serving memory or speed. Full T128 forward/runtime/model/HTTP remains
unimplemented/unqualified. Local224MBarchive transfer95279 remains live; no
second-machine full-state closure or performance claim until transfer ACK/hash/
same-checker replay. Preserve existing original handle; never restart for a wait.

Prepared separate vLLM observed-output policy/benchmark/probe by recorded exact
derivation from the already exercised SGLang methods, using actual vLLM JSON ID
schema and vLLM4K boundary controls. New policy record_vllm_stream_text_variants_v1
requires this engine's actual closed288diagnosis, serial restoration, color/
spatial/overlap coverage, every archive/source/PID/cleanup proof, and stable
vLLM0.32.0 settings/4096context/zeroreservedslots/37774cachetokens/config-source/
checkpoint/launcher identity. It does not inherit SGLang's4098/two-slot policy.
Original strict and SGLang code unchanged. Every later text variation gets a
sidecar after timing/sampling; counts, actual serial IDs, cache reset/counters,
full128length and raw stream accounting remain strict. No inferred streamed IDs
or numerical/quality equivalence. Direct admission of actual two-machine vLLM
diagnosis succeeds; this does not admit a benchmark result.

Nine CPU controls pass7.636s: closed-profile/source/context/cache/dtype drift,
serial-restoration and provenance faults, exact/changed text sidecars, strict
126token clamp rejection, plus synthetic124nonstream ID controls and124raw HTTP
streams spanning eight cohorts. Synthetic vLLM reset hooks assert controls precede
timing and all streams precede post-run counter check. Original protocol helper
AST remains identical; comparisons/hashes occur only after sampler closure.
These fixture timings/memory/requests are not real model benchmark evidence.
No actual observed-output vLLM benchmark requests. Owner/preparer/full closed
benchmark checker still need implementation before actual baseline collection.
Published comparison chart unchanged; overall2x/halfTTFT objectives unmet/unproven.


### 2026-10-25 — T128 commit full-state closure on both machines; matched observed-output vLLM4K trial started

T128 component224386502byte archive transfer finished and ACKed. Same independent
checker exits0 on local CPU and RTX5090host, identical proofa892c87a48b64fb1a82d7491b8d3078e2197f7efbaa7a14b93ff1c94225943d9;
archive2f28fd531f0d01ffd8c267be355d8a9102cd626f8c7aa9f369b78ebc820d0eea/90members. All16valid/35invalid full canonical states,
5448010944words and3replays of one captured graph replay exactly; full target
input/final and per-call C++ readonly observations retained as scoped. Source24,
runtime6/environment15/SDK8 unchanged, owner terminal0/cleanup. Complete raw RLE/
binary/build/CPU/GPU/resource data retained in output archive; selected logs,
workflow/cases/manifests/proofpair and recipe retained on branch. Component is
qualified for isolated final-state commit only. Full T128 forward/runtime/model/
quality/serving integration remains unimplemented/unqualified; no speedup claim.

Completed separate vLLM observed-output runner/preparer/full raw checker. Owner
inherits exercised SGLang lifecycle but uses actual vLLM launcher/metadata/port/
PID/stop, exact same ordinary0.32.0 profile as the independently replayed288request
diagnosis. Latest T128 component full-state proof/archive/current closed workflow
must pass before any next GPU owner, plus native fixed4K parent and observation
provenance. Archive/pins include all inherited proofs; require15named runtime
dependencies including shell launcher/metadata inspectors/new client/boundary/
cache/stop/helper files, not just Python module imports. Full source816/runtime6
hash checks and original exclusive idle guards remain mandatory. Local/remote
ledgers use explicit owned input inventories and exclude unrelated untracked files.

Checker preserves SGLang's exercised full805screen/124serialID/124rawstream/eight
cohort replay, replacing only engine-specific metadata/ID schema/port/boundary
and closure. Exact vLLM JSON/version/74installed-source/eight checkpoint/fiveconfig
and dual cache-reset/raw metrics replay fragments come from existing vLLM methods.
All output-variant sidecars and measured cohort exact/variant counts must replay;
no individual streamed IDs inferred. New result status is explicitly
reference_fixed4k_vllm_observed_profile_replayed, distinct from strict acceptance.
Original strict failure and SGLang sources remain unchanged. Source self-hash,
current replay-helper hashes, owned PID/start/stop, actual cache idle/reset/counters,
frozen bytes/full3968+128usage, metrics and sampled memory gaps remain required.

Initial new CPU owner import failed because an absolute sibling import did not
resolve under package unittest; corrected conditional package/CLI import before
deployment. Initial source-presence test expected encoder_hits_delta, whereas
the actual copied vLLM parser uses hit_delta and two explicit reset endpoints;
corrected the test to those actual invariants without weakening the parser.
Recorded every exact source transformation and final helper/source binding.
Four owner/dependency/derivation/closed-component control families pass. Clarified
an inherited control-report sentence that still described exact text matching;
new vLLM report explicitly records variants. This is a scope-text correction,
not a changed request, timer, token check or output-admission rule.

Combined72CPUcontrols pass20.597s, including old strict reference identity/length/
cache/owner gates, actual-source derivatives,288request synthetic diagnosis and
124ID+124stream observed-client fixtures/126token clamp rejection/closed component
and dependency faults. No fixture metrics promoted. Preparation
60d1b8208edcb5d8431c77d4443d912769e798f7273a97c15bac56752b835621; thirteen-target bundlede97148ffb523cd40b92cdec0e0b3d5ab2333379a8a370808ec526668054dd79,70653bytes.
Actual absent-only installer verifies all816source/proof pins and runtime6;
existing files require exact bytes and are never replaced. Installation receipt
downloaded/ACKed and checked before original48577 starts
vllm-observed-fixed4k-ordinary-attempt1. Owner observed live; all prior transfers,
raw CPU replays and component/reference owners closed. Source frozen, no second
GPU model/transfer/replay during this trial.

Planned real sequence: two full4K text/image boundaries, fourVLM/16short-ID controls,
fresh805adapted screening,124actual serial prompt/completion-ID controls, then
124warmup/timedstreams across text/image C1/2/4/8. Same pinned checkpoint,
modelopt_mixed/BF16KV/F32SSM,4096total context/no reserved slots,4GiB KV budget,
max8sequences/TP1/full decodegraphs1/2/4/8 and prefix/MM cache controls. Every text
variant retained with record_vllm_stream_text_variants_v1 after timing/memory
sampling. Actual metadata/controls/timings/terminal cleanup/full independent replay
are pending. Current chart and overall2x/halfTTFT claim remain unchanged/unproven.
Strongest tuned/MTP baselines/repeats, native B8 math/MTP/HTTP and T128 forward
integration, matched INT4/4B remain open. Objective is unchanged.


### 2026-10-25 — Complete matched ordinary vLLM 4K trial and five-series comparison

vllm-observed-fixed4k-ordinary-attempt1 completed with cleanup and preserved source/runtime pins. Raw archive eee2f5e3e699868e306f53eeadf63fd8e0cc6efa4585222bb9ac7e6ade62e696; identical remote/local full replay proof c67d7a4912a4a28f0956b2be918da6ea1474c55ea8c5c55860336f611d627eee. All 4151 archive members replayed, including 805 adapted screening responses (489 correct), 124 actual serial ID controls, 124 warmup/timed streams, eight measured cohorts and raw memory/cache/config/lifecycle evidence. The observed-output policy records every text variation; individual streamed IDs and long-output semantic equivalence remain unproven. Original strict failure remains preserved.

The new chart requires complete two-machine proof pairs and original archive hashes for native ordinary/MTP4/MTP16, SGLang ordinary and vLLM ordinary. It refuses missing/failed strict vLLM results before plotting; both real-file rejection controls passed. Actual 3968 input plus 128 output tokens on the same RTX 5090, text/image with client concurrency 1/2/4/8, eight measured requests per cohort. Native physical batch is one with queueing; reference servers admit up to eight requests. Separate single trials, no confidence intervals or broad cross-engine quality equivalence.

| Input | Clients | Garnet MTP16 decode tok/s | vLLM decode | SGLang decode | Garnet total tok/s | vLLM total | SGLang total |
|---|---:|---:|---:|---:|---:|---:|---:|
| text | 1 | 75.693 | 69.483 | 64.073 | 18.194 | 59.616 | 55.797 |
| text | 2 | 83.515 | 59.760 | 61.584 | 18.445 | 96.536 | 95.316 |
| text | 4 | 78.194 | 49.351 | 50.529 | 18.255 | 153.636 | 148.851 |
| text | 8 | 73.751 | 42.460 | 44.385 | 18.235 | 226.945 | 151.791 |
| image | 1 | 88.257 | 69.536 | 60.684 | 17.663 | 59.135 | 52.620 |
| image | 2 | 89.384 | 60.292 | 56.019 | 15.450 | 96.461 | 90.707 |
| image | 4 | 88.135 | 51.778 | 48.369 | 18.976 | 152.466 | 142.794 |
| image | 8 | 91.453 | 42.460 | 42.606 | 19.080 | 224.971 | 146.537 |

All eight panels (text/image TTFT, decode, total throughput and sampled adapter VRAM) and machine-readable data retained in chat outputs. vLLM ordinary and SGLang ordinary variation counts are {"SGLang ordinary": {"long_output_semantic_equivalence_proven": false, "policy": "record_stream_text_variants_v1", "serial_text_exact": 26, "serial_text_variants": 98, "streams": 124}, "vLLM ordinary": {"long_output_semantic_equivalence_proven": false, "policy": "record_vllm_stream_text_variants_v1", "serial_text_exact": 20, "serial_text_variants": 104, "streams": 124}}. Decode gains do not establish latency or end-to-end throughput gains. Overall 2x objective remains unmet/unproven; tuned/MTP references, repetitions, T128 forward integration, matched INT4/4B and wider native batching remain open. No new model execution by chart renderer; no benchmark invented or omitted.


### 2026-10-25 — T128 attention: full GPU outputs and independent FP64 replay on both machines

Completed standalone B1/T128 attention after the actual native CUDA phase diagnosis and closed matched vLLM baseline. Core and witness collector previously compiled to private SM120 objects locally, without linking or executing any local GPU/model/target executable or mutating runtime/SDK. Local CUDA13.2.78/MSVC14.29; actual remote build/link CUDA13.2.78/MSVC14.44.35207. Original offline ledgers, source/SDK hashes and object hashes retained. Geometry: 48MiB probability workspace; canonical bytes427294720; raw complete output3145728bytes per case. Production/default4/32 runtime admission remains unchanged.

Initial installation stopped in the preflight before writing any target because the remote production overlay retained a different version from the local optional tiled candidate. The exact downloaded source-hash observation is retained; no production source was overwritten. The candidate derivation source is now an exact inert study snapshot. Original unrun preparation/bundle/installer and corrected preparation are preserved. The corrected absent-only install added17new targets, kept11existing targets byte-identical, checked32source/proof pins, six runtime files and environment/SDK pins. Fresh11CPU controls pass0.173s locally and0.200s remotely; owner rebuilt and linked the standalone collector in the fixed private folder. No Garnet DLL loading/replacement.

Actual GPU collection passed36cases covering all16full-attention layers, empty/partial/full128 sequences, cache/page bounds, full4096end and128active queries. Complete plain/tiled outputs and probability workspace compared bit-exact in source-pinned C++; nine launches of three captured three-node graphs also compare exactly. Cached-prefix last-token invariance, complete canonical/small inputs readonly and output/workspace guards pass;256malformed actual invocations reject. These probability/readonly observations are checked by the source-pinned C++ collector; the corresponding probability/canonical buffers are not serialized.

Raw all786432F32 output values per case are serialized (113246208bytes total), including all24heads/all256dimensions and inactive exact zeros. Independent dense FP64 replay checks all28311552output scalars at ATOL3e-5/RTOL2e-5; max absolute error 2.2405192290353426e-07, max tolerance fraction 0.0047849146225023755. Both checkers exit0 against the same archive 22486615afd3cdbbaf75e7d27276abdb2cc828e73de51271f1a408ea08147934 (19590264bytes/45members). Local/remote proof hashes {"local": "a8df1c6f8a9516625be540733ee22cd70a01f24598083660fd1183475e969246", "remote": "1c573ec6ca2e6ec9915fc70f2e0e1f4431f0c01fa3c8e1ab28ac54cb278aea0f"}. Source/runtime/environment/SDK preserved, cleanup complete; original logs, manifest, proof pair, sampled memory and recipes retained. Sampled memory is correctness-only and does not establish serving memory savings.

The initial local reporting wrapper incorrectly required proof bytes to match, after the original full scalar checker already passed. The two actual proofs differ only in last digits of two per-case FP64 error summaries; all non-summary fields are strictly identical JSON, the archive and original verifier hashes match, and both original checkers independently accept all scalars at unchanged tolerances. Original wrapper retained. A separate comparison helper records every summary difference, permits only finite nonnegative statistics (fraction at most1), bounds statistic differences by1e-12 absolute error/1e-8 tolerance fraction, and rejects all structural/source/acceptance changes. Two CPU fault controls pass0.003s. Cause of tiny summary variations is not established; no GPU output or original proof/checker was modified or rounded to force equality.

This qualifies the isolated T128 attention component. It does not qualify full decoder forward, GDN, parser/plugin/state integration, model output, adapted screening or HTTP performance. Next integrate the qualified attention and final-state commit into a distinct opt-in T128 forward/runtime path, verify against retainedT32 complete model IDs/state, then repeat805quality and matched4KHTTP. Strongest tuned/MTP reference profiles/repetitions, native wider batching and matched INT4/4B remain open. Current comparison chart stays unchanged; overall2x/halfTTFT objective is unmet/unproven.


### 2026-10-25 — Fresh B4 same-input reference for B8 started after T128 attention closure

T128 attention GPU owner and both independent CPU replay processes completed; archive/proof transfers ACKed and both complete checks accepted before the next model owner. Isolated T128 component is correctness-qualified only; full-forward integration and performance remain open. Returning next to wider-batch serving qualification because current physical-batch1 queuing limits total throughput.

Revalidated the queued fresh B4 full-FP32-cache/BF16-value reference for the closed ordinary B8 input cases. Original model/vision requests, owner lifecycle and raw checker remain byte-identical. Added a source-pinned entry wrapper that binds run_logged.source_identity to the explicit expected owned inventory before calling the original owner, excluding unrelated untracked files. Two wrapper dependency/copy-isolation faults join all43existing controls. Original unrun695pin/43test preparation and package retained; new697pin/45test preparation and bundle e5386bb7cbdebf745193191ea24bc44a1fed25f64ca58356513a23c4580b71e4 (60192bytes/14targets). Actual local45CPU tests pass16.884s.

Actual absent-only installation verifies684existing sources, adds14targets, verifies697total source/proof pins plus runtime/SDK; receipt downloaded and ACKed before original supervisor48577 starts bf16-batch8-external-reference-attempt1 through the bound entry. Workflow is observed live. Planned model reference:18same-input four-row cases (halves of all9closed B8 cases),58complete captures/116raw vectors and all actual token IDs. The reference shares projections/linear arithmetic and isolates cache representation/batch geometry; it is not independent-framework/all-model/B1 equivalence. Source frozen, one model only, no transfers/replays/second GPU job while live. Raw failures, sampled resource gaps, cache cleanup and same checker on both machines required before any qualification. No new performance result, B8 MTP/quality/HTTP or overall2x claim. Strongest tuned/MTP references, repeated trials, T128 full forward and matched INT4/4B remain open.


### 2026-10-25 — Strict B8/B4 reference failure preserved; full T128 runtime candidate passes CPU compilation

Actual bf16-batch8-external-reference-attempt1 stops in the first short-mixed four-row half on generated_ids. All45remote CPU controls passed18.722s; strict owner exits1 with source697/runtime/environment preserved, cache retired, library1a1d unchanged and cleanup complete. Archive eaddf8987a88bfa37e422433c753399f42539bd378c1795fb8606e1cbc75e624 (680157bytes/37members). Same original full checker returns expected1 on both machines, identical failure proof dbdc67c1bb23f479f5cc7b039db0a0ac4e2743513133fc3fa3a4a63e3fb08735. Original strict gate is unchanged. Remaining17cases are unrun; B8 external cache arithmetic/MTP/quality/HTTP admission remains false.

Independent inspection of raw actual first-case data confirms identical inputs with prompt lengths41/104/93/93 and8outputs each. Output row0 matches; first differing output indices for rows1/2/3 are0/5/5. Reset complete logical-state hashes and raw hidden/normalized vectors match exactly. Post-prefill, the first six stored GDN component hashes (layers0/1/2, both roles) still match all four B8 rows; the first differing component is layer3 full-attention key, all four rows.122of128post-prefill components differ. Raw final-hidden relative L2 differences versus B8 are [0.17609598700399556, 0.643559669362636, 0.4944135110904901, 0.29034161993283875]. This is a boundary observation, not a cause, harmless-rounding or semantic-equivalence conclusion. Complete raw caches are not archived; hashes and raw hidden/normalized vectors are.

Implemented a separate local T128 native/Python forward candidate using byte-identical copies of the already GPU-qualified attention and final-state commit kernels. New T128-specific TensorRT plugin/wire type, strict bounded integer keyword, canonical binding, native commit class/API, Python layout/admission/program/decoder routing and27Bfactory. Prefill usesT128 while retained speculative decode remainsT4, one shared128NVFP4/208FP8 projection pool and canonical state. Old4/32andB4/B8 guard functions remain unchanged. Existing GDN source already accepts up to128tokens, but full T128 GDN/model equivalence remains unproven.

Eight new pure CPU controls cover layout, strict flags, complete64layer symbolic forward, no legacy KV slices, native capability/factory failures, copied kernel bytes and exact recorded source transforms; combined with18existing B1/B4/B8 regressions,26tests pass0.161s. Private CPU-only MSVC /c compilation passes for the new attention plugin, commit wrapper, actual modified TensorRT builder and actual app entry. Four object hashes, compile recipe, inputs and selected SDK before/after hashes retained. No linking, target executable, SDK/runtime/production build mutation or local GPU/model execution. Native/Python runtime candidate is local and not installed on RTX5090. Actual native DLL build, plugin/commit contracts, full forward/model IDs/state,805quality and HTTP performance remain required.

Next controlled diagnosis compares the same actual first short case with B4 full-cache and B4 packed-cache, sequentially on unchanged1a1d. Require full-repeat comparison against this preserved failed run and compare both to closed B8, retaining every state/vector/ID difference. This separates observed cache-mode and batch-geometry contrasts without relaxing the original failed strict gate or claiming a cause. Overall2xdecode/throughput and halfTTFT remain unmet/unproven; strongest tuned/MTP references/repetitions, wider-batch serving and matched INT4/4B remain open. Comparison charts remain unchanged.


### 2026-10-25 — Controlled B4 full/packed cache-mode characterization started

Preserved strict B8/B4 failure and both original replay processes are closed, with identical expected-failure proof. All old archive/proof transfers and CPU inspections completed before the next owner. New characterization keeps unchanged1a1d runtime/source and the exact first four-row short input case, then runs full-FP32-cache/BF16-values and packed-BF16-cache sequentially, with exclusive idle checks between processes. Each collects eight actual generated IDs, full expanded logical-state hashes and raw hidden/normalized vectors at reset/prefill/final plus empty/final-reset controls. Full-repeat versus failed-full, each mode versus closed B8 and packed versus fresh-full are independently replayed. Every difference is retained; no original gate relaxed and no tolerance, cause or quality/performance admission.

Exact source transforms from exercised owner/probe/launcher retained. New standard-library-only helper validates complete128component coverage/slots/rawF32hashes/finite values and reports exact counts/max differences, without NumPy or reconstructed IDs in the embedded runtime. Five faults/control families join the45existing controls; local50tests pass16.562s and actual remote50tests pass19.294s. Preparation/bundle source706, nine-target absent-only install, SDK/runtime unchanged. Bundle f8415071eb2109e98b18eb21c4d9d0f737f1ee5210d136704839426388a1c1b6 (48858bytes); install receipt downloaded/ACKed before supervisor48577 starts b4-cache-mode-characterization-attempt1. Exact observed handle remains live after CPU controls, constructing the first mode. No second GPU model, transfer or independent replay while live.

The full T128 forward/native candidate remains local only, CPU compiled and unqualified on GPU. Reconfigure the existing native build tree to collect its new globbed source files before the eventual preserved-DLL candidate build; do not infer runtime/link/model qualification from the four private /c objects. New50test diagnosis source/pins and current remote1a1d are frozen. Complete raw replay/cleanup and both-machine proof closure are required before another GPU owner or any performance promotion. Overall2x/halfTTFT objectives remain unmet/unproven; current charts unchanged.


### 2026-10-25 — Dual cache-mode failure replayed; packed-only retry and T128 wire CPU objects

Actual dual-mode owner exits1 after full mode completed0 and packed mode constructed an empty state but the old full-only representation control rejected it before any packed case. All50remote CPU controls passed; source706, runtime/SDK and1a1d library preserved, private engine cache retired and cleanup completed. Archive 2ba19852899ddcf9759bac284a59cebbeb9a38d26ea062123edcf5d0774bcf3c (783690bytes/41members), original expected-failure proof 25a55d425eb066b99cb2a064aca7bc569f6abf262ef4cc27324bf0959469ae51 exactly matches on both machines. Original gate and failed raw evidence remain unchanged.

New scoped partial checker independently replays the actual completed full case against both earlier failed full and B8. Full repeat exactly reproduces prior full IDs/state hashes/raw hidden and normalized vectors; it remains different from B8. Packed empty state is exactly equal to B8 under typed packed-versus-packed comparison, but no packed model case completed in that archive. Both-machine exact partial proof bdfd2ddb424eaed8d0c5d46c82e48778e8c28839a13db8a6f83991d8ed593bf1. Cause, harmlessness, arithmetic equivalence, model quality and B8 MTP/HTTP qualification remain unproven.

Prepared separate packed-only retry with correct typed empty/final-reset controls, unchanged request/decoder/model source and no relaxed comparison. Reuses the actual completed full companion via archived partial proof; added archive dependency fault control. Initial unrun52test prep/derivation retained, then explicit partial-proof archive binding fixed before deployment. Final53local CPU controls pass17.641s (logged process17.75s).716source pins; nine bundle targets, eight new/one identical, runtime/SDK unchanged. Bundle 8ddf968cd61c2cd98fade6d4e95e20a14c5a5f24d537216398ee7bb87abe7e6c (48976bytes); downloaded install receipt acknowledged before owned supervisor48577 starts b4-packed-cache-characterization-attempt1, confirmed running by live handle. Remote source and transfers frozen during live trial; no new performance result.

Separate T128 wire contract/stub sources pass private MSVC /c compilation for four objects, reuse the compiled new plugin object, no linking/target execution/local GPU or runtime/SDK mutation. Actual local toolchain14.29.30133, cl19.29.30159.0, WindowsSDK10.0.28000.0; benign NOMINMAX redefinition retained.16profiles,832bad wires,496descriptors,336formats,128cross-dialect rejections and96old roundtrips are planned executable checks, not runtime passes. Missing private fixture header was caught before compilation and replaced by explicit16qualified FA offsets. The future native runtime source union needs19files:18T128 targets plus the paired previous tiled-scores overlay implementing the API inherited by its entry. Audit retained; no candidate deployment yet.

Next: close actual packed case, cleanup and independent replay on both machines; use its observed cache-mode/batch-geometry contrast to choose the next correctness fix. Then preserved-DLL native T128 candidate build and plugin/model/805quality/HTTP qualification. Current verified charts unchanged; overall2xdecode/concurrency throughput and halfTTFT goals remain unmet/unproven.


### 2026-10-25 — Full T128 native build transaction and independent replay prepared locally

Prepared complete native T128 build owner using the fixed existing5090study/build directory. Thirteen source candidates comprise12native T128 sources plus the paired inherited optional tiled-scores overlay; six Python candidates remain separate for later model deployment. Owner saves every original byte, checks absent new targets, reconfigures existingCMake/Release/MT/sm120, builds actualGarnet, runs an isolated plugin CPU contract with stubs never linked intoGarnet, ten external+ten embedded selector checks and955native tokenizer checks, saves a separate candidate DLL and restores original sources/library even on build failure. Source/environment/runtime/parent hashes, failures and actual raw output are archived. This owner remains unrun remotely.

Five pure source transaction/archive controls cover existing+new restoration, unexpected outside edits, unrecorded sources/wrong base, partial installation and malformed source archive; two pure wire replay controls reject missing accounting, enqueue/quality promotion, foreign DLL/version and boolean counts. With49retained constructor/transaction controls,56local tests pass0.299s. Earlier54test inputs and both ledger event pairs retained. Source bundle a8e55d7679c9dcac401d81f3f77aeb3b4bab66a3603453c7543dc916d1314076; independent full archive verifier prepared, requiring baseline source/runtime/SDK restoration, actual native build and all CPU raw cases. No local native executable/link/GPU run, no remote transfer during live model, and no model or performance admission.

The sole packed cache diagnostic progressed from construction to reference_cases_running and saved one empty capture, with about22.7GB VRAM in use; actual case and both-machine replay remain pending. After terminal cleanup, bind current remote build/cache snapshot and closed proofs before deployment/build. Then full T128 model equivalence/quality/HTTP checks and wider batch arithmetic repair remain required; overall2x target remains unmet.


### 2026-10-25 — Packed B4 matches full B4; both differ from B8, complete T128 native build started

Packed-only actual four-row short8-output owner completed0,53CPU checks19.310s, full empty/reset controls and one complete case. Source716/runtime/SDK and1a1d library unchanged, cache retired, cleanup finished. Archive b6dafde2728730b8996907043f61be665f888b489f6de309c484f8cec1807900 (667361bytes/36members), original full checker returns0 on both machines with identical proof 2afe8281117495f8479712f0b536d3a159cc94e6573694439652875beb6cc8ce. All transfers acknowledged before new work.

Actual packedB4 and fullB4 generated IDs, complete expanded-state hashes and raw hidden/normalized vectors match exactly throughout this first short case. The full repeat also reproduces the original failed full case exactly; both B4 modes remain different from B8. This controlled observation narrows the batch-geometry/cache-mode contrast; it does not establish a cause, arithmetic/semantic harmlessness or B8 MTP/HTTP qualification. Original strict failure/gates and all prior archives remain intact. No new benchmark or chart admission.

After terminal cleanup, collected fresh idle remote snapshot: unchanged717baseline source pins (including closed packed prep), six runtime dependencies,15environment/tool/SDK pins, four existing and nine absent native candidate targets,32existingCUDA objects and actualCMake cache. Final56local CPU tests pass0.357s after adding precise CUDA object inventory checks; earlier54/56unrun inputs and ledgers retained. Deployment eight absent targets,717base+10method pins, SDK/runtime unchanged; downloaded receipt acknowledged before exact supervisor48577 starts bf16-prefill128-native-build-attempt1. Actual56remote CPU controls pass0.298s, VS14.44.35207/WindowsSDK10.0.26100.0 observed; CMake4.4.4 reconfigure completes and native build is live.

Complete native build requires exactly two new attention/commit CUDA objects; only the coupled tiled overlay may change among oldCUDA objects. Real creator/wire contract executable, ten external and ten embedded selector controls,955tokenizer controls, baseline source/DLL restoration, separate saved DLL and both-machine archive replay remain required. No model or HTTP runs during this build. Six PythonT128 candidates stay local until next model admission. Overall2xdecode/concurrency throughput and halfTTFT remain unmet/unproven; matchedINT4/4B, tuned/MTP/repeated references and B8 arithmetic diagnosis remain open.
