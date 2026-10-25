# Qwen3.8-27B on one RTX 5090: implementation task

Status: current8ae B1 model consistency passes independent remote/local replay
(4B4/8B1 cases). Actual HTTP attempt1 is terminal FAILED before any benchmark:
12protocol controls passed;805screen accepted10responses, then prompt IDs differ
at request11. Immutable304file archive and both independent failure replays agree;
source/runtime unchanged and cleanup complete. Actual partial HTTP/CPU reanalysis
reproduces all11native input sequences and identifies native tokenizer splitting.
Private CPU Unicode/whitespace/case candidate reduces frozen805mismatches122to0;
all124fixed4Kprompts still match. One of26added edge cases remains failed because
native NFC normalization is missing. This candidate is unqualified/un-deployed.
Next: implement NFC, rerun strict CPU differential gates, preserve/rebuild and
qualify the new runtime/model before a fresh HTTP attempt. Keep all frozen inputs.
Matched4K/tuned/MTP reference, INT4 W4A16, matched4B and repetitions remain open.
Original2x decode/concurrency and half-TTFT objectives remain unmet.

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
