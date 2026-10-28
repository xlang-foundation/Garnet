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

SSH target `CD-RTX5090` / `192.168.193.113` reports Windows hostname
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


### 2026-10-25 — Complete T128 native candidate builds and passes both-machine CPU replay

Actual bf16-prefill128-native-build-attempt1 completed0.56remote CPU controls0.298s; existing CMake tree reconfigured, fullGarnet DLL linked with two new T128 attention/commit CUDA objects; only the coupled existing tiled overlay changed among oldCUDA objects. Candidate DLL 897d2fc05f55229416cca8f1b420e6c27a70b1707298c24c311eab5db7a22da4 saved separately. Baseline original four source files restored, nine new native targets removed, original1a1d DLL restored; all717source/runtime6/SDK15/method hashes preserved and cleanup completed. Archive bf46f67c88bdedb762095c084f4e36e18eb7424d527835b25d44370122955332 (3453235bytes/83members).

Actual isolated plugin CPU executable passes16profiles,16clones/roundtrips,832malformed wires,496descriptors,336formats,128cross-dialect rejections and96oldB1/B4/B8 roundtrips; zero operator/enqueue calls. Runtime/header TensorRT versions101601 agree and actual loaded DLL is pinned SDK/bin. CPU-only stubs are linked solely into this contract executable, never intoGarnet. Ten external and ten actual embedded selector cases retain exact/default/frozen behavior. All955actual native tokenizer cases match frozen oracle, no mismatches. No GPU model or HTTP executed by this build.

Original full native-build checker passes remotely. Initial local outputs-directory replay fails its raw parent-byte identity assertion because the attention parent has the already documented tiny cross-machine FP64 error-summary differences. Failure log retained. Replayed unchanged checker in one fixed CPU-only work view using hard-linked immutable archives and exact actual archived parent-proof copies. Existing compare_replays validates both original local/remote full-FP64 attention proofs, including unchanged full28311552scalar acceptance and all non-summary fields. Original parent proofs remain untouched. Local native-build replay then passes0 and is byte-identical to remote proof e35b93c4dd7f5188f7e2aa12148bc32e28ef487c6e81552149a022562b8adc14. Canonical-view recipe/binding receipt and both checker outputs retained; no source checker or arithmetic tolerance relaxation.

This closes build/CPU integration only. Full T128 forward GDN/projections, model IDs/state correctness, image/reset/prefix behavior,805quality and matched HTTP timing are still unrun. Six Python candidates remain local; next model owner must temporarily pair the complete native/Python source profile with saved candidate DLL, preserve old paths/runtime and restore them on exit. B8 versusB4 strict mismatch remains a separate open arithmetic/geometry diagnosis. Current charts unchanged; overall2xdecode/concurrency throughput and halfTTFT remain unmet/unproven.


### 2026-10-25 — T128 full4K model pipeline; missing provenance preflight preserved and corrected retry started

Added a separate exact-derivative T16draft session restricted to the explicit T128target/T4decode profile; old T32class and guards remain unchanged. Draft prefill hook/reset/memory/close ASTs are identical to exercised old implementation. New constructor requires true T128opt-in, batch1/cap4096, unchanged depth3/full-head/prefix replay/T4/128MiB policy. Eight binding/exception/ownership/old-guard controls pass. Added strict full4K T128 replay wrapper and five source/profile/raw-evidence faults; frozen6case ORDER and exact ordinary/MTP/retained32state/64raw-vector/ID/callback gates remain. Actual ordinary observations saved before equality assertions, so a genuine math failure is retained rather than discarded.

Prepared55local CPU controls pass1.776s; native/Python20source bundle pairs CPU-qualified897d candidate with unchanged1a1d baseline oracle. First remote attempt stops at55CPU tests (1error,1.611s), before any embedded model process: recorded native derivation needs m/qwen38_bf16_prefill128_layout_contract.h, omitted from installer. All eight original native/Python source files and baselineDLL restored,12new candidate paths removed, source/runtime/SDK unchanged, cleanup done. Archive 71cd10af75dd7443f91d702c122599221eccc84bdef5cc39ecb2e728a3ff0907 (534225bytes/30members). Original full checker returns expected1 on both machines, byte-identical failure proof d97138e27a1bac030f7681f675157c0c3f8128af59ea87876da1718e3536f96e. No model or timing result from attempt1.

Corrected separate bound preparation declares every recorded provenance source, including four already-present component CUDA files and the missing header. Three controls reproduce all five omissions in actual old preparation, remove each required input in turn, and check exact recorded method-only retry transforms. All recorded source dependencies are now in installed inventory.58local tests pass1.908s; candidate20sources/nativeDLL/source bundle and all math/request/state/ID gates are unchanged. Retain original immutable owner/probe/preparer/archive and use distinct attempt2/preparation/launcher/checker paths.

Actual absent-or-identical corrected installation:35targets,9new/26identical,757baseline/769temporary-installed source pins, sixruntime and15SDK/environment dependencies unchanged. Receipt downloaded/acknowledged before supervisor48577 starts bf16-prefill128-model-admission-attempt2. Actual58remote CPU tests pass1.582s and embedded collector starts; exact job handle confirmed running. Remote source/transfers frozen and no second GPU model while live. Native runtime optional hashlib warnings are known and preserved; SHA256 works. Required next evidence: actual six3968+128text/image ordinary/MTP/reset-repeat completion, restoration/cache cleanup and both-machine full replay, then805quality/HTTP performance. Overall2xdecode/concurrency throughput and halfTTFT remain unmet/unproven; current charts unchanged.


### 2026-10-25 — Actual T128 first4K output mismatch replayed; layer1 recurrent-state boundary localized

Actual corrected attempt2 passes58remote CPU controls1.582s, constructs complete T128target/T4decode/T16draft session and runs first actual3968input+128output text case. The strict ordinary retained-ID assertion stops after saving actual prefill/final state hashes, raw hidden/normalized vectors and128IDs. No MTP generate or image case runs; remaining five cases unrun. Owner exits1, all757baseline sources/8original source bytes restored,12new candidate source targets removed, baseline1a1d and six runtime/SDK dependencies restored, cache retired, cleanup complete. Archive a25a2ca4f9ee908a1ae80e2c5754dc2283dc465438bf6ce5430ff89b02dd926b (832550bytes/57members). Original full checker returns expected1 on both machines, identical failure proof 4b79a5fa4f17bfd9131dca464fdf0122dc9fdc81b23d64dc9476f2732e8f29b7.

Independent scoped analyzer validates the unchanged retained baseline full32state corpus and first actual case metadata,3968prompt IDs,128output IDs, complete128component geometry/hash coverage/slots and all40960finite raw hidden/normalized scalars across empty/reset/prefill/final. Both machines produce identical analysis 57baf5446f4f64643976e9581de184f0b46920ff11bbf0ce002279ec9fc9f79d. Empty/reset exact. First differing output zero-based44 (45th token);44of128IDs equal. First differing stored prefill component is layer1 linear-attention recurrent state role2; layer0 conv+state and layer1conv hashes still exact.125of128prefill components differ; final-hidden relative L2 0.24171460250450066, maximum absolute difference48.96868896484375. This is an observation boundary, not harmless-rounding, cause or semantic equivalence. Raw full caches are not archived.

Pinned source audit observes existing FP8 canonical policies act only at flattened input rows32, and native helper requires32rows with M16selected algorithms35(output)/67(query). B1/T128 yields128rows and does not use this old intervention. This is a candidate mechanism to test, not a proven cause; no production policy changed and original guards/gates remain. Next bounded untimed control reuses the real layer0 FP8 input/weights, first reproduces old M16/M32 outputs, then compares identical replicated128row inputs under default versus AlgoCheck-qualified M16algorithm, every BF16 output word and cast/quantized byte. Only after actual component success should a separate M128 opt-in DLL be built and strict full-model/805quality/HTTP repeated.

No new benchmark or chart admission. Native T128 component/build CPU successes remain correctly scoped; first full-model qualification is FAILED. Current verified charts unchanged, overall2xdecode/concurrency throughput and halfTTFT objectives unmet/unproven; B8 arithmetic diagnosis and matched tuned/MTP/repeated/INT4/4B references remain open.


### 2026-10-25 — Real FP8 M128 algorithm control:4MiB rejection preserved,8MiB exact outputs verified

Reused the existing real layer0 GDN FP8 fixture without checkpoint cloning or fixture mutation. ActualM16 input16x6144 repeated8times suppliesM128. Same native FP8 helper and quantization kernel; archived M16/M32 outputs must reproduce first, then defaultM128 is observed without equivalence admission and a saved exactM16opaque algorithm is AlgoCheck-qualified for32/128. Full cast/quantized inputs and all BF16 outputs serialized, two repeats. No GarnetDLL load or production build/source/SDK mutation, no model or timing.

Original4MiB control exits2 after fourCPU controls0.548s and actual standalone build: default M16/M32 reproduce retained outputs, defaultM128 differs64of655360BF16words on each repeat. CheckedM16algorithm35 works atM32; M128AlgoCheck status/state0/0 requires5246879bytes, exceeding declared4194304budget, so M128shared matmul is not executed. Archive 72f704a39d4ce59f997173763bebf4b45a2329add7dfd724db19767bfa4a8880 (11047914bytes). Original full checker raises KeyError for two CUDA source keys absent from its unified inventory; local raw stderr retained. Those raw sources were already archived and hash-pinned as artifacts at all four original process starts. Separate bound checker verifies every such digest and final body, replays all default outputs/shared16+32outputs/shared128input bytes and preserves the actual workspace rejection. Original archive/checker unchanged; both machines match failure proof deeac880675ade77395cdde2a31a2d03cf3c6ffff4a47cae7af313afd6ddfc91.

Separate workspace8 trial changes only private allocation and sharedM128 call budget to8MiB; M16/M32/defaultM128 calls retain4MiB. Adds complete primitive source declaration plus two CPU controls for exact source transforms/workspace isolation and missing CUDA input rejection. Sixlocal CPU checks0.493s and sixremote0.573s pass; source770/runtime6/selectedSDK20/fixture preserved. Standalone build uses actualCUDA13.2.78/VS14.44.35207; known SDK C4819codepage warnings retained, headers untouched. Actual loaded cuBLASLt DLL path and version130400 verified against selected SDK/runtime hash. No application library loaded or changed.

Actual baseline algorithms16/32/128 are35/67/67; checked intervention35/35/35 uses the exact saved opaque algorithm, valid workspace checks and8388608bytes only atsharedM128. Shared128outputs exactly reproduce all repeatedM16 BF16words on both repeats; default128still64differences. Independent CPU replays on both machines verify every3604480outputBF16word, complete F32-to-BF16 round-to-even casts, E4M3 quantization, actual duplicate rows, repeated outputs, source/SDK/runtime/fixture hashes and algorithm logs. Archive 23799699c3a5bc81e60fa529f614db29e8b6c95d02313b5d1998afa23e5f9026 (13157388bytes), identical full proof 281d6c5ce91ab234f79b947402201ecc3a75abfb9b65f97e790804e990d91641.

This proves the bounded real matrix intervention only; it does not prove the cause or fix of the fullT1283968+128failure, other FP8 matrices, quality, performance or2x. Next concrete implementation uses a distinct strict default-off M128 GDN output plugin/creator/wire and8MiB helper, preserving old M32/default paths. Require actual policy selection/cache identity, native CPU compatibility, then unchanged strict sixcase model/state/ID gate and805quality before HTTP timing. Do not extend untested query geometry. Next plan retained; current charts unchanged and all original throughput/decode/halfTTFT/INT4/4B/reference goals remain open.


### 2026-10-25 — Distinct native M128 FP8 helper and CPU wire verified; full model still pending

Added default-off GARNET_FP8_GDN_M128_CANONICAL for original rank3 B1/T128 GDN output only, exact5120x6144 projection geometry; B4/T32 andB8/T16 flattened128 remain old path. Separate typed shared per-device native context, strictSM120/cuBLASLt130400, original4MiB M16 preference then actualM128 AlgoCheck with8MiB scratch. Separate GarnetModelOptFP8M128Canonical creator/version1/20byte policy128 wire, BF16 activation/output andINT8/opaqueINT32 weights,9175040total workspace. Four legacy plugin/helper files unchanged byte-for-byte. New graph stats count actual selected M128 projections; existing fresh per-graph engine-cache constructor unchanged. This is candidate native integration, not full model qualification.

Private local CPU-only compilation produces9objects for actual helper/plugin/lowering/builder/compiledgraph, old plugin/helper and wire probe/stubs. No link, target runtime, model or local GPU. 28CPU checks pass0.171s; exact composed originalT128+newM128 source derivation checks replace the single-stage provenance check only in this combined suite, original check untouched. SDK/input hashes unchanged. Initial preparation target-count and assumed unittest class-name errors were corrected before compiler execution and summarized separately. NOMINMAX redefinition warning retained.

Remote fp8-m128-native-control-attempt1 exits0:8CPU controls0.531s, actual distinct plugin CPU wire contract with33bad wires/18bad constructors/23bad descriptors/11bad formats/8policy rejects/3cross-wire rejects, clone androundtrip, zeroGPUstub calls. Standalone actual typed helper compiled in a separate translation unit from exact production source copies, cublas calls observed without algorithm substitution. Real retained layer0FP8 weights and16token rows repeated for32/128, two repeats, unchangedCUDA cast/quantizer. Actual default algorithms35/67/67 reproduce archived16/32 and observed128differences64words eachrepeat; nativeM16/M32/newM128 allalgorithm35 with exactsameopaque descriptor, checkedM128 requirement5246879bytes and8388608supplied. Every native128word exactly matches repeatedM16 and qualified8MiB parent output.

Independent replay on both machines checks complete3604480BF16 output words, RNE BF16casts/E4M3quantization/repeatedinputs/allactualoutputs, real helper context/actual matmul algorithm/opaque/workspace/DLLidentity, source787/runtime6/selectedSDK20/fixture preservation. Archive eb942175496d7c6a7163ba0bfbadb224587f9b02c3b95849c91bdaca530b76d1 (13410018bytes/75members); identical proof 1daf2b05a784031d4b88b859d321af14ec9159a9f2ec645136fa47f528dc44b2. Existing SDK C4819codepage warnings retained; no SDK modification or applicationDLL load/change. All raw words remain in output archive, metadata and raw ledgers retained in Git.

Next is combined22native-file candidate build on the5090 with original source/DLL restored afterward, actual native CPU compatibility, then strict paired29native/Python source sixcase3968+128 ordinary/MTP model replay. Require actual graphcounter48forT128 and0forT4decode/T16draft. Full-model mismatch cause/fix,805quality, HTTP speed/VRAM and2x are unproven. Current comparison charts remain unchanged; original throughput/decode/halfTTFT/INT4/4B/strongest-reference goals remain active.


### 2026-10-25 — Combined T128 and M128 FP8 native build verified; strict full-model retry pending

Built complete22native-source union in the existing CD-RTX5090 study CMake tree:13T128 native files and10new M128 policy/helper/plugin/counter files with trt_builder.cpp overlap once. Fresh authoritative baseline validates790source pins,6runtime and20selectedSDK/tool inputs,8existing target bytes and14absent targets,34existingCUDA objects. Packaged source only; original baseline source restored after actual build. Distinct derived owner/verifier preserve old transaction behavior, add source-proposal binding, exact34object provenance and separate FP8 CPU wire gate.

60local CPU tests0.331s and60remote0.276s pass;56original transaction/regression checks plus2ownedCUDA/source-derivation fault tests and2M128wire/algorithm parser checks. Actual Windows Release/MT/sm120 build links fullGarnet candidate with new helper/plugin translation units usingCUDA13.2.78,VS14.44.35207,WindowsSDK10.0.26100.0,CMake4.4.4. Existing34CUDA object set unchanged; exactly three intended objects recompiled with differentbytes (T128attention/T128commit/inherited lowbatch tiled overlay); no new/missing/unrelated CUDA object. Candidate DLL 2fc02e31ca89f7d07daace6205e12934ff6c731a87021a3237d3cc2e4cd2f132 saved as build/bin/garnet-native-bf16-prefill128-fp8-m128-attempt1.dll.

Actual CPU plugin compatibility: T128 overlay16profiles/832badwires/496bad descriptors/336badformats/128crossdialect rejects/96oldroundtrips and0operator calls; distinctM128FP8wire33badwires/18badconstructors/23baddescriptors/11badformats/8policyrejects/3crosswire rejects and0GPUstubs. TensorRT header/runtime101601 and loaded pinnedDLL match. Candidate passes10external selector cases,10embedded selector cases and955native tokenizer cases (805quality templates/124actual4K/26edges), zero mismatches. No GPU model/inference executed in this build job; these tokenizer cases do not constitute805model quality. Known SDK C4819 andNOMINMAX warnings retained, SDK untouched.

Owner exits0 and restores all8existing source targets, removes only14owned new targets, restores1a1d library and verifies source790/method19/runtime6/SDK20unchanged. Installed source inventory809fully pinned. Archive bed7521830630fa3b71b8ab68e2d7546afa03e412f0af286b77c1a18d1e754a3 (3826817bytes/100members); original new full checker independently passes on both machines with identical proof 75a3c4f9b7c84372bc7acf38f9186ecbbe3c449605d6fde01b4eff9ce16576be. Raw build/nativeCPU/source snapshots/proposal/derivations/ledger metadata retained; full source bundle and candidateDLL remain in archive.

Next is actual29native/Python paired model trial with strict original six3968+128ordinary/MTP cases, complete target states/rawvectors/IDs and805quality before timing. Add explicit default/off graphcounter0 construction control then enabledT128counter48/T4anddraftcounters0 proof. Full-model mismatch cause/fix, new HTTP performance/VRAM and2xunproven. Currentcharts unchanged; preserve complete original decode/throughput/halfTTFT/INT4/4B/strongest-reference scope.


### 2026-10-25 — Combined T128 FP8 strict model trial failed; complete evidence preserved

Distinct model methods derived exactly from preserved attempt2; original ID/state/vector oracle, six3968+128cases, T4verification andT16draft unchanged. Adds separate switch0 target construction then switch1 selection counters, saves graph metadata and completed ordinary observations before assertions. Paired29native/Python source candidate;12existing bytes restored/17new removed.64local CPU tests1.836s pass; combined source suite substitutes3exact composedT128+M128 provenance checks for old single-stage check, original test untouched. All49declared method/provenance inputs bound before owner; source812/installed829/runtime6/selectedSDK20.

Actual owner bf16-prefill128-fp8-model-admission-attempt1 exits1. Switch-off result {"128": 0, "4": 0}. Enabled native selection {"auxiliary_counts": {"draft16": 0, "draft4": 0, "prefill_norm": 0, "vision": 0}, "native_selection_verified": true, "target_counts": {"128": 48, "4": 0}}. Completed ordinary observations 1, fullcases 0. Model report status failed, raw error 'AssertionError()'. No broad805quality or timing admission from this run.

Archive 2ff12f3f31f399ce43ff4397c130d66e0a88c31b34659e51d68deac33576714f (1081900bytes/78members); original new full checker independently replayed both machines, identical proof fd476e05b1addadd87f399c6ea25e0dd7a0b90e8ccc543258af360b1e78455ce. Source, baseline1a1d library, runtime andSDK restored; only owned new cache entries retired. Raw vectors/resources/logs/selection metadata and all failures retained. CandidateDLL 2fc02e31ca89f7d07daace6205e12934ff6c731a87021a3237d3cc2e4cd2f132. Known embeddedPython optional hash import warnings remain in raw logs; SHA256checks work, SDK untouched.

Analyze retained actual failure before further operator changes; preserve strict full4K state/vector/ID gate and all original goals. Existing performancecharts unchanged; overall2xdecodeANDthroughput andhalfTTFTadditional, matchedINT4/4B andstrongest feasible tuned/MTP/repeated reference goals remain active.


### 2026-10-25 — M128 native policy selected but full4K output fails at token12; first prefill state difference layer2

New full checker preserves actual owner1failure on both machines, identical failure proof fd476e05b1addadd87f399c6ea25e0dd7a0b90e8ccc543258af360b1e78455ce. Actual default/off target graph counters0/0 and enabledT4/T128 counters0/48; draft4/draft16/vision/anynorm0. First3968+128text ordinary observation is saved before assertion; only11of128IDs match, firstdifference zero-based11 (token12), targetslots4095correct. MTPgeneration andimages are not executed;5remainingcases unrun. Baseline source/runtime/SDK restored, allrawinputs/vectors/resources/logs retained.

Distinct external analyzer is an exact recorded derivative of preserved original failure analyzer. Independently validates original retainedT32 complete32state oracle, actual3968prompt and128IDs, full128component state geometry/slots/hashes and40960finite hidden/normalized rawscalars on bothmachines. Identical analysis proof e2fd23cd25535a5b4d3fea312d67eaeff16b4bdf50fed2357f9b7e7dbe416380. Empty/reset exact. Firstdifferent prefill component now layer2linear_attention role2,786432words;123of128components differ. Prefill hidden maxabs17.0196533203125, relativeL2 0.22054152219154508; normalized maxabs2.732177734375, relativeL2 0.23471643950999108. Final state firstdiff layer0conv follows divergent decoding. Raw full caches are not serialized; these are complete hash boundary observations plus raw hidden vectors, not a root-cause proof.

Prior unchanged attempt2firstprefill difference was layer1role2 and outputfirstdifference zero-based44; current boundary shift tolayer2 does not establish numerical correctness or output improvement (current output diverges earlier). Existing repeated16row matrix control was bounded; native algorithm35selection alone has not repaired fullmodel. Keep defaultoff and original gates unchanged, no805quality/HTTP/timing promotion.

Next sourceaudit confirms old diagnostic_linear_stages supports embedded layer0B1/2T16 only, rejects register128/joinedMLP/externalweights and enforces1Miword rowbounds. Plan a separate opt-in T32/T128 GDN layer0/1 stage profile with matching actualexternal compressed pool/primary-output witness; preserve oldguards. Use real128distinctprompt tokens and equal token/state sequences, compare1x128vs4x32through first differing prefix boundary. Capture fullstage words in boundedpieces (primary/QKV exceed1Miword); require stage primary equal ordinary graph before attribution. Isolate actual non-repeated projection inputs/weights/scales as necessary before any policyextension. All original2xdecodeANDthroughput/halfTTFTadditional/INT4/4B/strongest-reference goals remain active; currentcharts unchanged.


### 2026-10-25 — Separate T32/T128 GDN stage profile built and CPU verified; GPU primary witness pending

Added strict separate diagnostic_gdn_chunk_stages integer0/1 on current candidate source, defaultoff. Pinned B1/T32-or-T128/layer0-or1/H5120/heads16and48/d128/conv4/RMS1e-6 with exactlyfive named external compressed projections, allF32hidden/history/state/active, fused+register128+joinedMLP, no payload/joinedinputs/BF16externalbinding/workspace override. Existing diagnostic_linear_stages embedded layer0B1/2T16 guards and all actual GDN/FP8/MLP/CUDA math bodies unchanged. New pure header plus onlytrt_builder.cppmodified. Newstage capture limit tokens*10240; legacy1Miword bound andglobalF32reader unchanged. Collector must split oversized primary/QKV reads.

66local composed model/source/routing/oracle CPU checks1.831s pass; two actual CPU /c objects (builder+profileprobe) compiled privately without link/runtime/GPU/SDK/productionbuild mutation. Original single-stage provenance tests unchanged; exact newstage+priorFP8+originalT128 source composition checks in combined suite. Native build prep62local CPU tests0.349s and62remote0.269s pass;3postinstall source checks pass beforeactualbuild.

Real CD-RTX5090 build uses existing Release/MT/sm120 tree for23native source union, fresh814baseline source pins/30methods/840installed pins/runtime6/selectedSDK20/34CUDA objects.8existing targets restoredbyte-for-byte,15owned new removed; baseline1a1dDLL andSDK preserved. ExistingCUDA set34unchanged; only intended T128attention/commit/inherited tiled-overlay objects recompile. FullGarnet links candidate 0ec79f33c94c756907af9454271e7a27edfb8765a9c5f820fa84f6bdf4380d96 saved as build/bin/garnet-native-gdn-chunk-stage-attempt1.dll. Actual pureCPU header contract accepts4valid profiles, rejects50profilefaults/6flagfaults/1legacyconflict. It uses no CUDA/TRT/GarnetDLL and does not exercise nativefrontend orproveGPUprimary equivalence.

Actual old16profile overlay wire, newFP8wire,10external+10embedded selector and955tokenizer checks pass; zeroGPUoperator stub calls/zero tokenizer mismatches. Known SDK C4819/NOMINMAX warnings retained, SDK untouched. Owner0restores source/runtime. Archive f106585e76a87e7d49d73bea40218ae8dfac680046896fadfb0300f975a000fc (4218252bytes/118members), identical independent fullchecker proof ee19c1de27ed76e87a62b757602b9fb80dedf51b2acf196d5d386389c8eeda85 onbothmachines. Rawsourcebundle/candidateDLL/cache/source snapshots/ledgers/contracts retained. Captured rawCMakeCache-before is preserved verbatim, including generated trailingblankline; exclude only this artifact plusrawlogs from whitespace checks.

Next is actualGPU frontend rejection andprimary-output witness using real128distinctprompt tokens and1x128vs4x32 sequential GDN controls. Needall11stage words, offsets/lineage, repeats andbounded <=1Miword readpieces, no largefullKVdump. Separate layer1 commoninput control from actual chained input. Do not attribute an intermediate difference unless diagnostic primary equals ordinary same-profile bytes. Currentfull4K candidate stillfails token12/firstprefilllayer2state; no new model/805quality/performance/2xadmission. Existingcharts unchanged and alloriginal goals remain active.


### 2026-10-26 UTC — Real GDN stage primary witness passes; Z projection differs on actual128 inputs

Preserved actual attempt1collector failure after firstlayer0T32chunk:18CPUchecks,10actualfrontendrejects and9complete finite arrays; generator readback raises TypeError(object is not subscriptable) before stage comparison. Not a numerical/primary mismatch. Original failure checker independently agrees on both machines, proof 4420f01d08d25fdd41324140b0a68af4efe526f7cd941f6f4965af909368f7ee, archive 986462fb514b5c5cbbc43a30795f176c02d3566256d32ecc1cc05c556215c731. CPU-only actualembedded interpreter control reproduces the generator failure on ASCII testbytes, explicit list-loop and concatenation both exact; no Garnet/CUDA import, no SDKchange. Distinct boundretry changes readback only, exact five-method derivation retained, originalsource/checker/prep immutable. Originalowner inherited T128model CPUlabel/docstring is historical; actual launcher/profile/outputs/scope identify stage witness.

Attempt2owner0:20CPUcontrols pass remotely2.355s andlocally3.244s. Same23native source candidate 0ec79f33c94c756907af9454271e7a27edfb8765a9c5f820fa84f6bdf4380d96,843source/858installed pins/runtime6/selectedSDK20;8originaltargets restoredbyte-exact and15new removed, baseline1a1dDLL/cache/config/index restored.10actualfrontendrejects flag2/negative/bool/float/legacyconflict/batch2/T16/layer2/noactive/BF16hidden pass.8profiles (layers0/1, T32/T128, ordinary/diagnostic),3warmups and2repeats use realfirst128of3968frozenprompt; layer1uses commonretainedT32layer0ordinaryoutput. No repeated16rows. Allinputreadonly/carriedstate/repeats/graphcounts hold. All diagnostic primaryhidden/history/state bytes equal ordinary same-profile output. Original fullverifier independently replays509complete finite arrays/98,205,696outputF32words on bothmachines, identicalproof ca7d3f095d1a0e48adfb1297a6c7233936925fdd48f079be6a1f6f1e6818e78a.

Firstdifferentstage Z: layer0 27/786432words first5510, gatedoutputsame27, attention/residual/MLP/hidden exact. Layer1controlledcommoninput Z74words first19654; gated74; attention/residual681; MLPnorm3779; hidden4910. Inputnorm/QKV/beta/a andfinalhistory/state exact inbothlayers. These first128component observations do not establish full3968model cause. Correct next geometry is in_proj_z N6144/K5120, distinct from existing out_proj N5120/K6144.

Immutable archive 0faf8a625ba25befbeae0877bdf0a08a7696dd3ed19845f7590238f58e1e7de3 (485580421bytes/555members). Lossless transport removes duplicate compressedZIPpayloads only:124598047bytes reconstruct exact485580421byte originalSHA before originalchecker. Interrupted183795712byte originaltransfer retained as workpartial. Initial localtransfer guard used stale observed46825472size and failed beforemutation; actualterminated size inspected then guard corrected. No sciencebytes changed.

NewCPU fixture extraction reads actualcheckpoint layers0/1 in_proj_z weight/scales andstageinputnorm/Zoutputs.12files/74448912bytes remain onstudy machine; fixturemanifest 81fa473159a724647197aa3ddc099210e9a7a6f7629006d5719d683c13855acd. SixnewCPU fault tests pass local0.240s/remote0.198s; composed15CPUtests3.218s pass. Source/runtime/SDK/checkpointmetadata unchanged. Independent local extractionchecker rebuilds allsixstage-derivedinput/output vectors againstparent bytes andcheckslogs/provenance/metadata, proof 617989e903d3ddf8c52a791da241f970a4b71677c667f1ba800f7dfa9ba8f82e. Externalweight/scale payloads have not been independently reread locally; nativecontrol unrun. First checker Windows pathnormalization failure retained, corrected beforeproofcreation.

Next: reproduce both actual fourM32/M128Zprofiles with actualnativehelper, complete BF16/FP8 conversion/output replay, observeM32algorithm and AlgoCheck actual128workspace beforeintervention. ExistingGDNoutpolicy not extended. No full4Kmodel/805quality/HTTP/2xadmission. Currentcharts unchanged; overall2xdecodeANDthroughput/halfTTFTextra andmatchedINT4/4B/strongestfeasible tunedMTP/repeatedreference goals remain active.


### 2026-10-26 — Actual real128 Z control reproduces both profiles; M32 algorithm restores complete M128 parity

Private explicit sm120 standalone CUDA control includes actual unchanged modelopt_cublaslt.cpp andactual modelopt_fp8.cu. No GarnetDLL/applicationmodel load, no productionbuild/SDK/source mutation.853sourcepins/runtime6/SDKenvironment20 and12fixturefiles unchanged. Actualindexed checkpoint weights/scales/headers reread againstfixture beforeprepare andexecution; pinnedconfig/index exact. Samefirst128realinputs fromclosedparent stagewitness, bothGDNlayers0/1, distinctN6144K5120 geometry. EightCPUfaultcontrols pass locally0.196s/remotely0.212s. Realnvcc13.2.78/MSVC14.44 buildsprivateexe 45030e9bd809fc841fd01528b71ed6e049986fa4da1f60a67d1be1abb42fd865; knownSDKC4819warnings retainedwithoutSDKedit.

Defaultbothprofiles gate passes beforeanyintervention: foursequentialM32chunks andoneM128 exactlyreproduce allarchivedBF16outputs atbothlayers/tworepeats. Perprocess observers seeeightM32 andtwoM128calls. CanonicalM32 anddefaultM128 bothalgorithmID67 butdifferentopaqueconfiguration; IDalone doesnotidentifyequivalent arithmetic. Sharedintervention appliesactualM32opaquealgorithm toactual128descriptors onlyafterAlgoCheck status/state0. Requiredworkspaceobserved0bytes, supplied8MiB; no claim8MiBnecessary. AllM128outputwords nowexactlyequalfourM32atbothlayers/tworepeats. No fallback, actualloadedcuBLASLt13.4DLL hash/locationpinned onSM120RTX5090.

Originalcompleteindependentverifier passes onbothmachines withidenticalproof 34ef3f622bfc5edaf3f0274f1e01e061cf32146146acad4282136cd3ec77501c. Fourprofiles,40actualmatmulcalls,12,582,912BF16outputwords andallF32-to-BF16/FP8quantizationbytes independentlyreplayed; allcompleteoutputs finite/repeatable. Archive 6cdf4eb999eb4a53649d32dd8857f780d7e5c6969dde75f9a99a95f3ceebb41b (36072997bytes/67members), rawsource/logs/casts/quantizedactivations/outputs/prep/exe/fixturemanifest/checkers retained. Raw31MBweights remainexternalpinnedcheckpointfixture, notduplicated inarchive; actualcontrol rereads/provesfixturebyteequality onstudy host. Source/runtime/SDK/fixture unchanged; nativeapplicationDLL baseline1a1d preserved.

This qualifies the focusedreal128Zalgorithm intervention, notfull3968cause/model/quality/performance. NextseparatetypeddefaultoffZ128context/plugin must preserveoldGDNout/generalFP8helpers, enforce originalrank3B1T128/N6144K5120/name/SM120/version/wire/workspace bounds, selectcanonicalM32descriptors andproveopaqueagreement inactualnativehelper testbeforefullintegration. Thenrepeatstageprimarywitness andunchangedfull4Kstate/vector/ID and805quality gates beforeHTTPspeedcharts. Currentcharts unchanged; overall2xdecodeANDthroughput/halfTTFTextra andmatchedINT4/4B/strongestfeasibletunedMTP/repeatedrefs remain active.


### 2026-10-26 — Typed native Z128 helper passes actual real128 controls; builder route CPU compiled

Added separate fivefile ModelOptFP8Z128 nativecontext/plugin/profile family derivedexactlyfrompreservedGDNoutM128 family,9legacyhelper/plugin files unchanged. Strict defaultoff GARNET_FP8_GDN_Z128_CANONICAL B1/T128/originalrank3/flattened128/exactGDNZsuffix/N6144K5120 only, notB4T32/B8T16. Newcreator GarnetModelOptFP8Z128Canonical/version1/20bytewire marker0x5a313238; finitepositive scales/product, BF16activation/output andINT8oropaqueINT32weights. Packedactivation655360B plus4MiBBLAS=4849664Bpluginworkspace. Newtypedperdevicecontext selectsactualM32descriptors at4MiBpreference,SM120/cuBLASLt130400; exactqualifiedopaque64byteconfiguration andID67 requiredbeforeactual128AlgoCheck, failclosedwithoutfallback. No opaque-field reinterpretation orfutureSDKcompatibilityclaim.

Local11provenance/conversion/logCPUchecks0.171s pass; sevenprivateCPU /c helper/plugin/oldcomparison/probe/stubs/observerobjects compilewithoutlocal link/runtime/GPU. Latercopy-aware provenance reading usesexactisolatedcandidatecopies onlywhenproductionnativefiles areabsent, nevermaskspresentsourcedrift. Original11test source preservedbyteexactagainstcompileinputhash. Current13localcontrols0.190s and13remotecontrols 0.199s pass. DistinctactualremoteCPUwire control passes34badwires/18badconstructors/23baddescriptors/11badformats/8policyfaults/5crosswirerejections,0GPUstubcalls; actualTensorRT10.16.1header/runtime/location pinned, enqueue/modelunqualified.

Exclusive standalonehelper owner0 uses878sourcepins/runtime6/SDKenvironment23/12fixturefiles andexact10candidatecopies (5new+5oldcomparison); source/runtime/SDK/fixture/config/index unchanged. No productionnative target/source/DLL/build modifications andnoGarnetapplicationDLL/model load. Actualrawindexedcheckpoint weights/scales rereadagainstfixture beforeprepare/execution. Realprivate sm120 CUDAexe 26181694cf2d5dc485579bbb503f27fa59047b06da1dd860c9298cf1865f1bfe linksactualnewhelper inseparateobservedTU andactualunchangedFP8quantization; CPUstubs onlyinwireexe. Baselinebothdefaultprofiles reproducepriorclosedoutputs first, bothlayers/tworepeats. NativeactualM128outputs exactfourM32/allpriorqualifiedsharedoutputs.

Actualtypedcontext algorithm67/opaqueexact/device0/requiredworkspace0, supplied4MiB;15realinvalidargchecks perlayer (30total) rejectnullcontext/pointers, wronggeometry/workspace andnonfinite/nonpositive scales with0matmulcalls. Fourprofiles/40positiveactualmatmuls;12,582,912BF16outputwords plusallF32RNEBF16casts/FP8quantization independentlyreplayed againstfixtureandpriorarchiveonbothmachines. Identicalcompleteproof a67758dd75aaa6bd68783b0fda0da3ed92ea93cb568c6a24b7c5daf2462a3c0b, archive 594b2248b5b5cf67e5ea719153841470392b2ad0bf4ecc3ceff964afacdbbbe8 (36394440bytes/92members). Rawweights remainexternalpinnedfixture, notduplicated inarchive. Fullrawsource/casts/quantizedinputs/outputs/exes/CPUwire/SDKhashes/failclosedcontrols/ledgers retained; knownSDKC4819andNOMINMAXwarnings retainedwithoutSDKedit.

Afterbothmachinesfullhelperproof, integratednewdefaultoffroute andindependent native_fp8_z128_canonical_projections builder/graphcounter, withcopy/reset/stats paths, in5existingfiles; all5beforeimages/exactderivation retained. ExistingGDNoutM128/M32 routes andpergraphfreshcache unchanged. Oldsingle-stage provenance tests leftunchanged;79composedCPUchecks2.027s pass byreplacing2supersededsourcechecks with exactoriginal+GDNstage+Zcomposition. Threeprivateactualbuilder/lowering/graph CPU /c objects compile; no local nativeexecution/model/GPU orSDK/productionbuildmutation. ExistingCMakeBackendCPPglob includesnewfiles, noCMake/CUDA source changes.

WholeGarnetDLL withnewZroute hasnotbeenbuilt/deployed yet. Next28native union/8existing+20newtransaction, actualwire/selector/tokenizer/fullbuild, thennewcounter eightstageprimarywitness, unchangedstrictsixcasefull4Kstate/vector/IDs and805quality beforematchedHTTP. First128component/nativehelper success doesnotprovefull3968cause/model/servingperformance. Currentcharts unchanged; original2xdecodeANDthroughput/halfTTFTextra/INT4/4B/B8arithmetic/strongestfeasibletunedMTPandrepeatedreferencegoals remain active.


### 2026-10-26 — Full native28 Z128 DLL builds and CPU contracts pass; actual GPU stage witness next

Distinct build/prepare/fullverifier methods areexact recordedderivatives ofclosedGDNstage nativebuildtransaction. Newsource union28=closed23plus5typedZfiles, replace5existingbuilder/lowering/graphfileswithqualifiedadditivecounter/routingbytes. ExistingCMakeBackendCPPglobfindsnewfiles; noCMake/CUDA/math/SDKedits. Fresh879baselinesourcepins/69methods/913installed/runtime6/selectedSDK23;8existingtargets restoredbyteexact,20ownednew removed; baseline1a1dDLL restored. Source/SDK/checkpointmetadata hashes checkedbeforeinstallandafterrestore. ReusedfixedROOT/build Release/MT/sm120tree, retained allsources/DLLs/cachemetadata/ledgers.

77prebuildCPUcontrols pass local0.527s/remote0.489s, including62retainedtransaction/contracts,13typedZ/real128faults,2newreport/source-derivation checks. Sixpostinstall native-source/route/fullpriorcomposition/legacydiagnosticguard checks pass. FullGarnetlinks candidate 6d4abad7ac368ce4e5ac99b46bd3afd53e6d73e0c22c4754c8ffc0158f57bebe saved build/bin/garnet-native-fp8-z128-attempt1.dll. CUDAobjectset remains34with0added; onlypermitted priorT128attention/commit/coupled-overlay objects mayrecompile, actualchangednames retained. KnownSDKC4819/NOMINMAXwarnings retainedwithoutSDKmutation.

Actualold16profile overlaywire compatibility passes832badwires/496baddescriptors/336badformats/128crossdialectrejections/96legacyroundtrips/0operatorstubs. OldM128FP8wire contract passesunchanged. NewZwire contract passes34badwires/18badconstructors/23baddescriptors/11badformats/8policyfaults/5crosswirechecks/0GPUstubs, totalworkspace4849664B. PureGDNprofile4valid/50invalid/6flag/1legacyconflict passes. Actualcandidate10external+10embeddednative-selectorcases and955frozen tokenizer cases exact,0mismatches; no model/GPUforward.

Originalcompleteindependentnativebuildverifier passesonbothmachines withidenticalproof 88a909c0cebcb7353b57c261ee1339a01e7d3835005a20e41802dad2400bab6b. Archive a0d05e5d5b0d5448c294a1acb13e91b9257d80f08a4e038a9f50ed52063f21f1 (4736840bytes/167members). FullcandidateDLL/nativebundle/8originalsources/rawSDK/cachebefore/CPUcontracts/tokenizercases/methods preserved. RawgeneratedCMakeCache-before preservedverbatim, includingtrailingblankline; onlythatnamedartifact plusrawlogs exemptfromwhitespacechecks. This isbuild/CPUqualification, notactualGPUfrontend/newrouteprimary/model/805quality/performance admission.

NextactualGPUstageowner willrun10freshprocessnewZpolicyfrontendcontrols (default/empty/off/on/6malformed), then8realfirst128layer0/1 T32/T128 ordinary/diagnosticprofiles. Freshprocess controls avoidassumingcross-runtime environment changes arevisibleafterDLLload. Requireold/newprojectioncounters, exactordinaryprimarywitness, completeinput/carriedstate/repeat/all11stage words andboundedreadpieces; preserveanyremainingnumericaldifference. Full3968modelcause/strictsixcase4K/805quality/HTTP/repetitions/strongestfeasibletunedMTPrefs remainopen. Original2xdecodeANDthroughput/halfTTFTextra/INT4/4B/B8arithmeticgoals stayactive; currentchartsunchanged.


### 2026-10-26 — Integrated Z128 stage attempt: fresh policy controls pass, main T128 graph initialization fails

Exclusive native28 stage owner used closed DLL 6d4abad7ac368ce4e5ac99b46bd3afd53e6d73e0c22c4754c8ffc0158f57bebe. All29CPU tests passed locally4.246s/remote2.962s. Ten fresh-process policy frontend controls passed: unset/empty/0/1 plus six malformed strings. Default/off graph counters stayed0; on T128 Z counter1 and oldM128 counter1. These policy graphs did not execute inference. All10 malformed diagnostic frontend attributes also rejected.

The main actual-real128 collector completed only layer0T32 ordinary and diagnostic profiles (3warmups/2repeats/four carried32-token chunks), then failed building first layer0T128 ordinary graph. TensorRT reported PluginV2Runner.allocateResources ErrorCode5 and buildSerializedNetwork failed. No main T128 graph output or stage array exists. Fresh T128 policy-only construction had succeeded in another process; cause of the later failure is unproven. Preserve opaque/shape/workspace/numerical gates and diagnose initialization before modifying code.

Full frozen failed-run verifier replays on both machines with identical proof 49aeabbf91895a0c15c7747b21a7a5e9aba24882f4464eee0f5b2a990b872258, expected exit1. Separate postfailure analyzer passes on both with identical proof c19788deb3f80da24c26802033b51bf1e92055c35b12188b403590b10a6835db:203complete finite entries,29,515,776F32 output words across two completed T32 profiles, exact ordinary/diagnostic primary, readonly inputs/carriedhistory/state/repeats, all200completed input/state/output arrays exactly equal oldstage baseline. This establishes only partialT32 preservation, not T128 qualification or full-model cause.

Original archive a6015fcb5695fa6f153a6ca11ae063b20d5f4c77bb5b67688ca4c2ff2740c851 (175457103bytes/282members) reconstructed losslessly from deduplicated transport; original ZIP bytes/hash/CRC and every manifest file checked. Source905/installed925/runtime/SDK/checkpoint pins restored;8existing native files restored/20new removed, baseline1a1d DLL restored, owned SSD engine files retired. Raw failures, original methods, source/prep/ledgers/resource observations and both proof pairs retained.

Next distinct diagnostic will enable cuBLASLt API/heuristic logging for same DLL and graph order; logged execution is correctness diagnostics only, never a benchmark. Full3968+128 state/vector/ID,805quality, HTTP/repetitions/tunedMTPreferences and original2xdecodeANDthroughput/halfTTFTextra/INT4/4B goals remain active. Published comparison charts remain unchanged.


### 2026-10-26 — Same native28 DLL reproduces T128 initialization failure; trace narrows selection boundary

Distinct exact-derived owner/bat retain same DLL/collector/graph order and all numerical gates. Only runtime additions: CUBLASLT_LOG_LEVEL=5, MASK=31, unique owned log path, as documented at https://docs.nvidia.com/cuda/cublas/index.html#cublaslt-logging . Ten alreadyclosed policy processes omitted explicitly; main frontend10rejections and originalstage sequence retained.32CPU controls pass local4.377s/remote2.93s. No cache capacity/selection/math/SDK changes or benchmark admission.

Reproduced same firstmainL0T128ordinary build failure after twoL0T32 profiles.86875-byte API trace final Z query uses actualM32 K5120/N6144/4MiB preference and reports IDs[67,67,35,35,35,35,67], then destroys descriptors/context without logged M128AlgoCheck. This narrows initialization to selection-before-check; exact rejected guard/full opaque bytes and root cause remain unproven. No OOM or numerical divergence claim.

Original verifier failed because frozen prompt fixture lives only in linked parent archive. Original checker/preparation/evidence preserved; separate exact-derived linked checker corrects lookup and verifies fixture hash, no gate relaxation. Corrected checker passes bothmachines with identical proof 39de5809ef16448716cb7a00e4574945e4ad14774b1793a1687164a8adfa88d5. Separate strictpartialT32 replay also passes both with identical proof 7d42fb819e025d114726fe36a38b8535be4a4470116a05fc3dbd891846ed485d:203finite entries/29,515,776F32 words, exactreadonly/carriedstate/repeats/ordinary-diagnostic primary/all200priorbaseline arrays. No T128graph inference orfullmodel/quality/performance qualification.

Original archive 47a28a878771e342419231ee802d4798f8f7a33e64737a23b71148e9bf24a18d (174652941bytes/228members) reconstructed byteexact fromdeduptransport andfullCRC/manifestreplayed.912source/932installed pins,8old/20new nativefiles,baseline1a1d DLL/runtime/SDK/checkpoints/ownedcache restored. Original checker traceback, repaired checker derivation, all raw logs/resource samples and proof pairs retained. Next audit actualtypedhelper opaquechoices in samewarmCUDAcontext before any selectionchange. Full4K/805quality/HTTP/repetitions/strongestMTPrefs/2xdecodeANDthroughput/halfTTFTextra/INT4/4B/B8 goals remain active; charts unchanged.


### 2026-10-26 — Observer builds; embedded libffi unavailable; existing native package ABI works without SDK changes

Original CPU ctypes capability probe imported successfully but incorrectly assumed no tensor/Garnet module preload; first guard failed. Separate bound probe then failed on staticbuiltin _ctypes lacking __file__. Both original methods/logs preserved; bound2 records actual tensorpreloaded=true/Garnetpreloaded=false, builtin _ctypes, hosthash2e2db1 and baseline1a1d. Attribute/import/dlopen capability did not prove foreign-function invocation. NoSDK edits.

Actual observer DLL 1b421e7946bc8b7959b3ec23ace61080b56cacb5ef22b1e452f832c5b926ba63 builds remotely from exact typedZhelper CPP/header in separate observedTU; wrappers call actual heuristic/config/check/matmul APIs once andreturn unchanged, record all64-byte candidates/status/workspace.37CPU source/candidate/replay/fault controls pass local4.199s/remote2.968s. Cold embedded process failed with NotImplementedError:libffi is unavailable for ctypes, before any helper report. Main GPU graph sequence never started. Original full expected-stage verifier remains unchanged; distinct failure verifier checks all source941/installed961/runtime/SDK/checkpoint restoration, rawCPU/build/cold ledgers/resourceGPU identity and complete archive.

Existing Garnet src/entry/main.cpp andcpu_main.cpp establish XLang package ABI Load/BuildAPI/Create. Separate fixedobserver nativeadapter uses that same pinned existing SDK/runtime, no productionGarnetDLL change. Actual embedded CPU invocation passes ABI24 plus3outside/traversal/empty-path guards returning20 before LoadLibrary/observer/CUDA,4nativecalls total. Source946/runtime/selectedSDK/checkpointmetadata unchanged. Compiler C4530 warning in first CPU prototype retained; next GPUadapter build mustadd /EHsc and repeat CPUcalls before cold/warm audit. Import-only success wasinsufficient; this actualmembercall proves usable alternative.

Both original archives fullyCRC/manifest/source replayed on bothmachines with identical proof pairs: failed observer f65215efdbc21a90062b5957d56424934f80aacc0daa3ab7037c31da4d2ec572 (archive bb6dbd37c913187854cc6a56c833a7f2e183581d0e137ec9f739e45a0393cf1c), successful nativeCPU a2dd2da7ddc1426ddd98c338d7db731c2a083c5966a705250127dcb6d1d52aa8 (archive 00ac34e90768ce23b73ca03daae1b6421593a1113b34538b5f9465ff93e16bd5). Fullbinary/source/rawlogs/failed capability probes retained. Neither qualification supplies actual warm opaque candidates or T128 outputs.

Nextdistinct native-package owner willreuse exactobserverDLL, compileEHsc adapter, validateCPU ABI/guards, then coldactualhelper and same28DLL real128 mainsequence withpostfailure nativecall on same thread/process/CUDAcontext. No ctypes/libffi/SDK mutation/DllMain GPU work. Preserve strictopaque/device/version/shape/workspace gates; onlycandidateevidencecanjustifyselectionfix. Full4K/805quality/HTTP/repetitions/strongestMTPrefs/2xdecodeANDthroughput/halfTTFTextra/INT4/4B/B8 remainactive; publishedchartsunchanged.


### 2026-10-26 — All8 real128 GDN profiles now replay exactly; sameDLL initialization failure did not recur

Private native adapter now uses /EHsc and adds readonly ownmodulepath, derivedexact from CPUqualified ABI24 adapter.42CPU controls pass local4.217s/remote2.962s. Actual5nativeCPUcalls (ABI,loadedpath,3reject-before-loader guards) pass; actualmodulepath/SHAmatchesownedbuild, noC4530. Existing observerDLL 1b421e7946bc8b7959b3ec23ace61080b56cacb5ef22b1e452f832c5b926ba63 unchanged. Cold actualhelper constructor succeeds, expectedfullopaque atfirstsuccessfulindex0, actualM128AlgoCheck succeeds, ID67, requiredworkspace0/4MiBsupplied,0matmulcalls. No libffi orSDK edits.

Unexpectedly, unchangednative28 GarnetDLL 6d4abad7ac368ce4e5ac99b46bd3afd53e6d73e0c22c4754c8ffc0158f57bebe andsame maincollector/order completed all8 layer0/1T32/T128ordinary/diagnostic profiles. All10frontendmalformedattrs reject;3warmups/2repeats. Fullindependent replay509finiteentries/98,205,696F32 outputwords: all11stages, readonly/carriedinputs, repeats andordinary-diagnostic primaryexact, finalhistory/stateexactbothlayers. All400T32 input/state/outputarrays exactlyequal priorclosedoldstagebaseline. NewZcounter1atT128only, oldGDNoutcounterunchanged. Actualfirst128nonrepeatedtokens fromfrozen3968prompt; layer1commoninputretainedT32layer0output, notfullcandidatechain.

Mainrun succeeded, so plannedpostfailurehook neverinvoked; no warmfailure opaquechoices captured. Earlier two sameDLL PluginV2Runner.allocateResources failures remain unexplained. Do notclaimrootcausefixed, deterministicinitialization, full3968modelcause,805quality orperformance. Original failure-specific verifier/preparation/source preserved; distinctpost-observation full-success verifier adds mandatoryall8profiles/allwords/400oldbaselinecomparisons andallnativeABI/restoration checks, no numerical gate relaxation.

Fullsuccess verifier passesbothmachines withidenticalproof d7557096d81a7e95327bfbe13f8910c1ed40b29c49888219ce76852f36f8580b. Original archive aad7fbece8b229c50a028106e4a241bd568556e025090982305730735bb341a6 (486237308bytes/583members) reconstructed byteexact fromlosslessdeduptransport, fullCRC/manifest checked.966source/986installed/runtime/selectedSDK/checkpointpins restored;8oldtargetsrestored/20newremoved/baseline1a1dDLL/ownedSSDcache restored. Allsource/binaries/rawarraypieces/logs/resources/coldcandidateopaques retained.

Nextfullmodel35source trial =native28plus7existingPythonprofilebytes,12old/23new restoration. Separateoff/on graphcountsforbotholdGDNoutM128andnewZ128 (target12848eachon/0off,decode4/draft4/draft16/vision/norm0), then unchangedstrictsix3968+128text/image/reset-repeat ordinary/MTP oracle. Keep exact128IDs/32statewitnesses/64rawvectors/callbacks; saveordinarydata beforeassertions. Onlyafterfullmodel/805quality/initialization qualification: HTTP/repetitions/strongestMTPrefs/INT4/4B/B8. Original2xdecodeANDthroughput/halfTTFTextra remainactive; chartsunchanged.


### 2026-10-26 — Composed Z128 full4K gate fails at first ordinary text case; layer3 key and value boundary retained

The same closed native28 DLL plus seven unchanged Python profile files (35 candidate sources) constructed separate off/on full-model graphs. Both old GDN-out M128 and new Z128 counters are 0 off and 48 at target128 on; decode4, draft4/16, vision and optional norm counters remain 0. All 70 CPU controls pass locally (1.965s) and remotely (1.844s). Only three superseded single-stage source tests were replaced by five current composition tests; all 61 prior model/state/vector/token/MTP controls remain, plus four new tests. Initialization succeeded this run, so the failure observer was not invoked; earlier allocation failures remain unexplained.

The first actual 3968+128 text ordinary observation was saved before the strict token assertion failed. First mismatch is zero-based output 57; 60/128 IDs agree. Empty/reset state and vectors are exact. At prefill, the first different complete state component is layer3 full-attention KEY (role1), and its VALUE (role2) also differs; 121/128 components differ. Layer0/1/2 state hashes match, which does not establish hidden-input equality. Complete 40960 finite hidden/normalized scalars and 128-component geometry/slots are independently checked. Prefill hidden maxabs31.052886962890625/relativeL2 0.23968307780975426; normalized maxabs3.811379909515381/relativeL2 0.25409461762346935, no exact scalar matches in either prefill vector. MTP generation, image and remaining five cases were not run. No quality/performance admission.

All 978 source/1001 installed pins, runtime, selected SDK, checkpoint and environment restore checks pass; 12 existing targets restored and 23 new targets removed. Full archive ae8ebbfac5d0abb94977873d67e5a325aa09d643d067e1013a3c80d82c580d68 (1520402 bytes/91 members) passes all CRC and file SHA checks. Failure verifier expected exit1 and independent analyzer exit0 on both machines, with byte-identical proofs ac88c51f39a114ec0f5289d86c6d1c154ad4eb461631fe55978ad07607190e6e and 3bd9b35325b015b9e82f1a7145221b1cad4b62a164b5f89c7b543ab46e836859. Raw errors, resources, inputs/IDs/vectors, derivations and initial generator source retained. The initial generator expected one launcher reference but found two; corrected exact count2 before deployment, no GPU work from that preparation error. Sampled adapter memory is correctness-only, not a performance result.

Next: isolate the layer3 input/projection/cache boundary with ordinary T32/T128 real-input controls and bounded complete cache reads. Existing full-attention diagnostics are T16 embedded-only and reject canonical KV; no T128 diagnostic support is assumed. Identify first differing token/head/dimension and reproduce the retained baseline before attribution or code changes. Then preserve unchanged full-model/quality gates and all original HTTP/baseline/INT4/4B/concurrency objectives. Existing charts remain unchanged and the overall 2x decode AND throughput target remains unmet.


### 2026-10-26 — Complete layer3 key/value prefix captures reproduce retained T32 and failed T128 ordinary controls

Same native28 DLL plus seven unchanged Python profile sources; no native/model math changes. Separate exclusive ordinary-only T32 and T128 processes use the actual first text3968+128 case. Preserve all70 earlier model/state/vector/ID/MTP CPU controls plus4 bounded-cache/geometry/finite/index/raw-vector faults: local2.534s, remote2.338s. Candidate policies/counts are0 for T32/T4 and48each for T128 only. No vision or MTP graphs are created in this diagnostic. The full six-case admission gate remains separate and failed.

Both processes capture COMPLETE layer3 key and value caches at prefixes128/256/512/1024/2048/3968 through <=1048576 physical FP32-carrier word reads. All24cache arrays (201326592 physical bytes) are retained, including inactive tail slots. Each process reproduces its retained empty/prefill/final128component state hashes, all hidden/normalized vectors and128ordinary IDs. The candidate reproduces the previous failed T128 result, including its divergence; it is not admitted to performance. Independent replay checks all finite BF16 values, inactive tails, bitwise differences and head/token/dimension mapping. Earliest sampled different prefix: 128. Prefix128 is the earliest observation, so this does not claim the precise first divergent operation.

Additional independent analysis verifies all35 source-bundle members,12 original-source members, and four complete final key/value captures against the corresponding layer3 logical component hashes in the retained full-state witnesses. Full archive 0158289d1b6706905dc10fc101f3064eed332a76ebb591c4c6ccf85c87031592 (53806808bytes/154members) passes CRC and all file hashes. Both replay tools pass on both machines with identical proofs de6af0d8d13116a38dd2036e12dc9d7d9f90e3a78f8261b73bd98aa19077d91d and 3b593d0a016f43123ddad4ef28d8048e0fe2bae8985a03b256cb4aee3792676b.989source/1012installed pins plus runtime/selectedSDK/checkpoint/environment/cache restore;12existing source targets restored/23new removed. Source/preparation/deployment/initial generator fault source/logs/resources/raw arrays retained. Initial installer-derivation guard expected4filename references but found3; corrected before installer generation/deployment, no GPU work from that preparation failure.

Source inspection: full-attention M32 canonical policy covers Q/O only, not K/V or T128. Existing diagnostic_full_stages is embedded T16 only and excludes canonical BF16 KV. These are leads/constraints, not causes. Next prove layer2 ordinary hidden output with common real layer1 input; then, if needed, extend a separate strict diagnostic opt-in for layer3 T32/T128 with mandatory same-profile ordinary primary parity and complete finite stage replay. No algorithm patch from cache differences alone. No new benchmarks or chart values; original overall2xdecodeANDconcurrency-throughput target and halfTTFT remain unmet.


### 2026-10-26 — Layer2 ordinary T32 result retained; subsequent T128 initialization fails while separate postfailure helper succeeds

Current native35 source and closed6d4DLL unchanged. Extract actual complete first128 T32 layer1 ordinary hidden output from the closed all8profile stage archive aad7fbece8b229c50a028106e4a241bd568556e025090982305730735bb341a6; re-run full parent eleven-stage/ordinary-primary replay before extracting. Both parent repeats match; common input SHA f855b583bd2f23912afd08e808488e77029feac0e376dc6797cb42a3f82a3895 (2621440bytes). Explicit ordinary layer2 only, old diagnostic-layer2 rejection preserved. All78CPU controls pass local3.635s/remote3.065s, retaining all74 model/cache controls plus4 new lineage/profile/complete-array/derivation fault tests.

Actual ordinary T32 layer2 graph completes3warmups/2repeats/fourcarried32chunks. Independent partial analysis checks58complete finite input/state/output entries and7,929,856F32 outputwords, readonly inputs, all carried history/state and exact repeats. Subsequent T128 graph fails TensorRT PluginV2Runner.allocateResources/buildSerializedNetwork before any T128 output. Layer2 T32-versus-T128 arithmetic remains unobserved. Keep the successful T32 output; do not promote a partial result to model/MTP/image/quality or performance.

The unique postfailure native ABI24 hook executes before collector cleanup, calling the unchanged observed Z128 helper in the same process/thread but a separate translation unit. It succeeds: firstsuccessfulindex0, expectedfullopaque atindex0, IDs[67,67,35,35,35,35,67], algorithm67, requiredworkspace0/4MiB supplied, actualM128AlgoCheck succeeds, CUDAerrorsbefore/after0,0matmulcalls. Raw report fa1049569ab43f21a6a86eb515db688bb4ac9dd90a9aa4a794dc168063517812 retained. This contradicts any claim that the postfailure helper itself failed; it does not identify the actual Garnet plugin or its earlier failing call. No opaque-selection rootcause or initialization-reliability fix is established.

Fullfailure verifier expectedexit1 and separatepartialanalyzer exit0 pass on bothmachines with identicalproofs 0abe430aa87746d0d26eac795fd4d62f10fb4088959df1774d7624cd9198e626 and 951ad430af3c1fabe8b4a5c85f1ed30604195fa1017c7a3c18daee226e854c3f.1002source/1025installed pins plus runtime/selectedSDK/checkpoint/environment/cache restore;12existing targets restored/23new removed. Original archive e6f2817400bcfe0ea082858e6259275eea45e31474e4efc81e5703b940dc1ced (76306493bytes/160members) reconstructed byteexact fromlosslessdeduptransport, fullCRC/allfileSHA verified. Preserve all raw arrays/logs/resources/constructor report, fixture/derivations/source and deployment assets.

Next implement a default-off trace in the actual Garnet FP8 ordinary/M128/Z128 and NVFP4 plugin initialize paths, with unchanged factory/SDK/algorithm/descriptor/workspace/wire/return behavior, CPU stub/source parity tests and separate remote DLL build. Determine the real failing route before a helper-level patch. Then resume layer2 arithmetic, layer3 controlled-stage/full-model linkage and unchanged full4K/quality/performance gates. No new benchmarks; charts unchanged; original2xdecodeANDconcurrency-throughput and halfTTFTextra remain unmet.


### 2026-10-26 — Actual four-plugin initialization trace builds and passes CPU controls; GPU failure route still unobserved

Added private exact CPP derivatives for ordinary FP8, NVFP4, GDN-output M128 and GDN-Z128 initialize only, plus default-off synchronized stderr header. Only exact GARNET_MODELOPT_INIT_TRACE=1 emits paired records with route/instance/shape/policy/context/factory/return/exception. Disabled branch preserves original factory/cached/null/catch/return behavior. Four complete CPP compositions and all other prior35 candidate source bytes verified unchanged; no helper/algorithm/descriptor/workspace/wire/enqueue implementation changes. Local production source is unmodified by this trace work. Candidate38 =native31+seven unchanged Python files;14existing/24new remote restore.

Preparation-only old28 bundle-count guard corrected before installation, and inherited prepared-status proposal corrected before the second proposal was uploaded; initial code/packages/metadata remain immutable. Executed native-build attempt1 stopped after81CPU tests with one old whole-file M128 integration-provenance assertion, before any C++ compilation or GPU work. Preserve archive 152d616458b28f71458eecc8ca39031bb0db35fc745361d3e1081c85ba82c791 and identical failure-proof pair 35a2be986e284fe0c391543ad90e77ef4d51bf8c8c33c551ddc436dd61f35d39. Both historical policy/integration byte-provenance assertions now execute in a composed original-source view whose originals and exact trace-only substitutions are independently validated; numerical assertions remain unchanged. The81-test installed-source-view simulation passes without source writes/native/GPU calls.

Attempt2 passes81CPU controls (remote3.131s; local3.625s plus installed-view3.646s). Actual four-plugin CPP CPU-stub executable passes10fresh policy processes:360initialize calls/280factory calls/0GPUoperators. Nine disabled policies including unset/empty/0/malformed values are silent. Enabled process emits72 paired records for36calls;8cached-context reuses,16standard/nonstandard exception returns and24intentional simulated failures. Wire/clone bytes unchanged. These are CPU fault controls, not real GPU initialization failures. /EHsc used and noC4530; existing SDK charset/deprecation warnings retained, SDK unmodified.

Full remote reused Release/MT/sm120 build succeeds; saved DLL d3f15b2e3e5bf4f73042883c620bf74805dcb75f3561c9b62ab01078c4c74516. Existing sixteen-profile T128/B1/B4/B8 wire/descriptor contracts, actual M128/Z128 wire/default/cross-family tests, pure GDN guard, ten external+ten embedded selector cases and955tokenizer cases pass. CUDA object set remains34 with no additions; changed objects recorded in raw workflow.1011source+33method pins/1068installed pins, runtime, selectedSDK/checkpoints and CMake cache restore;14old targets restored/24new removed, baseline1a1dDLL restored. Success archive 6e007e86fbe4d04469fe4ad26567b266f6b6cf6f52864b7778c81827e5e5b0f3 (5162495bytes/167members) CRC/allfileSHA/sourcebundle38/originalsource14 verified. Both-machine success proof identical 5653932def2b6efd68540b1f6007a6bc9984b09a62e04d553aac01c49f350102. Raw source/binary/build/fault logs and unsuccessful/preparation assets retained.

No GPU model or trace-on numerical test has run with this DLL. Next source-frozen actual layer2 trace-on observation with profile markers, complete event pairing and retained T32 raw-array parity; if successful, fresh trace-unset output parity. Identify the real failing initialize route before helper-level diagnostics or an algorithm patch. Then preserve unchanged full4K/quality and all original performance/baseline/INT4/4B/concurrency objectives. No benchmark or chart values changed; overall2xdecodeANDthroughput and halfTTFTextra remain unmet.


### 2026-10-26 — Actual Z128 plugin initialize returns1 from an empty context; underlying helper guard still unproven

Source-frozen native38 trace-on real-input ordinary layer2 observation uses qualified d3f15 DLL.83CPU controls pass local3.614s/remote3.13s. T32 completes with58complete finite array entries byte-exact to retained earlier control; T128 TensorRT build fails before outputs. Twelve actual native initialize calls have24complete begin/end events:7ordinaryFP8,4NVFP4,1Z128. First11return0. Call12 fp8_z128 N6144/K5120/policy0x5a313238 calls its factory and returns1 with no context and no exception. This directly locates actual Garnet failure at CreateModelOptFP8Z128Context returning empty. Underlying CUDA/cuBLAS status, device/version guard, opaque heuristic mismatch or algorithm-check cause is not yet identified.

Postfailure sameprocess/thread separate-TU helper succeeds on firstcandidate0/expectedopaque/ID67 with0matmul; it is separate evidence and cannot explain actual plugin failure. Python stdout BEGIN/READY markers arrive after native stderr despite flush=True; all call profile fields remainnull. Preserve this ordering limitation; do not claim chronology worked. Trace-unset process was not run after failure, so no on/off numerical parity or reliability claim.14existing/24new source targets, runtime, selectedSDK/checkpoints restored. Failure archive 9168007668d03d4971f11ca380ee8abb000d3ee639125ccea94d9fe87bf9acc6 (76037709bytes/96members) passes CRC/allSHA/sourcebundle38/original14 and both-machine frozen checker with identical proof 8b9db3523e0db49a428da0a11b935833df03b38cb28367e091ff7f8961399cea. Raw arrays/logs/failure and transport retained; no benchmark/chart values changed.

Next: actual constructor SDK/guard telemetry with no extra SDK queries or algorithm changes; CPU forwarding/fault and exact-source controls, then remote native build and controlled real-input rerun. All original full4096 quality/MTP/baseline/INT4/4B/concurrency objectives retained; overall2xdecodeANDthroughput and halfTTFTextra remain unmet.


### 2026-10-26 — Actual Z128 constructor trace builds;48CPU fault modes preserve original SDK call transcripts across10policies

Private exact derivative of real Garnet typedZ128 helper adds defaultoff native stderr records at17existing SDK expressions and10original empty-return guards, cache/created returns and full already-returned64byte heuristic choices. Every SDK expression forwarded once in original short-circuit order. No extra SDK queries, opaque search/substitution, descriptor/workspace/math/wire/enqueue changes. Other37prior candidate files remain byte-exact; one new header yields native32+samePython7=39sources. Local production source unmodified by this trace.

86CPU source/model/cache/failure controls pass local3.726s, simulated installed-source view3.706s and remote3.121s. Historical policy/integration assertions still execute through separately verified exact helper composition. Remote CPU-only fixture compiles actual original/traced helper CPP into two executables without any CUDA/Garnet/SDK DLL linkage.48modes cover every original SDK failure/exception position, device/version, no/all-failed/skip-failed candidates, opaque mismatch including later matching candidate, ID/length, state/workspace, success and weak-cache reuse. Across10policies,48original+480traced processes have identical complete SDK call/argument/order/cleanup/return/exception/cache transcripts and0matmul.432disabled-policy processes silent;48enabled processes contain49factory calls. CPU synthetic faults do not identify the actual GPU failure. Existing360actual plugin CPU initialize contracts retained.

Full remote Release/MT/sm120 build against real SDK headers succeeds; DLL 8f5a6c3cb9252f322e536c1d0a5ba9baf1eebf71eeae32f54910502de8c14365. Original T128/B1/B4/B8 wire/descriptor, M128/Z128 wire/default/cross-family, GDNprofile,10external+10embeddedselector and955tokenizer CPU contracts pass.34CUDAobjectset/noadd; only priorallowed3objects rebuilt.14old/25new source, runtime, selectedSDK/checkpoints and CMake cache restored. Archive 7ba7692cd0dd277475bbf76ded2396ed9d087ca1fce8058067e23b6ab040c2ce (6110718bytes/1172members) passes CRC/allSHA/exact39source and14original-source checks. Both-machine proof identical e5a8fc1098688371003918ef17cb56b01daab8ac04e711f16ad61ea2bebc93fa. All480rawstdout/stderr pairs retained in immutable archive; report/48originaltranscripts/48enabledtraces also inGit.

Next actual source-frozen real-input layer2 trace-on observation will join native stderr factory records to the single unambiguous enclosing Z128 plugin call, rather than delayed Python profile markers. Keep all finite/carry/input/retainedT32 controls; if complete, require fresh trace-unset parity. Then unchanged full4096 ordinary/MTP/quality gates and all matched baseline/INT4/4B/concurrency objectives. No new benchmark/chart values; overall2xdecodeANDthroughput and halfTTFTextra remain unmet.


### 2026-10-26 — Actual Z128 constructor rejects first heuristic on opaque bytes34/35; every preceding SDK call succeeds

Real-input ordinary layer2 trace-on run with native39 DLL8f5a6c passes88CPU controls (local3.649s/remote3.088s). T32 completes and58full finite array entries remain byte-exact to earlier retained control. T128 TensorRT initialization fails; actual plugin call12 contains factory1 with26native stderr records unambiguously nested.15existing CUDA/cuBLAS expressions succeed (cuBLASLtGetVersion130400); heuristic returns7successful candidates. Firstcandidate fails unchanged64byte memcmp; offsets0based34/35 expected00/00, actuald3/bb. Constructor returns empty, pluginreturns1/noexception before ConfigGetAttribute/AlgoCheck. Actual algorithm ID/configuration is not established by interpreting opaque bytes. Separate postfailure observer succeeds firstcandidate0/expectedopaque/ID67/0matmul; preserve its different scope. No off process ran after failure.

Source1080/installed1105pins, native32+samePython7,14existing/25new source targets/runtime/selectedSDK/checkpoint restored.98member archive eb08622e56ddd3d96eaf1d03de2e09f36d836df762612560a3813cec06a1e931 (76050695bytes) reconstructed byte-exact and replayed on both machines with identical proof d9e239748ce3ee3b8744e78508fb29dcbd3f6163fa2c760fec097080c4e1f787. Native marker attribution uses one unambiguous enclosing plugin call and does not use delayed Python stdout profile markers. All raw arrays/logs/SDK candidates/failure and deployment methods retained. No kernel/algorithm change or benchmark admission.

Next read nine public configuration attributes from both retained opaque descriptors using pinned NVIDIA SDK getters. NVIDIA cuBLAS13.4 documents opaque descriptor serialization and getters: https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgoconfigattributes-t (accessed2026-10-26). No assumption that differing bytes are padding/reserved; even equal public fields do not establish functional or numerical equivalence. Only after actual configuration evidence consider a separate strict candidate and unchanged controlled/full4096/MTP/quality gates. All original baseline/INT4/4B/concurrency objectives retained; overall2xdecodeANDthroughput and halfTTFTextra remain unmet.


### 2026-10-26 — Both retained Z128 descriptors report identical nine public configurations; opaque bytes34/35 remain unexplained

Read-only standalone SDK audit preserves exact expected and actual-first-candidate64byte descriptors from paired actual plugin failure evidence. NVIDIA cuBLAS13.4 documents opaque serialization and nine public configuration attributes (https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgoconfigattributes-t, accessed2026-10-26). Query program links pinned realSDK library only, verifies loaded root build/bin/cublasLt64_13.dll path/hash/version130400, and performs GetVersion+18ConfigGetAttribute calls. No setter/AlgoInit/AlgoCheck/context/GarnetDLL/model/matmul call.

Audit attempt1 stops before SDK compilation/query:89CPU controls on restored baseline report1failure/17missing candidate-source errors because these model tests require installed39candidate source. Failure preserved, not treated as model or SDK result. Archive b0f864b32520f79b72851749ca6d881e728aa05f21ff684660d7d87dfebdbfca / proofpair f8c1b65259f1952c8dadf751f2b289461c916baf81f0482bfafba6488e1a8bf1. Separate attempt2 leaves all model assertions unchanged and instead replays complete frozen parent88 installed-candidateCPU/source/raw58T32/actualfailure/restoration evidence with exact parent proof d9e239; then runs query-only fault unit on restored baseline. Query invalid/incomplete statuses cannot be promoted to equality. No production source/runtime/SDK writes.

Attempt2 realSDK compilation and all18queries succeed. Both descriptors report ID67, tile13, stages17, splitK1, reduction0, CTA-swizzle1, custom18, inner-shape0, cluster-shape0. All requested/written bytes match documented4byte/2byte types. Raw descriptors still differ only at34/35. This establishes equal public query values, not padding semantics or full functional/numerical equivalence. Actual plugin remains rejecting its original memcmp; no guard bypass or algorithm patch has been installed. Archive 283a6878eed4da05cc0a994ad62a32f31a295d8923878fb19b897f4513c5c3e9 (569928bytes/25members), both-machine proof b9bac466b1f72dc833ef15f8ca1f79662e94f7e4428354263a683af1efbcc53d. All build/query/fault logs, inputs, identities and first unsuccessful harness retained. Deployment receipts/native preparation logs for prior trace work also retained.

Next separate defaultoff public-API configuration reconstruction candidate, original type/layout/transpose/workspace/SM120/version guards and SDK checks, CPU factory/source/default/wire/fault controls, then unchanged real128 layer2/allGDN/layer3/full4096 ordinary/MTP/quality gates. No byte masking, no inferred equivalence, no oracle/tolerance change. Original strongest-baseline/INT4/4B/concurrency objectives and overall2xdecodeANDthroughput/halfTTFTextra remain unmet; no new benchmark/chart values.


### 2026-10-26 — Separate public-API Z128 factory passes CPU and realSDK construction; Windows hybrid runtime identified, model integration next

Appended a separate explicit CreateModelOptFP8Z128ReconstructedContext to a private exact helper derivative; no existing plugin/default caller is redirected. Legacy constructor body andGetInfo/RunMatmul suffix remain byte-exact. New factory has independent weak-context cache and originalSM120/cuBLAS130400/types/transpose/M32+M128layout/4MiB guards, AlgoInit67,8public setters,9exactvalue/width/status getters andtwoactualdescriptorAlgoChecks. No opaque byte masking/copy, heuristic search or fabricated equivalence.

Construction attempt1 fails CPU-source preflight before any native/SDK compilation because the exact public configuration proof was omitted from its evidence path; archive b034de9453d805a2ccee2278d8feff02c49603b106d639ea157e0fdbf429ea05 / pairedfailureproof 72a5dbc38900aecf273d8da8b3f936788d618779275fd926c577e24ceba42aa6. Attempt2 explicitly ships/pins that dependency, keeps all3assertions/candidateCPP/CPUdriver unchanged, and passes1400actual-helper CPU fixture processes:480legacy transcripts exact to retained originals,910new constructor32SDKfailure/exception/configdrift/cache/device/version cases,10unsupportedCUDART. Ten policies per mode preserve complete SDK call transcripts;1260disabled processes silent. Independent920case argument replay validates17210initialization calls including original layout/transpose prefix, FP32scale/compute, E4M3A/B/BF16C/D,67,8set/9getwidths. Both-machine argument proof bfdb662b10c24989d986979d51d461e3016bdbb6c309f0953988016815fc6afd. These are CPU fixtures, not GPUmatmul or model tests.

Attempt2 realSDK on/off constructor/cache/config/M32+M128 checks return0, but the original frozen archive checker rejects cudart module provenance: namedDLL handle wasnull and GetModuleFileNameA(null) reportedEXE. Preserve original failed checker and quarantine full1400CPU/SDKdata with pairedproof a31cefb83ecb78d58062977e557115793137bafd1bed02a8f127f4d221d9ea1c; no active-runtime identity or numerical claim from it. Later runtime-origin attempt1 records loader code module byaddress/linkmap/driver, but stops aftertraceon on a wrong header==runtime assertion. Its original static-implementation label and failed validator remain preserved, archive 6dff84f8ccf6706f56cd5aa52ceffef1c91f35c49accc019bb49d2546f8aa84d / failureproof 3506ffde3411900fa7695cc945828b77a93b1bc82ba57f4cbb3b7e3907557904. Headers13020, active runtime13040, driver13040. NVIDIA CUDA13Windows uses a hybrid statically linked APIloader that obtains runtime from display driver; runtime version need not equal toolkit version. Primary references: https://docs.nvidia.com/cuda/archive/13.2.0/cuda-compiler-driver-nvcc/index.html and https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART____VERSION.html (accessed2026-10-26). Correct the metadata protocol, not library/header/source/algorithm or model gates.

Separate hybrid-specific observation reuses1400CPU through full frozen post-replay and leaves constructorCPP/header/CPUdriver hashes unchanged. Two metadata source/fault controls then realSDK on/off fresh processes pass. APIloader code originEXE, actual cuBLASLtrootDLL hash/version130400, nvcudaSystem32module path/hash, exact header13020/runtime13040/driver13040, generated_cuda_runtime_api_loader.obj linkmap and PEimports (cuBLASLt+Kernel32, noCUDARTDLL) verified. Both SDK processes create/cache context, query all9configuration values, return0 onM32/M128AlgoCheck with0required workspace; traceon53events (47created+6cached), unsetstderr silent. Archive e0a65d8da7b0917d0650cbe5f5378d486f4847da5a57a43cdb10133e97ff2bbb (842101bytes/41members) and both-machine proof 634df123bf6c118bfaee95731d9ca49345f2c96185ce7df0b2b87f78df12166e. All unsuccessful/rawbuild/linkmap/provenance/resource logs and source identities retained; SDK-only checks are not arithmetic equivalence or initialization reliability inside TensorRT.

No production source/library was installed or modified, no model/matmul invoked and no benchmark/chart values changed. Next distinct R128registered plugin/wire/defaultoff lowerer/counter integration, actual CPU wire/descriptor tests and fullremote nativebuild, then real128 layer2/fullGDN/layer3/full4096 ordinary/MTP/quality gates before timing. Original INT4/4B/strongestbaseline/repetition/concurrency objectives and overall2xdecodeANDthroughput/halfTTFTextra remain unmet.


### Separate reconstructed Z128 plugin builds in actual native Garnet; real layer2 correctness pending

2026-10-26: New private ModelOptFP8Z128ReconstructedPlugin/Creator uses distinct type and20byte marker0x52313238, qualified explicit CreateModelOptFP8Z128ReconstructedContext, strict defaultoff GARNET_FP8_GDN_Z128_RECONSTRUCT requiring canonicalZ128=1 and exact B1/T128/128rows/6144x5120/logicalGDNZ. Independent reconstructed counter reset/copy/stats; old family/wire/default behavior and quantizer/enqueue/math remain exact compositions.43native/Python candidatefiles,14old/29newtargets restored after every attempt. Seven source/retry controls retain86historical assertions in explicit frozen parent-source view; this is source evidence, not new-route arithmetic admission.

Attempt1 passed newwire34badwires/18badconstructors/23baddescriptors/11badformats/8policyrejections/7crossfamily rejections/workspace4849664bytes; next initializationCPU build failed C1083 because inherited script selected earlier private source directory lacking newpluginCPP (and mixed canonical headers). No fullnative/GPU model ran. Retain original failed source/checker and raw logs: archive c62e40d9403e555989a5d4165a46933a23dd33872d7bc7306d5f0a153f03037f, pairedproof d7b36a6b84cef9a4a34ef9c62a11f723aaabf15040e0c34752cb69c43386a28e. Attempt2 changes onlyCPUbuild source directory to installedsrc/backends/tensorrt plus exact attempt artifact/test bookkeeping; all43candidatefile hashes unchanged.

Attempt2 actual remoteCPU controls pass: tenfresh tracepolicy processes450initializecalls/350CPUfactory calls, old/newroute counts, success/null/standard+nonstandardexceptions/cache/clone/wire,0GPUoperators; onlyexact1 emits90events, other9silent. Newprofile158cases/62rejections/onequalifiedselection; newandoldwirecontracts pass. Fullremote Release/MT/sm120 Garnet builds, no addedCUDAobjects, allowedunchangedkernel recomposition only; actual native/embeddedselectors andtokenizerpass. SavedDLL b8b01ca44c606e9f7d47ad72e2e0092d40b4376457667dcb6ee1ba8d1bda0e2f. Archive ae7b65ef87c9bbbb69a894ff6f4123c7d4754518d7502c228ad2c8729ba9c354 (5293828bytes/159members), both-machine proof 9e572f7ff25c298ff46ea422db6df65804aa4af5dd45c8e6c53db5cab9171be5. Source/runtime/SDK/checkpoint/CMake restored and allfailed/compiler/resource/provenance evidence retained.

No GPUmatmul/model or newbenchmark value follows from this nativebuild. Actual realinput layer2 T32/T128/new1old0 gate is next; require full73finite/readonly/carried-state/repeat controls, retained58T32entries, all655360hiddenF32words andfinalhistory/state byteexact, thenGDN/FA3/full4096ordinary/MTP/quality before timing. INT4/4B/strongesttunedMTPbaseline/repetitions/trueconcurrency and2xdecodeANDthroughput remainunmet.


### Actual reconstructed Z128 factory fixes layer2 initialization and preserves exact T32/T128 outputs

2026-10-26: Built candidate DLL b8b01ca44c606e9f7d47ad72e2e0092d40b4376457667dcb6ee1ba8d1bda0e2f ran inside actual native Garnet on the pinned RTX 5090, using the fixed study root and the same retained real 128-token layer2 input. The separate registered R128 plugin, explicit opt-in, and SDK-built descriptor initialized successfully in two fresh trace-on/unset processes. Each process completed both T32 and T128 profiles, three warmups and two repeats. Graph counters select reconstructed=1 and legacy Z128=0 for T128, both zero for T32. This resolves the observed real layer2 constructor failure for this candidate; it is not broad initialization or full-model qualification.

Independent replay on both machines verifies all 73 finite-array entries per process, 10,895,360 complete output F32 words per process, unchanged inputs and carried history/state, and exact repeats. All 655,360 hidden F32 words match between four T32 chunks and one T128 chunk: zero changed words, and final history/state are byte-exact. All 58 retained original T32 entries remain byte-exact. Trace-on and trace-unset processes match all 73 entries. Native stderr ties the documented 32-call factory/configuration/check sequence to the actual reconstructed Garnet plugin; this is actual execution rather than the earlier separate-TU observer.

The original frozen verifier rejected only its comparison of integer trace-map IDs in memory against JSON string keys. Preserve the original verifier and failed replay log. A separately named exact source derivative compares the complete JSON representation of that same validated mapping, with no array, state, counter, native event or parent-gate relaxation. No GPU rerun or model source change was needed. Archive 650746a83309b6c5c745b53e018aa37fbc461a42ced6bfbbe985bd5002b67186 (182951963 bytes / 194 members); paired proof 552425ad86b9f586ff938514b29342deeaa51d32938bc87467867ffb9c60334a. Source/runtime/SDK/checkpoint are restored; raw arrays, resources, traces, commands and source hashes remain in the immutable archive, with replay methods and summarized evidence in Git.

A 2,621,440-byte common FA3 input was extracted only after paired proof, all array gates, original T32 parity and cross-process byte checks; SHA256 1f5f7340cac51fc1c56157fcf10a623d901de81e1456a74674b6408d5282c72d. This is a controlled real layer2 output for a same-input layer3 witness, not a claim about the complete 3,968-token candidate chain. Source inspection confirms the current full-attention canonical policy covers Q/O only; K/V remain generic. The follow-up plan requires the original all-stage GDN controls, then a distinct explicit mixed-FP8 layer3 normalization/K/V witness using existing Garnet primitives, before any descriptor intervention or cache attribution. Preserve the old INT4-only projection and T16 diagnostic guards.

The unchanged six-case full 3,968+128 ordinary/MTP state/vector/token oracle, FA3 cache linkage, adapted 805 quality gate, native concurrency correctness, strongest feasible tuned/MTP reference repeats, INT4 W4A16 and Qwen3.5-4B comparisons remain required. No benchmark/chart value changed. The overall 2x decode AND throughput target and half-TTFT additional objective remain unmet.


### Explicit FA3 normalization and K/V projection witness builds in native Garnet

2026-10-26: Add a distinct qwen35_fp8_fa3_projection_witness unary node to a private derivative of the qualified 43-source candidate. The new 44-source bundle changes one CPP by additive branch/include and adds one header. Removing these additions restores every parent byte. Require explicit GARNET_FP8_FA3_PROJECTION_WITNESS=1, embedded modelopt_mixed weights, layer3, F32 B1 T32/T128 hidden5120, epsilon exactly 1e-6, and native FP8 K/V matrices N1024 K5120 with exact checkpoint names. Use the same ops.norm/ops.project primitives as ordinary full attention. Pack normalized5120 + K1024 + V1024; T128 output917,504 F32 words fits the existing bounded reader. Old INT4-only checkpoint projection and layer3 T16 diagnostic guards remain unchanged.

Actual remote header tests pass 47 cases: 45 invalid configurations rejected, two intended profiles selected. Ten CPU source/retry checks preserve 86 historical assertions in their explicit frozen parent source view. Existing actual plugin wire/default/cross-family and 450 CPU initialization/350 stub factory controls pass. Full Release/MT/sm120 native build, native/embedded selectors and 955 tokenizer cases pass. No new CUDA objects, only the earlier allowed unchanged-kernel recomposition. DLL e45950ed41b297cf04430d5835aa1fc5153def820b5bb239468a08d79453a4c0; archive 4c45c046b0c37cdf7578aef93c0521c149ba7959da6615c7a8a24ecac1260bb3 (5341591 bytes / 139 members); independently replayed on both machines, proof 263d92f38419cf746b17af2b5eb45d92804fd7e271b1c80a8695d90fa1e0d78c. Restore 14 old and 30 new targets, native library, runtime, SDK, checkpoint and CMake identities.

This build does not establish normalization/projection equality, the FA3 cache cause, full-model quality or performance. Observe the unresolved K/V boundary before rerunning unchanged all-stage GDN controls; this reorders diagnostics, and does not waive the original 509-entry GDN, six full4096 ordinary/MTP, adapted805 quality or concurrency gates before promotion/timing. Overall 2x faster-baseline decode AND throughput and half-TTFT objectives remain unmet.


### Actual FA3 K/V projection differences reproduce the retained VALUE cache mismatch

2026-10-26: Run the explicit actual Garnet layer3 normalization/K/V node on the admitted real layer2 output, first as four T32 chunks and then one T128 chunk, with three warmups and two complete repeats. Attempt1 fails before IO/graph construction because the collector imported a NumPy-dependent replay module in embedded Python. Preserve original source, raw ModuleNotFoundError, sampler, failed archive c78b06273762b449ac1c8b4378641fd34ef6c4eff09417fbb8f53d99ade80426 and paired failure proof b78ed608deda61d3d42008adb392b1a47ef32fc2c7d8d428c86c16613a7602ce. No inference arrays exist.

Attempt2 copies only the exact stdlib metadata body into an embedded-compatible module and changes that import, attempt paths, and a real embedded import-only preflight. All 44 native source hashes, DLL, matrix/data/shape/scales and numerical replay gates remain unchanged. The actual embedded preflight passes without requesting Garnet/GPU resources. Fifteen CPU controls, including 86 historical assertions in their frozen parent view, pass before the model. Archive 75ce1c0bac482b86f2fadd7445f7c3bb3ab32499946d7a87e72ef00f46a1732f (30991827 bytes), both-machine proof d6511e7dc3e4c818d7797970308c37c6fffa37fc48bdee4d57fefc6d7befde93. All 51 finite-array entries / 3,670,016 complete output F32 words are replayed; inputs are unchanged and repeats exact. All 655,360 normalized values match across profiles. Of 131,072 values in each projection, K differs at 8 values (max abs 0.00390625; relative L2 1.6888510416297236e-5), and V at 8 values (max abs 0.0001220703125; relative L2 6.995786540330331e-7). First K difference: token11/dimension882. First V difference: token11/dimension482. No tolerance/oracle was changed.

An independent byte mapping of the actual BF16-representable V outputs into head-major [4,4096,256] canonical storage reproduces both complete retained first128-prefix VALUE caches exactly: 4,194,304 BF16 values per profile, inactive tails zero. All 8 retained VALUE differences are already present before cache storage; first cache difference is head1/token11/dimension226 (BF16 bits48065 versus48066). This establishes the value-component linkage for this recorded first128 case, not KEY norm/RoPE or the remaining3968-token chain. Both machines replay the same linked proof 6e9c1f00cdc7ff8095b24e683284157d49f3dae66a4d73ba988832c2bd7bc8cc.

A third fresh process enables NVIDIA documented cuBLASLt logging only (CUBLASLT_LOG_LEVEL=5, LOG_MASK=31, new owned log file; https://docs.nvidia.com/cuda/cublas/index.html). Native DLL/source/data remain unchanged. All 51 input/output entries and metadata match the unlogged observation byte-for-byte. Vendor archive d0ce2d2992c27d4b782438cc45ac47630fba1d0ed59e349da4af3e124794e12d and paired proof cf47ddcd00b19ca2e4688584f3193f251633e6c7366f0a4b24df60397b8ada07. Eight actual Matmul trace records (four per profile, two weight buffers) use algorithm ID35 and tile64x64 / compute-type reduction. T32 uses stages64x6, splitK7, CTA swizzle1; T128 uses stages64x4, splitK5. Do not infer unprinted CUSTOM_OPTION, INNER_SHAPE_ID, CLUSTER_SHAPE_ID or omitted T128 CTA swizzle. Actual pinned SDK headers confirm tile64x64=15, stages64x6=18, compute-type reduction=2. Both-machine vendor selection proof 2af63301c4625896fd8d788278faae06b9934bf56ca810f5ba264c46351b6919. Clock values are not merged across vendor/Python/hosts.

The configuration difference is observed; a configuration intervention has not yet been tested. Next is a separate default-off, distinct-wire T128 K/V context/plugin using the live SDK M32 control, six typed observed-value guards, and AlgoCheck for both M32/M128 with 4MiB workspace. Retain the SDK-returned algorithm value, never synthesize/import/mask opaque bytes or invent unprinted parameters. Same quantizer/scales/output type; preserve old defaults and add a separate counter. Require original T32 arrays exact and all T128 normalization/K/V words exact to T32, then original full GDN, KEY/cache linkage, unchanged six full4096 ordinary/MTP state/vector/token oracles, adapted805 quality, concurrency and matched strongest-reference gates before timing. All source/library/runtime/SDK/checkpoints were restored. No benchmark/chart value changed; overall 2x decode AND throughput and half-TTFT objectives remain unmet.

Final CPU verification after the linked analyses: 20 controls pass, including the 86 historical assertions in their frozen parent view. These include actual-source composition, complete-array and metadata faults, readonly/repeat checks, vendor-log descriptor/query/process mutations, and independent token/head/dimension mapping with non-BF16/nonfinite rejection. The final command ledger is retained; these synthetic fault tests are not hardware benchmarks.


### Opt-in KV128 canonical control: actual native build and CPU contracts

2026-10-26: Implement a private default-off K/V context/plugin selected only by GARNET_FP8_FULL_ATTN_KV128_CANONICAL=1 for B1 T128, logical text K/V names in layers3,7,...,63, N1024 K5120. Query the live pinned cuBLASLt SDK's M32 control; require the six previously observed typed public values ID35, tile15(64x64), stages18(64x6), splitK7, reduction2(COMPUTE_TYPE), CTA swizzle1. Require successful AlgoCheck at both M32 and actual M128, each within4MiB, then retain the SDK-returned algorithm by ordinary typed assignment. Do not import, synthesize, mask, or compare opaque padding, or assume the three unprinted attributes. Keep the original quantizer/scales/output/alpha/beta behavior, previous defaults and all numerical/oracle gates. Separate plugin type GarnetModelOptFP8KV128Canonical,20-byte serialization marker0x4b313238, FP8_KV128 initialization route, factory trace and projection counter prevent cache/wire identity ambiguity. There are50 native candidate files (44 qualified parents plus6 additive files,5 additive compositions), no new CUDA kernel; original14old/36new targets are restored.

Build attempt1 fails before CPU compilation or GPU execution because source-derivation validation reads a mutable installed legacy header instead of the frozen original used to derive the new header. Preserve failed archive df2331065b827246ebd18c5250bea0fd85cef8b268d9333adc09463bd4b5ac39, paired replay and complete restoration. Attempt2 pins that exact derivation input in a private text fixture and adds an installed-header regression. Every one of the50 native source hashes remains unchanged (source ZIP a0063cb10bf32144058fee174c9e0959adcf91d21bf34a9901dfe2e291f17246); no SDK, matrix, numerical threshold or model change is made for the retry.

Actual remote CPU contracts pass:15 source checks plus86 historical assertions;158 actual helper fault/trace/unsupported-runtime processes with zero Matmul/GPU calls;278 pure-header cases covering32 exact K/V names and246 rejects; new wire34invalid payloads/18constructor/23descriptor/11format/8policy/7cross-family rejects;540 actual plugin initialize calls and420 CPU factory stubs across six routes/ten environment policies. Existing parent wire,450 initialize/350 factory and profile controls remain passing. Full Release/MT/sm120 build, native and embedded selectors, and955 tokenizer cases pass. Candidate DLL 5dff30dd8d2960bcfe64cc25d5014874bd893a3649407ea157eaa7759179350f; archive eb4a2bb2d97c1e4faa3a20136f8cfc67f6baff9f2bcd9beb24e05f63861f2842 (7323161bytes/546members); both machines independently replay byte-identical proof 797ec2870d3f21dcbc1a97de542f0f730dd8881c145234f136dfb9a49b52abd4. Preserve all baseline source/library/runtime/SDK/checkpoint/CMake identities; the saved candidate is separate from the restored default library.

These results qualify the native build and CPU contracts only. Next is the exclusive real-input registered-node test: require counter2 atT128/0 atT32, exact SDK checks and distinct native route, all51 finite entries/3,670,016 outputF32words, exact old T32/all normalized arrays, and every T128 K/V word equal to T32. Retain all SDK rejection/initialization/output failures. Original509GDN, KEY norm/RoPE/cache, six full4096 ordinary/MTP state/vector/token/oracle and adapted805 quality gates still precede performance. No benchmark/chart value changed; overall2x decode AND throughput and half-TTFT objectives remain unmet.


### Actual KV128 intervention removes the observed projection mismatch

2026-10-26: On the same dedicated RTX5090 and pinned mixed NVFP4/FP8 checkpoint, execute the registered projection witness using the new default-off KV128 path and unchanged admitted real layer2 hidden128 fixture. Source50/DLL5dff30dd8d2960bcfe64cc25d5014874bd893a3649407ea157eaa7759179350f remain frozen. Twenty-four CPU controls plus86 historical assertions and actual embedded stdlib-only preflight pass before GPU work. Actual graph counters are0KV128 atT32 and2atT128; native initialize traces identify both legacy fp8 and distinct fp8_kv128 routes with no failures. Four actual SDK factory calls comprise2fresh contexts and2cache hits; both fresh selections validate the six observed public attributes and pass M32/M128 AlgoCheck below4MiB. No opaque algorithm or padding value is imported/patched.

The new gate passes all51finite-array entries and3,670,016complete outputF32words, exact readonly inputs and two repeats. Every T32 control output and all normalized outputs remain byte-identical to the retained parent (3,145,728control output words checked including repeats). AcrossT32/T128 all655,360normalized words match. K and V each compare131,072completeF32words: changed-word counts fall from8K/8V to0K/0V, maximum absolute and relativeL2 differences both0. This establishes the controlled projection intervention result, without claiming KEY norm/RoPE/cache or full-model parity. No threshold, token/state/vector oracle or original admission gate was changed.

Archive 10db54e621fd408fd366733209b4cd26bfca7c7d1a697b50f51a07b1e66eb665 (29190891bytes/87members); paired independent proof 379505acdfb41f4e98393272d50d6f07137217a425a131904a4393773c75c610. The frozen original verifier fails because it expects a copy of the original input fixture inside this archive; the collector retained that fixture in the already linked parent archive instead. Preserve the exact original KeyError command/log on both machines. An exact source-derived postanalysis changes only its self-derivation verification and fixture lookup to the existing SHA-pinned control archive75ce1c0bac482b86f2fadd7445f7c3bb3ab32499946d7a87e72ef00f46a1732f. Require fixture length2,621,440bytes and SHA256=1f5f7340cac51fc1c56157fcf10a623d901de81e1456a74674b6408d5282c72d against both preparation fields. Every numerical, complete-array, control, route/factory and restoration check remains unchanged. Both machines produce exactly the same proof. No GPU rerun or raw-archive change is needed. Baseline source/library/runtime/SDK/checkpoints are restored and the job is terminal.

Next required sequence: rerun the original509-entry fullGDN gate using this exact50-source candidate, then actual KEY norm/RoPE and canonical cache-prefix/carry comparisons through3968tokens, then all six original4096total ordinary/MTP token/state/vector/oracle cases and adapted805 quality. Admit performance only after these pass. Establish matched stronger tuned/MTP vLLM/SGLang repeated reference baselines and true concurrency, complete INT4W4A16 and Qwen3.5-4B comparisons, and measure text/image TTFT/decode/aggregate throughput/serving peakVRAM. The requested2x faster-baseline decode AND throughput and half-TTFT remain unmet. No existing performance chart/data value changed during this intervention.


### Original GDN stage/state gate rerun on the exact KV128 candidate

2026-10-26: Rerun the complete original GDN correctness collection using the same50-source candidate and native DLL5dff30dd8d2960bcfe64cc25d5014874bd893a3649407ea157eaa7759179350f; no native rebuild or new kernel. Keep the original real prompt first128 embeddings, GDN layers0/1, retainedT32 layer0 common input for layer1, B1T32/T128 ordinary and diagnostic profiles, three warmups, two full repeats, bounded complete captures, readonly hidden/history/state, state carry and10negative frontend cases. The original gdn_chunk_stage_replay mathematical code and all numerical thresholds remain unchanged. Only explicit reconstructed-family counters replace legacy-family identity checks: legacyZ128=0, reconstructedZ128=1 atT128/0 atT32, newKV128=0 in GDN. Original M128 and other profile counters remain mandatory.

Twenty-seven source/metadata/corruption/replay CPU controls plus86historical assertions pass before the actual embedded stdlib metadata preflight and GPU run. All8ordinary/diagnostic profiles complete. Replay every509finite array entries and98,205,696complete outputF32words. Diagnostic primary outputs match ordinary, every carried input/state word and both repeats match, and all11stages plus final history/state match betweenT32/T128 for each layer. Additionally compare every509complete entries byte-for-byte to the retained successful original GDN control archive aad7fbece8b229c50a028106e4a241bd568556e025090982305730735bb341a6; all match. This is stronger control coverage than the earlier400T32-only array comparison. The scope remains first128 common inputs in layers0/1, not all48GDN layers or the full3968chain.

Actual native traces show80initialize calls:32fp8,32nvfp4,8fp8_m128 and8fp8_z128_reconstructed; no failed initialization. All8reconstructed factories create successfully with typed configuration getters and M32/M128 SDK checks; no imported opaque bytes or masks. No legacyZ128 or KV128 route is selected by these GDN graphs. Native resources are sampled for correctness only, not as serving peakVRAM/timing benchmarks.

Current raw archive 1db484eb08bbd209f4de1a1a67b78f8d7db5aef2f3e9affb4e2ce518e283f587 (485988065bytes/555members); independent proof f42f0bad61a602405a560c8455e79fa746e59a0f417b2dd9fc55ba5545974240, identical on both machines. Transport reuses only byte-identical compressed payloads from the retained control:483payloads/484,855,919bytes; 1,057,349-byte delta. Preserve every original ZIP header, payload, descriptor and central-directory byte and verify the whole reconstructed485,988,065-byte archive against the original SHA before independent CRC/manifest/full-array replay. Three CPU transport tests cover metadata/order preservation, wrong parent and corrupted/gapped segments. No report, array, threshold, oracle or native archive byte changes. The direct transfer was intentionally stopped after54,165,504bytes; remove only that redundant owned partial after proving it matches the corresponding prefix of the exact reconstructed full archive. Keep transport receipts and command logs. Original full raw archives remain on both machines.

Source/library/runtime/SDK/checkpoint baselines are restored and every job is terminal. This clears the original controlled GDN gate for this50-source candidate. Next is ordinary full-chain layer3 KEY norm/RoPE and both canonical cache captures at128/256/512/1024/2048/3968 prefixes, using the unchanged retained T32 state/vector/128-token oracle as the target for both profiles. Require physical counters: oldZ128=0, reconstructedZ128=48 and KV128=32 for the T128 full target; auxiliary/T4 and T32 counts0. Preserve failed observations if any cache/state/vector/token differs; do not require reproducing the old known-wrong T128 outputs. Then run all six original4096total ordinary/MTP state/vector/token/oracle scenarios and adapted805 quality before timing. Matched stronger tuned/MTP repeated vLLM/SGLang baselines, true concurrency, INT4W4A16 and Qwen3.5-4B comparisons, serving peakVRAM and the overall2x decode AND throughput plus half-TTFT targets remain required and unmet. No benchmark/chart values changed.


### Full-chain KV128 caches exact; original ordinary model gate remains failed

2026-10-26: Run ordinary real3968-token prefill with the same50-source candidate and DLL5dff30dd8d2960bcfe64cc25d5014874bd893a3649407ea157eaa7759179350f. No native rebuild or kernel/math change. Two sequential exclusive profiles T32/T128 use the SAME unchanged correct retained T32 mtp-prefill16-full4k-admission-attempt1 oracle, complete128component physical/logical packed-state hashes, raw5120word hidden/normalized vectors, and128 output IDs. The old known-wrong T128 run is not an admission target. Physical graph counts require M128=48, legacyZ128=0, reconstructedZ128=48 and KV128=32 for full T128; T4/T32 new counts0. Original logical projection predicates are checked on an explicit copy, retaining unmodified raw counters. All35 CPU controls plus86historical assertions and actual embedded stdlib-only metadata preflight pass before GPU. Trace flags are cleared for the ordinary whole-model run.

The current T32 profile reproduces the entire retained empty/prefill/final states, both raw vectors and all128 IDs exactly; all12 complete layer3 key/value prefix-cache arrays also reproduce the old correct T32 prefix control. T128 captures all six prefixes128/256/512/1024/2048/3968. Every complete layer3 KEY and VALUE canonical cache matches current T32 byte-for-byte:50,331,648 BF16 values compared, inactive tails zero, every changed count0. This confirms the new KV helper fixes the previously observed key/value cache differences through the full prompt.

The original full-model gate nevertheless FAILS at complete ordinary-prefill state comparison before native decode. First unequal stored component index9 is layer4 recurrent state (linear_attention role2); indices0..8 match, including layer4 convolution/history. There are119/128 differing components. All5120 hidden words differ (maxabs30.66162109375), and all5120 normalized words differ (maxabs2.7971408367156982). No native128 output IDs or final state were collected after the failure. Do not promote this candidate or run performance. First stored mismatch only bounds diagnosis; it does not prove Q/O GEMM, attention, MLP or recurrence is the cause. Earlier hidden differences can exist despite exact stored states/caches.

Preserve the original failed workflow/native traceback, successful baseline, command ledgers and resource samples. Source/library/runtime/SDK/checkpoints are restored; both GPU processes, remote verifier and transport jobs are terminal. Raw archive 12009807bcae0e3de5071010afbcba68f7b699f008c09c3656267d232b4a0ada (53476668bytes/96members). Full independent replay on both machines yields identical proof f8ec76d106bb910ff8d17feff9cbcc917323896863584c2c112f79467957a464. Lossless delta reuses54 identical compressed payloads/52,240,121bytes from the old prefix archive, with1,225,622bytes transferred; every original53,476,668 ZIP byte is reconstructed and whole SHA, CRC, manifest, full cache arrays and retained full-state/vector/token oracle replayed. Verifier exit0 means valid preserved failure evidence; workload exit1 and full-model admission failure remain explicit. No tolerances, vectors, array samples, oracle values or original math gates are changed. Correctness resource samples are not serving peakVRAM/performance results.

Next implement private read-only ordinary packed-program output traces after layers0..4, preserving original operations and FP32 barriers, without widening native diagnostic profile guards. Require same-profile primary full-state/vector/cache equivalence to this retained run before interpreting traces. Failed T128 output may serve only as instrumentation-equivalence control, never as the model-admission oracle (which stays correct T32). Find the first actual hidden difference using complete boundary arrays, then correct that cause. Full six original4096total ordinary/MTP scenarios, adapted805/broader quality, strongest feasible tuned/MTP repeated references, actual native concurrency, matched text/image TTFT/decode/throughput/serving VRAM, INT4W4A16, Qwen3.5-4B and production-cache/promotion checks remain required. Overall2x faster-baseline decode AND throughput and half-TTFT are unmet; no chart or benchmark values changed.


### Private full-chain hidden-boundary observation method

2026-10-26: Prepare and start layer-boundary-attempt1 on the fixed remote study root, using unchanged candidate50/sourcebundlea0063c/nativeDLL5dff30. The private Python subclass derives _forward exactly from qualified cuda_packed_program.py623d97063edf8e5dcb22e1558bc0996965bb85dfd8f53d2d93a0fbf68d362ec6 with only three additive replacements: create a boundary list, append flattened existing hidden outputs after layers0..4, append that list after the original complete output prefix. Preserve original Qwen35CUDAChunk arguments, FP32 barriers, all state/payload layouts and numerical operations. Expand shared graph storage to the actual wider output while copying the original-width prefix back to a separate primary tensor for unchanged native commit helpers. T4 remains original. The private constructor binding is restored immediately, including exceptions. No production/native source or diagnostic guard changes.

Four new CPU controls verify exact reverse derivation, original commit-prefix dimensions and bytes, readonly caller inputs, wide storage and boundary offsets, T4 preservation, constructor restoration, complete-array corruption/nonfinite/repeat rejection and precise changed-token/dimension detection. Combined39current CPU tests plus86retained historical assertions pass locally before deployment. Frozen preparation/source hashes and raw command ledgers are retained. Remote owner must rerun these controls and embedded stdlib preflight before GPU, run T32 then T128 exclusively and repeat each full3968 prefill. Every complete boundary array and repeated digest must match within profile; complete primary states/vectors and all12cache arrays must reproduce the saved same-profile current run, with128IDs/finalstate also required for T32. Known failed T128 may serve only as instrumentation equivalence control; the original correct T32 full-model admission oracle is unchanged and T128 remains unqualified. Complete cross-profile boundary comparisons are diagnosis, never model or timing admission. Retain raw arrays, failures and restoration proof before interpretation. Outcome pending at method preparation; later evidence records the actual result. No benchmark values or2x target claim.


### Actual full-chain traces locate the first observed hidden difference inside layer3

2026-10-26: layer-boundary-attempt1 completes with workload exit0 on CD-RTX5090. The39CPU controls plus86retained assertions pass remotely, followed by actual embedded stdlib metadata preflight. Both exclusive T32/T128 profiles complete two full3968-token prefills. Every captured boundary array is byte-exact between repeats, as are complete repeated prefill states and raw5120word hidden/normalized vectors. Crucially, instrumentation primary outputs reproduce the complete same-profile retained current run: empty/prefill states/vectors and all12layer3 key/value prefix arrays for both profiles, and all128 IDs plus complete final state/vectors for correct T32. The current T128 remains the known FAILED full-model admission candidate; matching its saved output proves only observation equivalence. No admission oracle, tolerance or original guard is changed.

Full-array remote replay compares101,580,800 finite F32 words across layers0..4 and all3968tokens. Each layer has20,316,160 compared words. Layers0,1,2 are byte-exact. Layer3 has1,116,934 changed words, maxabs0.029296875; first difference token1, hidden dimension4305 (flattened word9425), T32=-0.11088943481445312 versusT128=-0.11082839965820312. Layer4 has19,932,759 changed words, maxabs0.083984375, first difference at the same token/dimension. This bounds the first observed hidden drift to inside layer3, before the previously observed first stored-state failure at layer4. Exact layer3 canonical key/value caches and exact layer2 input do NOT identify Q/Gate, Q norm/RoPE, attention reduction, O projection, residual or MLP as the cause; actual intermediate observations are required before intervention.

Full original raw archive 43f1cf82ce8757ae94229f9db82bc1c9908c27786b8f6f420462237ca3d10983 (610840032bytes/878members), retained on the remote fixed study root. Remote full-array verifier proof 00b900582dcdd80a7ad0acea72a299401c7684b327995a99bfbed45538fdf5d9; no full3968 boundary-array replay is claimed locally. A19,573,604-byte SHA-linked capsule retains all original metadata/source bundles/commands/resources and raw primary vectors, plus complete first128 boundary arrays for all5layers. Independent local replay compares all3,276,800 F32 words in that first128 scope and verifies the exact first divergence. It also replays complete primary-state/vector/128ID equivalence and all12canonical cache arrays per profile by reusing byte-identical SHA-checked parts from the already independently replayed current prefix parent. Local scoped proof 73789b6fae3dd4bd7e390f32551ae48651d2b7062c2924d725424c016cbc5ade. Full remote raw-array hashes remain in the preserved original manifest, including omitted later frames. Do not call the local capsule a full3968-array replay or compare the two scope-specific proof files as if they had equal coverage. A premature transfer checksum check rejected an incomplete capsule before any numeric verifier ran; wait for SFTP DOWNLOADED acknowledgment and then whole SHA before successful independent replay. Retain this transport observation without treating it as a model failure.

Source/library/runtime/SDK/checkpoints restored, private cache retired, all GPU/CPU verifier/capsule jobs terminal. No production/native source, model/math, numerical tolerance, full-model gate or benchmark/chart value changed. Next add a separate default-off strictly bounded layer3 T32/T128 ordinary full-chunk stage witness to the qualified50-source candidate, preserving original T16 diagnostic and INT4-only projection guards. Retain exact ordinary math and primary commit prefix; capture existing normalized input/QG/K/V/core/O/residual/postnorm/output on the real first128 layer2 output, with full-chain same-profile primary parity and repeats. If a projection differs, collect actual vendor configurations before selecting a typed SDK control. Then correct the demonstrated cause and rerun all six original4096total ordinary/MTP/image/state/vector/token/reset cases and805/broader quality before performance. Strongest feasible tuned/MTP repeated references, true native concurrency, matched text/image TTFT/decode/throughput/servingVRAM, INT4W4A16, Qwen3.5-4B and production fingerprint/promotion remain open. Overall2x faster-baseline decode AND throughput plus half-TTFT remain unmet; goal stays active.


### Native read-only layer3 full-chunk stage witness built; actual stage/model parity still required

2026-10-26: Prepare51-source observation candidate over exact qualified50-source bundlea0063cb10bf32144058fee174c9e0959adcf91d21bf34a9901dfe2e291f17246. New sourcebundle86d4fb95b4df9f5b0fb8a1422cbd192da4c04cb4511dba5dd15e67ed8e78840c contains1new profile header and5exactly derived modified files (Qwen35 lowering, TRT builder header/body and CUDA compiled-graph header/body). No new CUDA kernel, plugin, wire field or model arithmetic. Original diagnostic_full_stages embedded T16/B1/B2 guard, original INT4-only checkpoint projection guard, canonical KV validation and old plugin source bytes remain unchanged. Exact reverse-derivation checks recover every original source byte. Original plugin/factory/wire contracts stay closed in the exact prior native parent; no unchanged plugin-control suite is silently redefined as new numerical admission.

The new witness_full_chunk_stages integer keyword defaults off and requires explicit GARNET_FULL_CHUNK_STAGE_WITNESS=1, shared mixed compressed joined F32 input, onlylayer3/B1/T32lowbatch orT128prefill, exact24Q4KV256D/rotary64/11,11,10/theta1e7/epsilon1e-6 and canonicalwidth106823680/key2482176/value4579328. It uses a separate chunk-stage vector; the original diagnostic vector retains its old restrictions. Append ten existing stage tensors (input, normalized, qg, projectedK/V, core, O attention output, residual, MLP input, output) after the unchanged ordinary complete packed prefix. Maximum2,097,152words per appended stage; host readers still split into<=1,048,576word blocks. Distinct native_full_chunk_stage_witnesses counter has explicit reset/getter/graph-stat propagation. Actual first128 stage captures and full3968 primary parity are not yet executed and no cause is claimed.

Build attempt1 FAILS in cpu-before before installing candidate source or invoking a compiler/model: restored remote production baseline lacks37research-only parent components, causing historical imports/source assertions to fail. Local baseline contains the qualified components, so the earlier local CPU pass did not establish this remote setup order. Preserve original traceback/archive c837cc827d8f42efb5a220966616407d3da8a0cedd0193781f4e345aaf29d1f0 and identical independently replayed failure proof 84638d28523e92c45f24602435c8ac311ece97e8329a16d99dfea91551a5c55c. Source/library/runtime/SDK/checkpoints/CMake are restored; no native compilation or GPU model ran. Do not waive or rewrite the failed old tests.

Attempt2 changes owner/verifier setup order only. Install the exact immutable50-source parent, run42current CPU tests plus86historical assertions unchanged (including a new regression forbidding mutable installed-header reads during source derivation and requiring parent installation before controls), then apply the same byte-identical51-source observation candidate. All42+86pass on both machines. Actual MSVC header-only profile executable passes142cases:2valid/140rejected across policy strings/null, precision, geometry, offsets, dtype, mode exclusivity, theta/epsilon, heads/dim/rotary/sections and layer names. The native Release/MT/sm120 build completes; CUDA source/kernel bytes unchanged, only3existing allow-listed CUDA object artifacts differ. New saved DLL 3574f8cc1f772933f117b484d9a57844fb3846a4e1ada59f62399876c31fa48a at build/bin/garnet-native-full-chunk-stage-attempt2.dll. Actual rebuilt native selectors pass and tokenizer matches all955frozen cases (805screen templates,124fixed4K templates,26edges); these are CPU-only API smoke checks, not image/model quality.

Successful raw native archive 9f1c58609254fe154b5bd63825334c80a2f621bc860712eeed24b9c72cd2f54e (4307269bytes/89members); full independent replay on both machines yields identical proof 4c1b2af709f49ab3138a359e9bfe17b1fd9b05e221e162bf71f5fe592f06bb75. Retain exact51-source bundle membership,14original source snapshots/37removed new targets, source and command hashes, raw C++/compiler/test logs, failure, install receipts,34CUDAobject hashes and source/library/runtime/SDK/checkpoint/CMake restoration. Every build/verifier job is terminal. Model weights/checkpoints and production deployment are unchanged; study uses the fixed remote folder with no new clone/model copy.

Next wire a private exact-derived Python packed program that requests this witness only atlayer3 ofT32/T128, expands shared storage for appended stages and returns only the original primary prefix to unchanged commit helpers. Require defaultoff counter0 and enabledcounter1 with old M128/Z/KV counts unchanged, complete first128 input/stages, readonly graph controls/canonical state and two full-prefill repeats. Both defaultoff/enabled profiles must reproduce complete same-profile full-chain states/vectors/all12KV cache arrays from the retained current run; correctT32 also all128IDs/finalstate. Known failedT128 is only instrumentation-equivalence control; the original correctT32 model-admission oracle remains mandatory. Replay all finite stage words, locate the first arithmetic difference and observe actual vendor/typedSDK controls before any projection algorithm intervention. Then correct the demonstrated cause and rerun all six original4096total ordinary/MTP/image/state/vector/token/reset cases and805/broader quality before timing. Strongest feasible repeated tuned/MTP vLLM/SGLang, true native concurrency, matched text/image TTFT/decode/throughput/servingVRAM, INT4W4A16, Qwen3.5-4B and production-cache/source promotion remain pending. Overall2x faster-baseline decode AND throughput and half-TTFT remain unmet; no benchmark/chart values changed and goal stays active.


### Actual full-chain defaultoff/enabled layer3 stage collection method

2026-10-26: Prepare and start full-chunk-stage-attempt1 with nativeDLL3574f8cc1f772933f117b484d9a57844fb3846a4e1ada59f62399876c31fa48a and exact51-source bundle86d4fb95b4df9f5b0fb8a1422cbd192da4c04cb4511dba5dd15e67ed8e78840c. Private Python chunk derives exactly from qualified ordinary cuda_chunk.py, changing only private fusion/function names and one layer3 witness_full_chunk_stages=1 metadata keyword. The packed forward method differs only by appending existing10stage views after the complete original output prefix. Original math/arguments/FP32 barriers/state layouts/T4 are unchanged. Expand actual shared graph storage, copy only original-width primary output back to unchanged commit helpers, then copy stages to separate study buffers. Restore private constructor binding immediately even on exceptions. On capturing first128 of BOTH full3968 repetitions, hash complete canonical graph-input state (all128components/physical/logical rows) and all original hidden/active/positions/slots before/after; require exact readonly and repeated incoming values. Complete stage arrays use<=1Miword parts, including every QG1.57Miword array. Repeat checks compare every raw part byte to the retained first repetition, not samples.

Four new CPU controls verify exact original/private chunk and forward derivations, exact collector/batch derivations, physical counters0/1, widened shared-storage and unchanged primary-prefix offsets, all10stage buffer boundaries, readonly mutation rejection, T4 preservation, constructor restoration on exception, and original complete-array finite/parts/path/type/SHA/repeat fault rejection. All46current tests plus86historical assertions pass locally before deployment. GPU owner installs exact parent50 for these old assertions, then exact51source, runs actual embedded stdlib metadata/AST before requesting GPU, and executes control32/control128/stages32/stages128 exclusively. Each full3968prefill repeats; complete primary states/vectors/all12layer3 cache arrays must reproduce retained same-profile current prefix run, correctT32 also128IDs/finalstate. FailedT128 is ONLY instrumentation equivalence control; original correctT32 model-admission oracle unchanged. Explicit status-only metadata copies reuse the unchanged original complete primary comparison; raw reports/arrays never rewritten. Compare all6,553,600finite F32stage words, require stageinput equal SHA-linked real layer2 fixture and stageoutput equal prior complete first128 layer3 boundary. Actual Garnet-frontend negative graph cases remain pending;142compiled pureC++profile cases are retained, not relabeled as frontend tests. Full raw collection outcome pending; no numerical/model/quality/performance admission or benchmark values.


### First actual layer3 stage collection preserved; canonical-width closure retry

2026-10-26: full-chunk-stage-attempt1 runs46CPU controls plus86historical assertions and embedded stdlib preflight, then actual exclusive control32/control128/stages32 profiles. Both defaultoff profiles report complete same-profile empty/prefill/repeated-prefill states/vectors and all12KV cache arrays exact; correctT32 also128IDs/finalstate. EnabledT32 builds with native_full_chunk_stage_witnesses=1, captures40complete first128 stage arrays and completes124ordinary prefill calls through3968 with all six cache prefixes. Four before/after complete canonical graph-input state/control observations pass for first128. However its post-prefill state witness fails BEFORE the comparison: the collector main-scope stage loop rebinds width from106823680 to5120, and the nested state() closure incorrectly passes that stage width into the unchanged strict complete-state reader. The reader rejects incomplete canonical geometry. No enabled repeat, T128stage capture, cross-profile stage diagnosis, full-model admission or benchmark is claimed. Original matrix math/native code was not changed by this collector defect.

Preserve original traceback, reports/raw arrays/commands/resources/source inventory and closure failure. Raw archive dd3cae5dd192032ddfcddee8f31470d9ef3f5838fa7e95ceb133b4d1c38836b6 (119328607bytes/175members); identical independent failure proof 9fb5498a2ac0eb7eb468934e28b1826c98c4d090994b1b3e5883c5cb6825756b on both machines. Exact transport reuses85byte-identical compressed payloads/78,437,960bytes from prior prefix archive;40,848,385-byte delta reconstructs every original119,328,607ZIP byte and verifies whole SHA/CRC/manifest. Three transport CPU corruption/metadata/parent tests pass. No scientific array, numeric gate, report or original ZIP byte is edited. Source/library/runtime/SDK/checkpoints are restored; original GPU/verification/transport jobs terminal. Raw original archives remain on both machines.

Retry fixes only the collector's canonical width binding: name it canonical_width and have state() read that immutable value independently of per-stage width. Keep exact51sourcebundle86d4fb95b4df9f5b0fb8a1422cbd192da4c04cb4511dba5dd15e67ed8e78840c, DLL3574f8cc1f772933f117b484d9a57844fb3846a4e1ada59f62399876c31fa48a, original private chunk/program math, original complete-state/vector/cache/ID predicates and every threshold unchanged. Use new attempt2 child/owner/batch/verifier/preparation filenames; never overwrite the failed attempt. Regression recreates stage-loop width5120, verifies original reader rejects it, verifies canonical106823680 passes complete geometry before GPU access, and inspects AST for the single canonical binding read by state(). Exact derivation checks cover every retry replacement. All47current CPU controls plus86historical assertions pass locally before deployment. Rerun ALL FOUR defaultoff/enabled T32/T128 profiles and both full-prefill repetitions; do not relabel inherited observations as new runs. Native frontend negative graph cases, original six4096ordinary/MTP/image/state/vector/token/reset cases,805/broaderquality and all stronger-reference/concurrency/INT4/4B/servingVRAM/performance objectives remain required. No benchmark/chart values changed; overall2x faster-baseline decode AND throughput and half-TTFT remain unmet and goal stays active. Retry GPU outcome pending at preparation.


### Complete layer3 stage replay isolates the earliest observed difference to Q/Gate

2026-10-26: Corrected collector full-chunk-stage-attempt2 completes all four fresh exclusive processes (control32/control128/stages32/stages128), each with two actual3968-token prefills; correctT32 also128decode IDs/finalstate. The original47CPU controls and86historical assertions plus embedded stdlib preflight pass. Enabled captures retain all10complete first128 stage arrays, full canonical graph-input state/controls before/after and across repetitions (T32:248stage calls/8readonly records; T128:62/2). Defaultoff/enabled complete same-profile primary empty/prefill/repeated-prefill states/vectors and all12layer3 KV arrays match retained controls. CorrectT32 all128IDs/finalstate are exact. Known failedT128 remains only an instrumentation-equivalence control; the original correctT32 full-model admission gate remains FAILED for the T128 candidate. Source/library/runtime/SDK/checkpoints restored and private cache retired; no model arithmetic changed.

Both machines independently replay all6,553,600finite F32stage words from the complete raw archive. Input and normalized input (655360words each) and projectedK/V (131072each) are byte-exact. First differing stage is Q/Gate:102/1572864words differ, maxabs0.03125; first token0/projection dimension1585, T32=-0.080078125 versusT128=-0.07958984375. Exact ordinary-source reshape/slice decomposition [128,24,2,256] locates77query and25gate changed values, all102coordinates/values retained; both observed F32 arrays are exact BF16 round trips. First change is query/head3/channel49. This proves the earliest observed stage difference, not its algorithm/scale cause. Core changes14542words (maxabs0.00030332524329423904); O attention8857/residual8855 (both maxabs0.0009765625); MLP input66398(maxabs0.0019193142652511597); output45291(maxabs0.002685546875). First output difference token1/dimension4305 matches prior ordinary layer3 boundary. Stage inputs match retained real layer2 fixture and outputs match the prior complete first128 boundary capsule. No downstream independent arithmetic cause is excluded.

Full original ZIP 085ad330cde66fd755c383f05b7af16d610d7fec3464de6e4a2928a10da7a196 (163783460bytes/237members) is retained on both machines. Identical complete remote/local proof 1a009ce36e1d375b06ad53907ccd414c0a88e64c8487c9f1ed67ef3993f6b6f6; Q/G split replay291bce6af7f1336ce3980838234d52eecf4ca65a2300525c5a76982e50b5f8cf. Exact transport delta50f8c924206598d2d799edc01b0e5fbe4b12d87d8e77a9a916fd3cddf8f4c257 (17689940bytes) reuses174identical compressed payloads from the failed first archive; reconstruction verifies every original ZIP byte and whole SHA, with no scientific/archive byte mutation. Preserve raw source/command/resource/repeat/control metadata, original failure, install receipts, exact transport manifests and independent CPU command ledgers on branch. All collection/verification/transport jobs terminal.

Next observe actual cuBLASLt Q/Gate choices with vendor logging on the SAME51source/DLL3574f8cc1f772933f117b484d9a57844fb3846a4e1ada59f62399876c31fa48a. Require logged complete primary/readonly/repeats and every rawstage byte equal the qualified unlogged profiles. Retained defaultoff controls must remain explicitly inherited, not relabeled as fresh runs. Only after actual vendor configurations/weight scale metadata and typed SDK checks should a narrowly gated Q128 algorithm control be considered. Do not assume prior KV/GDN settings generalize to QG. Then rerun all10stages and original full-model/six4096ordinary/MTP/image/state/vector/token/reset and805/broaderquality gates before performance. Strongest feasible repeated tuned/MTP references, true native concurrency, matched TTFT/decode/throughput/servingVRAM, INT4W4A16, Qwen3.5-4B, actual frontend negative cases and production fingerprint/promotion remain open. No new benchmark/chart values; overall2x faster-baseline decode AND throughput plus half-TTFT remain unmet and goal stays active.


### Logger-only full-chain Q/Gate vendor study prepared

2026-10-26: Prepare full-chunk-stage-vendor-attempt1 over exactly the qualified51source/DLL3574f8cc. Native source/model math/weights/algorithm/thresholds unchanged. Exact derived child adds only vendor environment assertions and actual layer3 Q/Gate compressed-weight metadata; exact derived batch enables NVIDIA cuBLASLt level5/mask31 and an owned unique sibling log file. Owner checks log absence before child DLL import. Execute only two NEW exclusive enabled profiles T32/T128, each complete3968 prefill repeated twice, correctT32 also128IDs/finalstate. Defaultoff control32/control128 are explicitly inherited from SHA-linked complete qualified stage attempt2; never relabeled as new runs. Require original complete-state/vector/all12cache primary equivalence and all first12810stage raw bytes, complete readonly graph-input state/controls and repeats identical to unlogged same-profile observations. T128 full-model admission remains failed.

Three added CPU tests require exact source/batch/owner/verifier derivations; reject even a finite changed QG byte or changed complete readonly record; reject missing/wrong geometry/dtype/version vendor traces and malformed/nonfinite checkpoint scale metadata. Parser retains only actually printed QG N12288K5120 target-M trace configurations and line numbers; does not assume unprinted attributes or associate unobserved heuristic objects with trace calls. All50current controls plus86historical assertions pass locally with raw command/source hashes. Remote owner will rerun them on exact parent50, install same51 and run unchanged embedded stdlib preflight beforeGPU. Retain exact deployment/install/preparation hashes, full raw vendor logs, stages/primary/resources/source inventories and failures/restoration. Current preparation does not claim actual vendor-run success, algorithm cause, model/quality or performance qualification. Remaining full gates, true concurrency, stronger repeated tuned/MTP baselines, INT4W4A16/4B, servingVRAM, production promotion and overall2xdecodeANDthroughput/halfTTFT objectives remain unchanged and unmet.


### Actual Q/Gate vendor choices verified with complete logger parity; CPU reader failure retained

2026-10-26: full-chunk-stage-vendor-attempt1 runs unchanged50CPU controls plus86historical assertions on exact parent50, unchanged embedded stdlib preflight, then two NEW exclusive enabled profiles T32/T128 on same51source/DLL3574f8cc. Each completes two actual3968-token prefills with all six layer3KV cache prefixes; correctT32 also all128decode IDs/finalstate. The defaultoff controls remain explicitly inherited from complete stage attempt2, never relabeled as new runs. Both children finish exit0 with full same-profile primary equivalence/repeats. Native source/model math/weights/thresholds/algorithms unchanged. The owner exits1 afterward in CPU vendor parsing: it wrongly requires reductionScheme and numSplitsK in every printed algorithm. Actual non-split traces omit both. Preserve original failure, all raw arrays/vendor logs/resources/source inventories and restoration; no GPU rerun.

Separate exact-derived optional-field CPU reader changes only the required vendor-field set from {algoId,tile,stages,reductionScheme,numSplitsK} to {algoId,tile,stages}. All printed optional fields are retained verbatim; omitted attributes stay unknown. Every original numeric primary/state/vector/token/cache/readonly/repeat/array predicate remains byte-identical. Two new CPU regressions reproduce the actual omitted-field failure and require exact derivation/no guessed defaults; retained three vendor controls continue rejecting finite changed stage bytes, readonly changes, wrong geometry/dtype/version and malformed scale metadata. All5pass on both machines. CPU replay first reproduces the original AssertionError in original full_chunk_stage_vendor.py:33, then fixed replay verifies complete logged primary/readonly/repeats and ALL50stage entries/13,107,200finite F32 words against unlogged same-profile data, byte-exact. Cross-profile first128 stage result is identical to qualified unlogged proof (including102QG differences); no new model-admission claim. Both QG metadata records match: FP8W8A8 logical[12288,5120], packed[12288,1280]INT32/62914560bytes, input_scale0.2064732164144516, weight_scale0.0007149832672439516, scale_bytes0. No scale/activation/weight intervention tested.

Actual vendor logs each contain32target QG matmul traces over16weight buffers, with one printed configuration per profile. T32: algoId67, tileMATMUL_TILE_64x32, stagesMATMUL_STAGES_64x5, ctaSwizzling1, customOption18. T128: algoId67, tileMATMUL_TILE_128x64, stagesMATMUL_STAGES_64x4, ctaSwizzling1, customOption18. Both use N12288/K5120, E4M3 input/weight, BF16 output, F32 compute/scale,4MiB workspace. Printed reduction/split fields are absent and NOT populated with defaults; no heuristic-to-trace object association is inferred. This identifies a concrete algorithm-configuration candidate, not a proven cause of QG or downstream arithmetic drift.

Original full archive fe1f015002e8bc4ff78211983c87c0ee202d49f7551bc5cbadaafd5d898410fc (111339157bytes/174members) is retained on both machines. Identical independently replayed original failure proof7361cb9316005adaac98fd16948e03a7b428d616ce5a527efc76831f7cbe73f5 and distinct CPU-only recovery proofbd87e5b133fea32d135d414d98ad55c2fe84aca968d953b29c0bf23b7ce12696. Exact1,520,256-byte delta1af1b8941e61cebdd20f7cda7fa3cb82d3c924d5b34d6a3287a2a7db2b032ee2 reuses128identical compressed payloads/109738574bytes from prior complete stage archive and reconstructs every original ZIP byte/wholeSHA. Source/library/runtime/SDK/checkpoints restored and private cache retired. CPU repair receipts verify baseline source/runtime/SDK/checkpoints unchanged. A local retention guard rejects a case-insensitive CPU/cpu ledger filename collision before overwrite; use distinct reader-CPU label and preserve both ledgers. All GPU/replay/transport jobs terminal.

Next build a standalone typed native SDK probe for actual QG N12288/K5120: query canonicalM16 with4MiB preference (the existing T32 control path), require first successful SDK candidate to match all FIVE actually observed T32 public attributes (67,64x32,64x5,swizzle1,custom18), record other queried public attributes without assuming values, and AlgoCheck the same SDK-owned object for M16/M32/M128 within4MiB on pinned SM120/cuBLASLt130400. Then a separate defaultoff Q128 context/plugin/strict text-Q route may test this observed candidate. Keep original KV/GDN guards/wires/plugins unchanged, distinct type/wire/counter and native CPU source/profile/wire failure controls. After intervention rerun complete first128 stages plus unchanged original full-model/six4096ordinary/MTP/image/state/vector/token/reset and805/broaderquality gates. Do not assume correcting QG also corrects an independent core/O difference. Strongest feasible repeated tuned/MTP references, true native concurrency, matched text/image TTFT/decode/throughput/servingVRAM, INT4W4A16/4B, actual frontend negative cases and production fingerprint/promotion remain open. No new performance/chart values; overall2x fasterbaseline decode AND throughput plus halfTTFT remain unmet and goal stays active.


### Standalone Q/Gate SDK probe: first deployment audit failure preserved; exact geometry checks unchanged in retry

2026-10-26: Compile standalone probe_qg128_sdk_control.cpp with real NVCC13.2/MSVC /MT, -archsm120 and requested -cudartshared, dynamic cuBLASLt import. No Garnet native source/library build or model weights/matmul. It requires pinnedGPUUUID/SM120/cuBLASLt130400, queries actualM16 QG N12288K5120 with4MiB preference, requires first-success SDK object match observed T32 ID67/TILE64x32/STAGES64x5/CTA1/CUSTOM18, queries other public attrs without guessing, and calls AlgoCheck on exactly that SDK-owned object forM16/M32/M128 with<=4MiB. The attempt reaches module-path audit after those checks but exits1 because it requires a separate loaded cudart64_13.dll. Its catch report contains only resolved_module_path; no successful compatibility proof or numeric data is claimed from the lost partial report. Actual PE dependency inspection shows only cublasLt64_13.dll andKERNEL32.dll. Compiler explicitly warns that shared is deprecated and maps to hybrid; do not infer a separate dynamic cudart module from the command flag or claim fully static CUDA execution.

Preserve original archive4ec5eed0659f5d58622d0e510abce44159c17ce295387c3fc28f7e0b1f1601fd (284234bytes/15members) and identical independent failure proof4e2026fa3d1b57ec43883c8fdddeeb944cc86a6975272e4a54c39c10dcf72fd5. Production source/library/runtime/SDK/checkpoints unchanged; native compile/run/replay jobs terminal. Retry uses separate attempt2 source/batch/owner/verifier/preparation names, preserves the first failure, and changes ONLY the deployment audit: require actual loaded cuBLASLt SHA andSystem32nvcuda path/SHA before/after; report optional cudart path/absence and validate its SHA if present. Record original failed executable PE imports with exact command/executable/tool hashes. Exact source derivation and AST checks require the entire main() selection/geometry/attribute/AlgoCheck prefix byte-identical. No API, threshold, shape, algorithm predicate, model weight or production runtime change. Retry outcome pending; no model/numerical/quality/performance admission. Original full-modelT128 admission, full six4K/image/MTP/quality, trueconcurrency, strongest repeated tuned/MTP references, matchedINT4/4B/servingVRAM/TTFT/decode/throughput and2xdecodeANDthroughput/halfTTFT remain open and goal stays active.


### Actual Q/Gate canonical SDK control compatible with M128; numerical intervention still required

2026-10-26: qg128-sdk-control-attempt2 compiles and runs actual native descriptor probe after exact module-audit-only retry. Original main() selection/geometry/attribute/AlgoCheck prefix is byte-identical. Actual pinnedGPUUUID/SM120/cuBLASLt130400. M16heuristic returns7candidates, first-success index0; supported queried public attrs allsuccess/4bytes: ID67, TILE_ID13(64x32), STAGES_ID17(64x5), CTA_SWIZZLING1, CUSTOM_OPTION18, SPLITK_NUM1, REDUCTION_SCHEME0. Unlike absent logger fields, split/reduction values are now actual SDK observations. INNER_SHAPE_ID andCLUSTER_SHAPE_ID returnstatus7/bytes0; zero-initialized local buffers do NOT establish attribute0. Leave both unknown and do not fill guessed values. The same live SDK-returned algorithm object passes AlgoCheck forM16/M32/M128, allAPI/state0 and requiredworkspace0<=4MiB; public queried attrs remain unchanged after each check. No model weights or matmul executes, no opaque algorithm bytes imported/set/patched. This proves descriptor compatibility, not QG numerical equivalence or arithmetic cause.

Compiler/headerCUDART13020, actual runtimeAPI13040/driverAPI13040. Loaded cuBLASLt D:/CantorAI/5090-study-20261023/build/bin/cublasLt64_13.dll SHA matches pinned baseline; NVIDIA driver loaded C:/WINDOWS/SYSTEM32/nvcuda.dll path/SHA audited before/after. Optional cudart64_13.dll module is absent. Actual PE dump of original failed executable (exact archived-exe hash verified) shows cublasLt64_13.dll/KERNEL32.dll imports; retain raw shared-deprecation/hybrid warning. Record observed binding rather than inferring fully static or a separate dynamic cudart dependency from compiler flags. No production source/library/runtime/SDK/checkpoint bytes change. All nativeSDK/replay jobs terminal.

Successful raw archive19f0d1da2bd6ad0fc7b41041182fbb4e3765f0309b233d5b0b109598c42538c85 (292308bytes/19members), retained both machines. Identical remote/local independent compatibility proofbba6603e9704787c857846871737afa51653e94ce9527b917b1e36da6b5310da. First failed audit archive/proof remain unchanged and linked; preserve source/compiler/API/dependency logs, actual executable hash, public attrs/status/bytes, loaded driver/vendor SHA and source/runtime/SDK/checkpoint before/after. Next implement a separate defaultoff Q128 context/plugin/text-Q selector and distinct wire/type/counter on frozen51-source parent. QueryM16 liveSDK and require all7observed supported attrs plus actualM16/M32/M128 AlgoCheck; leave unsupported inner/cluster fields unpopulated. Keep original quantizer/scales/alpha/BF16 output and all old KV/GDN gates/plugins/serialization untouched. Require strict native source/profile/factory/wire negative controls, full first12810stage and original correctT32 full-model gates; QG correction may leave independent core/O drift. Original six4K ordinary/MTP/image/state/vector/token/reset,805/broaderquality, frontend negatives, production fingerprint/promotion, trueconcurrency, stronger repeated tuned/MTP references, matchedINT4/4B/servingVRAM/TTFT/decode/throughput remain required. No new speed/chart values, T128 full-model admission remains failed, and overall2x fasterbaseline decode AND throughput/halfTTFT remain unmet; goal stays active.


### Defaultoff Q128 native candidate implemented over frozen observation51; real native build pending

2026-10-26: Implement57-file candidate (6new/5modified) over exact immutable51-source observation parent. New separate ModelOptFP8Q128Context and GarnetModelOptFP8Q128Canonical plugin, version1/20byte wire/marker0x51313238, private Q128 factory trace and native_fp8_q128_canonical_projections counter with reset/getter/graph-stat propagation. GARNET_FP8_FULL_ATTN_Q128_CANONICAL defaults off and accepts only exact0/1; selected only B1/T128/inputRows128, N12288/K5120, exact model.language_model.layers.{3,7,...,63}.self_attn.q_proj.weight. Vision/MTP/malformed/otherprojection names, foreign batch/tokens/geometry do not select or are rejected as appropriate. New context queries live M16 SDK first-success candidate, requires all7actually observed supported attrs ID67/TILE13/STAGES17/SPLITK1/REDUCTION0/CTA1/CUSTOM18 and AlgoCheckM16/M32/M128<=4MiB onSM120/cuBLASLt130400. Retain SDK-returned object; no opaque import/mask/init/set and no queried/set guessed INNER/CLUSTER attributes. Same activation quantizer, BF16 input/output, weight/input scale multiplication and matmul alpha/beta. Only new text-Q route and counters touch existing native files; all original KV/GDN plugin/factory/wire/guard and CUDA kernel bytes unchanged. Existing nonce-based private cache root preserved; production fingerprint/promotion remain pending.

Add strict source/frozen-parent derivation, actual prior parent50-before-tests ordering and CPU-control source regressions. All56current CPU tests plus86historical assertions pass locally after final build-owner composition; original controls never waived. Prepare actual native278header cases (16Q names selected/262rejected);182factory CPU-SDK fault/trace cases cover7attrs/status/bytes, allM16/M32/M128 API/state/workspace failures, first-success/no-silent-fallback, cache, unsupported headers and trace parity; wire contracts reject35bad wires/18constructors/23descriptors/11formats/8policies and9cross-wires including unchanged oldKV/Z/M128/plain; initialize controls extend retained6routes withQ for630actual initialize calls/490CPUfactories over10policies, clone/cache/null/standard/unknown exceptions, zero GPUoperators. These native executions are pending at preparation; do not claim their success from source-only local tests. Release/MT/sm120 native build and native selector/tokenizer955CPU smoke must pass before a model run.

Read-only remote inventory initially exits1 because its path guard incorrectly restricts all candidate targets to Garnet/src although the immutable parent also includes preserved xModel Python sources. No source/GPU mutation occurs. Preserve actual failure/replay command logs and exact one-guard retry to the intended resolved Garnet checkout root; retain all57hash/existence checks,1373baseline files, runtime/SDK/checkpoints/CMake/34CUDAobject assertions unchanged. Corrected inventory passes and verifies original targets14present/43absent, CMake cache and CUDA objects match the prior qualified native build; sufficient free space. Bundlea786becf195c68c9c5f51c56eeadfe15bce4be2176f63fd058684c70c653274e; deployment01f6022977c07e272eff489afe42fcbc78d2a23e80438e5964d8035922b85b9e (472507bytes/66files); preparatione1e0d95f47b71aabeeb0f9ed0fdb39a7d7f1ac5080ef7301cb5833c191249417. Install only methods/bundle now; the exclusive owner installs exact parent50 for56+86old/new controls then57candidate, runs native contracts/build, preserves originals/failures/CUDA object whitelist and restores source/library/runtime/SDK/checkpoints/CMake. Reuse fixed study root, no clone/model copy.

Native build outcome and all actual model numerical gates pending. If built, require defaultoff profiles reproduce original complete primary/vector/state/cache/ID data and Qcounter0, enabledT128 counter16 with oldM128/Z/KV counts unchanged; compare complete first12810stages and original correctT32 full-model gate. SDK compatibility alone does not establish arithmetic cause; QG correction may leave independent Qnorm/RoPE/core/O drift. Original six4096ordinary/MTP/image/state/vector/token/reset and805/broaderquality before timing. Actual frontend negative cases, production fingerprint/promotion, true native concurrency, strongest feasible repeated tuned/MTP references, matchedINT4W4A16/4B/servingVRAM/TTFT/decode/throughput and2x fasterbaseline decode AND throughput plus halfTTFT remain open. No new speed/chart results, T128 full-model admission remains failed, goal stays active.


### Defaultoff Q128 native build and strict contracts independently verified; model gates next

2026-10-26: q128-canonical-build-attempt1 completes real native Release/MT/sm120 build on CD-RTX5090. Exact historical parent50 installed before56current CPU tests plus86historical assertions, then exact57source (6new/5modified). Actual MSVC profile278cases:16valid/262rejects. Actual context factory CPU-SDK182cases/86modes validates7attrs/status/bytes, allM16/M32/M128 API/state/workspace checks, first-success fail-closed, cache, unsupported headers and trace observation equivalence. Actual TensorRT CPU wire contract passes35bad-wire/18constructor/23descriptor/11format/8policy/9cross-wire rejections,20byte marker0x51313238, workspace4849664; legacyKV/Z/M128/plain types/wires remain unchanged. Actual seven-route initialize controls pass630initialize calls/490CPUfactory calls across10policies, clone/cache/null/standard/unknown exception paths, zeroGPUoperators. Actual rebuilt native selectors pass; CPU tokenizer matches955frozen cases (805screen templates/124fixed4K/26edges), no mismatches. These CPU token templates do not establish image/model quality.34CUDAobjects remain, only the same3previously allow-listed overlay/attention/commit artifacts differ; all CUDA source/kernel bytes unchanged.

Saved candidate build/bin/garnet-native-q128-canonical-attempt1.dll SHA584be8a125d0e2334d5d3d1bcb2ecd8f807cae0b42b047e4198f8830f800347e. Sourcebundlea786becf195c68c9c5f51c56eeadfe15bce4be2176f63fd058684c70c653274e. Original native archive714f46cd6253ff18aa97ce1a05df796cf87174ef730f08530c7bfc8b4cc844a0 (6178680bytes/562members), retained both machines. Original build exit0 is unchanged. Initial verifier exits1 on KeyError: it expects full-chunk-stage-source-derivation.json inside the new archive, although that unchanged baseline metadata is in the already SHA-linked original parent build archive. Preserve the actual original reader failure on BOTH machines. Separate exact-derived CPU reader changes only parent lookup/self-provenance/result disclosure; every native build/CPU contract/hash/restoration predicate stays unchanged. Read the metadata from the parent whose whole archive SHA and proof are already verified. No native build or GPU rerun. Corrected replay passes on both machines with identical proof3de346be45f21942ae412cdf253869ad4c3eee8105aa4bdba236df1e1c61e9d99. Preserve original/corrected reader sources, exact derivation, raw failure/recovery command logs, and original read-only build-inventory guard failure/retry logs.

Production source/library/runtime/SDK/checkpoints and CMake cache restored; all57candidate targets audited (14originals restored/43new removed), private native candidate DLL retained. All build/replay jobs terminal. NO GPU model or matmul, image quality or speed benchmark runs in this build; current T128 full-model admission remains failed until demonstrated otherwise. Next execute same57source/DLL defaultoff T32/T128 controls to establish complete same-profile primary/state/vector/cache/ID and first128stage equivalence with Qcounter0, then enabledB1T128 Qcounter16 with oldM12848/reconstructedZ48/KV32/fullstage1 unchanged. Enabled candidate must use ORIGINAL correctT32 full-model oracle, not knownfailedT128 instrumentation control; require all128IDs/finalstate and repeats. Capture all10complete first128 stages/readonly input state/controls with bounded host reads, preserve any admission failure without weakening gates or claiming timing. QG may become exact while independent Qnorm/RoPE/core/O drift remains; isolate and observe vendor data before another patch. Original six4096ordinary/MTP/image/state/vector/token/reset and805/broaderquality, actual frontend negatives, production fingerprint/promotion, trueconcurrency, strongest repeated tuned/MTP references, matchedINT4/4B/servingVRAM/TTFT/decode/throughput and2x fasterbaseline decode AND throughput plus halfTTFT remain open. No chart values changed; goal stays active.


### Q128 actual model admission prepared against the original correct T32 oracle

2026-10-26: Prior native build and independent CPU replay passed (57-source bundle a786becf195c68c9c5f51c56eeadfe15bce4be2176f63fd058684c70c653274e, DLL584be8a125d0e2334d5d3d1bcb2ecd8f807cae0b42b047e4198f8830f800347e, native proof3de346be45f21942ae412cdf253869ad4c3ee8105aa4bdba236df1e1c61e9d99). This does not admit model arithmetic or speed. Add exact derived probe/batch/stdlib metadata/exclusive owner/independent verifier plus explicit reference/counter layout and original-state/stage replay. All59current Python tests plus86historical assertions pass locally; all original state/vector/cache/token/finite/repeat predicates preserved. EnabledQ128 is bound to original correctT32 reference, never the known-failed T128 profile. Physical newQ count16 only enabled128, zero defaultoff/T4, originalGDN/Z/KV counters unchanged.

Prepare three fresh exclusive sequential profiles off32/off128/enabled128, each with all ten complete first128 stage outputs. Recheck original complete3968prefill state/vectors/all12cache arrays/two repeats and correctT32 128IDs/finalstate in defaultoff controls; independently replay complete stage bytes/read-only observations against qualified old51/DLL3574 before enabledQ. Then enabled128 must pass original correctT32 empty/prefill/final packed states/rawhidden/normalized vectors/all12cache arrays/128IDs/two complete prefills. A first failed assertion terminates the child, preserving partial raw arrays and failure; no repeat claims or fabricated missing fields. One successful3968+128text case would still leave fullsix/quality/performance outstanding. Native source/runtime installation deferred to exclusive owner, full restoration/source hashes and fixed private cache ownership preserved. Deployment050b7c0021fea548a16de473ef76d7df1b388f34fd543f27b2944c76a81dd10d,496292B/76files; preparationc5bf6943cb264057c5c507bcd4a819682b237c9cf625284072d42d932ceb5f1d. No model run or new speed result claimed at preparation.

Current saved HTTP charts still show only imageC8 per-request decode above2x ordinary references (91.4533 vs42.4598/42.6058tok/s); aggregate throughput19.08vs224.97/146.54 and TTFT28.6246vs1.48645/1.60399seconds do not meet targets. NativeWindows physicalbatch1queues vsreferenceWSLbatching and singletrials disclosed. Fullsix4096 ordinary/MTP/image/state/vector/token/reset and805/broaderquality, actualfrontendnegatives/productioncachefingerprint, strongest feasible repeated tuned/MTP references, trueconcurrency, matchedINT4W4A16/4B and all timing/VRAM comparisons remain required. Goal stays active; no performance claim from contracts/builds or one-case admission.


### Q128 removes observed Q/core drift, but original full model gate fails at prefill; isolate O128 next

2026-10-26: Run actual q128-model-admission-attempt1 on dedicated RTX5090 using exact57source a786becf195c68c9c5f51c56eeadfe15bce4be2176f63fd058684c70c653274e and DLL584be8a125d0e2334d5d3d1bcb2ecd8f807cae0b42b047e4198f8830f800347e. All59current+86historical CPU controls pass on exact parent50, embedded stdlib metadata passes, then sequential fresh off32/off128/enabled128. Off32 passes original complete3968prefill,128IDs/finalstate/two-prefill repeats. Off128 reproduces retained known-failed same-profile instrumentation data only. Both defaultoff full states/vectors/all12cache arrays and all50first128stage entries/13107200finiteF32words/readonly records match earlier qualified51/DLL3574 bytes exactly, Qcounter0. Original knownfailedT128 is never admission oracle for enabledQ.

EnabledQ128 selects exactly16Q routes while oldGDN48/reconstructedZ48/KV32/witness1 andT4zero remain. It completes3968prefill and all6cache-prefix captures, then FAILS the unchanged same_packed_bf16_row predicate against correctT32. Child aborts immediately before128decode/finalstate/secondprefill; those gates are unrun, not waived or fabricated. Full packed state119/128components differ, firststoredcomponent9/layer4recurrent. Both5120rawhidden and5120normalized vectors differ (maxabsolute10.257614135742188/2.4893076717853546). All12complete layer3cache arrays remain exact. Baseline source/library/runtime/SDK/checkpoints/privatecache restored. Original raw155555172B archive5641e31a8b378f5fab423df84bfba38fccd9a0f184ad0bbd2d1c60f41828ab4a retained on both machines; paired failureproof51d928d17396304e3c495d44c23ba334314ca06ba5b277e7f07fe674f2622b16. Exact ZIP compressed-payload delta15076634B reuses155identicalpayloads/140422922B, whole original ZIP byte-for-byte reconstructed and independently replayed locally. No scientific data recompression/mutation or GPU rerun.

Separate explicitly SINGLE-observation reader reuses original strict finite/part/hash decoder and repeated correctT32 baseline. It refuses fabricated repeats/decode/admission metadata, incompleteQG parts, corruption/nonfinite arrays, readonly mutation or wrongroute/reference.3newCPU tests pass on both machines. Both machines replay all6553600finite first128stageF32words to identical diagnosticproof1dd915edba0fcfb8aa5f5bc2d3ba5e057c2a4012e812accf021d1ae31f48b2a2. Complete input/normalized/QG/key/value/core are BYTEEXACT to correctT32; QG previous102changedwords and core14542changedwords are now0 in this bounded single observation. First remaining difference is post-O attention at token1/dimension4305: -0.011962890625 vs-0.01190185546875. Attention66/655360changed(max0.00048828125), residual64, MLPinput13451(max0.0012344121932983398), output64(max0.00048828125); beforeQfix attention8857/output45291. This supports targeted nextO intervention, not full numerical admission or any performance claim. Candidate has one first128capture and one fullprefill only; no repeated-candidate claim.

CPU-only re-read of already SHA-pinned pre-Q128 vendor logs observes P geometryN5120/K6144: T32all128calls/64weightbuffers useID35,tile64x32/stages64x5,splitK3/reductionCOMPUTE_TYPE/custom1; T12896calls use same35and32use67/tile64x64/stages64x6/CTA1/custom18. Complete PPPQP pattern repeats twice with identicalweightbuffers;32P calls/16buffers immediately followQ. Identifying these as fullattentionO is an explicit inference from unchanged source order, not logical names printed inlogs. These historical logs precedeQ128; no new Q-enabled O vendor log or causality is claimed. Preserve initial CPUreader failure from incorrectly assuming every Pworkspace4MiB; actualexistingGDN128uses8MiB while ordinaryOuses4MiB. Exact separate retry records actual positive workspace percall and keeps all geometry/type/order/algorithm/weight-repeat predicates. Preserve failure/recovery logs/derivation and originalreader. No model/gate changes or GPU rerun for that reader repair.

Concrete next plan: separate defaultoffO128 plugin/profile/wire/counter over exact57parent,3new+5narrowlymodified files =>60sources. Exact16textOweight names,B1T128,N5120/K6144; own0x4f313238wire and statnative_fp8_o128_canonical_projections. Reuse UNCHANGED existing ModelOptFP8M128Context/liveSDK-owned canonicalM16-ID35 selection plusM128 AlgoCheck and8MiBworkspace contract; no opaque import/patch or widened original GDN guard. PreserveQ/KV/GDN plugins/factories/math/CUDAbytes. Require new source/profile/wire/cross-family/clone/cache/initialize/nativeAPI/tokenizer controls, then defaultoff parity and enabledQ+O against ORIGINAL correctT32 state/vector/cache/token/final/repeat gates. If more drift remains, diagnose rather than weaken thresholds. Fullsix4096ordinary/MTP/image/reset/state/vector/token/805/broaderquality, actual frontend negatives/production fingerprint, trueconcurrency, strongest repeated tuned/MTP references, matchedINT4W4A16/4B and all servingmetrics remain required. No new speed/chart results; overall2xdecodeANDthroughput andhalfTTFT objectives still unmet; goal stays active.


### Separate defaultoff O128 source and strict native contracts implemented; real native build next

2026-10-26: Implement exact60-source candidate (3new/5modified) over immutable57-source Q128 parent. New ModelOptFP8O128Plugin, creator/type GarnetModelOptFP8O128Canonical/version1/20byte marker0x4f313238 and ModelOptFP8O128Selected strict0/1 defaultoff GARNET_FP8_FULL_ATTN_O128_CANONICAL. Only16exact text self_attn.o_proj.weight names atB1T128/inputRows128/N5120/K6144 select; foreign/vision/MTP/malformed names excluded, selected wrong geometry rejected. Reuse exact unchanged ModelOptFP8M128Context live SDK-owned canonicalM16-first-success ID35/actualM128AlgoCheck and8MiB matmul-workspace contract (4MiB selection preference), unchanged original quantizer/input-outputBF16/scales/alpha/beta. Distinct native_fp8_o128_canonical_projections reset/getter/stat; old Q/KV/GDN plugins/factories/wires/guards/CUDAkernelbytes unchanged. Five shared files change only O routing and counters. Private graph cache nonce preserved; production fingerprint/promotion still pending. No new SDK factory or CUDA kernel.

All68current Python CPU tests plus86historical assertions pass locally after new owner/verifier derivation. Reversible source composition verifies immutable57-to60bytes and rejects mutable installed-header reads; all original source/route/state/vector/cache/token/repeat gates retained. Exact eight new CPU-method derivatives extend original Q templates: O profile278(16selected/262rejected); O wire expects36badwire/18constructor/23descriptor/11format/8policy/11cross-family rejections including originalQ, workspace9175040B; O initialize adds eighth route with72initialize/56CPUfactory perpolicy, tenpolicies =>720initialize/560CPUfactory calls. These native executions are NOT yet claimed from local source tests. Native owner retains originalQ278profile/182factory-fault/wire/630initialize contracts and adds all O contracts, then Release/MT/sm120 build/nativeAPI/selectors/tokenizer955CPU smoke. Exact historical parent50 installed before68+86tests, then60source; all failures/source/runtime/SDK/checkpoint/cache/CUDAobject checks preserved. Corrected earlier Q parent-reader metadata lookup is inherited without changing old build predicates.

Fresh remote read-only input inventory passes60targets (14existing/46absent),1373baseline files,6runtime/23environment/2checkpoint metadata pins, CMakecache and34CUDAobjects against actual qualifiedQ128native build; no source or GPU mutation. Inputinventory114e5d51b0c95977ca49175fb6c25803e103a69efc2fba32044dacda2aa43c67. Sourcebundlec44138f5e64b684d556830f58f09c2332d767155669c6642ba799b8445493997. Native deployment37a6670c12ab6deae71e5cbd6d35ca0c0581a91f3561ff41865edfcf5f3979cc (697604B/109files); preparation8d6c1b6ff5d20a34f1059c6dc38b82d70079ab4403ae6519d841ae625ee8928e. Install only method/sourcebundle assets before the exclusive owner; saved newDLL and all native contracts/build outcomes pending. No model/performance result from this preparation.

After native success, require same60source/DLL O-off controls to reproduce correctT32 and preserved Q128-only failure/stage data, then Q+Oenabled with O16/Q16/oldGDN48/reconstructedZ48/KV32 andT4newcounters0. Enabled admission uses ORIGINAL correctT32 complete empty/prefill/final packed states/rawhidden/normalized vectors/all12cache arrays/128decode IDs/two-prefill repeats, no tolerance relaxation. Original Q-only capture was a failed single observation beforedecode/repeats, never relabel it as a repeated pass. Compare all10first128finite stages and readonly inputs; isolate further drift if present. Original fullsix4096ordinary/MTP/image/reset/state/vector/token/805broaderquality, actual frontend negatives/production fingerprint/promotion, trueconcurrency, strongest repeated tuned/MTP vLLM/SGLang, matchedINT4W4A16/4B/servingVRAM/TTFT/decode/throughput and2x fasterbaseline decode AND throughput/halfTTFT remain required. Charts unchanged and goal active.


### O128 real native build and retained/new CPU contracts independently verified; model admission next

2026-10-26: o128-canonical-build-attempt1 completes real native Release/MT/sm120 build on CD-RTX5090. Exact historical parent50 before68current+86historical Python tests, then60source (3new/5modified) over exact57Qparent. All original nativeQ contracts retained and pass:278profile cases,182factory fault cases/86modes,35badwire/18constructor/23descriptor/11format/8policy/9cross rejections,630initialize/490CPUfactory calls across7routes/10policies. New actual O profile278cases passes(16selected/262rejects); new wire passes36badwire/18constructor/23descriptor/11format/8policy/11cross-family rejections including unchangedQ and oldKV/Z/M128/plain. Wire20bytes/marker0x4f313238/workspace9175040B/GPUstubs0. Actual new8route initialize controls pass720initialize/560CPUfactory calls across10policies, including cache/clone/null/standard/unknown exceptions and all original7routes. SharedM128 SDK factory/source and8MiBworkspace contract unchanged; no new factory or kernel. Native selectors pass, tokenizer955CPU cases exact (805screen text templates/124fixed4K/26edges). Tokenizer templates do not qualify image/model quality.34CUDAobjects retained; same3allow-listed overlay/attention/commit artifact hashes differ, CUDA source bytes unchanged.

Saved build/bin/garnet-native-o128-canonical-attempt1.dll SHA8414795d2953feae49532340b206328972c8a23767276319e6e4fb3e1912c18a, sourcebundlec44138f5e64b684d556830f58f09c2332d767155669c6642ba799b8445493997. Original raw native archive45e7e7c90a0b367707db2e9e13636f9647537944314da74abb15c12f0b207948 (7850843B/657members) retained unchanged on both machines. Independent remote and local full raw replay passes with identical proof fd4dc3af8f7570bd53b778c77e5c2bf32f6a2db548ce71e0f20cc9572ddef06c; preserves all old build/CPU/nativeAPI/tokenizer/hash/CUDAobject/restoration predicates and inherited SHA-linked parent metadata lookup. Retain all raw command/fault/trace reports and inputinventory logs, source/reader derivations and hashes on qwen38-5090. Baseline source/library/runtime/SDK/checkpoints/CMakecache restored (14original targets restored/46new removed); saved candidate DLL remains. All jobs terminal and no transfer pending at closure.

No GPU model or matmul/quality/speed was run in this native build. Current full T128 model admission remains failed until demonstrated otherwise. Next use same60source/DLL for fresh correctT32 O-off control and Q128-only/O-off diagnostic equivalence to preserved failed observation, then Q+O-enabled ORIGINAL correctT32 full-model state/vector/all12cache/128IDs/finalstate/two-prefill-repeat admission. Known-failed Q-only cannot become candidate admission oracle. Its prior first128capture was SINGLE with no decode/repeat; only fresh repeats may be claimed, validating prior raw stages with original single-array decoder. Require O16/Q16 on enabledT128, Q16/O0 onQ-onlycontrol, oldGDN48/reconstructedZ48/KV32/witness1, newcounters0 onT4/T32. Complete first128 all10finite stages/readonly records supplement, never replace, fullmodel gates. Preserve any further failure without tolerance relaxation. Original fullsix4096ordinary/MTP/image/token/state/reset/805broaderquality, actual frontend negatives/production cachefingerprint/promotion, trueconcurrency, strongest repeated tuned/MTP references, matchedINT4W4A16/4B and servingTTFT/decode/throughput/VRAM remain required. No chart values changed; overall2x fasterbaseline decode AND throughput plushalfTTFT remain unmet, goal active.


### O128 full4K model admission prepared with strictly separate failed-Q-only control

2026-10-26: Prepare actual o128-model-admission-attempt1 from verified60source c44138f5e64b684d556830f58f09c2332d767155669c6642ba799b8445493997/DLL8414795d2953feae49532340b206328972c8a23767276319e6e4fb3e1912c18a, native pairedprooffd4dc3af8f7570bd53b778c77e5c2bf32f6a2db548ce71e0f20cc9572ddef06c. Three fresh exclusive profiles: off32(Q0/O0) correctT32; Q128_only_control(Q1/O0) compared solely for diagnostic equivalence to preserved failedQ128 archive5641e31a; Q128_plus_O128(Q1/O1) MUST use ORIGINAL correctT32 baseline. Explicit strict-typed reference policy refuses invalid combinations and binds Q+O to correctT32. FailedQ-only reference_admits_model=False, original_reference_primary_equivalence_passed=False, model_admission_passed=False and no decode/finalstate branch; fresh two-prefill repeats may be collected as new diagnostic evidence, never invented for the prior failedsingle observation. Native Ocounter16 onlyQ+Otarget128, Qcounter16 onlyT128 profiles, both0T4/T32; oldGDN48/reconstructedZ48/KV32/stage1 guards unchanged.

Exact-derived probe/batch/embeddedstdlib metadata/exclusive owner/independent verifier preserve all original complete state/vector/cache/token/finite/repeat predicates and immediate failure/save/rethrow behavior. New CPU tests cover exact derivations, failedQ-only reference cannot admit/decode, wrongflag/types/routes/Ocounter boolean/missing/wrongbranch rejection. All71current+86historical CPU tests pass locally. Owner installs exact historical parent50 before those tests then exact60candidate, checks embeddedstdlib metadata beforeGPU, runs both controls and independently replays complete primary/stage/readonly parity before new candidate. T32 control retains128IDs/finalstate. Q-only compares allcomplete empty/prefill state/vectors/all12caches and all10complete first128stages to retained failedQ; oldsingle-array decoder validates prior without repeat metadata and fresh repeat-array decoder validates only actual new repeats. Candidate full3968prefill and128decode IDs/finalstate/two completeprefills must satisfy unchanged ORIGINAL correctT32 predicates. Complete10first128finite stages/readonly data supplement fullmodel admission. First original numerical assertion failure aborts and preserves all partial evidence/restoration without timing. Passing would admit only ONE real3968+128text case, not fullsix/quality/performance.

Deployment632c854f911a0f23054ad7035050f49bb9b2f6806bae347c0bef182405cf6eac (717051bytes/119files), preparation22aa89266e659cff5d05705e7fb5fe507c4c41126ff7ad66a3967daf0917a213. Install only study methods before exclusive source/DLL installation; snapshot/restore all60targets, baseline DLL/runtime/SDK/checkpoints/privateCcache, sourcehash/raw logs retained in fixed studyroot. Actual model outcome pending at preparation. Fullsix4096ordinary/MTP/image/reset/state/vector/token/805broaderquality, actual frontendnegative cases/production fingerprint/promotion, trueconcurrency, strongest feasible repeated tuned/MTP references, matchedINT4W4A16/4B and TTFT/decode/throughput/servingVRAM remain required. No chart values changed; overall2x fasterbaseline decode AND concurrencythroughput/halfTTFT unmet; goal active.


### Q128 plus O128 passes original full4K text admission and all ten complete first128 stages

2026-10-26: o128-model-admission-attempt1 exits0 after three fresh exclusive profiles on exact60source c44138f5/DLL8414795d. CPU71+86 and embeddedstdlib preflight pass. off32 reproduces correct original complete empty/prefill/final states/vectors/all12cache arrays/128IDs and two completeprefill repeats,40stage entries/6553600finiteF32words/readonly records exact retainedcorrectT32. Q128_only_control reproduces preserved failedQ-only complete empty/prefill states/vectors/all12caches/10singlefirst128stage arrays and readonly input, and independently establishes fresh two-prefill/stage repeatability. It has reference_admits_model=False, original_reference_primary_equivalence_passed=False, model_admission_passed=False and no decode/finalstate claim; original failedsingle observation is never fabricated as repeated or admitted.

Q128_plus_O128 now passes unchanged ORIGINAL correctT32 full3968prefill+128decode checks: all complete packed states, all5120hidden/5120normalized raw vectors, all12layer3cache arrays,128rawIDs, finalstate and two completeprefill repeats exact. Physical O16/Q16 onlytargetT128, oldGDN48/reconstructedZ48/KV32/stage1 preserved, newcounters0T4/T32. All6553600finiteF32words in complete first128 input/normalized/QG/key/value/core/attention/residual/MLPinput/output stages are BYTEEXACT to correctT32, zero changedwords in every stage; readonly complete graphinputstate/control witnesses and repeats exact. This closes the observed QG/core and remainingO drift for this ONE real3968+128text case. It does not establish sixcase MTP/image/broaderquality or performance, and no causality beyond captured comparisons is assumed.

Original raw archive38596fef75982e918f064004ecb39638e588c721cc7f82059c86d2b1bad56a83 (155889980B/314members) preserved unchanged on both machines. Independent remote and local full original-gate/stage/restoration replay produce identical proof9dd543a1217d9d78a5dabc9cc87ffa08505212eebf38dc60412a2dd3afc093cf. Lossless compressed-payload delta9375938B reuses197payloads/146471488B from priorQ archive; reconstructs exact originalZIP bytes/SHA without scientific data mutation or GPU rerun. Source/library/runtime/SDK/checkpoints/privatecache restored, raw logs/source derivations/failures/proofs retained onbranch. All jobs terminal at closure; no serving speed or VRAM benchmark claimed.

Next fullsix normal-program(stagewitnessOFF)3968+128cases must preserve original32complete-state/rawvector/ID/callback/reset comparison to retainedT32 ordinary/MTP baseline and allfour text/image inputs plus text/image repeats. Wholemodel onecase admission is a prerequisite, not a replacement for fullsix. Actual frontend negatives/production cachefingerprint/promotion,805/broaderquality, trueconcurrency, strongest feasible repeated tuned/MTP vLLM/SGLang, matchedINT4W4A16/4B and text/imageTTFT/decode/throughput/servingVRAM remain required. Charts unchanged; overall2x fasterbaseline decode AND concurrencythroughput/plushalfTTFT remain unproven, goal active.


### O128 six full4K text/image ordinary and MTP cases prepared

2026-10-27: Following independently paired one-case Q128+O128 proof9dd543a1217d9d78a5dabc9cc87ffa08505212eebf38dc60412a2dd3afc093cf, prepare the SAME60source/DLL8414795d for the original six real3968+128 ordinary/MTP cases: four retained text/image inputs in order0,1,2,3,0,2, with text/image repeats. Stage witnesses are OFF. Preserve all original32complete packed-state witnesses, finite raw hidden/normalized vectors,128 ordinary/MTP token IDs, callbacks/reset/repeats and249draft-prefill-call checks against original correctT32 baseline92eef5433c946f4f2f9dc831902cb80f48b060a92bc6d91879f07137367b1ea5. Physical target128 route counters GDN48/reconstructedZ48/KV32/Q16/O16, oldZ0/stage0, newcounters0T4 and auxiliarydraft/vision guards remain strict. A metadata-only logicalZ view feeds unchanged legacy validators; raw reports and numerical evidence remain unchanged. One-case success is a prerequisite, never a substitute for this suite.

All74current+86historical CPU controls pass locally. Preparation originally refused the older baseline proof's missing exit_code field; retain this failure and exact one-string schema-reader derivation. Recovery reads the hash-verified original raw workflow and requires actual integer exit0, cleanup, source/runtime/library restoration plus original proof32states/sixcases. No missing exit defaults to success and no numerical predicate changes. Exact deployment 197bab37501fdccdf1d2b635118e3a109042fe3a9727be9c8d4dcff193e66646 (738241B/129files), preparation 2d7c799ba4cff94c8e96b88f035e8d781690eef52d0b7586b6e6188435a2bb6c. Same fixed remote studyroot; no new model/clone. Exclusive owner audits source/runtime/SDK/checkpoints/cache, runs CPU and embeddedstdlib preflight before GPU, preserves failure and restores source/library/runtime/private cache.

Actual six-case GPU outcome is pending at preparation. This is correctness work, not serving performance. Broader805quality, actual frontend negatives/production fingerprint, real batching/concurrency, strongest feasible repeated tuned/MTP vLLM/SGLang baselines, matched INT4W4A16/Qwen3.5-4B and text/image TTFT/decode/throughput/serving VRAM remain required. Existing performance charts unchanged; overall2x faster-baseline decode AND throughput plus half-TTFT remain unmet/unproven; full goal active.


### Q128 plus O128 passes all six original full4K text/image ordinary and MTP cases

2026-10-27: o128-full4k-model-admission-attempt1 exits0 with all six cases in retained order0,1,2,3,0,2 passing on the SAME60source c44138f5/DLL8414795d. Actual74current+86historical CPU and embeddedstdlib preflight pass remotely. Normal program has stage witnesses OFF. All original32complete packed-state witnesses, finite raw5120hidden/5120normalized vectors,128ordinary and128MTP IDs percase, complete original correctT32 baseline comparisons, first-token/all-token callbacks, final reset and text/image repeat checks pass unchanged. Each case records249draft-prefill calls. Targetprefill128/decode4/draftprefill16 retained. Physicaltarget128 GDN48/reconstructedZ48/KV32/Q16/O16, oldZ0/stage0; target4 and draft4/draft16/vision/new auxiliary route counters zero as required. Metadata-only logicalZ legacy view never alters raw stats or numerical evidence. Four distinct cases plus two repeats establish this suite, not population-level quality.

Original immutable raw archive e4d33e74c11ea2cd8b4229ea81b6407cb25b8e1ab4a6be78e4619612b2360ac2 (2998982B/223members) is preserved byteexact on both machines. Independent closed remote and local source/library/runtime/SDK/checkpoint and full-original-numeric replay produce identical proof 4058a4f1c4b542883b97c9dc046bebae440e96fb415e95f5952895ef2a9ac9d1. All60source targets/baseline DLL/runtime/SDK/checkpoints/private cache restored. Raw model report, resources, CPU/metadata/GPU ledgers, source derivations, closed workflow, manifest and both proofs retained on qwen38-5090. Correctness process wall time and sampled correctness VRAM are not serving benchmarks. No chart value or2x claim changes.

Next: actual Garnet frontend negative cases and production flag/cache-fingerprint coverage; broad805adapted quality screen and variable/short/tail/image prompts against correctT32, retaining original dataset/quality definitions; then exact candidate HTTP serving with repeatable matched3968+128 text/image C1/2/4/8, actual batching/concurrency work, strongest feasible tuned/MTP vLLM/SGLang repeats, INT4W4A16 and Qwen3.5-4B comparisons. Measure TTFT, per-request decode, aggregate output throughput, attributable/adapter sampled VRAM with sampling gaps and repeated uncertainty. Production promotion and broad quality/performance remain unqualified. Overall >=2x the faster matched baseline in BOTH decode and concurrency throughput plus half-TTFT remain unmet/unproven; full goal active.


### Actual O128 Garnet frontend contracts prepared after full-six correctness

2026-10-27: Prepare17actual frontend cases on same60source c44138f5/DLL8414795d, requiring paired full-six original4096ordinary/MTP proof4058a4f1c4b542883b97c9dc046bebae440e96fb415e95f5952895ef2a9ac9d1 beforeGPU. Use real Qwen35CUDAChunk capture/native lowering with pinned mixed checkpoint, not INT4-only generic projection operation. Six positive fused full decoder-layer engine builds: enabledB1T128 layer3/layer7 Q1/O1/KV2, policyoff, T32/T4 and nonmatchingB2T128 with zero new routes where required. Eleven actual native errors: malformed Q/O/KV/M/Z/reconstruct policy, reconstruction without explicit canonicalZ, T129 and missing full-attention weight. Strict exact case identity, observed flags, error substring, source/DLL and typed physical counters are independently replayed. Every case uses a fresh embedded process so Windows native environment tables are initialized from declared flags; one graph/process at a time. No model decode or numerical/performance admission by these graph tests.

Local77current+86historical CPU controls pass. First CPU import caught one missing generator parenthesis; failed raw source, original command/output, exact one-parenthesis repair and corrected passing CPU run are retained. No native run occurred for that failure and no predicate was waived. Deployment cf86a120d7fe80ec55124903c53fb1eba7a1e319f3c261251ab9d411178fadb9 (993531B/141files), preparation bc3c9d5bfbedeaed9404cbd114fec07f48a3b1629ea9e4633b662dca928385a5. Installer preserves existing files and defers exact60source/DLL installation to audited exclusive owner, which retains original source/runtime/SDK/checkpoint/privatecache restoration. Actual native outcome is pending at preparation.

Source audit records compiled_model_runtime.cpp MakeGraphFingerprint omits study FP8 policy material, while candidate CUDACompiledGraph uses unique temporary nonce paths. This is source evidence, not an executed production cache-collision test. Production policy fingerprint fix/cache test remains required. Following actual frontend contracts: broader805adapted quality and variable/short/tail/image correctness, research HTTP serving with exact candidate source/flags/identity, matched repeated C1/2/4/8 measurements and trueconcurrency optimization, strongest feasible tuned/MTP references, INT4W4A16/Qwen3.5-4B. No benchmark chart or overall2x decode AND throughput/halfTTFT claim changes; full goal active.


### Actual Garnet frontend passes six compiled controls and eleven required native rejections

2026-10-27: o128-frontend-attempt1 exits0 after77current+86historical CPU and embeddedstdlib preflight. SAME60source c44138f5/DLL8414795d, no native rebuild or arithmetic change. Seventeen fresh embedded processes call actual Garnet Qwen35CUDAChunk capture/native TensorRT lowering with the real pinned mixed checkpoint. All six complete fused decoder-layer builds pass: enabledB1T128 layer3/layer7 physically Q1/O1/KV2; policyoff, T32/T4 and nonmatchingB2T128 have zero new routes as required. GDN/reconstructedZ/stage route counters remain zero in these full-attention-only graphs. All eleven malformed Q/O/KV/M/Z/reconstruct flags, reconstruction/canonical coupling, T129 and missing full-attention layer errors are rejected with the expected native error text. Raw native errors, flags, graph stats and per-case command/source/exit records are retained and independently checked. Failed general acceptance or unrelated native error could not pass. This graph-construction suite is not model decode, quality or performance evidence; paired full-six original4K numeric proof4058a4f1 remains separately pinned.

Original archive 1963e4b4454cf9b1f6bddccdfcae27b8f2ac358ab9a8c0bee5bfa74a064be5c4 (3975117B/221members) preserved byteexact remotely and locally; independently replayed proof 3cb4f2f55ca6f10c714ef4e69156133a29061434ccf0eb7cb88a3add3d82ea58 matches on both machines. Full source/library/runtime/SDK/checkpoint/privatecache restoration is verified. First CPU syntax refusal and exact repair remain preserved. Actual frontend gating is now established for this explicit matrix. Source audit still finds production MakeGraphFingerprint omits FP8 policy material; study graphs' unique temporary nonce paths do not prove production cache separation.

Next production cache work: add stable versioned FP8 policy material to compiled_model_runtime.cpp before cache lookup, normalize documented unset/empty/0 semantics, preserve native flag validation/coupling, cover each relevant policy and ensure fresh-process cache miss on change and hit on same setting using a tiny compiled_xmodel addition fixture. A separate native candidate must preserve all60numeric source bytes and CUDA objects, then pass actual native/cache contracts before promotion. In parallel sequencing (one GPU owner only), prepare the exact60source research HTTP worker from the retained prefill16 worker, bind startup to full-six/frontend proofs and loaded841DLL/flags/source, and run frozen805adapted quality plus variable/short/tail/image/reset cases before new serving benchmarks. Then repeated matched3968+128 text/image C1/2/4/8, real batching/concurrency, stronger tuned/MTP reference repeats, INT4W4A16/Qwen3.5-4B remain required. No chart changes or overall2x decode AND concurrency-throughput/halfTTFT claim; full goal active.


### Production graph policy fingerprint candidate62 prepared

2026-10-27: Prepare cache-only62source candidate from immutable60numeric source c44138f5. No operator/quantizer/factory/plugin/wire/CUDA source bytes change. Add pure graph_policy_fingerprint.h and three exact compiled_model_runtime.cpp edits: include, versioned material at TensorRT MakeGraphFingerprint, invalid_graph_policy status before cache lookup on invalid canonical policy/coupling. Eight canonical FP8 controls normalize unset/empty/0 and reject non0/1; reconstruct requires canonicalZ. Four additional graph controls preserve their existing parser semantics and use length-prefixed material to avoid embedded newline ambiguity. Schema qwen38_graph_policy_v1 must change when serialized policy meaning changes. This candidate is not yet built, production-cache tested, or model qualified.

The original1373 model-source inventory omitted compiled_model_runtime.cpp: preparation first refused that absent key. Preserve original/helper failure/recovery output and exact repair to separately audited bytes; freshly verified actual remote production runtime is added to new1374baseline. Native/front-end parent records differed only by extra status metadata and one omitted SDK parent; preserve original assertion and exact retry that retains union of ALL parents and compares every shared hash/exit field, with exact nonmutating reuse of already-created62sourceZIP. No scientific gate defaults or data mutation. Remote inventory checks62targets(15existing/47absent), runtime/SDK/checkpoints/CMake and34CUDAobjects against closed O128 build. Local80current+86historical CPU tests pass. New actual pureC++ header contract will require72checks,43strict/coupled rejections,13distinctkeys, aliases/order/newline separation. All old Q/O profile/factory/wire/initialize and955native selector/tokenizer controls retained. Build requires all34CUDAobject bytes identical, stronger than old3object allowlist; source/runtime/SDK/checkpoints/CMake restored and candidateDLL saved.

Deployment c9ca00e354f160c4e820ad343ed3982e1803c3f519ce63fe1d79d9061bcc0a6e (1152750B/154files), preparation e1d9a80129da0efb418514a1e85e0f3b64252e9798b1033a302d10bd80a04ead, sourcebundle 03bdbf860783513d815c99a8aabfa08cfcfd60e62a7785f6383d66094a856016. Native build outcome pending at preparation. Next actual production load_model tiny addition fixture, fresh process perpolicy: same/default aliases hit, changed settings miss, invalid settings fail before cache lookup and preserve prior cache bytes, real tiny GPU output exact. That is cache behavior evidence, not VLM numerical admission. New binary then requires original fullsix4096state/vector/token/ordinary/MTP/image/reset and broader805adapted quality/variable short/tail cases before timing. Research HTTP identity must bind exact source/library/policies and use matched repeats C1/2/4/8, actual batching/concurrency and strongest tuned/MTP vLLM/SGLang, INT4W4A16/Qwen3.5-4B. No chart or >=2x fasterbaseline decode AND throughput/halfTTFT claim changes; full goal active.


### Cache candidate62 compiles and passes header controls; strict object identity fails and remains unqualified

2026-10-27: fp8-policy-cache-build-attempt1 exits1 at the explicitly stricter all34rawCUDAobject identity gate. NativeRelease/MT/sm120 compilation itself exits0. Actual80current+86historical CPU, new pureC++72checks/43rejections/13distinctkeys and all original Q/O profile/factory/wire/initialize command phases through native build exit0. Native selector/tokenizer955smoke did NOT run because the object gate correctly aborted first. Do not call this native build qualified or replace its failed status. Original raw archive ca152badb646fff5d0fa00da01a8602b784de4f2aca021046f5632a826e52a37 (8314261B/664members), paired independent failure proof e93e41e0aa59d687ca0bae2df25d27e22cacf1100f3a649ab6917c816c75c8c9 retained unchanged on both machines; failed compiled DLL dc15e4f53855fb91e182cf52e78eb3917068b8df80cde1242ff1799ce4fb99ba preserved only as a failed artifact. Source/library/runtime/SDK/checkpoints/CMake restore verified.

Read-only diagnosis captures three changed raw COFF objects. Two (prefill128 attention/commit) reconstruct EXACT historical SHA256 by changing ONLY the four-byte COFF timestamp to values within the SHA-pinned O128 native compile ledger UTC window. Full raw new/historical objects and evidence retained. The low-batch overlay object does not reconstruct by timestamp-only changes in known parent compile windows or the bounded2026-10-20..current stamp search; its remaining host-object difference is NOT explained. Do not call all three metadata-only. Independently parse old841DLL from the retained O128 native archive and failed new DLL from the failed archive: __nv_mod, __nv_rel, .nvFatBi and .nv_fatb raw PE sections are all byteexact. All60numeric source files are unchanged by construction; this does not waive the failed all34rawobject gate or qualify the new binary's model arithmetic. Original32complete-state/raw-vector/token/MTP/reset gates remain unchanged and pending for any new binary.

Prepare (not execute) a41case actual production load_model cache matrix with fresh processes:14misses/16hits/11invalid/coupled rejections,13distinct fingerprints, unset/empty/0 aliases, return-to-default, strict cache file chain, invalid-policy previous-cache preservation, and tiny GPU output [1,6,15,28]. Preserve complete before/after cache files so independent replay can verify bytes, not just reported key strings. Three CPU corruption/control tests pass. No production controller/preparation or native production-cache call has executed; these source helpers are preparatory only and fail closed without preparation. Any provisional use of the failed DLL must be labeled exploratory correctness diagnosis, never promotion or benchmark qualification. Resolve artifact/reproducibility evidence and run native API smoke, actual cache tests, original fullsix and broader805quality before new serving results. No benchmark/chart/2x claim changes; source candidate62 remains unqualified and full goal active.


### Exploratory native API and production-cache diagnosis prepared on preserved failed DLL

2026-10-27: Original candidate62 build identity gate remains FAILED; proofe93e41e0 and original failed archive unchanged. Prepare additional correctness DIAGNOSIS on preserved DLLdc15e4f53855fb91e182cf52e78eb3917068b8df80cde1242ff1799ce4fb99ba/source62bundle03bdbf86. Native compiler/80+86/header72 phase passed, four NVIDIA sections match old841, two COFF objects differ only timestamps, one host-object difference remains unexplained. No identity predicate changes or qualification claim. Testing an unqualified artifact here is explicitly exploratory correctness evidence only; no promotion or VLM/quality/benchmark permission follows.

The controller temporarily installs exact62sources/DLL, requires unchanged source1374/runtime/SDK/checkpoint and CURRENT34object hashes, runs3CPU cache mutation controls, actual20native/embedded selector cases and955frozen tokenizer C API calls, then41fresh-process production TensorRT load_model cache calls on a tiny addition fixture. Require14misses/16hits/11strict or coupled invalid-policy rejections,13keys, alias and return-default behavior, exact cache-file chain/full before-after raw file snapshots, rejection preserving prior cache bytes, and real tinyGPU output[1,6,15,28]. Per-case compiler backend/counters/source/DLL/flags and raw cache hashes replay independently. Fixed studyroot, one GPU owner, source/library/runtime/SDK/checkpoint restored afterward and no CUDAobject or build modification. Original build-identity gate staysfalse in preparation, workflow, report and proof; zero VLM or serving timing admission. The3local CPU controls pass after adding actual backend and scratch/source checks.

Deployment 44e7c8cf21c4d67f13d0d8ef507fc33cda50e10e1bcc47cbc01022f949c69ac3 (1101118B/162files), preparation 5c37fcb942861da155bb34d1edce1d18892a42a0902a0823eff085ce147bb298. Actual API/cache outcome pending at preparation. Broader source/host-artifact reproducibility evidence and original fullsix32state/vector/ordinary-MTP/token/image/reset plus805quality remain required for any new binary. Existing60-source841 candidate/fullsix4058a4f1/frontend3cb4f2f5 remain unchanged. No chart or2x decode-and-throughput/halfTTFT claim changes; full goal active.


### Diagnostic preflight oracle closure corrected before first API/GPU call

2026-10-27: Review finds the exploratory collector did not yet include p.audit_dependencies (the frozen955tokenizer oracle) in initial hash checks or raw archive common inputs. Add that exact dependency map in both places, preserve old owner/preparation bytes, and bump setup revision2. No case/predicate/native source/DLL changes. Controlled installer verifies prior SHA of both owned method/preparation files before replacement and reaudits baseline/method/checkpoint/oracle hashes. Original deployment44e7c8cf and preparation5c37fcb9 remain retained; corrected preparationaa1979840653cd61a1fcf6f24b2af0c8292fc69dc62c5cdb2c158bec5fc12583 and fix ZIP4e95b9dd16ac14e3f1ebcb2e43d824223f778da6bdc478a1e7183969cffbfc9d (131528B). AST check passes. No actual native API/GPU diagnostic ran before this correction; original raw object identity gate remains failed and no benchmark/quality/promotion claim follows.


### Native APIs pass; production fixture extension refusal preserved and Python retry prepared

2026-10-27: Exploratory production-diagnostic-attempt1 exits1 after3CPUcontrols,10native selector+10embedded selector cases and955tokenizer C API cases pass on preserved unqualified DLLdc15e4f5. Independently replay every native freeze/flag record and955actual IDs against raw archived frozen oracle. First production load_model refuses stale .x fixture with invalid_root_xmodel / requires existing Python model file; no cache lookup/graph compile/GPU addition reached (empty before/after cache, no output). This is fixture/API setup failure, not evidence that cache policies passed. Preserve raw archive1e42adac68f7fd3dc27eb0c24b22e3c71389d6671605d62e1820382b7ff62ff4 (2781749B/242members), paired failure proof7e3b45fd9b5252b47319f98ad9c7b747d4bd0c21ca0ff9bf7747edfd1d49847f and complete source/runtime/SDK/checkpoint restoration.

Inspect CURRENT compiled_model_runtime.cpp: rootPath.extension must equal.py, generic root arguments accept symbols without a model spec, tensor_from_host defaultsCUDA; old phase24.x fixture is stale for current loader. Prepare exact-derived retry2 using new Python fixture and current garnet/TensorCompat named-fusion import. Arithmetic (x+x)*x-x unchanged; original41cache/counter/alias/rejection/snapshot/GPUoutput predicates unchanged. Preserve all attempt1source files/preparation/output and use new method names, separate preparation, attempt2output. Bind first failed API fixture proof as parent, original native build identity remainsFAILED/false everywhere. Deployment 96517cf1abb6a1ca9842f2c4185785b257b67c1a37495f07f641e4c472cbc233 (1241950B/170files), prep cc2dc2a0d6ca6eb3aaa912d69d6e027417d78af279e2ac62bf794716d451d5ff. Actual retry pending at preparation; no VLM/quality/benchmark/promotion qualification or chart change. Full goal active.


### Production cache41 passes as exploratory; return to qualified60 model measurements

2026-10-27: Actual current Python compiled_xmodel/TensorRT fixture completes all41 fresh-process calls:14misses,16hits,11strict/coupled invalid-policy rejections before lookup,13distinct MD5 fingerprints. All valid GPU outputs exactly[1,6,15,28]; unset/empty/0 aliases hit;12changed policies miss then hit; returning default misses then hits. Independently replay complete raw before/after engine/cache bytes, chain, source/DLL/flags/counters,20native selectors and955frozen tokenizer cases remotely and locally. Source/runtime/SDK/checkpoints restored. Raw archive a26dcad153871335c433103cfe254e14ca5d4ad8444ef7657d246f42982435be (8168219B/536members), paired proof ab237d6aac7be8e7c4671749ac71734a0e7ba95f9c0e82a4879587c5e8959c9b. Preserve attempt1 fixture refusal and original62-source build raw34object failure unchanged: this diagnostic remains exploratory, no VLM/quality/performance or production promotion.

Resume requested model quality/performance using already qualified60-source bundle c44138f5e64b684d556830f58f09c2332d767155669c6642ba799b8445493997 / DLL8414795d2953feae49532340b206328972c8a23767276319e6e4fb3e1912c18a. Six actual full4K ordinary/MTP text/image/reset cases and17frontend contracts already pass original complete gates. Use existing per-run nonce CUDACompiledGraph engines. Production cache patch/object reproducibility remains a separate unfinished candidate. Next: explicit source-bound128-prefill HTTP worker, adapted805 quality/token comparison and matched repeated HTTP measurements; charts unchanged and overall2x target unmet.


### Qualified60 O128 research HTTP measurement prepared

2026-10-27: Prepare explicit MTP-only HTTP worker on the already validated60source/DLL841 candidate, targetprefill128/decode4/draftprefill16. All request-validation and token-notification function ASTs exactly match retained prefill16 worker; original JPEG/image positions/token IDs/EOS/callback/response paths retained. Actual source/runtime/SDK/checkpoint and three pinned native/fullsix/frontend certificates checked before model construction; actual physical Q16/O16/KV32/GDN48/reconstructedZ48 routes onlyT128, allnewrouteszeroT4/draft4/draft16/vision/norm. Private per-run nonce engines only. CPU21controls pass locally, including corrupted restoration/certificate data and original805quality/protocol controls. New exclusive owner saves originalsource, installs exact60bundle/DLL, supervises four full3968+128text/image/reset HTTP controls and805 adapted low-resolution screening requests, restores baseline and retires only owned engine files even on failure. Every prompt/output ID must match complete saved correctT32 ordinary805 collection; its enclosing old HTTP benchmark FAILED later, so no oldHTTPsuccess claim. Immutable raw outcomes and independent verifier prepared. Native/model run pending at this preparation; no new speed, broadquality, official900MMMU or2x claim. Next matched performance remains required.


### O128 actual HTTP805 token parity and four4K controls pass; performance still pending

2026-10-27: Original o128-http-quality-attempt1 exits0. Same60-source/DLL841 research worker passes four actual3968+128 text/image/reset HTTP boundary controls, then all805 frozen low-resolution adapted screening requests. Every prompt and output token ID matches independently replayed complete correctT32 ordinary collection. Adapted strict-MC score 502/805 (not official900 MMMU); old enclosing tokenizer-nfc HTTP timing attempt remains failed, only its complete805 collection used. Independently replay all809native requests and full token callbacks against raw HTTP replies, original four-boundary model IDs, raw805wire/responses/corpus and exact original control. Source/runtime/SDK/checkpoints restored, owned private nonce engines retired. Raw archive 0bd29f0fb3c811aa4c3095b59e5d07cdd2043999be610d361cc8a883a8d777a6 (32992991B), paired remote/local proof 6853b3033d23a2dceac963bce3f08548fa1efdbdfb11cf3222f5746a942e2ed4. Full raw ZIP retained in outputs and fixed remote study; per-case records, worker logs, methods and checksums retained in branch. No new performance measurement, broad-quality/official900/strongest-reference/2x admission. Next three repeated matched fixed-length3968+128 HTTP trials; retain predeployment CPU-detected context-vs-fixed-length corpus error and repair, no native benchmark ran on wrong corpus.


### Three matched O128 fixed-length4K HTTP timing trials prepared

2026-10-27: Require independent805HTTP proof6853b3033d23a2dceac963bce3f08548fa1efdbdfb11cf3222f5746a942e2ed4 before fresh source-bound60/DLL841 worker constructs. Prepare three complete fixed-length3968+128text/image C1/2/4/8 trials,124requests (64measured/60warmup) each,372total. Same original fixed-length corpus596991dd/manifest25808fff and unchanged existing fixed-length HTTP timing harness. Replay every native callback and all124prompt/output IDs eachtrial against retained correctT32 prefill16 MTP archive; recompute rawSSE timings and adapter VRAM prefix snapshot bracketing requests, preserve trailing samples and gaps. No runtime/source defaults changed and physicalbatch1queue remains explicit.

Predeployment CPU positive control caught wrong earlier context corpusca78/829d: retained reference corpus differs by explicit ignore_eos fixed-length policy. Preserve original benchmark methods, failed CPU ledger and exact source repair. On correct original fixed-length corpus, actual retained124request positive replay passes; self-consistent changed warmup token, altered VRAM summary and altered latency summary each rejected. No new GPU benchmark ran with wrong corpus. Deployment 7c150bddefe52d053cc960d0d72a13b1d301466bbd2056ec51d108adbca23b50 (614090B/149files), prep 50940694f3aa0949c45b132ee48cfde4b255e5d641a3d3aaceb24ff0d275a2af; methods/source/restoration/quality parents pinned. Repetitions within one worker; nativeWindows vs referenceWSL, strongest repeated tuned/MTP baselines, fullINT4/4B and2xdecode+throughput/halfTTFT goal remain open. New benchmark is pending at preparation; no new speed claimed.


### Prepare explicit B4T32 flat-M128 FP8 eligibility hypothesis; no native or model qualification

2026-10-27: While original O128 three-trial GPU benchmark remains exclusively live and its deployedsource frozen, independent LOCAL source inspection identifies a potential batching arithmetic contrast: qualified canonical FP8 selectors accept logicalB1T128/flat128rows, but legacy physicalB4T32 also has128flatrows and currently falls through. This is a hypothesis, not established cause of prior B4/B8 vector/token failures. Prepare immutable61-source candidate from exact60parent: seven host eligibility files modified and one shared strict0/1 defaultoff GARNET_FP8_BATCH4_PREFILL_CANONICAL header added. Preserve physicalbatch4/tokens32 in all calls; no fake B1shape normalization, recurrent-state flattening or new SDK descriptor. Optional new parameter defaultsnullptr, preserving original function arity; existing per-family policy/name/N/K/flatrows checks remain, B1T128 remains originaleligible, and B4T32 requires explicit newflag. B2T64/B8T16/B4decode are excluded. ReconstructedZ coupling remains required. Actual128-row plugins still use identical qualified M128 SDK-owned context, canonical M16 selection/actualM128AlgoCheck and8MiB workspace. All plugin bodies, quantizers/scales/alpha, context factories, wires, CUDA sources, state/cache/commit implementations and private nonce engines byteunchanged. No production source or running model altered.

Three logged CPU SOURCE composition/unchanged-math/tampered-source controls pass locally. These do not compile or execute C++ behavior. Prepare remote-only header C++ control probe for all six selectors, original arity/defaultoff/B1 compatibility, B4 admission, otherbatch exclusions, malformed flags/geometry, canonical reconstruction coupling, all16text layer names and foreignweight exclusions. It remains UNCOMPILED/UNRUN until current GPU benchmark closes. Next must verify actual native header controls and actual flat128 projection outputs/SDK selection on5090, then build candidate and compare physicalB4 with qualified per-rowB1 complete state/finite hidden+norm vectors/all ordinary IDs across variable short/tail/full4K text/image/reset/repeat. Original failed B4/B8 controls remain preserved; no tolerance/ID/state gate relaxation, no MTP/quality/HTTP/performance or cause claim. Memory fit and production shared-cache policy compatibility remain pending. Use fixed existing studyroot; no parallel GPU jobs. Overall2x decode+throughput/halfTTFT goal active.


### Remote-only B4 header control executable prepared; wait for benchmark closure

2026-10-27: Prepare13file CPU-header deployment b6221c3f7a3a4f80950d81dd2478eece3ba01ca629e01a1b7f354b3f2e2e1925 (20633B), preparation469ae6c11cb206f3902232e0ae813ace41fbbd6b0442a2f643f40ee9dd61d6e7. Native benchmark remains the sole live GPU job; do NOT upload/deploy/compile these methods until its terminal/restored state and independently replayed complete372request proof are verified. Installer and owner enforce that prerequisite and idleGPU; preserve all pre-existing bytes. Header-only remote MSVC/stdlib executable uses the eight candidate host files and original per-family selectors, records compiler banner/tool hashes/source hashes/raw success or failure. It does not compile Garnet, link CUDA/SDK, execute a GEMM, change production code/library, load a model or produce benchmarks. C++ predicate counts and rejections are pending actual execution; preparation expectation235/61 is not a measured result. Actual SDK/projection math, full per-rowB1/B4 state/vector/ID/MTP/quality/memory and repeated HTTP performance still required. Full goal active.


### Independent timing replay/figures and CPU-header verifier saved before outcomes

2026-10-27: Original o128-http-benchmark1 remains live; first two complete124request trials passed local raw replay and allretained T32prompt/output IDs, third trial pending. Save independent remote/local raw benchmark replay wrappers and figure renderer before final outcome. Figures require three complete independently paired proofs and preserve median/min-max ofnative trials, maximum sampled adapter VRAM, rawtrials/CSV/gaps, with retained single-trial ordinary references clearly labeled. No confidence-interval or strongest-reference/2x claim. New artifact names preserve original chart files. No renderer or new numeric replay executes while GPU benchmark live.

Save separate B4 CPU-header verifier before actual control execution: re-derive61source composition from archived60source bundle, validate raw compile/control ledgers, executable SHA/AMD64 PE target,235expected predicates and61expected rejections, compiler/source/library preservation and closed benchmark parent. Expected counts remain expectations until remote executable runs. This CPU test does not executeGEMM/SDK/GPU/model or qualify arithmetic/MTP/quality/performance. No remote deployment until sole GPU job terminal/restored and independently verified.


### O128 repeated fixed4K HTTP outcome independently replayed

2026-10-27: Original o128-http-benchmark-attempt1 exits0; status actual_O128_three_fixed4k_HTTP_trials_raw_stream_memory_and_IDs_replayed. Paired remote/local proof c86b1a10553dbb83d5753495c3e6ede4411187ee1909e2fc05ee48a7eef9005c, immutable raw archive 0233565d6de5c4973760e4f618e91d192a070f342cfc3b8967facd4d61fb3206 (6698840B). Independently replay source/runtime/SDK/checkpoint restoration and all archived methods/ledgers. Full goal remains active; no strongest-reference/2x promotion.

Three complete124request trials (372actualrequests/192measured/180warmups), all8text/image C1/2/4/8cohorts each, every actual3968prompt+128output ID exact retained correctT32 MTP control. All372actualnative callback streams/final IDs match raw HTTP streams. Recompute all cohort SSE timings and exact memory-summary snapshot prefixes that bracket measured requests; retain later samples and observed gaps. Statistics are median/min-max across3cohort statistics, not confidence intervals or pooled-request statistics. VRAM uses maximum sampled adapter usage across3trials, includes background memory, not process/instantaneous peak. Repetitions share one nativeworker. Separate new8panel/all4individual PNG/SVG figures, HTMLreport, raw-trial CSV andJSON retain oldnative and ordinary vLLM/SGLang single-trial baselines labeled accordingly. Nativephysicalbatch1stillqueues; referenceWSLup8requests and output text variations persist. Cross-engine long-output equivalence and strongest repeated tuned/MTP references remain unproven. Matched INT4W4A16/Qwen3.5-4B still required. No benchmark rerun hidden or inferred from process-walltime. CPU B4eligibility candidate remains separate pendingactualSDK/model/quality/performance admission.


### Actual B4T32 CPU header235 controls pass; GPU/model validation still pending

2026-10-27: Original b4-prefill128-header-attempt1 exits0 on remote pinned MSVC14.44.35207 (actual compiler19.44.35229.0). Actual compiled header executable passes235checks including61expected invalid-policy/geometry/coupling rejections across all six selectors. Original function arity/defaultoff/empty-zero aliases/B1eligibility, explicit B4T32flat128 admission, otherbatch/decode exclusion, N/K/flatrow guards, reconstructedZ coupling, all16text layer names and foreignname exclusions exercised. Independently replay raw compile/control ledgers, nativeEXESHA/AMD64 PE target, immutable61source composition from archived60bundle and compiler/source/library preservation remotely and locally. No native executable runs locally. Paired proof 5e54d813544059562fd7b4b9a24323775f0ef3395c370bb771d75bb637933826, raw archive 2b1471b800a78eac8e55e516a69bc8b2e95ef725f9f01018143338e9981bfe95 (267267B). Sourceheaders/controls/manifest/logs/hash proofs retained, raw executable/object bytes preserved only in immutable ZIP.

No GEMM, CUDA/SDK or model executed by this header test. Existing Garnet baseline1a1d remains restored; no new Garnet binary built or qualified. Next actual SDK/projection and full physicalB4vsper-rowB1 complete-state/finite hidden+norm/token/short-tail/image/full4K/reset gates remain required before MTP/quality/performance. Preserve original failedB4/B8 contrast without claiming its cause. New measured O128 three-trial figures/evidence are separately retained on same source60/DLL841; they do not qualify this61source candidate. Overall2xdecode+throughput/halfTTFT, strongest repeated tuned/MTP references and matchedINT4/4B remain open.


### Descriptive MTP counters/client arrivals replayed; no call-count speed proxy

2026-10-27: Replay all372complete fixed4Krequests including180warmups from paired verified O128 archive,186text/186image. Weighted draft acceptance is0.5909517112text/0.6985889731image. All requests use249draftprefill calls and headreuse0; target full-forward count equals verification count, prefix fullrecomputes0. Source-pinned packed-prefix replay uses retained authoritative projections and reruns no target projections; do not mistake target_prefix_replays for extra full-model forward calls. Descriptive client first-to-second arrival median is44.97ms text/44.86ms image; median of per-request mean later-gap is12.72/11.10ms. These include runtime/transport scheduling, are not CUDA/kernel phase times or a new benchmark. Counters do not establish a speedup and do not justify re-enabling earlier failed head-reuse optimization. Controlled phase profiling and actual B4 SDK/model validation remain useful next steps.

Preserve first new counter-diagnostic failure and exact source repair: it incorrectly required final verification-iteration sampled128. Original generate emits a final remaining token and breaks without appending another verification iteration, so last recorded iteration127 is valid. Record observed127/128distribution; require unchanged actual128stream IDs,127processed target tokens and exact verification-iteration counts. No official benchmark/model/state/vector/token assertion changed or waived. Retain original failed source and ledger alongside corrected success.


### Standalone chart presentation v2: checkpoint label and explicit sampling gaps

2026-10-27: Export separate -v2 combined8panel/4individual PNG+SVG, HTML, CSV andJSON artifacts; preserve allv1files and original measured trials. Add standalone Qwen3.8-27B mixedNVFP4/FP8/RTX5090 labels, complete per-native-trial memory snapshot metadata, CSV memory sample/gap columns and explicit largest observed native sampling gap4.284s. Verify all plotted values, min/max ranges, ratios and original proof/source pins EXACTLY equal v1. CSV48data rows includes24newnative rows (3trials*8cohorts), plus single-trial oldernative/vLLM/SGLang rows. Visually inspect combined and TTFT layouts. This is presentation/metadata only: no model, GPU or new benchmark run; memory remains sampled adapter usage, native3trials vs reference1 each, bands are min/max not CI. Reported2xgoal remains unmet/unproven.


### B4T32 native build prepared with original guards and frozen oracle

2026-10-27: GPU idle and fresh remote baseline1374/runtime/SDK/checkpoints/CMake/current34CUDAobjects independently inventoried before deployment. Separate61-source B4T32 eligibility candidate package prepared after paired235CPUheader controls. Native owner preserves original O128 profile/factory/wire/init/selector20/tokenizer955 and all original numerical/model gates; 71current plus86historical CPU controls. Source bodies/quantizers/scales/SDK contexts/CUDA bytes unchanged. Owned frozen tokenizer oracle ZIP explicitly included and hashed. All baseline sources and active DLL restored after build even on failure. Current object ancestry from the failed cache62 build is recorded; the original three-object native-build whitelist is preserved, and the cache-only stricter all34identity failure remains failed. No new native binary/model result or performance claim exists at this preparation point.

Package SHA256 f7b3fbb1cd57492933990870c8c2da601a528dea64208d7318a4a5bcbe5a6fbe, preparation 41ac759a54a79c4539c2f8c338f5689d9e07b4536ab38f35591b1d5be4ee550f, source61bundle 23adffb664de5ade36d76b7377011e96a42de669a91a486cb39bb84e678bdd63. New binary must pass actual SDK/projection and physicalB4 vs per-rowB1 complete-state/finite hidden+norm/IDs/short-tail/full4Ktext-image/reset/repeat gates before batching/MTP/quality/performance admission. Strongest repeated tuned/MTP references, matched INT4/4B and >=2xdecode+throughput/halfTTFT goal remain open.


### B4T32 native DLL built and paired CPU replay passed; GPU math remains unqualified

2026-10-27: b4-prefill128-build-attempt1 exits0 on CD-RTX5090, actual Release/MT/sm120 native build. 71current+86historical Python controls, original Q/O278profiles,182factory fault cases, original Q/Owire and630/720initialize controls, native selectors and955frozen tokenizer cases pass. All34CUDAobject hashes recorded with original3object whitelist unchanged. Full source1374/61targets, runtime, SDK, checkpoint and CMake restoration independently verified remotely and locally. Raw archive 8ec8188e932c80c732d4ae5a9dec830854f8eff78b9ed2c160b3bddc1285b4cf (8732965B/726members); paired proof a11e469a50acdebddab252c6793bb62adb9054b02e87c2b742af3ebe9705c465; separate candidate DLL acb9ff9e12efb808f498d2bfc049d5a98c22c3ae0bddbc08b9675b8d5acfb5b6. No model or matmul executed; existing source60/DLL841 numerical/quality/HTTP results cannot qualify source61.

Retain raw native/CPU/fault/trace reports, inventories, reader derivation, transport failure and receipts. Initial local closure SHA transcription failed (extra PTY-wrapped character), rejected before verification; original draft retained and repaired using downloaded machine-readable remote receipt. Final raw bytes and paired verifiers pass unchanged gates. Next actual SDK/model physicalB4-vs-four-B1 short/tail/full4K text/image/all complete states/finite raw hidden+norm/IDs/reset/repeat admission before MTP/805quality/HTTP/performance. Cache62 strict object-identity failure stays failed. Overall2x strongest-baseline decode+throughput, halfTTFT, repeated tuned/MTP references and matchedINT4/4B remain open.


### Fresh four-B1 versus B4 short mixed-tail gate prepared

2026-10-27: Paired native61build success allows first actual GPU numerical experiment, not timing. Fresh B1reference uses exact qualified60source/DLL841 and B4policy0; candidate uses exact61source/newDLL and policy1. Four archived INPUTS only (41/104/93/93prompt tokens, text/image) exercise heterogeneous short tail;8generated IDs each. OldB8 differences remain preserved, not reused as a correctness oracle. Complete128statecomponents and physical/logical row hashes plus finite raw5120hidden+norm bytes at reset/prefill/final/finalreset. Four CPUcontrols reject lastcomponent corruption, rowpermutation, signedzero byte difference, NaN, missingcoverage, slot andID corruption. Source/runtime/SDK/checkpoint hashes, resource samples, raw failures, cacheownership/96GiBfree, exclusiveexecution and restoration retained. Preparation/deployment 6789e96becd90309453a165efb157298b40beafe3d6cfe6a0d582d989b2e0983/3364f4737474210d93c55dfbe16bbc787d003e9d92aaa5bf94baa0d9e5145e83.

A short pass is preliminary: original full4K text/image, allIDs/complete states/rawvectors/shorttail/permutation/reset/repeat, originalB1/fullsix/805, MTP andHTTP gates remain unchanged and required. No new benchmark/quality promotion or2xclaim. Next record actual physical canonical selector counts and fresh B1/B4 raw witnesses, then replay independently before broadening.


### Remote short B4 replay narrows remaining failure to decode; local transfer pending

2026-10-27: First four independent B1reference oracles complete and physical B4 GPU capture completes all8IDs plus complete reset/prefill/final/final-reset states and finite raw hidden+norm vectors. Original model job exits1 on embedded comparison harness NameError (free variable reference), after all raw witnesses were saved; cleanup/restoration completes. Terminal independent CPython archive replay recovers exact original comparisons without GPU rerun or failed-job promotion: all4rows reset/prefill/finalreset exactly match B1, including all128statecomponents and raw vectors; final decode state fails all4rows, generated ID sequences exact only for first2rows. Canonical B4T32 routes selected48GDN/48reconstructedZ/32KV/16Q/16O, new0onB1/T4. This is evidence for short-prefill matching and remaining decode/token-selection divergence, not proof of its cause or model/performance admission.

Raw86,216,211B ZIP and remote proof remain immutable in fixed study/results. Full original archive transfer is live; local paired replay/proof SHA and final retention remain PENDING. Optional dedup transport preparation stopped before producing files because corpus ZIP lacks a local mirror; preserve draft and use full transfer. Next await existing transfer ACK, obtain closure receipt/assets, exact manifest/SHA plus local same-reader replay, then inspect first per-token/first-layer decode differences and repair harness separately. Keep original failure, no tolerance relaxation, no speed/quality promotion or chart update. Existing measured source60/DLL841 serving results stay separate; full numerical/quality and strongest matched baseline2x objective remain open.


### Actual short B4 versus four fresh B1 result and paired raw replay

2026-10-27: Model job b4-b1-short-admission-attempt1 exits1; status actual_B4_short_vs_four_B1_failure_replayed. Four-B1 source60/DLL841 reference then source61/DLLacb9ff9e12efb808f498d2bfc049d5a98c22c3ae0bddbc08b9675b8d5acfb5b6 physicalB4 uses same exact short mixed41/104/93/93text-image prompt IDs and8outputs, capacity4096/T32/T4; preserve heterogeneous tail behavior and all complete128component state/physical/logical hashes and finite hidden+norm rawbytes. Actual statuses {"reference": "four_B1_short_oracles_collected", "candidate": "failed"}. Error RuntimeError('candidate numerical admission failed; preserve raw witnesses and errors'). Original raw archive 19056990075e46080af4c417f623f5e578dcc1c90a4785ac957670df4dba51f9 (86216211B/1595members); paired proof c707c7a28bcae3b6238ffee82828b1cd75cf9c03ce0f9622825486d5218f37c8. All source/runtime/SDK/checkpoints restored and private nonce engine artifacts retired. Raw reports, logs, resource samples, source/reader hashes and any numerical failure retained; raw F32 vectors preserved in immutable ZIP on both PCs.

This result is restricted to the first short case. It cannot qualify full4K text/image, repeat/permutation/reset/broader quality, MTP orHTTP performance. Any mismatch remains failed without tolerance relaxation. Original fullsix/805/allIDs/state/vector gates remain required. No plotted metrics or new speed claim. Next action follows actual raw differences on failure, or original full numerical admission on pass; strongest repeated tuned/MTP references, matched INT4/4B and >=2xdecode+throughput/halfTTFT remain open.


### Local archive transfer and paired failure replay closed; first token IDs all exact

2026-10-27: Full original86,216,211B/1595member archive transfer completes. Independently validate exact closure receipt/manifest and original archive CRC/all1594member hashes; CPython local original-reader replay matches remote proof byte-for-byte (c707c7a28bcae3b6238ffee82828b1cd75cf9c03ce0f9622825486d5218f37c8). All jobs including postprocess terminal; fresh remote idle check passes. Prior interim local-transfer-pending state is now superseded by this closure, not rewritten. Original GPU job remains FAILED; embedded comparison NameError and all subsequent raw numeric mismatches retained. No tolerance or original-state/ID predicate changed.

Additional original raw ID inspection: all4firsttoken IDs exact, same saved prefill normalized bytes exact. Text and first image rows match all8IDs; red/green first differ at zero-based index5 (sixth sampled token). Complete final decode state/raw vectors differ for all4rows, while complete prefill/reset/finalreset match all4. This does not prove logit equality, first-state-difference step, cause, semantic equivalence or harmlessness. Existing native FP8 cuBLASLt function selects ordinary heuristics using actual M; B1T4 flattensM4 and B4T4 M16. That source fact is a candidate explanation only. Next isolate same-input first-decode layer0 normalization/projections/state at actualM4/M16 and head outputs as needed, before selecting any new mathematical policy. Preserve physical batching, original fullscope gates and failedB4/B8 controls; never substitute serialized B1 as a throughput win. Full4K/MTP/805/HTTP/performance qualification and overall2x goal remain open; charts unchanged.


### Existing GDN layer0 decode projection diagnostic prepared; no new native math

2026-10-27: After paired short failure replay and fresh idle check, prepare actual existing GDNlayer0 retained-projection path with original shared5compressed matrices (2NVFP4/3FP8), source60/DLL841 B1T4/M4 versus source61/newDLL B4T4/M16. First5consumed input token vectors are from original agreed samples at indices0..4, before red/green sixth-token divergence. Zero conv/recurrent states for this component; QKV/Z/beta/a payloads are before the state-dependent core. Separate normalization graph is NOT a direct witness of normalized tensor inside the fused layer. Need controlled direct SDK input test if attribution remains ambiguous. Generic checkpoint_projection operation was inspected and remains INT4-only; do not bypass its original guard. No new native build, plugin, kernel, algorithm, quantizer or wire change.

Capture full finite raw input/normalization/QKV/Z/beta/a arrays twice plus requested actual cuBLASLt logs. Four CPUcontrols cover whole-array/offset geometry, active/inactive/signedzero errors, NaN/Inf/truncation/swapped rows, token and input mutation and repeat instability. CPU tests pass; actual component GPU work UNRUN at preparation. Exact owner derivation retains original snapshot/restore/SDK/checkpoint/cache ownership checks. Original failed short job staysFAILED; all original complete-state/IDs/raw vectors/full4K/MTP/805/HTTP gates remain unchanged. Source/preparation 3a397d3376c80b559975083607b14dcc8da148fb11338af84e1000446ea628b0; deployment 5600e0f7277be29f7464e1cfb9a17d6d2c2c5a6f1195acbbbf207b70477c57bf. New archive keeps new methods/raw arrays plus linked parent proof/sourcebundle pins, avoiding duplicate unchanged large corpus bodies. No new benchmark/chart/2x claim; strongest repeated tuned/MTP refs and matchedINT4/4B remain open.


### decode-projection-diagnostic-attempt1 original decode projection outcome independently closed

2026-10-27: Job exits1; status actual_decode_projection_diagnostic_capture_failure_replayed; phase errors {"reference": "RuntimeError('Host embedding requires an independent native NVFP4 head')"}. Raw archive 047f1980f40e7ec4281a28b67c0193ecdc93971e34c392a6e51b169e42a06332 (1131138B/28members), paired proof 312bc9e31c8d1e83de1f844e1b7d43b0172e877c4ca0438549f19be7da219425, identical independent remote/local CPU replay of manifest/source/method/parent/raw arrays or capture failure. All source/runtime/SDK/checkpoints restored. Capture result No matrices captured: host embedding required native NVFP4 head first.. First failed model short gate remainsFAILED; this diagnostic cannot qualify fullmodel, quality or timing. Raw arrays/logs/SDK traces retained in immutable ZIP on both PCs, text reports/methods/hash proof retained on branch. No new native math or2x claim.


### Required native-head-before-host-embedding setup corrected for diagnostic attempt2

2026-10-27: Original attempt1 ends before projection capture on strict IO guard: host embedding requires independent native NVFP4 head. Preserve failure/raw archive and byte-identical remote/local proof312bc9e31c8d1e83de1f844e1b7d43b0172e877c4ca0438549f19be7da219425; source/runtime/SDK/checkpoints restored. Separate versioned attempt2 probe adds only io.enable_nvfp4_head(True) before host embedding, matching actual full model IO setup. Head is unused in projection capture. Original probe/owner/launcher remain byte unchanged. Source60/61 bundles, DLLs, quantizers/scales, algorithms, all states/IDs/finite vector/full4K/MTP/quality/performance gates unchanged. Same first5agreed token inputs and four CPUcontrols pass. Preparation cea6ad94ada6d4075f09ec9c7ddb5e3b8d571b9ac70b69d7d73a43463d8d50e2; deployment 60634cdeb59c75f02ed44cf28efdfe83a42f192fdd14c0ab48282ff94a6adf75.

Attempt2 actualGPU arrays/SDK logs UNRUN at preparation. Keep standalone normalization distinct from fused internal norm; zero-state retained projections characterize same-input M4/M16 behavior without a full-model cause/admission claim. Original short model decode failure remainsFAILED; no new native build or speed/2x claim.


### decode-projection-diagnostic-attempt2 original decode projection outcome independently closed

2026-10-27: Job exits1; status actual_decode_projection_diagnostic_capture_failure_replayed; phase errors {"reference": "AttributeError(\"'TensorCompat' object has no attribute 'fusion'\")"}. Raw archive 26a554450c08cd18fd142813532a6d1c8163c4bdae6e8f160b2afab18842beaf (1144335B/33members), paired proof e0bb0d3c2fd3b922d497ba435e33e64d1e3c82c75eda89163e66516ef0c098ee, identical independent remote/local CPU replay of manifest/source/method/parent/raw arrays or capture failure. All source/runtime/SDK/checkpoints restored. Capture result No arrays captured; exact phase errors above preserved.. First failed model short gate remainsFAILED; this diagnostic cannot qualify fullmodel, quality or timing. Raw arrays/logs/SDK traces retained in immutable ZIP on both PCs, text reports/methods/hash proof retained on branch. No new native math or2x claim.


### Supported XLang fusion decorators used for component attempt3

2026-10-27: Independent remote/local replay preserves attempt2 capture failure: TensorCompat has no runtime fusion attribute. Correct versioned harness uses existing supported @T.fusion decorator syntax, as the working repository probes do. Preparation AST requires both fusion calls to occur only in decorator positions. Single GDN layer omits ignored position/slot inputs, passes None for those unused linear-layer arguments; actual hidden/history/state/active/weight shapes, primitive operators, retained payload offsets and all array gates unchanged. Keep required nativeNVFP4head-before-hostembedding setup from attempt2. Original first2failed probes/owners/launchers remain byte unchanged, all raw failures/proofs retained.

Same four CPU diagnostic corruption controls pass; source/preparation 890c1b22bb01feead1f7123f61bc564cb62053f2240e31c3dab5851743c45e17, deployment 9022449e023b0c18e686172d93d7a3ead67d5acec90f8b948267479f4571c9b8. Actual component GPU arrays UNRUN at preparation. No new native build, DLL/sourcebundle, quantizer, algorithm or mathematical guard change. Zero-state component and separate norm cannot alone prove full-model cause or qualify model/quality/performance. Original short B4decode failure/fullscope gates, matched strongest references, INT4/4B and2xgoal remain open.


### decode-projection-diagnostic-attempt3 original decode projection outcome independently closed

2026-10-27: Job exits0; status actual_decode_projection_diagnostic_replayed; phase errors {"reference": null, "candidate": null}. Raw archive 0fd7721f6e6f012dfe26caf8432568acd05a62a4d8c80aba91350ceba5d703c1 (6133765B/320members), paired proof bbc3fdde79d4194948b738c3261b8679a20271b9283a47bf03f67349d2bb1c40, identical independent remote/local CPU replay of manifest/source/method/parent/raw arrays or capture failure. All source/runtime/SDK/checkpoints restored. Capture result {"status": "actual_first_five_decode_projection_diagnostic_replayed", "all_input_hidden_bytes_exact": true, "all_normalized_bytes_exact": true, "all_projection_bytes_exact": false, "repeats_within_each_geometry_exact": true, "full_model_or_performance_qualified": false}. First failed model short gate remainsFAILED; this diagnostic cannot qualify fullmodel, quality or timing. Raw arrays/logs/SDK traces retained in immutable ZIP on both PCs, text reports/methods/hash proof retained on branch. No new native math or2x claim.


### Actual first-layer FP8 decode discrepancy and SDK profile change observed

2026-10-27: Versioned attempt3 completes actual one-layer GDN shared-weight capture at B1T4/M4 and B4T4/M16. Same first5agreed consumed token inputs, two identical repeats pergeometry, full input and separate-normalization bytes exact, all padding exact, beta/A exact. Across20active rows (one repeat), QKV differs in16/204800F32words and Z in6/122880; first decode input already shows QKV mismatch. These are retained BF16 projection outputs restored as F32; this diagnostic does not prove the separate normalization is the fused internal tensor or the complete-model cause. Full zero-state component arrays and SDKlogs retained; paired original raw proof bbc3fdde79d4194948b738c3261b8679a20271b9283a47bf03f67349d2bb1c40.

Actual vendor traces report FP8 QKV/Z algorithm35 at M4 and67 at M16; output projection stays35. Parse actual printed tile/stages/splitK/reduction/customOption/swizzle only when present, never infer omitted attributes. Same compute32F/BF16 output and4MiB workspace. Algorithm correlation is not yet causal proof. Next controlled direct SDK experiment: real normalized arrays and actual checkpoint QKV/Z matrices, four independentM4 outputs vs ordinaryM16 vs M4-selected SDKalgorithm AlgoCheck at actualM16. Capture FP8activation bytes, all7queried algorithm attributes/checkworkspace, full BF16 output bytes and repeats; validate against actual Garnet outputs and preserve unsupported/negative controls. No new Garnet production/native policy until actual SDK and original complete model/full4K/MTP/805/HTTP gates pass. Source/runtime/SDK/checkpoints restored, no new serving benchmark/chart or2x claim; strongest repeated tuned/MTP references and matchedINT4/4B remain open.


### Actual M4/M16 FP8 SDK intervention prepared

2026-10-27: Eight CPU replay tests pass (four new complete-output, baseline, quantization, repeat and actual-shape/config/AlgoCheck controls plus four unchanged independent cast/FP8 controls). One initial synthetic test fixture changed only one repeat and correctly hit repeat-stability guard before intended changed-word observation; fixture corrected to corrupt both repeats, original guard retained. Actual SDK/GPU test unrun at preparation. Pinned deployment f24f5c21cf3b748ed9a1b8b7d5514190aeae6402bdcbe1cd89b302a82607cffe, preparation 61a4cb795ca6cea241baa2d1e66ec62da1d9a8f33a224b842ddcb5ca3ef42ee3. Seven new owned methods,1548-source baseline hashes and all34current production CUDAobjects pinned.

Direct SDK compares original real first5decode inputs and indexed checkpoint QKV/Z FP8 weights/scalars at four independentM4, defaultM16 and actualM16 using selectedM4algorithm. Require actualM16 AlgoCheck within original4MiB, query all7attributes, full finite input/quantized/output arrays, exact repeats, savedGarnet M4/M16 baseline reproduction and independent remote/local replay. Unsupported or numerical failures retained with no fallback. The4compiled production files are absent from both immutable60/61 overlays, so the exact baseline files are shared by both prior DLLs: no source overlay, GarnetDLL swap or fullmodel run required. Private SDK compile only on5090; all source/runtime/SDK/checkpoint/34productionobjects must remain unchanged.80MiB actualweightfixture stays remote in fixed studyfolder; origins/hashes/smallarrays archived. Original B4shortdecode and full4K/MTP/805/HTTP gates remain open; no new benchmark, chart value or2xclaim.


### Actual M4/M16 FP8 SDK causal projection control passed

2026-10-27: Exclusive remote private sm120 SDK compile and capture exit0. Raw archive 2c4d684fb182dc3597e87ff72f09da3aa954e74e6cb1c1537d2d9ecd4f78c657 (6233486B/154members), independent remote/local identical full-array proof a1f20382c9c53b6377c3764c98fcee0150e9ae627d490a886e4ce5ec1ae25b92. All1548source pins, sixruntime files,23SDK/environment inputs, checkpoint config/index and all34production CUDAobjects unchanged. No GarnetDLL loaded/swapped, no fullmodel/quality/serving benchmark.80MiB real indexed FP8weights stay in fixed remotefixture; scalar/tensor origins/headerSHA/offsets/weightsSHA retained.

Actual separate-normalization inputs exactly reproduce BOTH fourindependentM4 and defaultM16 saved Garnet outputs for QKV/Z, five real inputsteps and two repeats (20cases). Independent CPU BF16 cast and FP8 quantization exact; fourM4 andM16 packedactivations identical; complete finite output/repeat gates pass. DefaultM16 versusfourM4 activeword differences QKV[5,2,3,6,0], Z[0,1,1,2,2], matching prior16+6. ActualM16 with actualM4-selected algorithm eliminates ALL differences across every full BF16 output word, including padding, in both repeats. Actual20AlgoChecks state/status0 within original4MiB; selected7config attrs qkv/z[35,13,17,3,2,0,1] (id,tile,stages,split,reduction,swizzle,custom). BothdefaultM16 use67; all120calls retained. This establishes a causal algorithm effect for the reproduced component projection discrepancy. Internal fusednormalization was not directlycaptured and complete-model B4correctness/cause remains UNPROVEN.

Next concrete backend candidate: defaultoff physicalB4T4/M16 canonicalQKV/Z only, separate strictprofile/plugin/wire/context, original4MiB and actualM4seed/AlgoCheck; no genericFP8/wire/default/B1 changes or unsupportedfallback. Nativebuild/profile/factory/wire/initialize tests and originalcomplete B1/B4 short/tail/full4Ktext-image/packedstate/finitevectors/IDs/reset/repeats plus MTP/805/HTTP gates required beforetiming. No chart values updated or2xclaim; matched strongest repeatedtuned/MTP references, INT4/4B and fullobjective stillopen.


### Separate defaultoff B4T4/M16 GDN FP8 backend source prepared

2026-10-27:67-source candidate over61, only loweringsource modified plus6new profile/context/trace/plugin files. Exact48GDNlayer logicalQKV/Z weights, physicalB4T4/M16 only, N10240/6144 K5120. Defaultoff GARNET_FP8_GDN_BATCH4_DECODE_CANONICAL, own20B wire0x47313634/type, two geometry-specific device caches. LiveM4 SDKselected7attrs[35,13,17,3,2,0,1] mustmatchpaired SDKproof; actualM4/M16 AlgoChecks within4MiB beforecachedexecution, no opaqueimports/fallback. Old61nonlowering bytes/CUDA/quantizers/genericFP8/wires unchanged.

11localCPU controls pass:3immutable67source composition/each7bodycorruption/actualSDKlineage,8unchanged independentconversion/fullwordcontrols. Nativecontracts unrun at preparation:437actualC++profilecases96eligiblelogicalweights,402bothgeometry CPU SDKfault/trace/cache/runtimeguardcases, actualTRTpluginwire/scales/descriptors/formats/crosswire/clone/initialize/exception contracts withGPUstubs unused. All nativecompile/exe work remoteonly; privateCPUjob doesnotinstallsource/DLL ormutateproductionbuild. Sourcebundle 9a8ff087755fd4847d4da1f2cdcaef1bdb81c67f5c2a28a3f87d3293298b9208. First sourcegenerator rejected CRLF-only replacement against LF parent beforewritingcompletebody; originaldraft retained, finalgenerator normalizesreplacementnewlines to pinnedparent and independentlyverifies already-produced identicalheaders. No originalguards weakened.

Allsource/runtime/SDK/checkpoint/CMakeCache/all34productionobjects pinned. Candidate remains SDK/nativefullmodel/quality/MTP/805/HTTP/performance UNQUALIFIED; original B4shortdecode and strictercache62objectidentity failure stayfailed. Newnativefullbuild withalloriginalcontrols and actualprojection/fullB1/B4 short/tail/full4Ktext-image/fullpackedstate/finitevectors/IDs/reset/repeats gates requiredbeforetiming. Strongest repeatedtuned/MTPreference, matchedINT4/4B and2xfullobjective remainopen.


### Actual new B4T4/M16 native CPU contracts independently passed

2026-10-27: RemoteCPUjob exits0. Raw archive 2aa2b6d50e521ae9af89a6ce935bd8685a5a8ca6e5cf65d3d4e6f57c7dd86aeb (2119586B/864members), paired remote/local identical replayproof 54cd7d572b79e81376f5105b0787eb63023851c77416af5d12031d3c50d0d541.11Pythoncontrols pass. ActualMSVCprofile437cases:96eligible logicalQKV/Zweights,341rejected. Actualnewfactory402CPUfault/trace cases across93modes andbothN10240/6144 geometries: all7configattrs/bytes,24APIstatus+exceptionfaults, M4/M16state/workspace failures, unsupportedversion/SM/CUDART, exactcounts, no fallback, samegeometrycache reuse, crossgeometrycache separation, runtimeinvalidshape/pointer/scale/workspace/device guards and traceobservationparity pass. Stubmatmul forwarding testedonlyonCPU, notSDKmath.

ActualTRTplugin checks pass:82badwires,44badconstructors/scales/products,92baddescriptors,18badtypes,6legacycrosswire rejections,14initializecalls/10CPUfactorycalls includingstandard/unknownexceptions/cache/clone/terminate/reinitialize.0GPUoperators, actualTRT101601loadedpathpinned. All1570source/runtime/SDK/checkpoint/CMakeCache/all34productionobjects unchanged; no productionoverlay/DLLinstalled. PrioractualSDK mathcontrol remainsseparateproof, newcompiledSDKfactory andwholemodel stillUNQUALIFIED. Allraw402case stdout/stderr/privateexes retainedZIPonbothPCs; branchkeepsmethods/reports/ledgers/proofs.

Prepared exactguarded67nativebuild/replay derivativeoforiginal61owner: preservealloriginal Q/O profile/factory182/wire/initialize630+720/selector20/tokenizer955tests,74current(+3source)+86historical Pythoncontrols on exactparent50, original3CUDAobjectwhitelist andfullsource/DLL/runtime/SDK/checkpoint/CMake restoration. New402CPU/437profile/wire proof andactualSDKcontrol requiredparents. Fresh readonly67target baselineinventory next; nativebuild UNRUN atthisentry. Originalfailed B4shortdecode/strictercache62identity andfull4K/MTP/805/HTTPgates stayopen; no benchmark/chart/2xclaim.


### Actual67source native build prepared after fresh inventory

2026-10-27: Fresh remote readonly inventory verifies restored1374baseline,67candidate targets(14present/53absent),6runtime/23SDKinputs,2checkpointmetadata files,CMakeCache,all34current61build CUDAobjects andclosed402CPU/SDK/native61/failedshort parents.564.34GiBfree onstudy volume atinventory. Native67sourcebundle 9a8ff087755fd4847d4da1f2cdcaef1bdb81c67f5c2a28a3f87d3293298b9208, preparation 2e6651c068238516098fae868dd8899de7043cfe63791bd1feb6169bbd5ae3f1, deployment a05063e6d13c9ad0a88839acc969d901aa76a8f4ebc746e93cb16bcbbaeb28f2;201explicitownedfiles, no productionoverlay/DLLinstallation atpreparation. Frozen tokenizer oracle retained.

Originalnativebuild/replay guards preserved viaexactrecordedderivation:74current+86historicalPythoncontrols onparent50;originalQ/O278profilecases,182factoryfaults,Q/Owire/630+720initializecalls,20selector/955tokenizer smoke; original3CUDAobjectwhitelist andallsource/runtime/SDK/checkpoint/CMake/DLLrestoration. NewactualSDKcontrol andpaired437/402CPUcontracts requiredparents. Nativebuild UNRUN atthisentry; no mathematical/model/quality/timingqualification or2xclaim. Originalcache62strictobjectfailure andB4shortdecodefailure notcleared; fullB1/B4 short/tail/full4Ktext-image/state/vector/ID/reset/MTP/805/HTTPgates andstrongestmatchedrefs/INT4/4B remainopen.


### Actual67source native build independently passed; model admission pending

2026-10-27: Exclusive Release/MT/sm120 productionnativebuild exits0. Candidate build/bin/garnet-native-b4-decode-m16-attempt1.dll SHA 67711e36fc28da954925b02f09588e8c9a8b7667de0f4f2ceb34a2162e026e08. Raw archive 10ed09c87a2f2166b346352fe0cdb07fceedd06084e2c215dcacd039bdeba934 (9016488B/758members), identical independent remote/local proof 9e3b418c140edde68ae532f62c6e645f283b4ec92b97ac04304df2efbf9b7275. Alloriginalnativegates retained andpassed:74current+86historicalPythoncontrols onexactparent50, Q/O278profiles,182factoryfaults, Q/Owire/crosswire,630+720initializecalls,20selector/955frozen tokenizer smoke. Paired437profile/402newfactory/newpluginCPUparent andactualSDKcontrol independentlylinked. Native67source9a8ff... andnewDLL remainwholemodel/quality/MTP/805/HTTP/performance UNQUALIFIED.

All67targets restored(14baselinepresent/53absent), activegarnet.dll restoredto1a1d..., allbaseline1374source/199methods/runtime/SDK/checkpoint/CMake hashes verified.34CUDAobject hashes recorded; changednames ["build/garnet/CMakeFiles/garnet.dir/src/cuda/qwen38_bf16_kv_low_batch_overlay.cu.obj", "build/garnet/CMakeFiles/garnet.dir/src/cuda/qwen38_bf16_prefill128_attention.cu.obj", "build/garnet/CMakeFiles/garnet.dir/src/cuda/qwen38_bf16_prefill128_commit.cu.obj"] arewithinUNCHANGEDoriginal3-objectwhitelist. This isnotall34identity anddoesnotclearoldstrictercache62failure orestablishall3changesmetadata-only. Saved67DLL remainsseparate, no activeproductionpromotion. Allrawnative/per-case logs/source snapshots/privatebinaries retainedZIPonbothPCs; branchkeepssource/ledgers/reports/proofs. No newservingbenchmark/chartvalues/2xclaim.

Next actualGPUgate: source60/DLL841 fourindependentB1 oracles vs67DLL physicalB4, same41/104/93/93token text+image short/tail inputs and8IDs, full128statecomponents perrow/all106823680physical and173932544logical words plus finite5120hidden+norm bytes, reset/prefill/final/finalreset and8IDs exactly. Preserveoriginalfailedshort capture/proof. Correctits embedded-comprehension lambda scope failure by moving identical numerical comparisons to the CPython exclusiveowner after rawcapture; independently replay everyoriginalexactgate, no tolerance/sampling/waiver. Enable newdefaultoff B4decode flag onlycandidate andrecordactualGDNM16factory/SDKselection witnesses. Originalcache/source/DLL/restoration/ownedSSDcache guards unchanged. FullB1/B4tail/full4Ktext-image/reset/repeats/MTP/805/HTTP/strongestmatchedreferences/INT4/4B andoverall2xobjective stillopen.


### Source67/DLL677 B4-vs-four-B1 short mixed model gate prepared

2026-10-27: Original failedshort/source61capture andproof retainedunchanged. Versioned newprobe/launcher/owner uses qualifiedsource60/DLL841 B1oracles vsactualnative67/DLL677 B4candidate, same41/104/93/93token text+image inputs, capacity4096/prefill32/decode4/8generatedIDs. Source67newGDNdecodepolicy1 andactualfactory/inittrace onlycandidate; referenceflag0. Alloriginal pool/head/vision/sixFP8route/full128statecomponents/perrowphysical106823680/logical173932544words/fullfinite5120hidden+norm/reset/prefill/final/finalreset/IDs gatesretained. Embeddedprobe capturesrawonly; exclusiveCPythonownerenforces identicalcompare_capture/require_ids afterbothGPUcaptures andbeforepass, resolvingoldembeddedcomprehensionlambda freereference NameError withoutnumericalwaiver.

7localCPUcontrols pass (4original unchanged plus3newallphase/ID/geometry/count/live7config/workspace/route faults). Newactualroute requires>=48newplugininitializations foreachQKV/Zgeometry, liveM4seed7attrs[35,13,17,3,2,0,1], exact24APIsequence/status, M4/M16AlgoCheckswithinoriginal4MiB, opaquealgorithm preservedfromliveSDK, nofallback. All1534baseline pins and34currentnative67CUDAobjects/CMake auditedbefore/after, source67/DLL/runtime/SDK/checkpointrestoration andownedprivateSSDretirementunchanged. Boundedarchive keeps1187consumed/sourceinputs+rawcaptures/sourcebundles/parentproofs; oldinputarchive/unrelatedbaselinecorpus linkedbySHA ratherthanrecopied.

Preparation fa951587580798745a929875f9e9cfc147cdd900784e673a0442403a6acf206f, deployment 4dc03a01a8aec10f430563e62a71daeade3bdf414edba198a553e387517296d1. GPUgate UNRUN atthisentry. Numericalorroutefailure remainsfailed evenifbothcapturechildrenexit0; observationwrittenbeforestrictadmission andindependentremote/localreplayrequired. Shortpass cannotqualifyfull4K/MTP/805/HTTP/performance. Originalfailedshort/strictercache62 gate andfullstrongestrefs/INT4/4B/2xobjective remainopen; no newbenchmark/chartclaim.


### Source67/DLL677 original complete B4/B1 short gate independently closed

2026-10-27: Job exits1, status actual_B4_M16_short_vs_four_B1_failure_replayed, strictshortadmission False. Rawarchive 75c7fe213bd3e2108827d594e7f36e2f90fd2ed82ac1f130a4a5236f456f65e3 (13483512B/1261members), paired identicalremote/localproof d96dfc99e40eefa925f10ec82e68af80a0b3b6c82e526f3b29ebd9be72cfed55. Phase statuses {"reference": "four_B1_short_oracles_collected", "candidate": "B4_short_mixed_tail_raw_capture_needs_CPython_admission"}; errors {"reference": null, "candidate": null}; ownererror AssertionError('B4 complete state/raw vector mismatch against four independent B1 oracles').7CPUguards pass; embeddedcomprehensionfailure removed, unchangedexactnumericalgates enforcedbyCPythonowner afterrawcapture. Capture successcannotpromotefailedmath/routegate.

Actual complete-state/finitevector observations {"reset_state": {"all_exact": true, "rows": [{"physical_row": 0, "different_component_indices": [], "finite_raw_vector_bytes_exact": {"hidden": true, "normalized": true}}, {"physical_row": 1, "different_component_indices": [], "finite_raw_vector_bytes_exact": {"hidden": true, "normalized": true}}, {"physical_row": 2, "different_component_indices": [], "finite_raw_vector_bytes_exact": {"hidden": true, "normalized": true}}, {"physical_row": 3, "different_component_indices": [], "finite_raw_vector_bytes_exact": {"hidden": true, "normalized": true}}]}, "prefill_state": {"all_exact": true, "rows": [{"physical_row": 0, "different_component_indices": [], "finite_raw_vector_bytes_exact": {"hidden": true, "normalized": true}}, {"physical_row": 1, "different_component_indices": [], "finite_raw_vector_bytes_exact": {"hidden": true, "normalized": true}}, {"physical_row": 2, "different_component_indices": [], "finite_raw_vector_bytes_exact": {"hidden": true, "normalized": true}}, {"physical_row": 3, "different_component_indices": [], "finite_raw_vector_bytes_exact": {"hidden": true, "normalized": true}}]}, "final_state": {"all_exact": false, "rows": [{"physical_row": 0, "different_component_indices": [7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127], "finite_raw_vector_bytes_exact": {"hidden": false, "normalized": false}}, {"physical_row": 1, "different_component_indices": [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127], "finite_raw_vector_bytes_exact": {"hidden": false, "normalized": false}}, {"physical_row": 2, "different_component_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127], "finite_raw_vector_bytes_exact": {"hidden": false, "normalized": false}}, {"physical_row": 3, "different_component_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127], "finite_raw_vector_bytes_exact": {"hidden": false, "normalized": false}}]}, "final_reset": {"all_exact": true, "rows": [{"physical_row": 0, "different_component_indices": [], "finite_raw_vector_bytes_exact": {"hidden": true, "normalized": true}}, {"physical_row": 1, "different_component_indices": [], "finite_raw_vector_bytes_exact": {"hidden": true, "normalized": true}}, {"physical_row": 2, "different_component_indices": [], "finite_raw_vector_bytes_exact": {"hidden": true, "normalized": true}}, {"physical_row": 3, "different_component_indices": [], "finite_raw_vector_bytes_exact": {"hidden": true, "normalized": true}}]}}. Eight-ID equalitybyrow [true, true, false, false]. ActualGDNM16route {"status": "actual_B4T4_GDN_M16_route_and_live_SDK_selection_replayed", "created_contexts": 4, "cached_factory_calls": 188, "initializations_by_output_rows": {"10240": 96, "6144": 96}, "created": [{"configuration": [35, 13, 17, 3, 2, 0, 1], "checks": [{"id": 1, "seq": 40, "event": "check", "name": "M4", "a": 0, "b": 2623519, "c": 0, "opaque": ""}, {"id": 1, "seq": 42, "event": "check", "name": "M16", "a": 0, "b": 2623519, "c": 0, "opaque": ""}], "opaque_live_SDK_selection_preserved": true}, {"configuration": [35, 13, 17, 3, 2, 0, 1], "checks": [{"id": 2, "seq": 41, "event": "check", "name": "M4", "a": 0, "b": 1574175, "c": 0, "opaque": ""}, {"id": 2, "seq": 43, "event": "check", "name": "M16", "a": 0, "b": 1574175, "c": 0, "opaque": ""}], "opaque_live_SDK_selection_preserved": true}, {"configuration": [35, 13, 17, 3, 2, 0, 1], "checks": [{"id": 97, "seq": 40, "event": "check", "name": "M4", "a": 0, "b": 2623519, "c": 0, "opaque": ""}, {"id": 97, "seq": 42, "event": "check", "name": "M16", "a": 0, "b": 2623519, "c": 0, "opaque": ""}], "opaque_live_SDK_selection_preserved": true}, {"configuration": [35, 13, 17, 3, 2, 0, 1], "checks": [{"id": 98, "seq": 41, "event": "check", "name": "M4", "a": 0, "b": 1574175, "c": 0, "opaque": ""}, {"id": 98, "seq": 43, "event": "check", "name": "M16", "a": 0, "b": 1574175, "c": 0, "opaque": ""}], "opaque_live_SDK_selection_preserved": true}], "all_seven_attributes_checked": true, "original_4MiB_workspace_preserved": true}; routeerror None. Source67/DLL677vsqualified60/DLL841, identicaltext/image41/104/93/93inputs, capacity4096/prefill32/decode4/8IDs; full128statecomponents/allperrowphysical106823680/logical173932544words and finite5120hidden+norm bytes atall4phases, no tolerance/sampling.

All1534baseline sourcepins/runtime/SDK/checkpointmetadata/CMake/all34native67CUDAobjects restored;activeDLLbaseline restored, ownednonceenginecaches retiredonlyafteridle. ImmutableboundedrawZIP onbothPCs retainsactualvectors/images/traces/sources; oldinputdataset/corpus linkedSHA, notrecopied. Originalfailedshort/source61 andstrictercache62objectfailure notpromoted. Thisresult isshortgateonly: fullB1/B4tail/full4Ktext-image/reset/repeats/MTP/805/HTTP/strongestmatchedrefs/INT4/4B/performance/overall2xremainopen. No newbenchmark/chartvalueclaim.

Before/after analysis independently confirms16complete B1reference snapshots (4rows x4phases) andallrawhidden/norm bytes unchanged. Matching-token row0(s3-text) changedstatecomponents128->115, firstpersistentdifference layer0conv->layer3fullattentionV(index7). Matching-token row1(s3-image)127->105, firstdifference layer0recurrent->layer11fullattentionV(index23). Red/green stillfirstdifferentgeneratedID index5(sixthtoken), first5consumeddecodeinputs match; laterlayer0state differences onthese rows cannotlocalizea same-input fault becauseconditioningtokensdiverged. Fullmatching-token boundary0/1/2/3/4/10/11/12 plusactualFA3normalization are next readonlyobservations; mustprove same-profile primaryfullstate/finitevector/ID endpointparity beforeusingthem. Existing B1T32/T128 boundary/witnessguardsstayunchanged; a newseparateB1/B4T4/cap4096 observer required, no originalguardwaiver. Earliestchangedpersistentcomponent isnotcausalproof; newactualFA3input/SDKVcontrol neededbeforeexpandingcanonicalmatrixpolicy. BothshortmodeljobsstayFAILED; no2xclaim.


### Readonly decode boundary diagnostic prepared; original batch4 gate still failed

2026-10-27: Source67/DLL677 complete shortgate failed and pairedproof d96dfc99e40eefa925f10ec82e68af80a0b3b6c82e526f3b29ebd9be72cfed55 remains authoritative. A separate readonly observer appends full T4 layer0/1/2/3/4/10/11/12 hidden boundaries for B1/B4 only. Original layer observer B1T32/T128 guard and forwardmath/FP32barriers byteidentical; prefill32 remains ordinary. All7decodecalls captured, including calls after red/green conditioning diverges. Full active/padding words kept, not sampled. Separate FA3/11 RMSNorm runs twice with unchanged input hashes. Actual Vcache active slot reads four heads x256BF16 perrow, twice, fullfinite replay including low packed halves.

BEFORE interpretation, require same-profile exact reproduction of all4phase reset/prefill/final/reset full128statecomponents and physical/logicalrowhashes, fullfinite5120hidden+norm bytes and8IDs against linked original source60/DLL841 and source67/DLL677 failedshort raw captures. Newdiagnostic cannot promote original failed model, 4K/MTP/805/HTTP orperformance. Originalcrossprofilegate observation recomputed without tolerance; inputs with divergentconditioning explicitly marked. Earliest observation alone cannot establish causality.

17CPUtests pass (original7 preserved +10new prefix/binding/derivation/offset/primary/lateword/signedzero/nonfinite/repeat/inputreadonly/row/slot/conditioning controls). 1547baselinepins, all34CUDAobjects/CMake, runtime/SDK/checkpoint guards preserved. Fixedstudyroot/cache/exclusiveowner; no native/CUDA rebuild. Preparation 559f0954c878c2566a01cdf29b483af366d7d3a519d961c7fb9a7b1c464f6840, deployment 47eff5252424521ed606ec53fe7d07c5ac2e6a6b51695ffc4ccaa839bd599125. GPUdiagnostic UNRUN atthisentry; independent remote/local archive replay required. Existing 4K charts unchanged, strongest tuned/MTP refs and INT4/4B comparisons and overall2x objective remain open.


### Readonly decode boundary diagnostic independently replayed; model failure preserved

2026-10-27: decode-boundaries-attempt1 exits0, diagnosticpass only. Paired identical remote/localproof 682f54de95a19b9ca6f2315059eddb05703129ce13a62868eda1322b500e91f0, rawarchive 95a52fcceceeeee6f74b2c9747c9fc363280821ba2d38d0805f3e7b3f1fa7dc1 (25360400B/1920members). Both original source60/DLL841 and source67/DLL677 same-profile primary endpoints exactly reproduce prior failedshortjob: {"reference": {"status": "same_profile_complete_primary_state_vectors_and_8IDs_exact", "mode": "reference", "physical_batch": 1, "complete_snapshots": 16, "components_per_snapshot_row": 128, "physical_words_per_row": 106823680, "logical_words_per_row": 173932544, "numerical_tolerance_applied": false, "model_or_performance_admitted": false}, "candidate": {"status": "same_profile_complete_primary_state_vectors_and_8IDs_exact", "mode": "candidate", "physical_batch": 4, "complete_snapshots": 4, "components_per_snapshot_row": 128, "physical_words_per_row": 106823680, "logical_words_per_row": 173932544, "numerical_tolerance_applied": false, "model_or_performance_admitted": false}}. No numerical tolerance, waived endpoint, sampled primaryvector or model promotion.17CPUcontrols pass onbothPCs. All7decodecalls for4rows, eight fullT4hidden boundaries andseparateFA3/11normalization twice plusactualactive Vcache4heads x256BF16/row twice retained. Fullfinite validation includesinactiveF32 andbothpackedBF16halves; observerinput hashesreadonly. ActualM16route/init/factory7attrs/status/workspace replayed.

Originalcrossprofile 4phasegate exactness {"reset_state": true, "prefill_state": true, "final_state": false, "final_reset": true},8-ID equality [true, true, false, false]. Newperstep28row table in evidence/Garnet-Readonly-Decode-Boundary-Progress.json explicitlymarkssameconsumed-inputprefix andactive/inactivewordcounts. FA3Vcache mismatch steps withALLfour actualpreceding T4hidden arrays andseparateRMSNorm arrays exact: [5]. This qualifies only the next actual-weight SDK baseline/control, does not prove internalnorm orcausalprojection. SDK must reproduceactualGarnetfourM4/defaultM16 activeVbytes beforeusinganyM4-algorithmatM16intervention. Ifinputsnotexact, trackearlierdependency instead. OriginalmodelFAILED; no newperformance/chartclaim.

All1547baselinepins/runtime/SDK/checkpointmetadata/all34currentCUDAobjects/CMake restored, activebenchmarkDLL restored, ownednonceprivatecaches retiredafteridle; GPUidle independently verified. Allrawarrays onbothPCs underfixedstudy root/localoutputs; linkedparentinputarchives notrecopied. Terminalreplayops v1unused typo inJOB detectedbeforepostprocess, immutablev2 changes onlyjoblookup; bothfilesandcorrectionhash retained. No liveGPUsource edits. Full4K/MTP/805/HTTP/strongest tunedreference/INT4/4B/overall2x goal remainsopen.

V2 same-proof summary extends exact-input qualification independently toFA11 aswell: {"3": [5], "11": [1]}. FA11step1 allfour layer10hidden andseparateRMSNorm fullT4arrays exact, row1hasoneactiveVBF16difference; FA3step5 allfour layer2hidden/normfullT4exact,row0hasoneactiveVBF16difference. PreservebothnotjustFA3. These tworealfixtures shouldqualifySDKfullM4/defaultM16activecachebaselinecontrolnext, no inventedinputs. Parsedremoteproof+closurereceipt validateexactdownloadbytes; initialextra hand-transcribed expectedhashassertion typo preservedintransferrecord, independentfullreplaysagree.


### Actual FA3/FA11 value-projection SDK control prepared

2026-10-27: Paired readonlydiagnostic682f54de95a19b9ca6f2315059eddb05703129ce13a62868eda1322b500e91f0 qualifies two real cases: FA3decode5 andFA11decode1. Allfour fullT4 precedinghidden andseparateRMSNorm inputs exact, activeVcache differsoneBF16word ineachcase; originalmodelgate remainsfailed. Direct SM120/cuBLASLt13.4 control uses actualindexed N1024/K5120 FP8E4M3 weights andpositiveF32calibrations. RequirefourM4/defaultM16 SDK ACTIVE outputs reproducebothactualGarnetcaches before admitting M4selectedalgorithmatM16 intervention. CompareALL16x1024SDKoutputwords includinginactivepadding,2exactrepeats, independentCPU BF16cast/FP8quantization, all7attrs and4actualM16AlgoChecks withinoriginal4MiB/no fallback. FullinactiveGarnetprojectionoutput notcaptured andinternalfusednorm notdirectlyobserved; no broadclaim.

14CPUcontrols pass (new6+unchangedoriginal8). Initialsyntheticfixture repeatedidenticalrows so intendedrowpermutation fault unchangedbytes; correctedfiniteperrow/token values, failureandhashderivationretained, no gatewaiver. NativeSDKsource derivedexactlyfrompriorqualifiedcontrol: onlytwolabels,N1024,onecaseeach,label-specificinput path. Original production4compiledfiles haveidenticalbaselinehashes andareabsentfromsource60/67overlays, so noGarnetDLL/sourceoverlay/modelrun. PrivateSDKbuild onlyremote5090; all1580sourcepins/runtime/SDK/checkpointmetadata/all34productionobjects/CMake auditedunchanged. Fullactualshardheader andtwo10MiBtotalweight slices archived ratherthanwholecheckpoint; onefixedstudyroot/fixture.

Preparation e722da85a5985e2872a25df366dee3ce824209eadb7298bc636b8734ddf9622c, deployment e2311e16bb6e8a836fb27675ab7511cbf6ab701aad9e8e51acc6666996eb0d0e. GPUcontrol UNRUN atthisentry; actualbaseline, unsupportedAlgoCheck, numerical orbuildfailures retainedbefore anyclaim. Independentremote/local archive replay required, downloadednativebinary neverexecutedlocally. Existing4Kcharts unchanged; fullB4decode/full4K/MTP/805/HTTP/strongest tuned/MTPrefs/matchedINT4/4B andoverall2xgoalremainopen.


### Actual FA3/FA11 value SDK control FAILED and independently preserved

2026-10-27: ActualprivateSM120SDKcapture exits0, CPU/fixture/build/SDK stages all0, but aggregatebaselinegate exits1: AssertionError('SDK baseline did not reproduce both actual Garnet active V caches'). Rawarchive afbecc5876164801284d28bffdbe04cccb27f8fa9f7f222ab4b7928c709bfa9d (10076454B/87members), paired identicalremote/localfailureproof b669ce653f945cebb87851e8aa096af00ca5566f869e8e24ea5898c920f7b8ff.14CPUguards pass onbothPCs; 1580sourcepins/runtime/SDK/checkpointmetadata/CMake/all34productionCUDAobjects unchanged; noGarnetDLLloaded/swapped ormodelrun. Full10MiBindexedE4M3weight slices/positiveF32scalars/shardheader, input/cast/FP8/fullBF16output arrays,2repeats/nativeexe/log/failures retainedonbothPCs; downloadednativeexe NEVERexecutedlocal.

Two actualcases: FA3step5 andFA11step1. Allfour realprecedinghidden andseparateRMSNorm arrays exact B1/B4 fromqualifiedreadonlyparent. SDK CPU BF16cast/FP8quantization exact. ActualdefaultM16 vsfourM4 differsoneBF16word eachcase,2exactrepeats. SelectedliveM4algorithm atactualM16 removesALL16x1024SDKoutputworddifferences includingpadding inbothcases/repeats. All4actualAlgoChecks status/state0 andworkspace<=original4MiB, details [["v3", "0", "0", "16", "0", "0", "262495", "4194304"], ["v3", "0", "1", "16", "0", "0", "262495", "4194304"], ["v11", "0", "0", "16", "0", "0", "262495", "4194304"], ["v11", "0", "1", "16", "0", "0", "262495", "4194304"]]. BothM4anddefaultM16 algorithm IDsare35, but7attrsM4[35,13,17,3,2,0,1] versusdefaultM16[35,15,18,7,2,1,0]; IDonlyinspection wouldmissdifference.

CruciallyFA11case reproducesbothactualGarnetactivecaches (all4096words) andsharedM4-at-M16activecache equalsGarnetB1, bothrepeats. FA3baseline doesNOT reproduceallactualGarnetactivecaches: 168differentwords, exclusivelyphysicalrow3, identicalcommonerrorforSDKM4/defaultM16/sharedM16; rows0/1/2 exact. Aggregate2casegate staysFAILED; it isnot relabeledpassed orpromotedbydroppingrow3/FA3. Thisdistinguishesa reproducibleFA11componentcontrol fromanunqualifiedwhole2casemethod. FullinactiveGarnetVprojectionoutputs andactualinternalfusednormalization werenotcaptured. Do notinferwholemodelcause/performance fromstandaloneSDKexactness.

Nextsafe action: qualifyactualFA3normalization-to-projection boundary, with allfourrealinputs andsameoriginalcachebaselines; sourcefullDecoderChunk appliesactive select beforeRMS andcastsRMSinputtoBF16 forprojection, whereas currentseparateobservable isFP32RMS output. Thosecompile/fusion/rounding differences arehypotheses, notestablishedcauses. Firstreproduceexistingunmaskednormalizer andthenevaluateconnectedmaskednorm/projection ornewexplicitreadonlyinternalwitness withsame-profile primaryendpointparity. Preserveoldwitnessguards/originalfailedSDK/modeljobs. No servingchange ornewtiming/chartclaim. Strongest repeatedtuned/MTPrefs/matchedINT4/4B/full4K/MTP/805/HTTP andoverall2xgoalremainopen.

Raw safetensors header 60f09f862dd9fc8c2594641081613cc3f12f162419cfa06f2e6ee3feac7bc5f5 (126536B) isretainedbyteexactinverifiedrawZIP onbothPCs; requiredwhitespacepadding notnormalizedforGit. Headerstoragepointer and fullindexedmetadata retainedinGit insteadofonehugesingleline. No numerical/witness change.


### Connected FA value-component baseline prepared after failed standalone SDK

2026-10-27: Original2caseSDKfailure b669ce653f945cebb87851e8aa096af00ca5566f869e8e24ea5898c920f7b8ff remainsfailed. New component-onlymethod preserves original fullDecoderChunk internalmask/RMS/sharedQKV/BF16cast/projection/attention/MLPpath andsixoriginalcompressedweights atsource60/DLL841 B1T4 andsource67/DLL677 B4T4. UseexactactualFA3step5 andFA11step1 precedinghidden rows frompairedreadonlyarchive, inputJSON F32bitroundtriprequired. OrdinaryzeroFP32KV/position/slot controls intentionallyisolatehistory-independent Vprojection; no actualmodelhidden/state/outputequivalence asserted. No native/CUDA rebuild ororiginalstageguardchanges.

Requireallfourrowsandbothcases matchactualoriginalB1/B4activeVcacheBF16bytes, unchangedstandaloneFP32normalizer reproduceallfullT4oldarrays, inputhiddenandcomplete2x4194304wordKVcachesperrow readonlyzero, fullcomponenthiddenandVexact2repeats,4FP8/2NVFP4/6externalweightcounts/alloldcanonicalstagecounts0, actualVmetadata/calibrations matchfailedSDKrealweightfixture. Internalactualfusednormalization stillnotdirectlycaptured.10CPUcontrols pass(new6+original4); neitherfailedSDK/modelgate isrelaxedbythisnewmethod.

Preparation 6f2fb835fd3b4a862cd8605411da5eaf89bab1d8209acad2f7f16c9f13a18902, deployment 7e558522a2579f6bd87659b80a90142b53e4608b108bb2894192606dcfaa253e. GPUcomponent UNRUN atthisentry.1588sourcepins/runtime/SDK/checkpointmetadata/all34productionobjects/CMake restored; exclusivefixedstudyroot/privatecache andindependentremote/localfailureorpassreplay unchanged. Existingcharts/performance unchanged, no2xclaim. Strongest tuned/MTPrefs/INT4/4B/full4K/MTP/805/HTTP/fullgoalstillopen.


### Connected component attempt1 import failure preserved; dependency-free attempt2 prepared

2026-10-27: Attempt1 exits1 BEFORE componentgraph construction: embeddedprobe importedCASES fromNumPy-dependent fa_value_sdk_control; NumPy intentionallyunavailableinembeddedruntime. Harnessmistake, no actualcomponentmathresult. Rawarchive 5a5a906281803cb85c0c96f905f724b898f0123752d0849c0eb4069aa5140f11 (13214899B), identicalremote/localproof 8db5d83989da0fbc43de622dc6177764f6818b912f9cb2670696a2465980ac0e. Source/runtime/SDK/checkpoint/all34objects/CMake restored andGPUidle. Originalfailedattempt andrawstacktrace retained; no reclassification aspassing or numericalfailure.

Version2probe changesonlypreparationpath andimportofidentical CASEStuple todependency-freeconnected_fa_value_layout. No graph/math/cache/gate/body changes; derivationbytechecked. NewstaticAST/sourcecontract verifieslayoutimportsnone andcaseidentity.11CPUtests pass (original10+1embeddeddependency). Existingexactfixture in SAMEfolder reusedafterbit/hashcomparisonwithverifiedattempt1archive; no newfixturecopy/folder. Prepared 1796ad3441133551186c5adc820ce78cc12c6659d6028838ab955777d974f1d4, deployment 5ba94be3d90db7ba8d96f9893d48ce15414e8bace2d54ccd2e65ef8c84a31c00,1596baselinepins. Actualv2GPUcomponent UNRUN atthisentry; requiresoriginalallfourrow/twocaseVcachebaselines, fulloldnormalizer/repeat/readonlycache/finite/metadata checks andindependentremote/localreplay. No installofnewembeddeddependencies. OriginalSDK/modelfailures andfull4K/MTP/805/HTTP/strongestrefs/INT4/4B/2xtargetremainopen; chartsunchanged.


### Connected attempt2 unused host-mode setup failure preserved; attempt3 prepared

2026-10-27: Attempt2 exits1 BEFORE componentgraphs: io.enable_host_embedding(True) requiresanindependentNVFP4head, but thiscomponentneverusesembedding/head. Rawarchive 19cf8a4c6d765d0c3198832f3eb6eb389cc5d643cfdf38941a91cec304352d9f (13239799B), paired identicalremote/localproof 26ad60631bcfa2010d7f1839761f4c58792032efd1d81eda3e4f389c7efa1344, exactphaseerror {"reference": "RuntimeError('Host embedding requires an independent native NVFP4 head')"}. Allsource/runtime/SDK/checkpointmetadata/34objects/CMake restored. This isanotherharnesssetupfailure, no componentnumericalresult. Originalattempt2/source/trace retainedunchanged.

Attempt3 I/Ohelper openscheckpointwithoutrequestingunusedhostembedding/headmodes; ordinarydeviceembedding andpackedhead storage loadedbyOpen remainsunused, no head/embedding inference or performance/VRAMclaim. Helperclosesfailedopens; newCPUstub checksmethodcallorder/no unusedmode andderivation. Graph/math/cache/normalizer/allfourrow/twocase/repeat/read-onlygates unchanged.12CPUtests pass(original11+I/Ocontract). Sameexactfixture reusedinoriginalfolder. Prepared 2881693ee3054c3ebe68ed25a1801e6b482ec74dbbcd2d38b8cd3b3322b84e45, deployment b9d2f3abd3c5bbd0fa76054c3163d93f08acda66f3d45ad0c9356b9cfceaa89d,1604baselinepins. Actualattempt3GPUcomponent UNRUN atthisentry. No newembeddeddependency/native/CUDA installation. OriginalfailedSDK/modeljobs remainfailed; full4K/MTP/805/HTTP/strongestrefs/INT4/4B/overall2xremainopen, chartsunchanged.


### Connected attempt3 observer rank failure preserved; packed-view attempt4 prepared

2026-10-27: Attempt3 constructed referencegraphs but failed BEFORE firstcomponentinference in completeinputcacheobserver: {"reference": "RuntimeError('slice requires packed rows and matching batch')"}. Rawarchive 7ec88c0e7d67092169a596f08226855e08fa167315b6e266faf4277de467757a (13291650B), paired identicalremote/localproof ce3803692ab08cf6eb71ac11915a9aad8bc7f01993aafee5f71c17b8d16a9895. Allsource/runtime/SDK/checkpointmetadata/34objects/CMake restored. No componentnumericalresult; originalfailedattempt/source/trace preservedunchanged.

Version4 only addsio.storage_view(cache,[batch,4194304]) beforeboundedreadonly slice_into in completehashreader; sourcecacheiscontiguousrank4, slice APIrequirespackedrank2. Samecomplete2KVcaches/allrows/allwords, no sampling. CPUtest compilesactualold/fixedcache_hashesASTagainststrictfakeIO: oldrank4call rejected, fixedcompletezero hashesvalidatedforB1/B4.13CPUtests pass(original12+actualobservercontract). No graph/math/gate changes, existingfixture andsource60/67 libraries unchanged. Prepared 9f617eed39b3fc363e49b5655c7a8a639e777e3c5bd15801bf331404331259f3, deployment 4c95d9937377b38a344bda7b14e04b1993866e9de4d6797fcfbeeb4e4092d9f3,1611baselinepins. Actualattempt4GPUcomponent UNRUN atthisentry. OriginalSDK/modeljobs stayfailed andchartsunchanged; full4K/MTP/805/HTTP/strongestrefs/INT4/4B/overall2xremainopen.


### Faithful connected FA3/FA11 value baseline PASSED; original SDK/model failures unchanged

2026-10-27: connected-fa-value-attempt4 exits0. Rawarchive a69928200275bbccd309870ab32b05bd6fac84b27c64e796f53ab042041984a0 (15046965B/1366members), paired identicalremote/localproof be940892762ad1700895c772111c35cfd23a57e898fee7a57c8d3c82bf7cb149.13CPUtests pass onbothPCs. Source60/DLL841 fourB1 groups perFAcase andsource67/DLL677 physicalB4, two realcases FA3decode5/FA11decode1, fullT4actualhidden inputs.20componentinferences(two repeats each8reference+2candidatecases), originalstandaloneFP32normalizer controls, allcomplete2x4194304wordinputKVcachesperrow remainzero/read-only; exactfullhidden/Vrepeats, allfullT4input/normbytes and2case/all4rowactualVcachebaselines passwithouttolerance. Correct4FP8/2NVFP4/6externalweightcounts andactualVcalibrations matchpinnedcheckpoint/failedSDKfixture; all1611baselinepins/runtime/SDK/checkpointmetadata/all34CUDAobjects/CMake restored, activebenchmarkDLL restored andprivatecaches retiredafteridle. No native/CUDA codechange orservingbenchmark.

Connected originalfullDecoderChunk preservesactive select/RMS/sharedQKV/BF16cast/projection andattention/MLP graph; ordinaryzeroFP32history/positions/slots deliberatelyisolatehistory-independent Vprojection. ALL4096activeVwords percase/profile nowreproduceactualpriorGarnetcache, includingFA3row3where priorstandaloneSDKhad168mismatches inbothM4/defaultM16. Before/after baselinefaithfulness [{"label": "v3", "mode": "m4", "old_standalone_SDK_vs_Garnet_active_word_differences_by_row": [0, 0, 0, 168], "connected_component_vs_Garnet_active_word_differences_by_row": [0, 0, 0, 0]}, {"label": "v3", "mode": "default16", "old_standalone_SDK_vs_Garnet_active_word_differences_by_row": [0, 0, 0, 168], "connected_component_vs_Garnet_active_word_differences_by_row": [0, 0, 0, 0]}, {"label": "v11", "mode": "m4", "old_standalone_SDK_vs_Garnet_active_word_differences_by_row": [0, 0, 0, 0], "connected_component_vs_Garnet_active_word_differences_by_row": [0, 0, 0, 0]}, {"label": "v11", "mode": "default16", "old_standalone_SDK_vs_Garnet_active_word_differences_by_row": [0, 0, 0, 0], "connected_component_vs_Garnet_active_word_differences_by_row": [0, 0, 0, 0]}]. Originaltwo-caseSDKjob remainsFAILED; thisdifferentfaithfulcomponent method passes its own completebaselines, doesnotpromotefailedjob ordropanyrow. Internalactualfusednormalization stillnotdirectlycaptured, no specificrounding/fusioncause proved.

ActualB4-vs-B1Vdifferences persist unchanged(oneword foreachcase), asreproducedbythisfaithfulbaseline. Additional hiddenwordcomparison inevidence/Garnet-Connected-FA-Value-Progress.json is ZERO-HISTORY COMPONENToutput only, neveroriginalcomplete-model equivalence. Nextbackendcandidate canusea separatedefaultofflogicalFA-V B4T4/M16 policy withliveSDKselectedM4algorithm(all7attrs, actualM4/M16AlgoChecks,original4MiB/no fallback), thenpassfaithfulconnectedcomponent andunchangedcomplete short/tail/full4K/state/vector/ID/reset/repeat/MTP/805/HTTPgates beforetiming. Originalsource67fullmodel FAILED; overall2xdecode+throughput/halfTTFTgoal andstrongesttuned/MTPrefs/matchedINT4/4B remainopen. No newchart values orperformanceclaim.


### Default-off physical B4T4 V projection M4-algorithm candidate prepared

2026-10-27:73-source candidate over source67. Exact16 FA V logical weight names, physicalB4T4 flatM16,N1024,K5120 only; defaultoff GARNET_FP8_V_BATCH4_DECODE_CANONICAL, separate20B wire0x56313634/plugin. LiveM4 SDKopaque algorithm selected by all7 actualattrs[35,13,17,3,2,0,1], M4/M16 AlgoChecks original4MiB limit, no opaque imports/fallback. Original67bytes unchanged except loweringdispatch; no new CUDA/quantizer/matmul kernel changes. PrivatefreshnonceCUDACompiledGraph cache only; productionfrontendpolicycache62 remainsunqualified. Sourcebundle f3bccec717a030f9bf855d493674cd46092a871bc9e46ecfae360a9bda4e08b5.

12 local Python controls passed. Prepared unrun remoteCPU contracts:1209 exactselector cases16selected,212 factory interface/fault/trace/unsupported/device-cache/geometry/runtime/opaque-retention cases, actual V+oldGDN+genericTRTplugin serialization cross-rejection, descriptors, scales, formats, clone/initialize/factoryexceptions withGPUstubs. All native compile/execution remote only. Preparation corrected expected literal counts in source/method generators before deploy/compile, byte-checked partial outputs; merged explicit owned historicalsourceinventory aftermissingcandidatecompileinput detection. Final derivations and source hashes retained. No source/DLL/productionbuild mutation by CPU job.

Faithful connected component baseline4 passed, original standaloneSDK and shortfullmodel gates remainFAILED. This candidate is numerical/model/quality/performance UNQUALIFIED. Next actual nativebuild withoriginalcontrols, faithfulconnectedcomponent off/on intervention, originalcomplete-model bit/state/vector/token/reset/repeat gates, MTP/805/full4KHTTP required beforetiming. Strongest repeatedtuned/MTPreference, INT4/4B andfull2x goal remainopen.


2026-10-27 V M16 CPU attempt1: FAILED before C++ compilation because deployment omitted M/evidence/connected-fa-value-attempt4-verified.json, required by unchanged source-lineage test. Both independentCPU archive replays agree proof ebdca317c8d3e2598314289ad00c20bd72df0e51244a223d50aaa286c1522073, raw40c405832f2a712ad1e034bbb702d33b3925a40e1810515acdd5f98eafeb3db4; source/runtime/SDK/checkpoint/all34productionobjects/CMake unchanged. Original failedjob preserved. Version2 package explicitly pins all three actual paired parent proof copies at required M/evidence paths and adds versioned owner/reader; same source/test/factory/plugin/math/gates.12local controls pass, remote retry unrun. No numerical/model/performance admission.


2026-10-27 V M16 CPU attempt2 PASSED, independentremote/localraw replays byte-identical proof 1349ebc273d463036ed2f04c7ea0179aa8bd906bade36b6be91d1465b616c5f1, raw ed43127619cc73811103e6c5d86cef41b5c423ad2bb27aa7ba324b9059eeaf75.12Python,1209actualC++profile cases16selected/1193rejected,212actualfactoryCPUinterface faults including94modes/7attrs/liveopaque preservation/device+geometryisolation/tracingparity/4MiBchecks/runtimeguards. ActualV+oldGDN+genericTRTplugin:45badwires24ctors46descriptors9formats7crosswirerejections7initialize5factorycalls0GPUoperators; TRT101601loadedpinnedDLL. All1660owned source/runtime/SDK/checkpoint/all34CUDAobjects/CMake unchanged. Originaldeploymentattempt1 remainsFAILED. Native73DLL/unrunGPU/math/fullmodel/quality/HTTP/performance UNQUALIFIED; proceed originalnativebuild gates then faithfulcomponentoff/on, originalfullmodelgatesbeforetiming.


### V M16 actual native build prepared after paired CPU qualification

2026-10-27: Fresh remote readonlyinventory pins1374baseline source files,73overlaytargets,SDK/runtime/checkpoint/CMake/all34currentCUDAobjects andclosedparents. D free563GiB. Nativepackage da2a67e89c1469a7aedde75fe7584ca4d45223dd2af15bff44e8192b766fa1f3, sourcebundle f3bccec717a030f9bf855d493674cd46092a871bc9e46ecfae360a9bda4e08b5. Originalnativeowner/replay exactderivations retain78current+86historicalPython controls, Q/O278profiles,182factoryfaults, allQ/Owire/630+720initialize/native-embeddedselector/955tokenizer gates; original3CUDAobjectwhitelist unchanged, all34recorded. New73source defaultoffV-only intervention notyetGPUqualified. Snapshot/restore all73targets andbaselineDLL/runtime/SDK/checkpoint/CMake guards retained. OriginalSDK,fullmodel,cache62 andCPUdeploymentfailures remainfailed. No model/GPUmath orbenchmark inbuild. Afterpassedbuild, faithfulconnectedcomponentoff/on andunchangedcomplete-modelstate/vector/token/reset/repeat/MTP/805/full4kHTTP gates beforetiming; strongest tuned/MTPrefs,INT4/4B,2xgoal open.


2026-10-27 V M16 nativebuild attempt1 PASSED, pairedremote/localindependent rawreplay proof 8599160df1dbdf98b197e8ca7b0b2d47b52c46713fe878e03c258461408dcc6c, raw 01c03b23baa67bf74f86a565a1f5e8252603a856ce1ea2d32c56d430b3c628be, DLL ab248b760f2491db3b4d75db34a8470c4cdf8de20e448c749725accd6086fc34. Source73Release/MT/sm120build retains78current+86historicalPython,originalQ/O278profiles182factoryfaults/Q-Owire/630+720initialize/native-embeddedselector/955tokenizer controls. All34CUDAobject hashes recorded; onlyoriginal3-objectwhitelist changes, unchanged. Source/runtime/SDK/checkpoint/CMake/baselineDLL restored, candidateDLL saved separately, GPUidle atclosure. No actualV GPUmath/fullmodel/quality/performance admitted. OriginalfailedSDK/model/cache62/CPUdeployment jobs remainfailed. Next faithfulconnectedoff/on mustprove V andcompleteT4hiddenbitwiseequalfourB1 withalloriginalbaseline/cache/input/normalizer/finite/repeatguards andactual7attr/M4-M16SDKrouteproof, then unchangedcomplete-model/MTP/805/full4KHTTP gates beforetiming. Strongest repeatedtuned/MTPrefs,INT4/4B,2xgoal remainopen.


2026-10-27: Prepared next connected V off/on gate andactualSDKtrace reader;10combinedlocalCPU controls passed andrawledgers retained. Off must first reproduce originalB1/B4 cachebaselines; on mustequalfourB1 activeV andall20480F32 hiddenwords perrow/tworepeats, withoriginalfullKV-input/read-only/normalizer/finite/repeat guards. Negativecontrols reject baselinewaiver/unchangedV/rowpermutations/inactive/signedzero/nonfinite/repeat/inputcache faults. Tracechecks retainall7liveattrs/APIstatuses/actualM4-M16checks4MiB/opaqueSDKretention andatleast2N1024projectioninitializations; reference/offmusthavezeroVroutes. TheseareCPU gate controls only; GPUintervention unrun, fullmodel/performance unqualified.


### Actual V M16 connected reference/off/on intervention prepared

2026-10-27:Source60B1/source73B4off/source73B4on, sameoriginalfullDecoderChunk AST/sixweights/actualFA3step5-FA11step1 precedinghidden/immutablefixturefolder/zeroordinaryFP32history-slot-position componentcontrols. Beforeon,reference/offmustreproduceallpriorpaired4 input/normalizer/fullT4hidden/activeV arrays andmetadata, plusactualcachebaselines. OnmustmatchfourB1 fullactiveV andall20480F32hiddenwords perrow/tworepeats. ActualVM16 factory/API/7attributes/liveopaque/M4-M16AlgoChecks4MiB/2N1024initcalls required; reference/offzeroVM16routes.24localCPU controls passed, includingnewpriorfullarraynegativecontrols, alloriginal13 retained. Threefresh embeddedprocesses/privatefreshnoncegraphs, guardedsource73/runtime/SDK/checkpoint/CMake/all34objects restoration. Newsourceinventory 1671 explicitowned paths, no protected bodies. New73candidateDLL ab248b760f2491db3b4d75db34a8470c4cdf8de20e448c749725accd6086fc34. Deployment e2e5e3e2d86dd4d28351fa2c4c1effdd28a75e8511c065ddf3822165d59c9f8a. GPUphasesunrun. This is component-only with zerohistory, notoriginalfullmodel equivalence. OriginalSDK/model/cachefailures remainfailed; fullmodel/MTP/805/full4KHTTP beforetiming,strongest tuned/MTPrefs,INT4/4B and2xgoalopen.


2026-10-27 actual VM16 connected off/on PASSED, pairedremote/local independentrawreplays byte-identical proof c4d6d1cf3765e9b7336c311262fc190b4f887ad595c88cc28fb1123b608b8926, raw 7448bd98108ebb89fbb97fe0825b99dc67a577d487aec48b9b3f02f7c37a3940.24CPU controls,3fresh embeddedphases. Source60reference56fullarrays andsource73off14fullarrays reproducepriorpaired4 exactly. Source73on removes bothobserved1BF16 Vworddifferences(FA3step5row0,FA11step1row1); all8case-row Varrays andall16fullT4hidden arrays equalfourB1 exactly(tworepeats), including4831FA3row0 hiddenF32word differences removed. AlloriginalinputKVfullcachezero/read-only/inputhash/normalizer/finite/BF16/fullpadding/repeat guards retained. ActualVM16 trace4createdcontexts/4N1024initializations,all7attrs[35,13,17,3,2,0,1],actualM4/M16AlgoCheck workspace262495B eachwithinoriginal4MiB,liveSDKopaquebytes preserved. Reference/offzeroVM16routes.1671ownedsource/runtime/SDK/config-index/CMake/all34objects restored,baselineDLLrestored/privatecache retired. Thisisconnectedcomponentwithzerohistory, NOTcomplete-modelhistory/state/output/performancequalification. OriginalSDK/model/cachefailuresstayFAILED. Nextunchangedfullmodel128component/finitevector/8IDs/reset/prefill/final/reset shortgate withbothGDNandall16FA Vroutes, thenfull4K/MTP/805/HTTP beforetiming. Strongest repeatedtuned/MTPrefs,INT4/4B and2xgoalopen.


### Original complete-model short gate prepared with corrected V route

2026-10-27:Pairedactualconnected off/on proof c4d6d1cf3765e9b7336c311262fc190b4f887ad595c88cc28fb1123b608b8926 qualifiesnextnativefullmodel gate, doesnotpromoteoldmodel/SDKfailures. Originalsource60B1 vs source73B4,4096capacity,prefill32/decode4,41/104/93/93prompts,8IDs. SAME128complete-state-component/fullfinitehidden-normalizedvector/rawsignedzero/row/slots/IDs/reset/prefill/final/reset exactCPython gates; modelprobe computation unchanged, onlynewpinnedpreparation andVpolicy. AlloriginalGDNroute proof retained andV>=16N1024initcalls with7liveattrs/checks4MiB required; 2componentroutes insufficientfor64layer model. Original7CPUguards+3newfullroutecontrols passed. Unrunreader draft got explicitverify_derivations import beforeanyGPU/replay execution; exactfinalderivations retained. Candidate ab248b760f2491db3b4d75db34a8470c4cdf8de20e448c749725accd6086fc34, deployment f96ca234c4b324cbbbe51b1402484f0b556213ad123e3e93a41237bcbc1a042b.1679ownedsourcepins/runtime/SDK/config-index/CMake/all34objects/hostembedding/NVFP4head/privatecacherestore preserved. GPUfullmodel unrun atpreparation. Evenpasscannotqualifyfull4K/MTP/805/HTTP/strongest refs/INT4/4B/performance/2x.


2026-10-27: Prepared CPU-only full4K B4 correctness input selector while shortGPUgate live; no remote source mutation. Actual immutable corpus596991dd0907ccb8ae8ffb090737f59ba8da3faac181bb18fadc74ae0e4ef3ce selected8realrequests(first4measuredC8text+first4image),3968prompt+128forcedoutput,4096total,greedy/thinkingoff/ignoreEOS. Actualrequestbytes/tokenIDs exactly match originalpairedpaged-model full4k cases. Puretext4/pureimage4/mixed4 reuseeightB1oracles; no newprompts/model/benchmark. SixCPUnegativecontrols pass(checksums/context/template/greedy/tokenrange-bool-truncation/payloadkind/duplicate-cohortidentity); rawCPU andactualdataset audit ledgers retained. Future GPUlonggate still requires current shortfullmodelpass, freshB1/B4 complete128statecomponents/finitevectors/128IDs/reset/repeats; MTP/805/HTTP/strongestrefs/INT4/4B/2xremainopen.


### Source73 complete-model short gate FAILED after faithful V component success

2026-10-27:BothGPUcaptures complete, originalstrictCPython admission fails: AssertionError('B4 complete state/raw vector mismatch against four independent B1 oracles'). Pairedindependentremote/localreplays agree proof d206de4ab0acc44e66581be42af90204c083a6305f5606bf6f363c535bc55a9b, raw 84a0f76b456e55686912a63306359cec5a62ffffefc9a00e77bb37cb43b26df5 (13957597B). Reset/prefill/finalreset all4rows exact; finaldecode fullstate/finitevectors fail.8IDs equality[true,true,false,false], red/greenfirstdifferenceindex5 persists. ActualGDN>=48eachN andV32N1024initializations/live7attrs/M4-M16checks4MiB proofs pass; failures notcausedbyskippednewroute. All1679ownedsource/runtime/SDK/config-index/CMake/all34objects/baselineDLL restored/privatecache retired afteridle. No full4K/MTP/805/HTTP/modelquality/performance admission.

Pairedbefore/after analysis confirms16completeB1reference snapshots andallrawhidden/normalizedvectorbytes/IDs unchanged. Common-input row0 finaldifferentcomponents115->113, earliestpersistentdifferenceFA3V(index7)->FA7K(index14); row1 105->104, FA11V(index23)->GDN12role1(index24). This corroborates Vcomponent correction butdoesnotprove K/Q/O isuniquecause; actualinternalquery/outputactivations/fusednormalization/full-history effects remainunobserved. Red/green havechangedconditioningafterfifthcommonconsumedtoken, so laterlayer0differencesnotrootcausalevidence. OriginalfailedSDK/model/cachejobs stayFAILED. Next verifyliveSDKM4/M16 FA Q/K/O geometries andsame-input actualattentionprojections; anynewobservermustfirstreproducecurrent same-profilecomplete-state/vector/IDendpoints. Preparedfixed4Kcorpus remainsGPUunrunpendingstrictshortpass. Strongest repeatedtuned/MTPrefs,INT4/4B andoverall2xgoalopen; charts/servingnumbers unchanged.


### Read-only liveSDK FP8 decode shape/configuration control prepared

2026-10-27:FollowingpairedVfullmodelfailure d206de4ab0acc44e66581be42af90204c083a6305f5606bf6f363c535bc55a9b, preparestandaloneMSVC/CUDA/cuBLASLt shapequeries: GDN_QKV10240x5120,GDN_Z6144x5120,FA_KV1024x5120,FA_Q12288x5120,FA_O5120x6144. BothM4/M16 heuristics, all7configattrs/liveopaque andactualM4/M16 AlgoCheckstatus/workspace4MiB forreturnedcandidates. Cross-checkknownM4GDN/V references, actualloadedcuBLASLtDLL/GPUUUID/SM120/SDK130400. No tensorinput/cudaMalloc/cublasLtMatmul/Garnet/TRT/model/benchmark; actualimportinspection required. Eligibilityisnotnumericalequivalence. Original Vcomponentpass/modelFAILED preserved.3localCPU identity/geometry/attribute/API/workspace/unsupported-no-admission controls pass.1686ownedsource/runtime/SDK/config-index/all34productionobjects/CMake preservation. Deployment 412bde38061483ff716ee067fd86bc72e05bdf55cd53a4f671baf92d8c60977d. SDKqueryunrun; actualprojection/fullmodel/4K/MTP/805/HTTP andstrongesttunedrefs/INT4/4B/2x remainopen.


2026-10-27 SDK shapequery attempt1: C++query0/no matmul, wrapperFAILED falseheader-runtime equality assumption (header13020,reportedruntime13040,driver13040,cuBLASLt130400). Independentremote/localfailureproof47735f83d175e4b1dd8f71ae7f8b8cb6c589283f6cbfbd8542e73acd15c2cb4b/raw e19c60d66328d1185653d8fc27216cea4a81fd2aee8d91536af8d1714ced85d5 retained, neverpromoted. All1686source/runtime/SDK/config-index/all34objects/CMake unchanged. RawSDKstdout keptbyteexact inimmutableZIP; derivedprettyJSON andstoragehash pointer retained. Version2 newfiles retainmatrixdescriptor/heuristic/7attrs/AlgoCheck computations; delay-load andexplicitabsoluteLoadLibraryExGarnetbuild/bin cudart+cuBLASLt, recordmodulepaths+actualSHAs againstruntimepins. SDKandGarnetcuBLASLtSHAs alreadyidentical. Separateheader/runtime/driver fields, no falseequality test, no genericcompatibility/math claim. FourCPUcontrols passed, source1695pins, newdeployment 71460465ec9771f67194da0ca5d791ce4149a03485845149f74ed70de3610b53. Oneearlylocalv2 unittest invocationusedchatcwd andcouldnotimporttest2026; rerunfromrepo andformalpreparation4controls passedbeforeanynativev2execution. v2SDKunrunatpreparation; originalmodel failure remainsfailed, no newbenchmark/2xclaim.


2026-10-27 actual live FP8 decode geometry query attempt2 completed0; independent local/remote CPU raw replays byte-identical proof 259fd065a553c50f206538f0bbc7180ef3bb495fcff5c8765936131008301d00, archive 88225b0df3638d54a46af78e584b6ba28beb8bec9a6bb1c9fc84fabbcc4f4710 (929562B). Four CPU controls;70validheuristics/140actualAlgoChecks/731APIevents. Live M4 eligible at M16 under4MiB for all5 geometries. Q N12288K5120 M4=[35,18,16,7,2,1,2], K/V N1024K5120 M4=[35,13,17,3,2,0,1]; do not reuse V tuple for Q. O N5120K6144 M4/M16 fullopaque equal; other4 differ. Eligibility/config differences do not prove numerical/model behavior or cause. All1695source/runtime/SDK/config-index/all34objects/CMake unchanged; no GEMM/model/timing executed.
Runtime-provider audit: explicit pinned DLL paths/hashes were collected, but linker LNK4199 says cudart delay-load ignored and import table has no cudart DLL. Therefore earlier v2 preparation/derivation scope language claiming actual runtime-call provider pinning is too broad: only the explicitly loaded cudart module is pinned, linked CUDA call provider is unproven. cuBLASLt delay-import/pinned DLL remains evidenced. Header13020 vs linked-runtime/driverAPI13040 are distinct recorded fields. Original first metadata-wrapper job staysFAILED; all SDK/model/cache failures stayFAILED. Next separate defaultoff Q/K route selected by exact32logicalFA names/physicalB4T4 only, geometry-specific live M4 configs and actual M4/M16 checks; original fullstate/vector/IDs/reset/repeat gates before4K/MTP/quality/HTTP timing. Overall2x and tuned/MTP/INT4/4B comparisons remainopen.


2026-10-27 source79 Q/K M16 candidate prepared, native CPU/GPU unrun. Separate defaultoff GARNET_FP8_QK_BATCH4_DECODE_CANONICAL, exact32FA Q/K names, physicalB4T4/M16,N12288/1024,K5120 only, wire0x514b3136. Geometry-specific live M4 configurations from pairedquery2; actualM4/M16AlgoChecks <=4MiB, no opaque import/fallback. Source73 parent unmodified exceptlowering+6newfiles; originalV/GDN/generic/NVFP4/CUDA/math unchanged.13localPython controls pass; remote nativeCPU selector/factory/plugin contracts next before actualbuild and unchanged wholemodel state/vector/ID/reset/repeat gates. QK fix remains unqualified; no new serving measurement. Localpreparationhelper postprocess-template final occurrence check failed after tests/package completed; exactremaining-name helper generated separately, recoveryrecord retained, no source/deployment/native rerun.


2026-10-27 QK initialinstaller preflight rejected stale pre-source73 CUDAobject snapshot inheritedfromV CPUprep, before any install/nativeCPU/ownerjob. Version2 corrects preparation only to exact currentall34 snapshot already measured by terminalSDKquery2; no objects changed, skipped comparisons or whitelist relaxation. Allsource79/selector/factory/plugin/math/13controls unchanged; initialfailure and versionedpackage retained.


2026-10-27 QK M16 actual nativeCPU contracts PASSED, pairedremote/localrawreplays identical proof 2026b9270697adacea0be81565593fb47b6ed340b67b584d3bc586753024e176, archive ad991cd1a6cfb164478ef0bc1127b2a08d3b4e9eec7c88529199da7620948dbd (2483301B,917members).13Python,1248selector32selected1216rejected,406factorycases94modes/twogeometries inclall7pergeometryattributes, liveopaque preservation, M4/M16checks4MiB, device+geometrycacheisolation, unsupportedSDK, tracingparity/API/runtimeguards. ActualQK+oldV+oldGDN+genericTRTplugin92badwires48ctors92descriptors18formats18crosswirerejections14initialize10factorycalls0GPUoperators. All1724ownedfiles/runtime/SDK/config-index/CMake/all34CUDAobjects unchanged; no productionoverlay/GPUmath/model/performance. Initialstale-objectinstaller failure staysrecorded. Nextoriginal83current+86historical nativebuildgates and unchanged strictfullmodel gates; source79 QK remainsnumericallyunqualified.


2026-10-27 source79 QK guardednativebuild preparedUNRUN afterpairedCPU406/1248 success. Freshremoteinventory confirms1374baseline/79targets/runtime/SDK/config-index/CMake/all34currentCUDAobjects and562.7GiBDfree; closedQKCPU/source73V/failedfullmodel/livequery parents checked. Candidate sourcebundle adbb29965abb939131fd942883dfe4b5c1ac72e25e0a48dcba824fa0019e1397. Alloriginalnativecontrols retained plus5QKsource=83current,86historical; original3CUDAobjectwhitelist unchanged. CandidatebuiltDLL willbesavedseparately, baselineDLL/source restoredbeforemodelgates. Localinventoryhelper initially expected one generic73targettoken butactualtwo; narrowedreplacementtospecificreadonlylabel beforeanyremoteinventorydeployment, partialhelper bytechecked. No numerical/model/performance admission; completegoal remainsopen.


2026-10-27 next source79 wholemodel short gate methods preparedCPU-only while remote nativebuild source frozen. Original fullmodel probe computations/inputs/all128rawstatecomponents/fullfinitevectors/8IDs/reset-prompt-final-reset and exactCPython gate unchanged; only candidateQK policy androuteverification added. OriginalGDN/Vactualroutes retained; newQK requires bothpergeometry7attrs/liveopaque/M4-M16checks4MiB/all16Qand16K initializationrecords, referenceV/QK absent. Geometryinferredfromuniqueconfigtuple+initialization/pinnedcode, notdirect internalprojectioninput witness; routeproofnotmathadmission.14CPUcontrols(original10+4new) pass; initialunittest11included7importedTestCasealiases, correctedtomoduleimports beforeformal14suite. No modeljob/deployment/benchmark executed bythispreparation; nativebuild andpairedreplays required first.


2026-10-27 QK M16 nativebuild attempt1 PASSED, pairedremote/localindependent rawreplay proof d2f4f80b1d98a4fa276208c626da3b47c340c1dd44ea224758943d66911f4f66, raw 9c6fa6e525d00d787933f55e4d30ca23f927a7e3df74144655793999411f45e3, DLL 2bafed218fbd1f55a20c0b3bb303638a0f135e12bd70be2884b4a098d4b7beb4. Source79Release/MT/sm120build retains83current+86historicalPython,originalQ/O278profiles182factoryfaults/Q-Owire/630+720initialize/native-embeddedselector/955tokenizer controls. All34CUDAobject hashes recorded; onlyoriginal3-objectwhitelist changes, unchanged. Source/runtime/SDK/checkpoint/CMake/baselineDLL restored, candidateDLL saved separately, GPUidle atclosure. No actualQK GPUmath/fullmodel/quality/performance admitted. OriginalfailedSDK/model/cache62/CPUdeployment jobs remainfailed. Next unchanged strictwholemodel shortgate: all128statecomponents/fullfinitevectors/IDs/reset/prefill/final/reset againstfourfreshB1 oracles, andactualGDN/V/QK all7attrs/M4-M16checks4MiB/all16Qand16Kinitroutes. Zero-historycomponent wouldnotprove query/full-historyeffects; modelgate required before4K/MTP/805/HTTP timing. Strongest repeatedtuned/MTPrefs,INT4/4B,2xgoal remainopen.


2026-10-27 source79 strictwholemodel short gate packagedGPUUNRUN afterpairednativebuild proof d2f4f80b1d98a4fa276208c626da3b47c340c1dd44ea224758943d66911f4f66 andDLL2bafed218fbd1f55a20c0b3bb303638a0f135e12bd70be2884b4a098d4b7beb4.14CPUcontrolspass.1756ownedfiles/runtime/SDK/config-index/latestnativebuildall34objects/CMake/79targets andbaselineDLL guarded. Fourfreshsource60B1 thenphysicalB4 source79, unchangedprompts41/104/93/93,4096capacity,prefill32,decode4,8IDs,complete128state/finitevectors/reset/prefill/final/reset. Source79QKdefaultoff exceptcandidateflag; actualpergeometryQ/Kconfig+GDN/Vroutesrequired. Originalfailedfullmodel/SDK/cachejobsremainfailed. Anyshortpass alone cannotqualifyfull4K/MTP/805/HTTP/strongestrefs/INT4/4B/performance oroverall2x.


2026-10-27 QK short initialowner preflight rejected KeyError status for3historicalnativebuild parent records beforeoutputdir/sourceDLLoverlay/model/GPU. Originalhash-pinnedpairedproofs containstatus; v2metadata prep fillsmissingstatus fromexactproofs andretains strictstatuscheck. Versionedprobe/launcher/owner/reader onlyswitchprep/filenames+verifyderivationchain; originalmodelmath/source79/DLL/14numerical-routecontrols unchanged. RawSSHrejection andoriginalpackage preserved, no failedmathjobpromoted, no timingdata produced.


2026-10-27 full4K B4 exactCPython admission comparator preparedCPU-only whileexclusive source79short GPUjobrunning; remote source remainsfrozen.11CPUtests pass(5newfullcontext+6existingpinnedinputcontrols), unchangedcompare_capture/require_ids. Eight independentB1oracles/three4-rowtext-image-mixedgroups, all48phase-row comparisons x128statecomponents, physical/logicalhashes/finitebitwisevectors,128IDs,3968prompt/4095finalslots/resetrestoration. Lastcomponent/lastID/missingoracle/permutation/shortcontext/truncatedoutput/nonfinite/signedzero/profile/slot/reset corruptions rejected. No full4K GPUprobe/owner/capture/quality/timing yet; currentstrictshortpass+pairedclosure stillrequired beforefuturelongrun.


2026-10-27 futurefull4K rawcapture probe draftedCPU/AST-only fromexactsource79shortv2probe. Byte-replayedderivation bindsoldprobe07a5b2d9455f7cb078a4665867ab3f0dd92496994ffc9bb91b7d0994f64be53c to68ee4339f77ae389d398ca5f62314293a899407ee5552c55d5348d71ab6ad2d3. Modelconstruct/weight/vision/tokenization/prefill/decoderstep/128statefinitevectors/resetmath unchanged; onlypinned3968promptcorpus8B1/3B4case selection,128forcedIDs/finalslots4095 andreportlabels changed. InputGoldenloader stillchecksrequest/template/token/imagepins. No deployment/owner/prep/GPUfull4Kcapture orperformance: currentshortpass andpairedclosure stillrequired; futurefull4Kowner/launcher/evidence plumbing unprepared. Remote source remainsfrozen forrunning source79shortjob.


2026-10-27 source79 short wholemodel exactgate PASSED afterv2metadata fix; pairedremote/local rawreplays identical proof 8507a283757be743980795d4e13bf6b5b7d227091b106208bedd827ded9252eb, archive d69a06752cebd0160a98e0f7debc63a79c68c0e0e51ab3050bcfa661f57f0891 (15040039B). Bothcaptures0,14CPU; all4rows all8IDs, all128statecomponents/physical-logicalhashes/fullfinitehidden+normalizedvectors reset/prefill/final/reset exactagainstfourfreshsource60B1. All1763ownedfiles/runtime/SDK/config-index/CMake/all34objects/baselineDLL restored; privateenginenonces retiredafteridle. ActualGDN/V/QKroutes validated; QK bothN1024/12288 all16+init withgeometryspecific7attrs/liveopaque/M4-M16AlgoChecks4MiB. Source79DLL2bafed218fbd1f55a20c0b3bb303638a0f135e12bd70be2884b4a098d4b7beb4 remainsdefaultoff outsidecandidate.
Before/after comparison firstproves16completefreshB1reference snapshots, allrawhidden+normalizedbytes and8IDs unchanged. Finalcomponentmismatches source73->79 rows[113,104,128,128]->[0,0,0,0];8IDs matches[true,true,false,false]->[true,true,true,true]. Prior source73FAILED andallSDK/cachefailures remainFAILED. This is a paired candidate result, not unique Q-vs-K causal isolation; sameDLL QKoff wholemodel ablation notexecuted. Short4cases/8IDs donotqualifyfull4K,permutation/tail/repeats,MTP,805quality,HTTP orperformance. Full4Kgolden8inputs/3groups/comparator11CPU/probeAST prepared butGPUunrun; execute afterpairedshortclosure. Strongesttuned/MTPrefs,INT4/4B,overall2x remainopen andservingcharts unchanged.


2026-10-27 progress comparison v3 chart updatedafterpairedsource79shortpass; generator/input/artifactSHAs retained under evidence/figures. Servingratios/data unchanged source60 repeatednative vsordinaryreferences; no newbenchmark. Engineeringrow nowSHORTPASS all4rows/8IDs/128components/fullfinitevectors, full4K/MTP/805/HTTP stillOPEN (11CPUfull4Kcontrols/probeAST only). Figurevisuallyinspected; allfigures/metrics excludeany unmeasuredcandidateperformance. Overall2xNOTACHIEVED; strongesttuned/MTP/INT4/4B pending.


2026-10-27 full4K exactgate actualowner/launcher/reader package preparedGPUUNRUN afterpriorgoalturn progress(source79short8507pairedpass/nativeCPU406/1248/source79build83+86).25CPUcontrols pass(original14+11full4K); exactowner/reader/probe derivations replayed. Eightfreshsource60B1oracles vs3source79B4text-image-mixedgroups, pinned3968prompt+128forcedIDs(cap4096,finalslot4095), all48phase-row comparisonsx128statecomponents/fullfinitevectors/IDs/reset, oldGDN/V/QKactualroutes remainrequired. Same source79/DLL2bafed, no newnativebuild orkernels. Guard1774ownedfiles/runtime/SDK/config-index/CMake/all34currentobjects/baselineDLL/privateSSD96GiB/ownednonce retirement. Parentstatusschema canonicalized fromexactpairedproofs. FailedCPU/model/route results canreplaywithoutadmission; successfuljobrequiresall25andfullnumericproof. Deployment427361B hashb37799f6ac8b36afbb47212a334459f73e07bb1bb3131e98144bf339e2eee3f4, source/candidateGPUinstall deferredtoexclusiveowner. Full4Kactualcapture/permutation-tail-repeats/MTP/805/HTTP/performance/strongestrefs/INT4/4B/overall2x remainopen.


2026-10-27 existingB4streamworker audit foundlegacyT4prefill/400projectionpool differsfromcurrentqualifiedT32/336joinedpool/BF16KVsource79; cannotreuseoldworker proofascurrentadmission. Newseparate defaultoff cuda_mtp_prefill16_b4_study module preparedCPU-only duringexclusivefull4Kjob; oldfactories/coreMTPprogram/policyguards untouched. TargetB4/cap4096/T32prefill/T4decode andallBF16KV/host/joined/retain/deferred/shared/register/flat flags required. OriginalB1T16temporaryswap/cache/memory/failure/reset/closehook body preserved;4-rowextraT16 engine/buffers, unchangedT4verifieddecode. Decodeworkspace256MiB remainsB4T4only, extraT16prefillusesallowed128MiB; no globalworkspace guard relaxation.27CPUcontrols pass(8newB4+19oldB1/provider), explicitduplicateddraftweights/memorynotGPUqualified. No deployment/nativecompile/MTPGPU/model/HTTP/timing; full4Kpairedpass andfutureprefix/mixedacceptance/reset/fullmodel/quality gates required. Currentremote1774source remainsfrozen.


2026-10-27 futureB4MTPprefill16 full4Kcapture+CPUgate draftedundeployedwhileexclusivegreedyfull4Kjobruns. Exactderivation preservesqualifiedtarget79/modelweights/vision/tokenizer/PagedPromptRows/corpus/snapshots; onlygreedyloopreplacedbyexplicitB4MTP3session(stops[]forignoreEOS, fullhead0/prefixreplay/T4decode256MiB/T16prefill128MiB). Actualgraphmetadata, target-draftslots, nonzeroproposals/verification and249MTPprefillcalls required; unchangedfull4Ktargetgate against8independentB1oracles stillrequiresall128IDs/all128state/finitevectors/48phase-rowcomparisons/reset.15CPUcontrols pass(4newMTPevidence+11existingfull4K), summedtargetcommits127perrow required. Doesnotqualifyallpartialprefix0-1-2-4/heterogeneousacceptance/tails/permutations/repeats/805quality/HTTP/speed; futureowner/launcher andcurrentgreedyfull4Kpairedpassrequired. No newGPUjob/nativecompile/deployment or measuredMTPspeed; remote1774sourcefrozen.


2026-10-27 source79 B4 full4K exactgate FAILED, pairedremote/localrawreplays identical proof 896e92de9686d938e22e4068fa697f21c086ea0635d92993b590d9ad940e8b57, archive 03fc1658efd1ec4b645ab952772732443be60a258affa09ec67211f9c6728ce7 (16308141B).25CPU,8freshsource60B1oracles/3source79B4groups text-image-mixed,3968prompt+128forcedIDs/cap4096/finalslots4095. Strictadmission=False, error=AssertionError('Full4K B4 complete state/raw vectors differ from independent B1 oracles'). Complete128component/physical-logicalhash/fullfinitehidden+normalizedvector/reset-prompt-final-reset/128ID gate neverrelaxed. ActualGDN/V/QKlive7attrs/M4-M16checks4MiB/all16eachroutes required. All1774ownedfiles/runtime/SDK/config-index/CMake/all34objects/baselineDLL restored andprivateSSDnonces retiredafteridle. Originalfailedjobs neverpromoted. See b4-full4k-exact-summary.json for exactpergroup/phase/row outcomes; changedconditioning canconfoundlatepersistentdifferences. No performancebenchmark oroverall2xclaim. B4MTPprefill16 CPU27andfullcontextMTPchecker15 preparedlocally butGPUunrun; prefix0-1-2-4/mixedacceptance/permutation/tail/repeats/805quality/HTTP/strongestrefs/INT4/4B/2xtarget remainopen.


2026-10-27 full4K failure follow-up: immutable source79failedfull4K captures show firstB1/B4 generatedID differences atzero-based text[118,52,52,91],image[120,60,84,41]; mixed[text0,image0,text1,image1]=[118,120,52,60]. Allreset/prefill/finalreset complete states andfinitevectors exact; finalchangedconditioning preventsuniquelayercause. Originalfull4K admission staysFAILED. Preparedseparate ordinarymodel first7decode-state diagnostic, fourfreshB1mixedreferences thenB4,3968prompt+8forcedIDs/cap4096; complete128components/rawfinitevectors aftereachstep1..7, exactsame-profileoriginalfailedparent reset/prefill/reset/first8IDs required. Noobservergraph/nativebuild/kernel/precision/head/tolerance changes.29CPU controls pass (old25+new4), explicitowned1782source/runtime/SDK/config-index/CMake/all34/baselineDLL/SSDnonce guards preserved. InitialnewCPUtest incorrectlycalledB4-only IDhelperforB1; correctednew B1validator andretainedoriginalB4helper. TerminalGPUdiagnostic result pending; no numerical/performanceadmission or2xclaim. Deployment 45f99fdb64967d2f32f8496068b12f4f36cec552a0f14514d9f0c021466e6e40.


2026-10-27 futureall64readonlydecodeobserver preparedlocally, undeployed/GPUunrun. OriginalForwardandRun inheritedunchanged, only64layerlist/offsetmetadata/privateconstructorbindingadded.8CPUoriginal+all64profile/row-suffixisolation/restoration controls, plus4newall64rawcomparison controls pass; sourcehashes/logs retainedseparately. Futureinterpretation strictlyrequirescompleteoriginalordinarysame-profile128componentphysical-logicalhash/fullfinitehidden+normalizedvector snapshotsafterEVERY1..7decodeandall8IDs exact, thenall64fullT4arrays atselectedsteps,5120active/15360inactiveworddifferences separated; layer63lastactiveword/NaN/missinglayer/changedprimarycontrols covered. No native/kernel/weight/precision/default/math changes. Originalfailedfull4K remainsFAILED, no GPU/model/MTP/805quality/HTTP/performance/2xadmission.


2026-10-27 selectedaffectedrow all64trace CPUpreparation.43prospectiveownerCPU integrationchecks pass, exactsource derivations retained. Runtimepackage stillrequires terminalordinaryprefixproofwithfirstnonexactstep/affectedrow; no remoteinstallationorGPUrun. AffectedselectedB1row andall4B4rows mustmatch everyordinarycomplete128componentphysical-logicalhash/fullfinitehidden-normalizedvector snapshotafter1..7+all8IDs before any64layerarray isinterpreted. OneB1rowreducesrepeatedprefill; allotherB1layeroutputs remainunobserved. Source60/source79nativeDLLs/weights/precision/kernel/math/defaults unchanged. Earliestobservedlayerboundarydoesnotuniquelyidentify causaloperator. Originalfull4K128ID failedgate neverpromoted; no MTP/quality/HTTP/timing/performance/2xadmission.


2026-10-27 ordinary full4K first7decode state diagnostic terminal: full4k_decode_prefix_diagnostic_replayed, diagnostic_completed=True, first_nonexact_step=None, pairedremote/localproof 75df1126bfa53d3b313a1e4b58d457c185685897705d4b1875bd3e8e0ecab5c4, rawarchive 7a1d91a1e3ac3483bcb1c6b10d76298ae2cd45cb66ee2b154dcbf6bb406bf26b.29CPU; unchangedsource60B1/source79B4 DLLs/graphmath/weights/precision/head,3968prompt+8forcedIDs/cap4096. Exactsame-profileoriginalfailedparentreset/prefill/reset/first8IDs required. See b4-full4k-decode-prefix-summary.json for all128persistentcomponentdifferences andfullfinitehidden/normbytes atsteps1..7. Driftcomponent isnotuniqueoperatorcausality. All1782ownedsource/runtime/SDK/config-index/CMake/all34/baselineDLL restored; ownedSSDnonces retiredafteridle. Original128IDfull4K staysFAILED, no MTP/quality/HTTP/perf/2xadmission. Privateall64observer CPU8prepared, GPUunrun; futureobserver requiresallseven same-profilecompleteprimarysnapshots+8IDs exactbefore layerinterpretation.


2026-10-27 nextselectedstep52 diagnostic preparedafterpairedprefix75df1126bfa53d3b313a1e4b58d457c185685897705d4b1875bd3e8e0ecab5c4 provedallfirst7steps exact. Originalfailedfull4Kmixedgroup IDdifferences[118,120,52,60] selectphysicalrow2 key ctx4096-s3-text-c8-r1. FreshoneB1selectedreference andwholeB4mixedgroup useoriginalsource60/source79nativeDLLs/modelgraphs/weights/precision/head,3968prompt+53predictions; ID0..51 consumedidentically, differingID52 remainsunconsumed. Capturecomplete128componentphysical-logicalhash/finitehidden-normalizedvectors after7and52. Reproduceallcurrentstep7B4rows+selectedB1againstqualifiedprefix, andall53same-profileIDs/reset/prefill/resetagainstoriginalfailedfull4K. Exactdecoderstate withdifferentunconsumedID isolates token-selectionpath; notuniqueinternalheadoperator.33CPU controls pass, no observer/kernel/nativebuild/default/tolerance changes. All1790ownedsource/runtime/SDK/config-index/CMake/all34/baselineDLL/SSDnonce guards preserved. GPUresult pending, otherB1step52states unobserved. Originalfull4K FAILED; no MTP/805quality/HTTP/performance/2xadmission. Earlierall64step1..7futuretrace staysundeployed becauseordinaryall7states exact, soitsfirstdriftprecondition isunsatisfied.


2026-10-27 static token-selectioncallsiteaudit retained:5sourcefiles matchbothsource60reference/source79candidate inventorypins. Decoder normalizeslast_hidden thenio.predict; nativeNVFP4head convertsinputF32toBF16, callsactivationquantizer andFP32compute cuBLASLt E2M1/BF16out usingactualM1/M4,4MiB, defaultheuristics (headdoesnotrequestseparateNVFP4downcandidate). Pinnedparallel-top1flag1 andvocab248320 selectsameparallelbatchsampler sourceimplementation forrows1/4. Noactivationkernel-body mapping orruntimeargmax-equivalence proof claimed. Matrixshape-dependent logits remainsunprovedcandidate pendingstep52 state/prediction result. No code/runtime/GPUchange orindependentnumericreplay duringlivecapture; no head/samplerbug orperformanceclaim.


2026-10-27 selectedrow step52 diagnostic terminal: selected_step52_diagnostic_replayed, diagnostic_completed=True, pairedremote/localproof 398fb2ab8795a8fb0edadcb0765227aa56b23feda34cc7e08976df76f9bcb5c2, rawarchive 1f5c177b1c8344073a0e6caa4e29e03e8892468f446093230337a827720af4f6.33CPU, unchangedsource60B1/source79B4 nativeDLLs/modelgraphs/weights/precision/head. FirstmixedbatchdifferingID52 row2selectedfromactualparentindices[118,120,52,60],3968prompt+53predictions;all52consumedIDs requiredidentical. See b4-full4k-selection52-summary.json for complete128componentphysical-logicalhash/fullfinitehidden-normalizedvectors after7and52, sameprofileparent53IDs/endpoints andwholecurrentB4step7parity. Observation={"status":"selected_step52_state_and_unconsumed_prediction_observed","selected_key":"ctx4096-s3-text-c8-r1","physical_row":2,"endpoint_comparisons":{"reset_state":{"physical_row":2,"slot_exact":true,"physical_hash_exact":true,"logical_hash_exact":true,"different_component_indices":[],"finite_raw_vector_bytes_exact":{"hidden":true,"normalized":true},"components_per_row":128,"numerical_tolerance_applied":false,"all_exact":true},"prefill_state":{"physical_row":2,"slot_exact":true,"physical_hash_exact":true,"logical_hash_exact":true,"different_component_indices":[],"finite_raw_vector_bytes_exact":{"hidden":true,"normalized":true},"components_per_row":128,"numerical_tolerance_applied":false,"all_exact":true},"final_reset":{"physical_row":2,"slot_exact":true,"physical_hash_exact":true,"logical_hash_exact":true,"different_component_indices":[],"finite_raw_vector_bytes_exact":{"hidden":true,"normalized":true},"components_per_row":128,"numerical_tolerance_applied":false,"all_exact":true}},"checkpoints":[{"step":7,"comparison":{"physical_row":2,"slot_exact":true,"physical_hash_exact":true,"logical_hash_exact":true,"different_component_indices":[],"finite_raw_vector_bytes_exact":{"hidden":true,"normalized":true},"components_per_row":128,"numerical_tolerance_applied":false,"all_exact":true}},{"step":52,"comparison":{"physical_row":2,"slot_exact":true,"physical_hash_exact":false,"logical_hash_exact":false,"different_component_indices":[70,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127],"finite_raw_vector_bytes_exact":{"hidden":false,"normalized":false},"components_per_row":128,"numerical_tolerance_applied":false,"all_exact":false}}],"all52_consumed_IDs_exact":true,"reference_prediction_52":264,"candidate_prediction_52":11352,"prediction_52_differs":true,"step52_state_hidden_normalized_exact":false,"token_selection_path_differs_with_exact_decoder_state":false,"original_full4k_admission_still_failed":true,"MTP_quality_HTTP_or_performance_qualified":false,"numerical_tolerance_applied":false,"scope":"One selectedrow: complete128statecomponents/physical-logicalhash/finitehidden-normalizedvectors after7and52steps. IDs0..51 consumed identically; differingID52 is unconsumed. Exactdecoderstate withdifferentID isolates token-selectionpath, notuniqueinternalheadoperator. OtherB1step52states unobserved; no full4K/quality/performanceadmission."}. Exactstate+differentunconsumedID isolates token-selectionpath, notuniqueinternalheadoperator. OtherB1step52states unobserved. All1790ownedsource/runtime/SDK/config-index/CMake/all34/baselineDLL restored; ownedSSDnonces retiredafteridle. Original128IDfull4K FAILED; noMTP/quality/HTTP/perf/2xadmission.


2026-10-27 key35rawcache52 diagnostic preparedfrompaired398fbselection52proof: consumed52tokens exact, step7fullstate exact, step52firstdifferentstoredcomponent70=K35, earlierstoredcomponents/V35exact; hidden/normdiffer. Thisdoesnotproveheadcorrectness oruniqueoperatorcause. Newprobe onlyreadscomplete8MiBkey35carriers twiceafterlastpredictionbefore reset viaexistingCUDAIO,2x4MiBblocks; expandedBF16SHA256mustmatchactualcompletecomponent70witness. All4heads/4096slots/256dims comparedbits includingprompt/inactive andsignedzero; rawslot scheduledwriterstep notproof ofactualmutationtime. Beforeinterpretation all53IDs/fivefullprimarycaptures forselectedB1/ALL4B4rows mustmatchordinaryselection52parent.38CPU controls pass (original33+new5), exactderivations/sourcehashes/logs retained. No graphobserver/kernel/nativebuild/modelmath/weights/precision/head/default/tolerance changes. All1798ownedsource/runtime/SDK/config-index/CMake/all34/baseline/SSDnonce guards retained. NumPyintegercache-replayversion printedintoseparateCPUlogs. GPUresultpending, original128IDfull4K FAILED; no MTP/quality/HTTP/timing/performance/2xadmission.


2026-10-27 separatedefaultoff K-M4x4 controlledintervention source85 preparedlocally, undeployed/nativeCPU-SDK-GPUunrun. Source79actualM16execution usesM4selectedalgorithm; proposednewK-onlyB4T4plugin preservesactualM4descriptors andexecutesfourorderedM4matmuls overphysicalrowfour-tokenblocks, liveM4SDKopaque/sevenconfigattrs/4MiB, weights/scalar/workspace/streamshared, stopsfirstfailure. ExactinputFP8quantizationouter16 unchanged; byteoffsetsgroup*4*5120 input/group*4*1024*2 BF16output. Source79->85 adds6privatehelpers andonlyloweringdispatch; alllegacyQK/V/GDN/NVFP4/oldwires/allCUDAkernels unchanged. NewflagGARNET_FP8_K_BATCH4_DECODE_M4X4 andnew20Bwire0x4b343434.5SOURCEONLYCPUcomposition/lineage/corruption/policyscope/wire/source-shape/failedparentchecks pass; notactualC++selector/SDKeligibility/numericequivalence orspeed proof. Existingkey35rawcacheGPUjob sourcefreeze/exclusivity retained. Rationale isactualcommoninputstep52firststoredK35diff; norm/rotary/othercauses notexcluded. Beforeanyuse requiresactualnativeCPU/API/runcontract/nativebuild/model128IDfull4K/MTP/quality/HTTP/performancevalidation. Originalfailedfull4K/cache62jobs stayFAILED, target2xunmet. Sourcebundle 822f01ed23b92542425ee21d1f9e3dda704f71d18e61866c268ba093b4bf486b (210891B).


2026-10-27 key35rawcache52 diagnostic terminal: key35_cache52_diagnostic_replayed, diagnostic_completed=True, pairedremote/localproof c497ad3bff282223ffd2f4f548004c9c9108a9619a97709acf1d771e9ef6b4ba, rawarchive ddbeaf66254886961f78c5df8f052a451d53d54c69db424346063b9ce3adaa38.38CPU; modelgraphs/nativeDLLs/weights/precision/head unchanged, onlyrawcachedataread. Complete8MiBkey35carriers twice-read/expandedSHAlinkedtoactualcomponent70; all4heads4096slots256dims BF16bits compared includingprompt/inactive. Fivecompleteprimarycaptures/53IDs forselectedB1/allfourB4 mustrepeatordinaryparent beforeinterpretation. Cachecomparison={"status":"all_key35_cache_words_compared","BF16_values":4194304,"different_BF16_values":1,"different_by_head":[1,0,0,0],"different_prompt_values":0,"different_decode_values":1,"different_inactive_values":0,"first_differing_slot":4012,"earliest_scheduled_decode_writer_step":45,"samples":[{"head":0,"slot":4012,"dimension":211,"reference_bits":16296,"candidate_bits":16295,"reference_value":1.3125,"candidate_value":1.3046875,"scheduled_decode_writer_step":45}],"all_bits_exact":false,"numerical_tolerance_applied":false,"scope":"All4heads/4096slots/256dimensions compared asBF16bits, includingprompt/inactive slots. Scheduledwriterstep derivesfromslot, notproof oftime oruniqueoperator thatchangedthevalue."}. Fullper-slotcounts/rawmanifest in b4-k35-cache52-summary.json. Scheduledwriterstep derivesfromslot, notprovenmutationtime oruniqueoperatorcausality. All1798ownedsource/runtime/SDK/config-index/CMake/all34/baselineDLL restored andownedSSDnonces retiredafteridle. Original128IDfull4K FAILED; noMTP/quality/HTTP/timing/performance/2xadmission.


2026-10-27 progress comparison v5: updated the report with the verified full-context early-decode pass (steps 1-7), failed 128-output batch-4 gate, and paired raw-cache diagnostic (one differing BF16 value out of 4,194,304 in key cache layer 35, head 0 / slot 4012 / dimension 211 at step 52). The proposed four-actual-M4-call correction has five source checks only; native compilation and GPU/model validation remain pending. Serving data are unchanged: image C8 decode 2.107x versus the faster ordinary reference, throughput 0.132955x, first-token speed 0.084248x. Native: three HTTP trials, physical batch 1; references: one ordinary trial each, up to eight running. Strongest repeated tuned/MTP references, matched INT4 and Qwen3.5-4B remain pending. Overall 2x decode AND throughput target is NOT ACHIEVED. No model, native binary or GPU job executed for this report. PNG/SVG visually reviewed; generator, input hashes, retention manifest and receipts saved under evidence/figures.


2026-10-27 K four-M4 actual native CPU contracts passed on CD-RTX5090, then independently replayed locally with byte-identical proof fb05df24875f635625b8a8d6cef3a134f60e7ac192a004e3b081fa8bae04c189. 13 source/SDK-evidence Python tests; actual C++ selector 1,209 cases (16 selected, 1,193 rejected); 232 actual factory/run cases under CPU SDK fixtures. Four ordered calls verified real host input offsets [0,20480,40960,61440] and BF16 output offsets [0,8192,16384,24576], actual M4 layouts, identical weights/scalar/beta/workspace/stream/opaque, every first-failure/throw group and strict runtime/shape/device/scale guards. Actual TensorRT plugin compiled against pinned headers/runtime: 49 malformed wires, 25 bad constructors, 46 descriptors, 9 formats, 13 legacy cross-wire cases rejected; 7 initialize calls / 5 CPU factories / zero GPU operators. All 1,827 guarded source files, runtime/SDK, config+index, CMake and all34 CUDA objects unchanged. This does not establish live cuBLASLt eligibility or numeric equivalence. Source85 production DLL, same-DLL flag-off/on controls, full128-ID/4K model, MTP, quality and HTTP tests remain required before performance admission. Attempt1 packaging failure occurred before any CPU subcommand/native compile: three missing evidence dependencies; partial archive/workflow/absence closure preserved, all method/compile-input dependencies added to initial guards in attempt2. Failure closure a43cae97968e0e076f962963ffe26faf756e9171070d8d4cb681a4f57626a326. Successful raw archive de1facae994314f32b75bad8119c9b68b339dafca815d7cda7004236f5897756 (3887680B). Original full4K and strict cache62 failures remain failures; serving figures unchanged and overall2x target unmet. Raw artifacts remain under the fixed study results folder and local outputs; methods, commands and proof receipts retained in branch evidence.


2026-10-27 source85 full native DLL build prepared after paired K-fourM4 actual CPU1209/232/plugin proof. Fresh remote inventory validates 1,827 owned files, 31 closed parent proofs/archives/workflows, runtime/SDK, config+index, CMake and all34 current CUDA objects; 85 production targets retain14 original files. Inventory1 stopped read-only on three older parent rows without status keys; failed command/manifest retained. Inventory2 adds those statuses only after original proof+archive SHA checks, then validates actual archived/live workflows; no source, runtime, CUDA object or gate changes. Native owner/reader exact derivations retain all prior controls, extend current Python controls83->88 with unchanged historical86, old Q/O profiles/factory/wire/initialize tests and native+embedded selector/tokenizer. Original3-object whitelist unchanged, all34 hashes recorded; no claim of strict cache62 admission or that all3 allowed objects are metadata-only. Default-off source85 differs from79 only in lowering dispatch plus6new K-only helpers; every CUDA kernel unchanged. Deployment 8711e516a9a162b6de4a0bcc25c1c6009394effcf8dec3fd8b207c016cca62a6 (286251B), source bundle 822f01ed23b92542425ee21d1f9e3dda704f71d18e61866c268ba093b4bf486b. Build/model unrun at preparation; exclusive owner will snapshot/restore85targets and baselineDLL. Production compile, same-DLL flagoff/on controls and full4K/MTP/quality/HTTP gates required before timing. Original full4K/cache62 failures preserved; overall2x unmet.


2026-10-27 source85 actual full native DLL build PASSED on CD-RTX5090, with independent remote/local byte-identical proof db32c32abad09ead9fbd7b4e57ca20480dee18a0860c806c16dc3762c8bbdb4d. Release /MT /sm120; candidate DLL cae02b4197db2ac0913fe2083184dac6eea1707683020daeafca90b78e1215e2 saved as build/bin/garnet-native-b4-k-m4x4-attempt1.dll. All88 current and86 historical Python controls passed on exact parent50; original Q/O278 profiles,182factory faults,Q/Owire,630+720initialize calls,native+embedded selector and tokenizer smoke gates retained and passed. NewK1209selector/232factory-pointer/plugin CPU parent required and verified. Baseline source85 target snapshot (14 original files), baseline DLL, all owned source/method/runtime/SDK, config+index and CMake restored. All34 CUDA object hashes recorded: the3changed objects are the original low-batch overlay/prefill128-attention/prefill128-commit whitelist; whitelist unchanged. This doesnot prove all3 metadata-only, and original strictcache62 remains FAILED. Newbuild post-state object hashes retained for subsequent model preflight; do not reuse stale pre-build hashes. Raw archive f931665ea80a26ab8d4a77396b104f26bb3d1bc7bbd515101013bb1743041655 (10509552B /884members), compiler and all15stage command logs/reports/source hashes retained. No live SDK matmul/GPU model/quality/HTTP/performance in this build; original full4K remains FAILED. Next: sameDLLflagoff parity against source79, newflag1 K-only intervention with actual live SDK route evidence and complete selected-row state/raw-cache checks, then all128-output/4K/MTP/quality/HTTP gates before timing. Strongest tuned/MTP references, matched INT4/Qwen3.5-4B and overall2x remain open; serving measurements unchanged.


2026-10-27 same-DLL source85 K-fourM4 off/on model experiment prepared after paired actual native build proof db32c32abad09ead9fbd7b4e57ca20480dee18a0860c806c16dc3762c8bbdb4d. Same candidate DLL cae02b4197db2ac0913fe2083184dac6eea1707683020daeafca90b78e1215e2 and source85 in both fresh physicalB4 mixed4 processes; only GARNET_FP8_K_BATCH4_DECODE_M4X4 changes 0->1. Immutable source60 B1/cache52/full4K/prefix oracles reused; no fresh B1 run, no full checkpoint shard hash claim (config+index pins). Flagoff must exactly reproduce source79 all4rows/53IDs/five complete 128component/physical-logicalhash/fullfinitevector captures/twice-read 8MiBkey35cache before flagon may start. Flagon requires all4 endpoint and step7 captures and53IDs against retained B1 oracles, selected row2 step52 complete state/vectors and all4,194,304 BF16 key values; only selected row has independent step52 complete-state/cache oracle. Actual newK live7attrs/twoM4checks/4MiB/all16initializations plus oldGDN/V and Q-only M16 required; flagoff newK absent with factory trace enabled. 47CPU controls passed (old38 +6route +3control/intervention), tests include other-row/lastcomponent/IDs/nonfinite/cachebit/binary/alias/oldroute corruptions; no tolerance or original gate relaxation. Current post-build all34 CUDA hashes used, not stale oldCPU snapshot. Guard1,849 files/42 closed parents, fixed C-drive cache96GiB/private-nonce retirement, source85/SDK/runtime/config-index/CMake/baselineDLL restoration. Package 9af6a9a760e538ad3f0bf0c1a85ac6d004eb2a270e54713b9cfc6605e91fde4c (209675B), methods/probe exact derivations/CPU logs retained. GPU unrun at preparation; not full128-output/4K/MTP/quality/HTTP/performance qualification. Original full4K/cache62 failures and overall2x unmet status preserved.


2026-10-27 same-DLL source85 K-fourM4 actual model intervention FAILED the selected step52 exactness gate, with remote/local byte-identical proof ff584e7ff834b7fa4697af3686a1ae5edeecce9fc3442782054c8dcaa515bad7. All47 CPU controls passed; both fresh GPU phase captures completed successfully. Flagoff exact parity against retained source79: all4rows/53IDs/five complete128component physical-logical hashes/fullfinite hidden-normalized vectors and8MiBrawkey35cache. Flagon actual newK liveSDK7attrs/twoM4checks/4MiB/all16initializations and unchangedGDN/V/Q-only routes replayed. Nonetheless direct off/on comparison is exact at ALL five captured primary checkpoints for all4rows, all53IDs and complete selected8MiBcache. Against source60 B1, same single BF16 difference persists (1/4,194,304): layer35 K/head0/slot4012/dim211, reference0x3fa8=1.3125, candidate0x3fa7=1.3046875; adjacent BF16 codes, not a one-bit difference. ID52 remains11352 versus264, earlier52consumed IDs exact, other3rows first53IDs exact. Selected final components70 and72..127 plus hidden/norm still differ. This K-only actual-M4 execution-shape intervention doesnot repair this case; doesnot prove all intermediate tensors equal or a unique cause. Scheduled writer45 still inferred fromslot, not proven mutation time. Next diagnostic should read this existing carrier coordinate after each ordinary decode step, require the same primary/fullcache parity, then localize upstream hidden/projection/normalization operands before another arithmetic change. Original step<=7 all-layer observer package precondition remains false; do not silently relax it or launch full128/MTP admission from this failure. All1,849 owned files/runtime/SDK/config+index/CMake/currentall34/baselineDLL restored; ownedSSDnonces retired afteridle. Raw archive 8ef4e3e367ade4b30bb34824afee2f0c7603421a3be553f02b3db82a2c47d22e (30071535B /1590members). Both raw captures, route evidence, resource logs, failures, reader and paper metrics retained. RemotePython3.12.3/NumPy2.4.4 and local3.12.10/1.26.4 replays identical. No new serving benchmark or performance gain; full4K/cache62 remainFAILED andoverall2x unmet.


2026-10-27 ordinary key35 coordinate-trajectory diagnostic prepared after negative sameDLL K-fourM4 proof ff584e7ff834b7fa4697af3686a1ae5edeecce9fc3442782054c8dcaa515bad7. Read-only existing carrier at layer35/head0/slot4012/dim211, including neighbor210: physical key word513641/stateword56407657. Reuse oneBx1 scratch allocation, read same4bytes twice after prefill and each ordinary decoder step1..52; no graph observer, model math, kernel, native binary or precision changes. Fresh source60 B1 selected text and source85 flagoff B4 mixed4,3968prompt+53IDs/52commits. Require complete53ID/fiveprimary 128component physical-logicalhash/fullfinitevector and all8MiBcache parity against same-profile retained B1cache52/B4source85off parents BEFORE interpreting any sample; step52 carrier links to complete cache bytes/hash. Derive first observed selected-coordinate and whole-carrier difference separately; scheduledwriter45 is not assumed mutationtime or first global decoder drift.52CPU controls passed (inherited47+5new offset/row/repeatedread/ordered53steps/nonfinite/fullcachelink/timing tests).Guard1,861files/43closedparents/currentall34/runtime/SDK/config+index/CMake/baselineDLL, fixed C SSD>=96GiB/private-nonce retirement. Exact probe/launcher derivations and CPU source hashes/logs retained. Package 8bc3b3c7d7c291e7236a4dc2bd2d6e35e9de116b5cc06ba46e4c0dd933b5714c (287104B). GPU unrun at preparation. Earlier all-layer step<=7 observer protocol remains inapplicable and unrun. Full4K/cache62/K-fourM4 failures remain failures; no model/MTP/quality/HTTP/performance admission or new2x claim.


2026-10-27 static key35 callsite audit during exclusive coordinate model job; no GPU/model numeric replay or remote source changes. Actual source60/source85 qwen35_lowering.h byte-identical SHA256 87698cfd3789816b0f5186791f11dcc8e41f3112502e18db434ad917bdafd439; use frozen bundle header, since worktree root header is a different baseline instance. Header lines and body hashes retained. Key path: normalized input -> K projection -> per-head FP32 RMS normalization (mean square + eps, sqrt, divide, offset scale) -> leading64 RoPE and unmodified suffix -> float key -> BF16 cache round-to-nearest. Pinned head_dim256/partial0.25 gives64; coordinate211 lies outside intended rotary prefix. Both low-batch/generic BF16 commit sources are pinned baseline files and use __float2bfloat16_rn(target[source]); they are not members of source bundles, so audited against guarded source hashes instead. This does not establish actual operand equality, TensorRT tactic math or unique cause. Coordinate model reference capture completed, B4 still running; no new numerical result or benchmark.


2026-10-27 ordinary key35 coordinate timeline and progress comparison v6: fresh source60 B1 and source85 flagoff B4 ordinary runs completed. Two identical reads per timepoint of the existing four-byte carrier, after prefill and each completed decode step 1..52. Before interpreting the trace, all53 IDs/five complete128-component captures/full finite hidden-normalized vectors and full8MiBcache reproduced each same-profile immutable parent. Actual coordinate layer35/head0/slot4012/dim211 is zero and equal through step44; first observed difference after step45: B1 code16296=1.3125, B4 code16295=1.3046875, persisting through52. This establishes the mutation time for this sampled coordinate only, not the earliest global decoder divergence or a unique causal operator. Scheduled write45 agrees with observed samples. All52 CPU controls passed; remote/local raw replays byte-identical, proof 614f9a17b2f8edd2059331584a212fd1369a458b7eecdd6fc87ff6bffa5c0639. Raw archive d2b8642ec7f1faaf23859a418431216c6d28a3383606921d90e231182ffc34a5 (29825783B). All1861 owned files/runtime/SDK/config+index/CMake/current34objects/baselineDLL restored and owned SSD nonces retired. No native/graph/kernel/precision changes in the diagnostic. Historical full4K/cache62 gates and same-DLL four-actual-M4 intervention remain FAILED. Next inspect upstream operands at step45 with primary parity before interpreting observations; old step<=7 observer precondition stays false. Updated v6 PNG/SVG plus coordinate PNG/SVG/CSV, generators, hashes, resource logs and paired proofs retained. No new serving benchmark. ImageC8: Garnet decode89.7712 tok/s/request, total29.9110 tok/s, TTFT17.6437s; ordinary vLLM42.4598/224.9710/1.48645s and SGLang42.6058/146.5366/1.60399s. Faster-reference decode2.107x but throughput0.132955x, first-token speed0.084248x. Native3trials physicalbatch1 queues clients, references1trial each up to8running; cross-engine full-output equivalence unproven. Strongest repeated tuned/MTP references, matched INT4 and4B pending. Overall2x goal unmet.


2026-10-27 separate step44/45 all-layer boundary diagnostic prepared after ordinary coordinate trajectory proof614f9a17b2f8edd2059331584a212fd1369a458b7eecdd6fc87ff6bffa5c0639. The sampled layer35K value first differs at45; responsible upstream operation remains unproved. Append all64-layer completeT4hidden outputs only to private decode4 graphs, collect at completed44/45 for selected B1text and B4mixed physicalrow2. Keep53IDs/52ordinarycommits and all original five complete 128component state snapshots/finitevectors/full8MiBcache plus53coordinate samples. Require exact same-profile ordinary parent parity BEFORE interpreting observer outputs; added outputs may alter TRT tactics/fusions despite original arithmetic AST equivalence. Frozen source60/85 arithmetic AST exact after removing only three observer suffix operations; step/chunk source proves one-token activecolumn0. 66CPU controls passed: inherited52 plus8originalobserver/all64suffix/binding/profile and6newprimary-first/timeline/ID/selectedrow/shape/hash/nonfinite/active-inactive checks. No native/kernel/precision edits, no relaxation of old step<=7 protocol. Guard1877files/44closedparents/current34objects/runtime/SDK/config+index/CMake/baselineDLL; fixed C SSD>=96GiB and owned nonce retirement. Package 2b7f66b2daa16f88802e3f2114edc7b8fd94b2e6a0fb854b15e6102efec04fdc (311838B). GPU unrun at preparation; full4K/cache62/K-fourM4 failures preserved, no MTP/quality/HTTP/performance admission or2x claim.


2026-10-27 step44/45 all-layer observer attempt1 FAILED at native graph construction in reference phase: Qwen3.5 variadic packed concatenation profile invalid. 66CPU passed; no layer observations or model equivalence result produced, candidate phase never launched. Native existing parts guard allows at most128 additional parts. Retained-input model primary has113 total parts (hidden1,state64,linear retained payload48); appending 64observations inone operation exceedsguard. Keep originalguard unchanged. Next private observer should separately join113primary and64observations then join2, preserving exact primary-prefix/suffix order and all arithmetic; same-profile primary/cache/timeline admission still required before interpreting anything. Remote/local raw failure replays byte-identical proof fbdd382d2e9e8bb02cf36d848c003d6026fa44011a723ed294dd225398fc1424, archive 1937cd41b7b2affdce93161b423749474923e7730949885cba5433ed33162516 (15482594B). All1877ownedfiles/runtime/SDK/config+index/CMake/current34objects/baselineDLL restored; fixedCSSDnonces retired. Original unused installer-v1 referenced the inherited preparation filename; corrected-v2 only fixedthatfilename before execution and installed the unchangedpackage successfully. Both installer sources/hashes retained. Full4K/cache62/K-fourM4 failures remain failures; no new benchmark or2xclaim.


2026-10-27 bounded step44/45 observer attempt2 prepared after retained attempt1 nativeconcat failure fbdd382d2e9e8bb02cf36d848c003d6026fa44011a723ed294dd225398fc1424. New private BoundedStep45BoundaryProgram keeps inherited64layer buffers/run/profile/binding contracts, changes only final joins: original113parts (1hidden+64state+48GDNpayload),64observations,then2. Originalnative128extra-part guard untouched, same dtype/precision/nativeDLLs/weights. Frozen source85 flat_concat and actual new_forward executed under faithful native-limit CPUstub; all primary and observation spans/order preserved forB1/B4, old177partconcat rejected. Striponly boundedjoin construction and original arithmetic AST equals prior observer, previously proven against source60/85. Freeze actual source85 cuda_packed_program text for AST-only CPU tests because restoredremote baseline differs; never assume root program is frozen candidate.69CPU passed (all66retained+3bounded). Require all53IDs/fivefull128componentcapturedstates/finitevectors/full8MiB cache/all53coordinate samples same-profile ordinary parity before any layer interpretation; extra outputs may alter TRT tactics/fusions. Guard1892files/45closedparents/current34objects/runtime/SDK/config+index/CMake/baselineDLL/fixedCSSD96GiB and nonce retirement. Package 15f094610e77330c20ce9a709bf167fa8917f1aa433ee4fe6aeb098fcdba3611 (301150B). GPUunrun at preparation; original all-layer step<=7 protocol remainsinapplicable. No full4K/MTP/quality/HTTP/performance qualification or2xclaim.


2026-10-27 bounded step44/45 all-layer observer attempt2 PASSED its diagnostic parity and independent remote/local raw replays, proof 3458627f8e735a84c178f585a55e2fe8313f445ec9fe19288550d397ab099f2c, archive 2f54cffd88074e4dfaa324dcd7dfcb126cf7efb0861a0898cc6c6a4e2ed28ac7 (41151451B). 69CPU passed onbothhosts. Original nativeconcat guard unchanged; bounded 113primary/64observations/2final joins compile andrun under source60B1 and source85flagoffB4. Before interpreting observations, all53IDs/five complete 128component physical-logical checkpoint hashes/fullfinite hidden-norm vectors/full8MiBkey35cache and all53coordinate samples reproduce each same-profile ordinary parent exactly. At completed step44 all64 complete T4layeroutputs exact. Atstep45 layers0..34 completeT4outputs exact; first differing observed active andcompleteT4output islayer35. Activecolumn0 is source-proved; allrawinactive words also retained. This narrows observation tolayer35 onthisstep, NOTuniqueoperatorcausality or earliest globaldecoder drift. Togetherwith earlier singleBF16K35/head0/slot4012/dim211difference at45 and failed actualK-fourM4 intervention, nextinspect actual internal layer35 inputnormalization/Kprojection/per-headKnorm/postRoPE operands with sameprimary/cache/timeline parity. Actual source already has full-chunk witness hooks but their existing guards are B1/specificprofiles: preserve them and use a separate explicit defaultoff decode35 protocol if native instrumentation is needed. No blind shape or precision change. All1892ownedfiles/runtime/SDK/config+index/CMake/current34objects/baselineDLL restored, ownedSSDnonces retired;45closed parents verified (postprocess ledger description inherited44 is atext typo only). PNG/SVG/CSV and generator/receipts/rawlogs/resources/proofs retained forpaper. Thisisdiagnostic, no full128/B4MTP/quality/HTTP/performance admission. Full4K/cache62/K-fourM4/observer1 failures stay FAILED andoverall2xgoal unmet. Staticqueueaudit also found native INT4 worker currentlybreaks atEOS evenfor bodyignore_eos=True, whilefixed4K reference requests ignoreEOS; reconcile andqualify explicit128-output policy beforematchedINT4timing. ExistingINT4/4B shortpilots donot satisfy matched4K requirement.


2026-10-27 source86 separate decode35 internal operand witness prepared after paired all64layer proof3458627f8e735a84c178f585a55e2fe8313f445ec9fe19288550d397ab099f2c (all64exact at44; first35 at45,26activeF32words). Modify only frozen source85 qwen35_lowering.h/trt_builder.cpp plus newguardheader, defaultoff witness_full_decode35_stages integer0/1 AND GARNET_DECODE35_STAGE_WITNESS=1. Require exact sharedmixed F32 joined/fused layer35 B1lowbatch orB4BF16/T4/4096 layout,heads24/KV4/d256/rotary64/sections11-11-10/theta1e7/eps1e-6 physicalwidth106823680/Koffset55894016/Voffset57991168. Exclude existing diagnostics, oldchunkwitness, nonBF16overlay/reference/prefill128 profiles. Capture seven existing tensorpointers without recalculation: maskedx, inputnorm,Qgate,Kproj,Vproj,KnormpreRoPE,KpostRoPE. AllFP32,widths 20480/20480/49152/4096/4096/4096/4096, sum106496words perphysicalrow. Validate allshapeproducts/overflow/batch/type before packing after originalprefix and emitting distinct nativepack trace. Existingguards, allCUDA,allmatrixmath and oldcounters byte-identical after removing newobservation-only additions; no newpubliccounter. Five local source composition/reversechange/defaultoff/type/operand/size/corruption checks pass inledger. Sourcebundle48f323c025cce22577072bc7c1819b62ade5ad08d37e94b2146ded75dac0e43f (86files). Actualnew/oldheader CPU109case MSVC probe isprepared butUNRUN atthisentry; fullDLL build/liveTensorRT/model/GPU unrun. Package c7b62924227250670f6227a423b7f5e0abed54b58e64fe4a4974c6aa81838128 (90834B). CPUowner guards/restores baseline1892files/runtime/SDK/config+index/CMake/all34 withoutproductionmutations. Before interpreting future internaloutputs, require original53IDs/fivefullstatevectors/fullcache/coordinate timeline plus bothsteps all64layer arrays sameprofileparent parity; extraoutputs may alter tactics/fusions. No cause/performance qualification. Full4K/cache62/K-fourM4/observer1 failures and2x goalopen.


2026-10-27 source86 decode35 native CPU profile contracts PASSED on CD-RTX5090. Actual MSVC C++17 /MT /W4 /WX compiles exact newguardheader and pinned original source85 fullchunkprofile header;109cases:2newvalid (B1lowbatch/B4BF16 T4layer35),103invalidflag/precision/binding/geometry/mode/finite/neighborfloat/offset variants rejected,2oldB1T32/T128layer3 valid,2oldguard rejects newprofiles. Zero failures. Thisvalidates only actualheadercontracts, NOTliveTensorRT typeparsing/stagepacking, fullDLL compilation, numerical tensors, modelquality orperformance. Five local source controls separatelypassed. Remote/local rawreaders byte-identical proof dde45b74d817426a54720f31448e70023c2e9252fad0bca1ddd82f6253108202, archive 6a90286fb95aec696ea8e56af98d02b3d3460ea3574601eedbc13298ea76c296 (487201B /14members). All1892 baseline files,guardedruntime/SDK/config+index/CMake/all34CUDA objects unchanged; no production source/DLL mutations. Allsix CPUinput hashes/actualbuild+contractcommandledgers/compiledEXEhash/sourcebeforeafter and rawworkflow retained. Localreader only hashes EXEbytes; neverloads or executesdownloadednativecode. Nextfullsource86nativebuild with all original88current+86historical controls plusnewsource5/header109proof, original three-objectwhitelist andcurrentpostbuildhashes, thenseparate modelobserver retaining original53IDs/fivestatevectorcaptures/fullcache/coordinate/all64layer44-45parity beforeinternaloperand interpretation. Originalfull4K/cache62/K-fourM4/observer1 failures preserved; strongest tuned/MTPrefs,matchedINT4/4B andoverall2xgoal remainopen.


2026-10-27 source86 full native DLL build protocol prepared after actual109headercontracts paired proofdde45b74d817426a54720f31448e70023c2e9252fad0bca1ddd82f6253108202. Fresh remote readonlyinventory PASSED:1892current guarded baselinefiles, 86overlaytargets (14existing originals), guardedruntime/SDK/config+index/CMake/all34currentCUDAobjects and47closedparents exact;Dfree561.265GiB. Nativebuild owner/reader areexactderivations of qualified85protocol, retaining all old93current(88+source5)+86historical Python checks and Q/O278/factory182/wire/630+720initialize/selector/tokenizer gates. New109profile proof/headerSHA/source86bundle linkage required. Snapshot/install/restore86targets; preserve baselineDLL and savecandidate separately asgarnet-native-decode35-stage-attempt1.dll. Record all34 CUDAhashes alsoafterfailedbuild andvalidate original3objectwhitelist; no claim ofall34bitwiseidentity ororiginalstrictcache62 admission. Inputinventory archive 8ce014e57c8fa72436bf14ae70cd9e94c32e7867e5a71b85de2005bbc28f0533 (192608B). Localprep attempt1 failedbeforeanydeployment becauseit compared seed status to intentionally differentverifiedinventory status. Original script/failure retained; fixexcludes onlystatus fromcomparison and still requires exact verifiedstatus plusallother seedfields. No native orGPUexecution inthatfailure. Newdeployment 6c6f6e1da36976fa8cd83e45ca30947c58872dc95d409f4a5df45282b24e6e2c (301869B),source48f323c025cce22577072bc7c1819b62ade5ad08d37e94b2146ded75dac0e43f. Fullbuild UNRUN atthisentry. Stagekeyword/policy remainsdefaultoff; newbuild cannotqualify liveTensorRT stagepacking, numerical tensors, modelquality/HTTP orspeed. Laterobservermustretain complete53IDs/fiveprimarycaptures/full8MiBcache/coordinate/all64layer44-45 parity beforeinternaloperand interpretation. Historicalfailuresand2xgoalopen.


2026-10-27 source86 fullnative DLL build PASSED and independent remote/local rawreplays byte-identical, proof 5a43ee92cbf045386d5b0de32d65839380a555cf0e1cdab995637ac4d202cce7, archive d9753a8221759be388ea3b9d22ec36020e46f6fd29939bc65236701f7fabc3d9 (11007274B /914members). Release/MT/sm120 candidatebuild/bin/garnet-native-decode35-stage-attempt1.dll SHA 27b8c8b97bbe43991c5a75e8d5c753e989f2da559d59f0c05b22c1690869fe91.93current Python controls (88original+5newsource)+86historical, originalQ/O278profiles/factory182/wire/630+720initialize/native+embeddedselector/tokenizer gates retained andpassed. New109actual CPUprofile/header proof required andverified. All86target source snapshots/restoration, baselineDLL/runtime/guardedSDK/config+index/CMake/method pins restored. All34CUDAhashes recordedbefore/built/closure; built equalsclosure. Changed onlyoriginal3objectwhitelist (lowbatchoverlay,prefill128attention,prefill128commit); NOTproofall3 metadata-only andNOToriginalstrictcache62 admission. Use theseNEWpostbuild hashes fornextmodelpreparation. No liveTensorRTstagepacking or GPUmodel/matmul/quality/HTTP/performance frombuild. Rawsource/SDK/commandreports and postbuildpins retained. Separate private PythonDecode35OperandProgram nowprepared: exactfrozenchunk math with only nativewitnessmarkerfor FA35, originalprimaryprefix/all64layerboundaries retained, sevenF32 operand suffixes total106496words perrow. Five symbolicCPUchecks passed: B1/B4 row/suffix/spans/prefix, oldprofile rejects, bindingrestoration/prefillunchanged, actualDSLmetadataforwarding and exactoriginalchunk/parentforward derivations. No native/GPU import inlocaltests. Program requires explicitnativepolicy; modelUNRUN. Nextcapture selectedrowat44/45 under source86DLL, require original53IDs/fivecomplete statevectorcaptures/full8MiBcache/coordinate ANDall64layer44-45 arrays exact against qualified same-profileparents before interpreting newinternal operands. Preserve declaredDLLchange explicitly, donotwaiveoldmodelgate; link rotatedK BF16rounding toactualcacheword. Originalfull4K/cache62/K-fourM4/observer1 failuresstayFAILED; strongesttuned/MTPrefs,matchedINT4/4B,overall2xopen.


2026-10-27 same-DLL seven-operand diagnostic prepared (GPU unrun). All 81 CPU tests passed: inherited 69, private frontend 5, analysis 7. The symbolic source test now reads the exact frozen cuda_chunk source used by source86, rather than the different restored checkout; assertions and the frontend implementation are unchanged, and the previous test is retained in commit e06cbaffe. Fixed source bundle 48f323c025cce22577072bc7c1819b62ade5ad08d37e94b2146ded75dac0e43f and DLL 27b8c8b97bbe43991c5a75e8d5c753e989f2da559d59f0c05b22c1690869fe91 are shared by four fresh processes: off-reference, off-candidate, on-reference, on-candidate. Off controls explicitly declare their DLL transition from historical source60/source85 but must reproduce all 53 IDs, five complete 128-component captures and finite vectors, full 8 MiB selected cache, 53 coordinate points, and all 64 complete T4 layer outputs at steps 44 and 45. On controls must match same-DLL off before interpreting any new operand. Trace must show one actual native stage pack only when on. Check seven F32 operands, link masked input to layer34, require unchanged unrotated RoPE suffix, and independently round active K to actual BF16 cache slots 4011/4012. Guard 1923 files, 48 closed parents and all 34 latest post-build object hashes; restore source, baseline DLL, runtime, SDK, config/index and CMake. Reuse fixed C cache with >=96 GiB free and retire only owned new nonce entries. Deployment 4875aa0355f83121ad316e70bad5a488c3c4b80849aa624e70bfa7e62792b305 (232240 bytes). No full4K, MTP, quality, HTTP, speed or 2x admission. Existing failures and the complete final objective remain open.


2026-10-27 local output-policy fairness fix prepared while the remote decode35 model diagnostic remains exclusively running and its source frozen. Existing INT4 and 4B workers accepted ignore_eos in requests but stopped at EOS anyway, incompatible with the reference forced128 policy. Native INT4 now honors a strictly boolean request ignore_eos. Model.infer/infer_batch and the 4B worker gain the same default-false option, retain all raw sampled IDs when true, hide stop markers from text, and report length at the limit. Default EOS termination, per-row masking and positional on_token callbacks remain supported. Speculative ignore_eos is explicitly rejected because that path is not qualified for this API. Ten CPU tests execute the actual Model method AST and both complete worker scripts with token-only stubs; forced128 with EOS at first and last positions, IDs/callbacks/finish reasons/text, batched masking and invalid flags pass. No Garnet DLL, numeric ML replay or GPU loaded locally; no remote files touched and no new or revised benchmark claim. Native/model/HTTP checks and matched INT4/4B 4K runs remain required before timing admission.


2026-10-27 same-DLL seven-operand diagnostic terminal outcome: same_DLL_decode35_operands_failure_replayed, exit 1. Independent remote/local raw replay proofs are byte-identical: 031df5a8fbd8888391d6e70ba4ac337df43b1fb6c6c4d5a5b16b61c5513ea5cd. Raw archive 42ce8c647ec46f618f0b65cc420a50f45fbe3eb17f7b28aab9d52a78610106d7 (67984931 bytes). All 81 CPU controls passed. Completed primary controls: off-reference, off-candidate, on-reference. Error: AssertionError('Observer or DLL changed complete primary state'). Operand summary: null. Source, baseline DLL, runtime, SDK, config/index, CMake and all34 latest CUDA object hashes restored; private owned cache nonces retired. Actual raw phase logs, workflow, manifests, resources, reports, replay and helper sources retained. Failed or incomplete controls do not admit operand interpretation. Full4K failure remains; no quality/MTP/HTTP/performance qualification or 2x claim.


2026-10-27 failed seven-output observer independently audited without opening its operand blobs. IDs remain exact for all four rows. Reset, prefill, step7 and final-reset logical states are exact; step52 differs in 56 component hashes, only physical row3, first component70/K35. This is an observer perturbation, not original-model diagnosis. Preserve the full all-row failed gate; do not reinterpret new operands. Next private Decode35EndpointProgram omits only K_normalized_pre_RoPE from final network outputs while retaining the exact native seven-stage fusion function, metadata, guard, source86 bundle and DLL. Advance through all seven native offsets, including the omitted 4096 words, so K_post_RoPE reads its original position; return six endpoints contiguously (102400 words). Hypothesis: exposing the internal normalization intermediate changed TensorRT fusion. Pruning, causal explanation and numeric parity remain UNPROVED. All87 CPU controls pass (81 unchanged plus6 topology, binding, native identity, source reversal and actual view-offset checks). Two read-only derivation assertions failed on mixed LF/CRLF before any target write; scripts retained, exact matching corrected and source line endings preserved. New model protocol can reuse immutable same-DLL ordinary off captures from the paired failed-parent archive, provided source86/weights config-index/corpus/flags/runtime/SDK/CMake/all34 objects remain exact; two fresh six-endpoint on processes must pass all53IDs, five all-row128-component states and vectors, fullcache, coordinate timeline and both44/45 all64 outputs before interpretation. Native trace still reports7; do not claim a new6 native hook or pre-RoPE suffix check. No native build, model run, performance or quality admission for this variant yet. Full objective and 2x target remain open.


2026-10-27 six-endpoint two-process model protocol prepared. All97 CPU tests pass: original81 retained,6 endpoint topology,7 endpoint primary-first/metadata/finite/cache-rounding checks,3 immutable-off identity/failure rejection checks. New endpoint analyzer has no pre-RoPE suffix assertion or claim because that tensor is omitted; the native7-stage trace and 106496-word hook guard remain unchanged, final six outputs102400 words. Two local preparation failures retained: ambiguous row_words replacement first created only a new analysis file before stopping; then a copied seven-file guard rejected the six-output fixture. Invalid source/record/generator versions retained; corrected only the new observer metadata and six-file guard. All original seven-output code/tests unchanged. Owner reuses only both immutable ordinary off captures from failed seven-output job after full source86/DLL/flags/config-index/corpus/selectedkey/runtime/SDK/CMake/all34 identity, and replays their original historicalsource60/85 controls. Two fresh six-output on processes must still reproduce all53IDs, five all-row128-component states and complete finite vectors, whole8MiB selected cache, all53 coordinate points and both44/45 all64 completeT4 outputs. Independent raw reader reproduces primary and endpoint-analysis failures without interpreting invalid results. Guard1948 files/49 closed parents; snapshot/restore86 source targets and baselineDLL, fixedSSD cache>=96GiB and ownednonce cleanup. Deployment 1f7b5a73185768e60bf48f4dfd800cb42ae65b2f833f46ef8be3d03bd2eb2fde (251793 bytes), DLL27b8c8b97bbe43991c5a75e8d5c753e989f2da559d59f0c05b22c1690869fe91. Fresh model processes UNRUN at this entry. No native build or math/precision change. Failed seven-output observer remains failed, and the full correctness/quality/MTP/HTTP/strongest-reference/INT4/4B/2x objective is unchanged.


2026-10-27 same-DLL six-endpoint diagnostic terminal outcome: same_DLL_decode35_endpoints_failure_replayed, exit 1. Independent remote/local raw replay proofs are byte-identical: f4cd3dbaa531eb38afb54b00ac94cea0467838a3bd33622687d5304dfdd30836. Raw archive 39a35e3b3d2c77946e4b960e4d279ed7138944e05b2487bd1631758e8343067a (42038263 bytes). All 97 CPU controls passed. Completed primary controls: off-reference, off-candidate, on-reference. Two fresh on processes; ordinary off captures reused under exact source/runtime pins. Native trace stayed7, final outputs6; pre-RoPE suffix not observed or claimed. Error: AssertionError('Observer or DLL changed complete primary state'). Operand summary: null. Source, baseline DLL, runtime, SDK, config/index, CMake and all34 latest CUDA object hashes restored; private owned cache nonces retired. Actual raw phase logs, workflow, manifests, resources, reports, replay and helper sources retained. Failed or incomplete controls do not admit operand interpretation. Full4K failure remains; no quality/MTP/HTTP/performance qualification or 2x claim.


2026-10-27 six-output observer failure replay is paired byte-identical (f4cd3dbaa531eb38afb54b00ac94cea0467838a3bd33622687d5304dfdd30836; archive39a35e3b3d2c77946e4b960e4d279ed7138944e05b2487bd1631758e8343067a, 42038263 bytes). Primary-only audit opened no operand observations: all IDs remain exact; reset/prefill/step7/final-reset logical states exact against ordinary off; step52 has56 changed components only row3. Six and seven observers have identical complete primary logical states, finite vectors, selected cache/timeline and all64 completeT4 layer outputs at44/45. This proves no observed numeric benefit from final selection omission, not identical compiled kernels or a root cause. Stop final-output observer variants; do not interpret their operands. Next source87 is a controlled K-only normalization shape intervention over immutable source86: same F32 RMS equation/epsilon/scale, split B4 projectedK along physical rows, apply existing RMS to four [1,4,4,256] slices and concatenate. Default false retains original single RMS call. Separate explicit env+integer-metadata policy rejects wrong profiles and all native observer modes; exact B4T4 mixed weights and sixteen FA layers with independent BF16 cache-offset guards. Trace counts RMS subgraphs, not executed kernel launches. No dtype, weights, RoPE, cache format, GDN, V or old native counter changes; all CUDA bytes unchanged relative to source86. Five CPU controls pass (source reversal/defaultoff/independent config offsets/high-precision equation), but native header compilation, full DLL build and model repair remain UNRUN. Source87 package fcffdf2455d188ca555e561ddb16d337d5f172c00b4dafdc7b90e2634651f190 (214290 bytes), derivatione50b67d6b044326849c9a6145c1e8c9a9c2cfd2b7c86a9b068ba47dee50d2f8c. This is a testable hypothesis based on the original K-cache mismatch and failed four-M4 projection intervention, not confounded new operands. Implementation plan retains full4K/MTP/quality/HTTP, strongest repeated reference, INT4/4B and2x requirements. Future prefill research cites FLA chunk_gated_delta_rule and https://arxiv.org/abs/2412.06464; no GDN performance claim. Local GDN source provenance must be checked against actual remote build before attribution. No new serving measurements.


2026-10-27 source87 actual native CPU header protocol prepared:184 cases (17 new valid,161 new rejected,4 retained original valid,2 original rejections). Covers all16FA layer offsets and every neighboring key/value/state span, strict env/mixed/F32/shared/joined policy, exclusion of native output observers, malformed layers/ranks/shapes, finite settings and cache modes. Include actual source87 rowwise guard and unchanged seven-stage/fullchunk headers. MSVC /Bv /MT /W4 /WX compile and CPU execution happen only on CD-RTX5090, no local EXE execution. Independent raw reader verifies input hashes/parent closure/source87/ledgers and parses PE imports as bytes to reject CUDA/Garnet linkage. Guard1948 baseline files, runtime/SDK/config-index/CMake/all34 CUDA objects unchanged; no production source/DLL modifications. Parent six-observer failed but fully restored; failure remains failed. Packet 3a9ef87d7847336578176cd3c7d1ac27356e56e032594b26e1f4754f655790c1 (102388 bytes), source87 fcffdf2455d188ca555e561ddb16d337d5f172c00b4dafdc7b90e2634651f190. Native guard run UNRUN at this entry. Full DLL, rowwise TensorRT path, model repair and performance remain unqualified; complete2x goal open.


2026-10-27 source87 actual native CPU profile PASSED all184 cases (17valid/161rejected new,4valid/2rejected original), zero failures. Remote/local archive and PE byte replays are identical, proof 3ce098a2ad9c8589511c2592de08217b5a0195012dedee46b67cb71438d21d85, archive f5093103b05b96a06057f59f9aa288e78788e7c74ce4c522d3ae51f0b010fb7d (511354 bytes). PE imports onlykernel32.dll; no Garnet/CUDA dependency or local EXE execution. MSVC /Bv identifies compiler19.44.35229.0 and linker 14.44.35229.0 in tools directory14.44.35207; localized console text is retained as captured. All1948 baseline files/runtime/SDK/config-index/CMake/current34 CUDAobjects unchanged. Header offsets/modes/policy exclusion and old guards compiled/tested; this does not qualify TensorRT rowwise RMS lowering, model repair, quality or speed. Source87 full DLL and model stages remain pending.


2026-10-27 private rowwise K decoder frontend prepared with ordinary and boundary-control variants. Actual source87 chunk arithmetic copied from frozen cuda_chunk; add only exclusive profile guard and integer rowwise_full_decode_k_norm=1 marker. Both forwards copy their exact frozen ordinary or existing64-boundary source and change only full-attention chunk selection. No new operand outputs. Constructor rejects all but B4/cap4096/T4 before allocation; factory leaves B1 and prefill untouched and restores binding on failure. Six actual CPU symbolic checks pass, including all16FA metadata offsets and exact reverse derivations. Target source1d468ba29c732cf29bcfc825d52fa074774325d365cfeb473e185c77996a6e45. Native CPU guard184 is already qualified, but full DLL and both model paths remain unrun. Qualify ordinary production outputs separately after diagnostic repair; do not infer no-observer equivalence from boundary controls alone. No quality or speed claim.


2026-10-27: Prepared source87 default-off rowwise K-normalization native build. The current RTX 5090 serving result remains 29.9110 aggregate tok/s at image C8; Garnet physically executes B1 and queues clients.89.7712 decode tok/s/request excludes prefill and waiting; the overall2x throughput target is not achieved. Batch4 remains correctness-unqualified. This controlled experiment changes only K RMS subgraph shape to four B1-shaped rows when explicitly enabled. Fresh remote read-only inventory verified1948 baseline files,87 overlay targets (14 existing),51 closed parents, runtime/SDK/config+index/CMake/all34 objects; D.free560.405GiB. Source bundle fcffdf2455d188ca555e561ddb16d337d5f172c00b4dafdc7b90e2634651f190, deployment bc59eccfd5d8f74fc4e5e54fb79a74f4008c7bf26d10ee7bc9d8833481550e44. Native184 CPU guard parent passed, original109 seven-stage guard parent retained. Full build retains104 current+86 historical Python controls and Q/O/factory/wire/initialization/native+embedded selector/tokenizer controls; original3-object whitelist and restoration remain mandatory. Build alone cannot qualify model repair or speed. Next: paired raw build replay; same-DLL default-off controls; explicit on B4 ablation and full4K/128 ordinary text/image/mixed regression, quality, then matched HTTP performance. All prior full4K/cache62 and seven/six-output observer failures remain failed. Local preparation assertions about newline/replacement counts were corrected before deployment; failed preparation logs retained. No model benchmarks run here.


2026-10-27: Source87 native build terminal result rowwise_k_norm_native_CPU_build_replayed. Independent local and remote raw replay proofs byte-identical: 2398a5d2f44ae4a9f6cf1b81cc09100dcdf893edef9d2efba8a62c6eef96c4a7. Raw archive c77b1e25055e998048c17735548699ab3d2619d36171d0fbfb1adc4af6877cc0 (11451146bytes). Candidate DLL e8431b9778f5ccf01b18a6799444d45d6463c7b6f6c8b0bdad984883adcba563. Current104/historical86 Python and original Q/O/factory/wire/initialization/selector/tokenizer gates are required for successful status; actual184 new native guard proof and original109 guard parent checked. All baseline source/runtime/SDK/config-index/CMake restored; all34 CUDA object hashes retained, original3-object whitelist preserved. This does not establish all3 changes as metadata-only or pass the old strictcache62 gate. No rowwise-on model, numerical repair, model quality or performance qualification follows from compilation. Full4K and six/seven observer failures remain failed. Serving measurements remain unchanged; imageC8 aggregate29.9110tok/s vs vLLM224.971 and SGLang146.5366; overall2x target unmet. Next use same-DLL default-off controls, explicit rowwise-on ablation, ordinary full4K/128 regressions and quality before HTTP timing.


2026-10-27: Prepared the source87 ordinary-output rowwise K-normalization GPU ablation after paired native build proof2398a5d2f44ae4a9f6cf1b81cc09100dcdf893edef9d2efba8a62c6eef96c4a7. 115 CPU controls passed: original97 unchanged plus5 source,6 DSL and7 new analysis/trace/all-row/consumed-trajectory guards. Three fresh same-DLL processes: off B1 selected row,off B4 mixed four rows,then on B4. Both off controls must reproduce immutable ordinary source60/85 all53IDs, five complete128-component state and finite-vector captures, selected8MiB twice-read cache and53 coordinate points before enabling on. On changes only K RMS graph shape into four B1-shaped row subgraphs at16 FA layers; trace counts subgraphs, not executed kernels. No added graph observation outputs; six/seven endpoint observer failures remain failed. Earlier consumed-token divergence blocks same-input cache interpretation. Only selected step52 has an independent complete-state/cache oracle; a pass cannot qualify full4K128/MTP/805/HTTP or performance. Pin1984currentfiles, 52closedparents, exact post-source87-build34objects/runtime/SDK/config-index/CMake; snapshot87 targets and baseline DLL; fixed Ccache96GiB admission and owned nonce retirement. Deployment f7d0efbd2505c223ca25a9c6e067702cefc0ff502021dae9efdfdc0f30394248 (358346B). DLL e8431b9778f5ccf01b18a6799444d45d6463c7b6f6c8b0bdad984883adcba563. Local probe derivation assertion failures occurred before remote deployment because the original argument line lacked a trailing semicolon; corrected exact match and retained all three logs. GPU experiments unrun at this entry. Throughput29.9110 imageC8 remains unchanged and overall2x is unmet.


2026-10-27: Prepared optional host-side stage timing for the next throughput diagnostic; no worker integration or GPU/profile run. Six CPU controls pass with explicit source ledger: result identity/kwargs/startup behavior, nested spans, original exception propagation, invalid/stale/reentrant spans, bridge lock lifetime and release on logging failures. Native and bridge clock domains are labelled separately; never subtract their absolute clocks. Host intervals include instrumentation and are not CUDA kernel elapsed time or unprofiled serving throughput. Embedded-native compatibility is unrun. Static exact ZIP audit proves source60 and87 have identical existing OneShotTRTLayerProfiler, enqueue/graph policy and attach/write regions. GARNET_PROFILE_DECODE_LAYERS profiles only the first eligible enqueue per enginePath and bypasses graph launch while active; use as first-enqueue diagnostics, never warmed full4K serving performance. Retained primary TensorRT10.x algorithm-selection and debug-tensor documentation: https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/precision-control.html and https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/engine-tools.html . Explicit debug marking can inhibit fusion, while unfused-debug marking is documented to preserve fusion; actual listener/graph compatibility and observable tensors still need validation. General PyTorch numerical context https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html does not establish harmlessness or a unique cause for Garnet differences. Current acceptance gates unchanged, original failures remain failed. After closing the current model evidence, profile qualified B1/MTP queue/dispatch, prompt/image preparation, target/draft prefill, decode verification/replay and output formatting. Preserve the full2x/halfTTFT and matched strongest references/INT4/4B objective. No new benchmark from these helpers.


2026-10-27: Source87 ordinary-output rowwise K-normalization model terminal result rowwise_K_norm_selected_model_replayed, exit0, errorNone. Remote/local raw replay proofs are byte-identical: 09e6862bffbe6ce21fc1e27f5718b25c2a6ada04924887d1c694d3ec7e0d3d69. Raw archive 6aca056e6d41715785091b2b3ca9fecdb1cf9d0f1558234979742f57697c6185 (38279283bytes). Default-off controls closed: ['off-reference', 'off-candidate']. Selected step52 intervention passed=True. All raw token/state/cache/coordinate and resource reports retained; source/baseline DLL/runtime/SDK/config-index/CMake/all34 objects restored. This ablation adds no graph observation outputs. Native rowwise trace counts graph subgraphs, not executed kernels. Any selected step52 result is narrow; full4K128/MTP/805quality/HTTP/performance remain unqualified, all historical full4K/cache62 and six/seven output-observer failures stay failed. No new serving benchmark; imageC8 aggregate29.9110tok/s remains the measured result and the overall2x target remains unmet.


2026-10-27: Prepared full4K128 source87 rowwise-K ordinary validation after selected repair paired proof09e6862bffbe6ce21fc1e27f5718b25c2a6ada04924887d1c694d3ec7e0d3d69. One fresh B4 process, original3968prompt+128forced output tokens for three text/image/mixed groups against eight immutable source60 B1 oracles. The original full4K observe/admit functions are unchanged:48 complete phase-row comparisons of128 components and finite raw vectors, plus all12x128IDs must match without tolerance. No new graph observation outputs; actual 16FA/four-row RMS subgraph trace and unchanged GDN/V/QK routes required. The exact selected same-DLL off/on53ID/step52 repair is a prerequisite, not evidence for the remaining128-token trajectories.115 original CPU controls retained;53 closed parents/current1997guardedfiles/all34 CUDA objects/runtime/SDK/config-index/CMake remain pinned. Same e843 DLL; no new native build. Full source/baseline DLL restoration and fixed Ccache 96GiB admission/owned nonce retirement unchanged. Deployment 0cb7d1df993de4846d42e8545dafd298c1310e30415145c1d80c24749aa9f4c0 (425527B). Full4K run unrun at this entry. A pass qualifies these ordinary cases only; short prefix/permutation/tail/repeat/MTP/805quality/HTTP/performance remain open. Original source79 full4K/cache62 and both output-observer failures remain failed. Overall2x and halfTTFT targets unchanged and unmet.


2026-10-27: Static MTP target-binding audit completed while the source87 ordinary full4K job remains exclusive. Exact installed-overlay identities were used rather than the restored baseline: cuda_packed_decoder isbe9198 while its restored base isba48f7; packed_program is623d97 while base isfc85e0. All six inspected installed sources hash-verify from source87 ZIP or the closed selected-model archive. For the declared packed/prefix-replay profile, CUDASpeculativeSession selects CUDAPackedDeltaReplayVerifier; verification calls decoder.chunk_hidden, replay refers to the existing decoder.programs[decode_chunk_size], skips attention replay in canonical KV mode and commits retained target K/V via decoder.delta_commit. These inspected sources construct no replacement CUDAPackedDecoderProgram. This supports using the private rowwise target constructor in a future MTP probe without changing core verification/replay arithmetic, but is only a conditional source audit, not runtime/MTP correctness proof. B4 draft-study module41656e4f1acd2bd1c9cca8e50549e5bb82e804e1bd2daa42396b2f019ecfc02b matches its historical27CPU preparation and is not in the current remote inventory. Existing15-control MTP full4K gate is unchanged. Wait for ordinary full4K128 paired pass before preparing/deploying/running MTP; actual graphs/memory/249prefill/127commits/target state+IDs and prefix/reset/permutation/tail/quality/HTTP/speed remain unqualified. No running job or native/model source/DLL changed in this audit.


2026-10-27: Source87 ordinary rowwise-K full4K128 terminal result rowwise_K_norm_ordinary_full4k128_replayed, exit0, errorNone. Independent remote/local raw replay proofs byte-identical: 4461f2cccf8481b5de85267dbd3e6ddb913ad947ac0b25d19cb7834f9f275c9e. Raw archive 1fff7ec277c246599f77803ef3e3352fb363cc25dbb6b99121f52689aa28e39f (17765493bytes). Ordinary full4K128 case gate passed=True. First differing ID indices by group={'full4k-text': [None, None, None, None], 'full4k-image': [None, None, None, None], 'full4k-mixed': [None, None, None, None]}. Original exact gate and115 CPU controls retained: text/image/mixed groups with3968prompt+128forced outputs vs eight immutable B1 oracles,48 complete phase-row/128component and finite raw-vector comparisons,12x128 IDs. No tolerance or added graph observations. All runtime/source/baselineDLL/SDK/config-index/CMake/all34 objects restored. Trace is16 FA layers with four K RMS subgraphs each, not an executed kernel count. A pass covers these ordinary cases only. MTP/partial-prefix/reset/permutation/tail/repeat/805quality/liveHTTP/performance remain unqualified; next retain this evidence and proceed to appropriately qualified MTP and throughput profiling. Original source79 full4K/cache62 and six/seven observer failures remain failed. No new serving measurements; overall2x/halfTTFT objectives unmet.


2026-10-27: Prepared source87 rowwise-K B4 MTP3 full4K128 validation after ordinary paired pass4461f2cccf8481b5de85267dbd3e6ddb913ad947ac0b25d19cb7834f9f275c9e. 146 local CPU controls passed with source ledger:115 unchanged plus27 original B1/B4 draft-prefill/workspace/ownership controls and4 MTP evidence guards. Remote owner retains115 before overlay and31 after overlay so the qualified T128 CPU source provider is present. Same e843 DLL/source87 and rowwise target program. No core MTP/verifier/replay arithmetic or production-factory changes. Explicitly deploy only the pinned defaultoff B4 draft-study module41656e and methods; installer retains absent-or-exact checks and allows exactly that additional xModel path. One fresh GPU process will collect three3968+128 text/image/mixed groups with actual MTP3/T16draft-prefill/T4draft-decode/full-vocabulary head. Require original 48phase-row128component/finite-vector and12x128 target ID equality vs8B1 oracles, actual draft graph/workspace/duplicated-extra-weight evidence, 249prefill calls per group, positive proposals/verification and127 committed target inputs per row with no target-prefix recompute.54closedparents/2025guardedfiles/currentall34objects/runtime/SDK/config-index/CMake pinned. Same full restoration/fixed Ccache96GiB/ownednonce retirement. Deployment 73497d773b01bc86600a817e9afcdbf9ac52789a25212c5e919b2fde09e16e15 (453374B), uploaded as rowwise-k-norm-mtp-full4k-deployment.zip from local rowwise-k-norm-mtp-deployment.zip. MTP GPU unrun at this entry. Even a pass will leave prefix0/1/2/4, heterogeneous acceptance/reset/permutation/tail/repeat/805quality/HTTP/performance pending. Strongest repeated vLLM/SGLang,INT4/4B and complete 2x/halfTTFT goal remain open; no new serving metrics or speed claim.


2026-10-27: Source87 rowwise-K MTP full4K128 terminal result rowwise_K_norm_MTP_full4k128_failure_replayed, exit1, errorAssertionError('Full4K candidate capture failed'). Independent remote/local raw replay proofs byte-identical: e3b567ca8b38c0eaa7b08d68eff2068d350a581d72bee1eb56dd4c2671dcfd30. Raw archive 16bbde098c8ef07f8d022f2aff1f07bbd0191c9310c91e4e7fb1af4c97a2839e (17115274bytes). MTP full4K128 target case gate passed=False. CPU controls=146. A candidate capture failure is not a numerical mismatch diagnosis and provides no serving benchmark. Actual draft/target runtime failures must be investigated from retained logs before another attempt. Source87 ordinary full4K128 pass remains independently valid; MTP and broader prefix/reset/permutation/tail/repeat/805quality/HTTP/performance remain unqualified. Complete runtime/source/baselineDLL/SDK/config-index/CMake/all34 restoration verified. Original source79 full4K/cache62 and six/seven observer failures remain failed. Historical image C8 aggregate throughput stays29.9110tok/s versus vLLM224.9710 and SGLang146.5366; native single-request decode89.7712tok/s excludes prefill and queuing. Native physicalB1 serialized eight clients while references batched; fast decode does not establish aggregate serving speed. No new speed claim; overall2x and halfTTFT goals unmet.


2026-10-27: Prepared source89 explicit256MiB B4T16 MTP draft-prefill validation after retained source87 MTP128 build failure. Reported TensorRT tactic requires134227968B, exceeding128MiB134217728B by10240B. New budget is a tested hypothesis, not proof a rebuild or model will pass. CUDAMTPProgram permits256MiB only for strict integer B4/cap4096/T4-or-T16. Default-off B4 study keeps128MiB default and accepts explicit256MiB for extra T16 engine. Actual graph metadata must report256MiB for both T4/T16; original128MiB evidence gate and historical failures preserved. Source89 bundle 69428d7375a320da3b2e4e248f456bae6f25d92433f3455b345f3eb874dd3bfc contains all87 source87 members byte-identically plus two Python workspace-policy files. Same e843 native DLL; CUDA/DSL/target/verifier/replay arithmetic and checkpoint unchanged. 156 local CPU controls passed (115 retained+31 existing MTP+5 workspace/ownership+5 new workspace/target-evidence controls). New remote owner will repeat115 then41 on installed candidate. 55 closed parents,2038 guarded baseline files,1692 archive inputs,89 source members; runtime/SDK/config-index/CMake/all34 pins retained with full restoration and fixed96GiB SSD admission. One fresh GPU process with text/image/mixed3968+128 outputs vs8immutable B1 oracles, original1536IDs/48complete128component state/vector comparisons. Deployment 4ac5468cacdd74088e6fe76d6eaeba05bb68f1ea3476110773244777343593d4; prep 1bfa45b63316fd75cc324f81bf974f1cf81200402dff4a6a94e6abdd3b64596c. GPU build/model unrun at this entry. Broader MTP partial-prefix/reset/permutation/tail/repeat/805quality/HTTP/performance and strongest references/INT4/4B remain open. No new throughput or2x/halfTTFT claim.


2026-10-27: Prepared optional host-stage timing attachment and private source60 O128 diagnostic worker.12 CPU controls pass, including actual speculative prefill method AST exercising temporary decoder.chunk_hidden binding and restoration, return/argument/exception identity, startup silence, nested phases, independent override preservation and lock release. Initial test fixture omitted batch; its error and corrected passing rerun are both retained. Model code was not changed for that fixture correction. Diagnostic worker is derived from retained qualified HTTP worker3430d9ad source bytes, adding request/input-preparation/token-delivery spans and Python call boundaries for reset, prefill, draft, target verification/commit/head, and program calls. Existing model arguments and output-ID checks preserved. All intervals are host wall time including logging; graph calls may enqueue asynchronously, so no CUDA kernel elapsed claim or clean throughput measurement is allowed. Bridge/native absolute clocks must remain separate; nested spans must not be summed as independent costs. Embedded native compatibility and actual GPU/HTTP diagnostic requests unrun. Helpers have not been deployed during the exclusive MTP model run. Native layer profiler remains separate and its first eligible enqueue bypasses CUDA graph replay; it cannot establish warm graph serving performance. Next use controlled diagnostics with exact retained IDs, then run uninstrumented matched serving benchmarks after batch4 quality/protocol checks. No new speed claim or2x/halfTTFT admission.


2026-10-27: Source89 rowwise-K MTP256 full4K128 terminal result rowwise_K_norm_MTP_full4k128_replayed, exit0, errorNone. Independent remote/local raw replay proofs byte-identical: 8cf95c663abf7697e2490c67b571449caef02f6ebce2637fd1079fcb0bd8fc83. Raw archive 4970bc128b79b00f62324921aef979fd5cb8fb474b77559968b33d1af90ec00a (18348515bytes). MTP full4K128 target case gate passed=True. CPU controls=156. All three text/image/mixed3968+128 MTP cases pass the original exact target gate:1536IDs and48 complete128component state/finite-vector comparisons vs8immutable B1 oracles. Actual T4/T16 graph budgets both256MiB,249 MTP prefill calls/group and127 committed target inputs/row verified. Source87 ordinary full4K128 pass remains independently valid. Broader MTP partial-prefix/reset/permutation/tail/repeat/805quality/HTTP/performance remain unqualified. Complete runtime/source/baselineDLL/SDK/config-index/CMake/all34 restoration verified. Original source79 full4K/cache62 and six/seven observer failures remain failed. Historical image C8 aggregate throughput stays29.9110tok/s versus vLLM224.9710 and SGLang146.5366; native single-request decode89.7712tok/s excludes prefill and queuing. Native physicalB1 serialized eight clients while references batched; fast decode does not establish aggregate serving speed. No new speed claim; overall2x and halfTTFT goals unmet.


2026-10-27: Updated comparison dashboard v8 after independently paired source89 B4 MTP256 full4K128 pass. 45 verified retained input artifacts. Benchmark values unchanged: same source60 B1/MTP3 three native trials versus one ordinary trial each vLLM/SGLang. New dashboard distinguishes historical source79/full4K, failed M4 intervention and rejected observers from successful rowwise-K selected/ordinary/MTP full-context cases. MTP correctness pass covers three groups only; broader partial-prefix/lifecycle/805quality/HTTP and new serving timing remain pending. Strongest tuned/MTP references, INT4 W4A16 and Qwen3.5-4B remain open. No new benchmark, throughput improvement or2x/halfTTFT claim. PNG visually inspected: all three ratio panels, target lines, ten status rows and method caveats visible. [Progress comparison v8](../../test2026/cuda/qwen38_5090/evidence/figures/Garnet-Progress-Comparison-2026-10-27-v8.png).


2026-10-27: Prepared source89 B4 padded lifecycle/public-event validation after paired MTP full4K proof8cf95c66. Parent actual iterations cover processed prefix lengths0..4 in all three groups, with116 heterogeneous steps; terminal full states/reset match B1. This is observed execution coverage, not intermediate-prefix snapshots or a waiver of public padding/lifecycle checks. New five-case selector from unchanged eight pinned3968-token inputs:1text,2image,3mixed,4permuted mixed,repeat3mixed. Missing physical rows duplicate first real input. Require all20 physical row sequences2560IDs and80 complete128component state/finite-vector comparisons vs8B1 oracles, including dummy rows. Public callback helper uses existing typed padding and ignore_eos policies plus actual native JSON formatter; independently replay1664 real-row IDs, prompts/images/counts/stop metadata and reject extra request IDs. Original three-group gates/corpus unchanged. Same89source bundle69428d73/e843 DLL; no native compilation or model arithmetic changes.170 final CPython controls passed (156 retained+7 public-stream+2 selector+5 full lifecycle gate). Same170 checks were rerun after inherited descriptive labels were clarified; initial unexecuted package0635d2d6 preserved locally/remotely and included in final raw-input pins. Fresh SSH read confirms existing padding policy7cc69ec4 matches local; now explicitly guarded. Additional9 actual embedded-runtime symbolic controls precede GPU model work and are unrun at this entry. 56 closed parents,2066 guarded files,1720 archive inputs, baseline source/runtime/SDK/config-index/CMake/all34 restoration, fixed96GiB Ccache admission and ownednonce retirement retained. Final deployment b24336f7346eb1f06cfab6c40d55f8db899b16e648bc0b1028d709ef1af2488c (265715B), prep bea7b2841a9a6784b505f6a4ad48c49d3a7e4e0edfd3ecaade8523eff4b141ee. One fresh GPU process planned. A pass covers these additional padded/order/repeat/event cases only; adapted805 quality, actual HTTP scheduling/streaming, new unprofiled serving timing and strongest tuned/MTP references/INT4/4B remain pending. No new throughput or2x/halfTTFT claim.


2026-10-27: Source89 B4 padding lifecycle terminal result rowwise_K_norm_B4_padding_lifecycle_replayed, exit0, errorNone. Independent remote/local raw replay proofs byte-identical: 432f22350418c9602572c5fe139acbebc26fd14d888bee93f29892ad8f807b51. Raw archive 6047bcfdb8823ef0a91a76b42a74bc25c1fd56a3dcf9aff09647ffb296de0b6e (19440214bytes). B4 padding/order/repeat/public-wire case gate passed=True. CPU controls=170. All five padding/order/repeat3968+128 MTP cases pass:2560 physical IDs including dummy rows,80 complete128component state/finite-vector comparisons vs8immutable B1 oracles, and1664 real public-wire IDs. Actual T4/T16 graph budgets both256MiB,249 MTP prefill calls/group and127 committed target inputs per physical row verified.170 CPython and9 actual embedded CPU controls pass. Source87 ordinary full4K128 pass remains independently valid. Broader MTP partial-prefix/reset/permutation/tail/repeat/805quality/HTTP/performance remain unqualified. Complete runtime/source/baselineDLL/SDK/config-index/CMake/all34 restoration verified. Original source79 full4K/cache62 and six/seven observer failures remain failed. Historical image C8 aggregate throughput stays29.9110tok/s versus vLLM224.9710 and SGLang146.5366; native single-request decode89.7712tok/s excludes prefill and queuing. Native physicalB1 serialized eight clients while references batched; fast decode does not establish aggregate serving speed. No new speed claim; overall2x and halfTTFT goals unmet.


2026-10-27: Prospective source91 B4 HTTP transport preparation: source89 model bytes and e843 DLL retained; two Python transport candidates align strict ignore_eos validation and permit an explicit16MiB joined frame, preserving the default2MiB bound. Eight source-bound CPU protocol controls pass; worker/launcher/805 collector are syntax checked only, actual HTTP and GPU deployment unrun. Full current inference-input manifest ad10ddf8e44385a2fe3e08498bd5a04781835b473d335be1e622266912de4210 records18 files/3 shards/21944954979bytes with streamedSHA256 and stable size/mtime. This is posthoc to previous model runs and does not retroactively attest historical shards or establish a fresh full HF LFS OID comparison. Future before/after checks will use all18 hashes. B4 HTTP correctness and throughput remain unmeasured; overall2x and halfTTFT unmet.


2026-10-27: Source91 B4 HTTP quality package prepared. Deployment c18cdbba2898754b251129a6699599085d0fafc39476950af2135a9ca004ee95, preparation 940e7fee408f3c2591b2a28ef49d7856433e57070323c2cc2905d7c2f1ac9088, source bundle 4ccee24a9d58cd8297ea6a44632d54d3efd608fe323f686b12dcb56d9a179779. Same89 model source members/e843 DLL; only two additional Python bridge/parser source members.14 source-bound CPU protocol/event/dispatch controls passed;14 actual embedded controls and836 actual loopback HTTP requests remain unrun. Planned controls:29 fixed3968+128 nonstream real requests across padding/order/repeat and C8, including a marked large ignored transport fixture;2 text/image SSE controls, HTTP400 invalid-flag rejection,805 adapted natural-stop requests with full IDs vs retained T32 subset. Independent reader checks actual dispatch frames>2MiB and suppresses public dummy rows. All18 current model-file hashes checked before/after, streamed to avoid loading full shards for logging;57 closed parents and all34 CUDA objects retained. Initial preparation bookkeeping lookup failure occurred before deployment and is retained; a fresh SSH read confirmed both original transport file hashes before extending pins. No native math/build change, serving timing or2x claim.


2026-10-27: B4 HTTP attempt1 stopped in preflight with exit1: historical candidate_bytes accepts only src/xModel, so rejected the two test2026 transport files. No original output directory/source overlay/DLL installation/model/GPU request occurred. CPU-only reproduction and fresh hashes of all owned source/runtime/environment/config-index/CMake/all34/full18 model files pass; audit 7d2b93b293023cdfdcb36cf1c4e291aa33f6f6432329e3b5539d6c55e48f4af3. Original c18c package and940e preparation are retained unchanged locally/remotely, with the preparation also in repository evidence. Attempt2 replaces only that incompatible loader call with a source91 loader accepting exact89 qualified model members plus exactly the two fixed transport paths, with hash/path/duplicate/traversal checks.17 source-bound CPU controls pass. New deployment 3ad3331cef854eedb43c913708f62306eb9afbc20e1caed67a4626830db8b2a0, preparation 2d15f461c0d179d883eb315dde15ce1ddd4eafbe2d05b19f21d94ed29bb949b4. Native/model bundle4ccee24a and DLL unchanged. Only exact expected prior method hashes may be replaced, and remote installer snapshots those methods first. Actual embedded17, HTTP805/controls and throughput remain unrun. No speed or numerical failure claim from preflight.


2026-10-27: B4 HTTP attempt2 closed exit1 before inference, caused by missing checkpoint_revision metadata in the new collector.17 remote CPython and17 embedded symbolic controls passed, but those did not exercise collector startup metadata. Raw archive f52200c61c1b57653a5cc6e743382aca1bb95fec15a36fe2287b9acdba1c83b5 (18096923bytes); independent remote/local failure proofs byte-identical ab33027aa5e5bff4840a7a080c58a8b9628458d0a7a200a307d2f90820b3b61c. All source/runtime/environment/config-index/CMake/all34/full18model hashes restored. Failure is setup, not numerical or performance evidence. Worker startup must move inside cleanup handling; a real-main CPU startup regression will cover missing metadata and failed-worker cleanup before attempt3. Native/model bundle and DLL remain unchanged; no new serving metrics or2x claim.


2026-10-27: B4 HTTP attempt3 package ec3b153453b13a72779b8c8e840e6b1e791f4a98a625122c1844e6a449e069fb, preparation dc5faa3df70d706f88331b651a3a6562f22193750b14895a3ca7f6cafb0ef82c. Adds pinned checkpoint revision482ca0 from full18file manifest, owner checks revision before source mutation, and collector instantiates worker within its cleanup guard.19 source-bound CPU controls pass, including two actual-main startup regressions with a controlled nonmodel worker. Missing revision prevents worker construction; a controlled startup failure closes the worker once and saves failure. Static audit verifies all26 literal preparation keys read by actual owner/collector/admission/launcher are present.17 embedded controls retained unchanged;58 closed parents include failed attempt2 with paired proofab33027a. Model source89 and native DLL e843 unchanged; source91 bundle4ccee24a unchanged. Actual HTTP805/836control gate and serving timing remain unrun for attempt3. Original failures preserved; no2x/halfTTFT claim.


2026-10-28 UTC: Current full checkpoint identity independently checked against pinned publisher revision482ca0f3832238542f8f5295dde86b5f22711d80. Three weight shards21921697280bytes match publisher LFS SHA256 and size; all12 other locally present published assets match pinned LFS identity or fetched bytes/SHA256 plus Git blob ID. Metadata response9a8f61e3 and full source/evidence archive 63ff77c018bbfa2a22532e7e7b91f414a293b92249f329a481e424e6115388dd retained. Clarification: current18-file manifestad10ddf8 is overinclusive:15published assets plus3local download/provenance bookkeeping JSON files, all still guarded before/after current runs. It does not retroactively attest older benchmark shards. Initial check expected all18 at publisher and failed on local bookkeeping; that result is retained alongside corrected15+3 scope. Publisher config architectureQwen3_5ForConditionalGeneration,64layers48linear/16full, hidden5120/Q24/KV4/d256; quant_method=modelopt and quant_algo=MIXED_PRECISION,208FP8 targets and193NVFP4 targets including lm_head, MTP excluded from quantization. Resolved vLLM runtime modelopt_mixed naming is a runtime adapter; published config literally says modelopt. Static saved-reference audit confirms SGLang0.5.22 disable_radix_cache=True, uses_mamba_radix_cache=False, mamba_ssm_dtype=float32, KVbfloat16, decode full graphs1/2/4/8. Historical vLLM MTP3 server did load/serve with acceptance metrics but its strict same-engine full-ID gate failed at s3-text-c1-r0; this is not a load/support failure and has no qualified throughput result. Fresh stronger references remain pending; no native math/source changes or new benchmark from these CPU-only audits.
