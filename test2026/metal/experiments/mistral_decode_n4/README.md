# Lossless N4 decode layout: native experimental checkpoint

This study interleaves four adjacent output channels at each packed Q4 word and
FP16 scale/bias group. Decode can use uint4/ushort4 loads and componentwise FP32
fused operations while preserving each output's word/lane traversal and scalar
SIMD reduction. Quantized values, coefficient bits and xModel operators stay
unchanged. Full Mistral MLP dimensions need no padding and retain the original
compact payload size. The standalone screen also checks padded N tails.

The native extension is **default OFF**. It applies only to four-active-row
decode with K4096/N14336 or K14336/N4096. Prefill, vocabulary, attention
projections and one/two/three-row decode keep their existing layouts. It retains
the row-major weights for those consumers and prepares immutable N4 copies
before execution. Decode performs no cache allocation or shader compilation.

Preparation reserves bytes transactionally under a3GiB process cap and checks
the recommended device working set with a256MiB margin. A shared weight owner
holds the new buffers; its mutex protects preparation/publication and bindings.
Owner destruction returns its charge. Inventory exposes the N4 payload and
FP16 coefficient representations; memory snapshots expose live/reserved bytes,
owner count and encoded dispatch count. All original11 dirty paths remain
unchanged; this package does not activate the production runtime.

## Operator evidence and comparison correction

Trained gate/up/down operands from layers0/15/31 cover full shapes, N1/7/17/33
tails, ordinary1–4 rows and all16 selected-row masks. Contiguous and padded
layouts pass1680 variant cases, with complete output/inactive/padding/guard
parents exact to the frozen scalar FP32 controls. Independent oracles verify
every repacked Q4/coefficient word and5292 FP64 dot checks using the unchanged
bound. Each layout round-trips297,672,192 original bytes into297,741,312 prepared
bytes, including tail padding. Sixteen deliberate evidence corruptions reject.

Four fresh serial processes produce3360 measured samples after full pipeline
preparation, with five warmups and20 measured rounds per cell. Two warm and two
pressure cohorts reverse pair order. Pressure is a separate256MiB read/write
command, excluded from projection GPU time; it does not prove every hardware
cache is cold. The declared C4 gate requires at least3% lower summed gate/up/
down median time in all processes, with no individual regression over2%.
The C4 operator ratios are1.043–1.120, so native integration screening proceeds.

The initial C1 timing control was the blocked4-column kernel. Source inspection
showed that actual occupancy-one decode selects the scalar kernel. That earlier
C1 result remains retained as a surrogate; it is not the native C1 comparison.
Four additional serial processes compare N4 against the actual scalar kernel
at physical rows0/3, adding960 measured samples. Ratios span0.998–1.149 and fail
the declared consistency gate. **N4 is therefore enabled only for occupancy4.**
The original C4 blocked-kernel comparison remains valid. Fifteen timing and
eight corrected-C1 evidence corruptions reject.

## Native qualification

Both enabled and default-off builds pass54 tests serially. A new native test
first computes original scalar decode before profile preparation, then checks
all16 masks and full four-row output against that exact control:1,253,376 FP32
words. It verifies N4 selection only at occupancy4, inventory membership, cache
charges and owner cleanup. The portable test covers packing, supported shapes,
occupancy guards, cache-cap arithmetic and working-set rejection. Invalid
configuration without the prepared reuse engine rejects.

One untimed full-model run then matches all32 output IDs/text,260 whole-engine
KV/hidden digest pairs and520 per-request digest pairs to the frozen FP32
control, plus98,304 hidden values. Six four-row decode frames execute576 N4
projections. Inventory observes96 N4 weight owners and288 distinct buffers,
with3,170,893,824 extra payload bytes. Readiness allocates13,211,467,776 Metal
bytes; shutdown releases all tracked owners and N4 charges, with1,900,544 Metal
bytes remaining under the4MiB bound. These are snapshots, not peak memory.
Ten deliberate model-evidence corruptions reject.

The first model attempt correctly rejected a relative inventory output path
before readiness or token generation. It and its clean shutdown remain retained;
the second configuration uses an absolute path. A saved-data audit correction
uses inventory `representation` for layout labels; `role` describes slot/held
use. No model was rerun to repair that auditor. Original stderr, including Apple
CoreAnalytics diagnostics, remains retained; the cleanup claims concern the
tracked native owners and measured allocation bound.

## Serving observations

One fresh process runs the unchanged frozen2K-input/C4/128-output requests in
cold/warm1/warm2 rounds, with prefix caching off. All12 responses and1536 output
IDs/text match the saved native control. There are303 actual four-row decode
frames and29,088 N4 projections. Native up-to512 profiles, the512-token actual
tick budget and prepared GPU workspace bounds remain enforced. No comparator
inference is rerun. Sources/models are hashed before collection and after
terminal shutdown; no builds or source/model hashing occur during collection.

| Round | Median TTFT s | Decode/request tokens/s | Aggregate tokens/s |
|---|---:|---:|---:|
| Cold |9.069 |9.189 |21.449 |
| Warm1 |9.362 |9.035 |20.970 |
| Warm2 |9.732 |8.826 |20.373 |

The prior wide candidate observed warmed TTFT9.34–9.61s, decode8.60–8.90 tokens/s
and aggregate20.15–20.81 tokens/s. Those are separate campaigns with uncontrolled
OS/thermal state. The new observations do not establish a repeatable serving
gain, and the extra2.95GiB is a material profile cost. No default promotion or2×
claim is made. The latest established comparison remains the
[selected fused-MLP campaign](../mistral_fused_mlp_serving_benchmark/README.md).
Prefix lineage for this new engine, coding qualification, repeated serving,
production integration, ANE/Gemma and original Qwen goals remain open.

## Reproduction and saved-data replay

Pinned sibling studies under `out/research/mistral7b-local-coder` are required.
The native generator clones the frozen wide source and applies guarded exact
edits plus the four supplied new files. All26,299 source file hashes must match
the frozen base plus eight changed/new paths. No build or inference is executed.
The operator generator reproduces both shader libraries and the host source.

```sh
python3 -B prepare_native1.py /absolute/path/to/mistral7b-local-coder/FRESH-NATIVE-SOURCE
python3 -B prepare_operator1.py /absolute/path/to/mistral7b-local-coder/FRESH-OPERATOR-SOURCE
python3 -B replay_saved1.py /absolute/path/to/FRESH-REPLAY-REPORTS
```

The replay checks packaged auditor/helper hashes against the frozen scripts,
runs all five audits without inference, and compares complete reports with only
auditor file locations normalized. The legacy admission auditor's output path
alone is redirected so the saved report remains untouched. Recorded collection
scripts, profiles and engine specs are included for review. Fresh native source
copies include the build and model-state runners; do not rerun unchanged
inference merely to validate saved evidence.
