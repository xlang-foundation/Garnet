# Half-QKV attention policy: qualified component, native integration pending

This standalone C++/Objective-C++/Metal study evaluates distinct numerical
policies `fp16_qkv_p16_v1` and `fp16_qkv_p22_v1`. Evidence is retained in
`out/research/mistral7b-local-coder/mistral-fp16-attention-policy1`.
The selected Garnet runtime, xModel operators and frozen reference engines are
unchanged. There is no full-model, cache, quality or serving qualification yet.

Q/K/V round once to IEEE half. Accumulators, online softmax, normalization and
output remain float. P16 uses one scaled half probability; P22 adds a second half
residual with a separate float correction accumulator. The probability scale is
32768. Half QKV is a different policy: a CPU FP64 probe records7,156/18,432
sampled outputs outside the old FP32 tolerance. These failures remain failures;
they are not evidence of coding-quality degradation or permission to select the
new policy without full-model testing.

Before any GPU collection, `numeric-policy-budget1.json` fixes the half-QKV
reference and probability-rounding allowance. The base allowance retains
`1e-5+2e-4*abs(reference)`. P16 adds `2^-11 * weighted_abs_V` and
`2^-40 * visible * max_abs_V`; P22 uses `2^-22` and `2^-39`. For complete-output
comparison, maximum absolute V upper-bounds its probability-weighted value.
These terms follow the half-conversion bound and exact binary scaling. The
remaining base is an empirical float-math check, not a universal proof of
hardware roundoff. No coefficients change after collection. This new contract
does not replace the original FP32 contract or qualify model quality.

Ten synthetic cells retain cold/2K/4K, B1/B4, ragged/empty owners, gating,
offset/padding, half midpoint and tiny-probability cases. Three trained cells
use the actual Mistral layers0/15/31 at B4/T128/C4096, offset1920. A second fixture
rounds only logical Q/K/V/cache values; every gate/control/guard bit is preserved.
Raw and rounded candidates must agree bit-for-bit. Full comparisons against
rounded-input four-product MPS and51,712 FP64 values per candidate pass the fixed
new-policy limits. Original MPS controls retain31,333,032 exact parent-word pairs
from the preceding study. Original-policy errors remain separately reported.

The initial float-storage/strict-descriptor implementation is slower than MPS.
The next implementation follows the pinned vLLM-Metal prefill source's relaxed
descriptor and vectorized register-load pattern. It uses half Q/K/V storage,
with Q packing inside each GPU call. Every Q/K/V input map, score/P/output map
and all2,560 cooperative coordinates are checked in the actual compiled library.
Both candidates' complete outputs remain exact across31,333,032 parent words
between the strict and packed implementations. The combined changes improve
speed; no isolated causal share is assigned to one of them.

The final source also includes GPU KV conversion inside each candidate call.
Every half scratch parent is poisoned before every call, outside timestamps;
afterwards complete Q/K/V parents must equal independent CPU conversions.
Synthetic/trained outputs and parents remain exact against the preceding path:
610,949,986 float-word pairs and270,209,172 half-word pairs. Input/cache guards,
tails, inactive owners and original float parents remain checked. The standalone
host retains float control buffers; it does not establish model-memory savings.

| Plain-half path | GPU ms, range of layer/cohort medians | Paired two-MPS/candidate |
|---|---|---|
| Float storage, strict descriptor | 8.85–9.05 | 0.794–0.803× |
| Packed half KV, Q conversion included | 2.69–2.82 | 2.511–2.655× |
| Q and KV conversion included | 3.54–3.91 | 1.816–2.004× |

The two-product MPS control is about7.1ms, original four-product MPS about8.6ms.
The two-probability-component path is slower: packed5.60–5.69ms, or6.45–6.61ms
including KV conversion. Each path has two serial, opposite-order processes,
three warmups per layer/variant and nine rotating four-variant rounds. In total,
648 measured command buffers and216 attention warmups are audited;15 deliberate
timing corruptions reject. Timing includes all candidate Q/KV encoders indicated
in the table; CPU setup/poison/readback and verification are outside GPU spans.
No source hashing, compilation or other inference overlaps timed collection.

A stale shader path causes the first packed timing attempt to reject a missing
function before any attention dispatch. That empty result and failed status are
retained; cohorts4/5 are the corrected packed measurements and6/7 include KV
conversion. No failed samples enter medians. These are attention-component
observations, not TTFT, token speed, throughput or the overall2× target.

Prepare exact final sources without building the dirty checkout:

```sh
python3 prepare_shader.py ADMITTED_GARNET_SOURCE FRESH_OUTPUT_DIRECTORY
```

The helper delegates the adjacent strict-attention source preparation and
appends the pinned final shader. Compile host5 for correctness or host6 for
timing using C++20/ARC, admitted `third_party`, and Foundation/Metal/MPS, as in
the saved compile5/compile6 commands. Host5 takes
`SHADER - FRESH_OUTPUT synthetic` or
`SHADER TRAINED_FIXTURE FRESH_OUTPUT trained`. Host6 takes
`SHADER AUDITED_RAW_TRAINED FRESH_OUTPUT forward|reverse`; it requires the saved
numerical audit3 in the fixture's parent directory.

The numeric, copy-path and timing auditors take an absolute retained study
directory. They deliberately depend on its saved revision1/2/3 files and the
adjacent frozen compensated/trained studies. Probe/conversion helpers require
fresh destinations. These are study-bound replay tools, not a public inference
API. Large inputs, all earlier sources/commands, failed attempts and raw results
remain local. Historical inherited probe comments do not override the explicit
half types, relaxed descriptors and metadata in the final source.

Next is an opt-in native implementation with preallocated bounded scratch,
explicit numerical/cache namespace and compiler-proved committed KV input.
Native state/lifetime/failure, full-model logits/coding, prefix reuse, continuous
batching and serving checks are required. The serving comparison, historical
strict coding0/8, ANE/Gemma and original Qwen27B/9B/4B goals remain open.
