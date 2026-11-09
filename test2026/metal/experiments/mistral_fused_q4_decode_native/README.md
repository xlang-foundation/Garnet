# Native four-row FP32 Q4 gate/up/SiLU fusion

This default-off experiment integrates the
[qualified operator screen](../mistral_fused_q4_decode/README.md) into the native
Metal runtime. It shares input loads between gate and up projections and writes
their SiLU product directly, preserving FP32 dot accumulation and the original
default-compiled SiLU arithmetic. Down projection, quantized checkpoint values,
coefficient bits and xModel operators are unchanged. The earlier operator
screen measured approximately 5% lower combined gate/up/down time at four rows;
that is a component observation, not an end-to-end speedup.

`GARNET_METAL_FUSED_Q4_DECODE` defaults to OFF. It requires the prepared affine
profile/index32, FP16 coefficients, Mistral decode reuse and active projection
options. Only four actual active rows at K4096/N14336 can use the new route.
This covers B4 decode and B1/T4 speculative verification. One/two/three actual
rows, partial verification and incompatible SiLU policy retain the original
route. No ANE, N4 weight cache or budget-clipped prefill policy is included.

Graph compilation proves that gate/up/SiLU intermediates have no other readers
or externally observable aliases. The execution session prepares its own
pipeline profiles before readiness, with no shader compilation or GPU allocation
during dispatch. Up to 64 input/output stride pairs are allowed per context.
A 65,536-byte private FP32 input buffer supports overlapping physical input and
output; it is copied only when their spans overlap. No extra weight payload is
retained. Owner readiness, pending-command membership, device, matrix extents,
unique row selection and checkpoint/private scratch aliases are checked before
writes. Inventory and read-only counters expose preparation, dispatch and copies.

## Correctness evidence

Enabled and default-off builds pass 53 and 52 tests, respectively, serially.
Invalid configuration without the required reuse engine rejects. The native
unit passes 457 checks and compares 4,180,320 FP32 parent words against original
projection/SiLU results. Trained layers 0/15/31, all 16 active masks, input alias
copy, invalid selections, readiness, private aliases, alternate SiLU fallback,
Abort/Flush recovery, stable pipeline/scratch identity and tracked cleanup are
covered.

An untimed full-model run matches all 32 output IDs/text, 260 whole-engine and
520 per-request KV/hidden digest pairs, plus 98,304 exposed hidden values, to
the frozen ordinary FP32 control. Six full decode frames execute 192 fused
blocks. Readiness allocates 10,040,639,488 Metal bytes; close leaves 1,900,544
bytes and zero tracked owners. Eleven deliberate evidence corruptions reject.

Native accepted-prefix verification then passes 11 cases at context capacity
256 and 13 at 4096: rejection after one/two/three tokens, full acceptance,
partial blocks, EOS, carried continuation, context boundary and full-prompt
state. All 1,560 complete layer/plane KV and accepted-hidden digest comparisons
match ordinary reference execution. Sixteen full four-input frames execute
512 fused blocks; eight partial frames use the original route. No input copy
is needed in these model layouts. Close leaves 720,896 Metal bytes and zero
tracked owners per capacity. Twelve evidence corruptions reject. These are
oracle proposals for correctness, not learned-draft acceptance or speed tests.

The original failed attempts remain saved. The first trained test reused its
context across layers while incorrectly expecting a shape profile to remain
unprepared; the fixture was corrected to use a fresh context per layer. The
first full-model attempt exposed a real integration defect: preparation was
performed in the graph-local context, while execution uses a session context.
Final source3 moves preparation to session Build/PrepareVerification and was
rebuilt and retested. An initial model2 audit used the wrong output filename;
the filename-only correction reproduced the complete report without inference.
Apple CoreAnalytics stderr is retained. Cleanup claims concern tracked native
owners and the measured allocation bound, not every system allocation.

## Diagnostic serving results

One fresh engine runs the unchanged frozen approximately 2K input / C4 / 128
output requests in cold/warm1/warm2 rounds with prefix caching off. All 12
responses and 1,536 output IDs/text match the frozen native control. There are
303 actual four-request decode frames, 9,696 fused blocks and zero input copies;
156 partial decode frames retain the original route. The native scheduler's
prepared up-to512 shapes and actual 512-token tick budget remain enforced.
Eighteen deliberate evidence corruptions reject. No builds, source/model
hashing or comparator inference run during collection.

| Round | Median TTFT s | Decode/request tokens/s | Aggregate tokens/s |
|---|---:|---:|---:|
| Cold | 8.830 | 9.327 | 21.817 |
| Warm1 | 9.129 | 8.990 | 21.078 |
| Warm2 | 9.689 | 8.783 | 20.362 |

The preceding wide candidate observed warm TTFT 9.34–9.61 s, decode 8.60–8.90
tokens/s and aggregate 20.15–20.81 tokens/s. These are separate campaigns with
uncontrolled OS/thermal state; they do not establish a repeatable improvement.
The frozen vLLM-Metal C4 medians are TTFT 11.570 s, decode 10.484 tokens/s and
aggregate 21.179 tokens/s. Lower candidate TTFT includes scheduling changes,
and its decode interval includes competition with unfinished prefills. It does
not prove an isolated prefill or decode kernel speedup. The
[selected comparison campaign](../mistral_fused_mlp_serving_benchmark/README.md)
remains unchanged. No default promotion, production integration or 2× claim is
made. New-engine prefix lineage, learned-draft performance, repeated matched
serving, coding quality, ANE/Gemma and the original Qwen goals remain open.

## Reproduction

Pinned sibling studies under `out/research/mistral7b-local-coder` are required.
The source generator clones the frozen wide engine, applies guarded native
edits and copies the three final new source files. Every one of 26,298 source
file hashes must reproduce the final source3 canonical digest. It copies fresh
qualification runners but does not compile or execute inference.

```sh
python3 -B prepare_native1.py /absolute/path/to/mistral7b-local-coder/FRESH-STUDY
python3 -B replay_saved1.py /absolute/path/to/FRESH-REPORT-DIRECTORY
```

Saved-data replay runs all three complete model/rollback/HTTP auditors, including
their corruption controls, and requires exact report equality. Only the legacy
HTTP auditor's output destination is redirected; its input root and validation
statements are retained. No saved report is overwritten and no inference runs.

For an explicitly needed fresh qualification, use the generated study in this
order: `build_native3.py`, `run_model2.py`, `audit_model2.py`,
`run_verification3.py`, `audit_verification3.py`, `refresh_serving1.py`,
`run_admission1.py`, `audit_admission1.py`. The refresh step binds the newly
built library and fresh certificates before serving; an original build's binary
hash is not assumed valid at a different source path. Keep GPU runs serial.
Do not rerun unchanged inference merely to replay saved evidence. The original
11 dirty paths in the working repository are unchanged by this package.
