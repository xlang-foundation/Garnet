# Shared-value Mistral verification attention screen

This isolated Metal operator shares V and FP32 probability tiles across four
query tokens and four query heads per KV head. Each output retains the ascending
key accumulation order. The prepared profile is B1/B4, T4, H32/KV8/D128,
C4096, ungated. Select tile64 for B1 and tile32 for B4 at visible prefixes
of at least 512. Short and unsupported native profiles retain their existing path.
Native integration is pending; these are operator timings, not serving results.

Two fresh processes ran serially. Each covers 468 cases, all uniform masks,
mixed owners and first positions, inactive NaN slots/value tails, capacity edges,
and zero or 64-byte offsets with padded strides. Native probabilities are
computed before value timing and preserved unchanged across all four methods.
The audit checks 119,328,192 whole-parent word pairs, 6,000 independent FP64
context samples and 12 corruption controls. All contexts match bitwise.
The full CPU audit reproduces its original report exactly.

At approximately 2K context, four active query tokens per request:

| Physical batch | Current GPU ms, cohorts 1 / 2 | Selected GPU ms, cohorts 1 / 2 | Paired speedup vs current |
|---|---:|---:|---:|
| 1 | 0.5036 / 0.5113 | 0.2871 / 0.2915 | 1.76x / 1.75x |
| 4 | 3.4273 / 3.4331 | 0.6110 / 0.6110 | 5.61x / 5.62x |

B1 control is the existing vector4/TG64 route; B4 control is the existing scalar
route at its native maximum threadgroup size. A separate plain-vector4 B4
comparator isolates the gain from shared data: tile32 is about 1.71x faster
than that comparator at this point. Compiler options match native defaults.
No global GPU workspace or extra weight cache is introduced.

`source-reference.json` pins the complete raw evidence outside Git. Run
`replay_audits1.py FRESH_OUTPUT_DIR` with the qualified asset Python environment
to verify all raw files and reproduce the CPU report. It executes no GPU work.
Frozen vLLM/SGLang results are unchanged. Full-model speed, C4 causal drafting,
HTTP TTFT/throughput, coding quality and the overall 2x target remain unqualified.
