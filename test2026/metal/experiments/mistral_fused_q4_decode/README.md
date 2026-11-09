# FP32 Q4 gate/up/SiLU decode fusion screen

This native Objective-C++/Metal experiment shares the input-vector loads of the
Mistral gate and up projections and evaluates SiLU in the projection output
stage. Normal execution writes the FP32 product directly, eliminating two FP32
intermediate stores and a separate pointwise kernel. The down projection is
outside this screen. Original Q4 words, FP16 coefficients, FP32 operands, FMA
traversal, scalar SIMD reduction, and logical model operators are preserved.
There is no extra weight representation/cache and no runtime integration yet.

The control uses the actual native route: strict MSL3.1 scalar decode at one,
two and three selected rows, and blocked four-column decode only at four rows
with K4096/N14336. The original native SiLU source compiles with nil/default
options. Shadow variants expose complete gate/up outputs for validation only.

## Exactness and compiler settings

Three candidate settings are retained: strict/exp, strict/fast::exp and
nil-default/exp. All preserve the tested gate/up projection shadows. Only the
default setting reproduces every FP32 SiLU-product bit. Strict variants each
produce more than 1.5 million product mismatches per layout and are rejected;
no tolerance is relaxed to admit them.

Trained weights and normalized input captures from layers 0/15/31 cover ordinary
one-to-four rows, all sixteen physical-row masks, N14336 and N1/7/17/33 tails,
and contiguous/padded parents. Two serial probes audit 3,600 candidate/shadow
cells. The exact candidate passes 27,737,280 full-parent word comparisons,
including inactive output and padding. Input and weight guarded-parent hashes
also match independently reconstructed fixtures. Independent CPU
oracles check 7,560 projection samples at the unchanged
`2e-6 + 4e-6*sum_abs` bound; maximum error fraction is 0.006058.
Twelve deliberate evidence corruptions reject.

## Measured operator span

Four fresh processes run serially: warm control-first, warm candidate-first,
pressure control-first and pressure candidate-first. Each contains five warm
pairs and twenty measured pairs per physical mask: 480 measured commands and
120 warm commands total. Every pipeline compiles before collection; normal
kernels alone run during timing and every measured product is checked exactly.
Pressure is a separate 256 MiB read/write command excluded from projection GPU
time. It does not establish hardware cache coldness. Thirteen timing
corruptions reject.

| Actual selected rows / mask | Control GPU median ÷ fused, all four processes | Declared ≥3% time-reduction gate |
|---|---:|---|
| One, physical row 0 / `1` | 1.007–1.251× | Fails |
| One, physical row 3 / `8` | 1.004–1.029× | Fails |
| Four / `15` | 1.050–1.058× | Passes |

The first warm row-0 observation is variable; it is retained, not selected as
the representative C1 gain. The overall gate fails because C1 does not meet
the original criterion. The same criterion passes for C4 in all processes.
These are gate/up/SiLU component ratios, not whole decode or serving gains.

## Why four-row work also matters to single-request speculation

A read-only census of the earlier qualified native Qwen2.5-Coder draft traces
finds 1,125 instrumented target-verification frames per cohort. All submit a
four-token profile; 1,121 use 161 blocked projection dispatches and four use
only one. Native integration must inspect actual active rows and keep the
narrower fallback; a submitted shape alone does not prove occupancy.

The covered target GPU intervals account for about 97% of their corresponding
wall intervals. Reconciliation clearing takes about five milliseconds across
the complete 1,179-round campaign. Draft decode GPU work totals about twelve
seconds inside thirteen seconds of measured draft decode wall time. Other
draft wall time mixes prefill, tokenization and other uninstrumented work.
Fifty-four target fallback frames lack GPU measurements and remain explicitly
excluded from GPU coverage. Five malformed cost traces reject.

The first census rejected an incorrect assumption that every instrumented frame
had the full blocked route. Its error output is retained; the correction keeps
all four narrower frames. No inference was rerun. These eight coding cases
differ from the frozen 2K/128 HTTP workload and establish no vLLM/SGLang speedup.

The next native candidate is restricted to four actual active Mistral gate/up
rows. It still needs logical consumer/alias proof, allocation-free preparation,
complete native parent and Abort tests, ordinary fallback, full model and
verification/rollback parity, and serial serving qualification. Ordinary C1
remains unchanged. Strict coding is still 0/8 and all global targets are unmet.

Prepare a fresh source-only study, without GPU execution:

```sh
python -B prepare_fresh1.py /Users/shawn/CantorAI/out/research/mistral7b-local-coder/FRESH-STUDY
```

Replay all three complete saved reports, without GPU execution:

```sh
python -B replay_saved1.py /Users/shawn/CantorAI/out/research/mistral7b-local-coder/FRESH-REPORTS
```

The saved study is
`/Users/shawn/CantorAI/out/research/mistral7b-local-coder/mistral-fused-q4-decode-screen1`.
`qualification1.json` pins the operator sources, complete audits and decision.
