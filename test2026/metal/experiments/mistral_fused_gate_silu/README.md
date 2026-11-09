# Fused affine gate/up, SiLU and half down-input screen

This isolated C++/Objective-C++/Metal experiment finds a useful block-level
candidate for the remaining Mistral prefill cost. It changes no runtime,
production default, public operator, xModel file or frozen comparator.
The retained study is `out/research/mistral7b-local-coder/mistral-fused-gate-silu-screen1`.

The earlier paired projection still writes two full FP32 matrices and runs
SiLU separately. This candidate retains the gate/up accumulators in registers,
evaluates the original SiLU expression in the GEMM output stage, and writes
only the compact half input that the down projection consumes. It removes
the two FP32 intermediate writes, subsequent SiLU reads/write and the final
down-input gather/cast. One input gather is shared by both projections.
The down projection itself is outside this screen.

## Numerical and ownership gates

The preparation script derives the candidate from the pinned paired shader.
It preserves strict cooperative descriptors, coefficient dequantization,
half operands, FP32 accumulation, K16 reduction order and the original row
maps. Only the output stage changes. M128/M256 tiles retain17,408 bytes of
threadgroup storage and256 threads. All buffers have bounded up-to512 arenas;
actual row counts vary without padding computation to512.

The original packed projections compile with MSL4 Safe. This standalone
screen extracts `silu_mul` verbatim but explicitly compiles it as MSL3.1.
The subsequent [real trained MLP replay](../mistral_fused_mlp_trained/README.md)
found that the selected native build uses nil/default options instead. The
control here is a compiler-setting surrogate. Both settings match numerically
on subsequent real captured data; performance equivalence is unproven.
The candidate is screened under both Safe and Fast options:

- Fast preserves every tested gate/up FP32 shadow and final half output bit.
- Safe preserves the gate/up shadows but changes14,121 half words across
  shadow/normal/writer observations. It is rejected under the exact-bit gate.
  No numerical tolerance is relaxed to admit it.

The complete screen contains36 cases: nine row counts
127/128/129/255/256/257/383/511/512, synthetic coefficients including signed
zeros/subnormals, and actual trained Mistral weights from layers0/15/31.
The input corpus is frozen synthetic finite input, not trained MLP activations.
The127-row case checks a partial kernel tile; it does not establish that
native routing should select this kernel below the existing128-row threshold.

Independent saved-array audits check1,585,460,736 half-word comparisons,
1,061,690,112 FP32 shadow-word comparisons,528,484,608 retained writer-map
words, and1,080 independently recomputed FP64 samples. The maximum sampled
projection error uses0.00804 of the unchanged2e-6+4e-6*sum_abs allowance.
Physical row permutations, strided parents, prefix/suffix guards, unused
output rows, full writer coverage and an intentionally truncated grid are
checked. Nine corruptions reject. The Fast candidate alone passes to timing.

## Corrected operator timing

Four fresh processes run serially in balanced order: synthetic-forward,
trained-layer0-reverse, trained-layer0-forward, synthetic-reverse. They compare
the candidate against both original separate and original paired projections,
each with ordinary SiLU and the actual existing profile32 SiLU source.
Each shape/method has three warmups and nine measured commands:540 measured
and180 warm commands total. No vLLM/SGLang inference is repeated.

The span includes gate/up projection, SiLU and preparation of compact half
down input. Down GEMM, other layer work, scheduling and HTTP are excluded.
Every command's complete half output is compared with the frozen correctness
control after the measured span. Only normal kernels run during timing;
shadow stores and atomic writer probes are disabled.

| Actual compact rows | Separate/explicit-MSL3.1 GPU median ÷ fused | Paired/profile32 with explicit-MSL3.1 GPU median ÷ fused |
|---:|---:|---:|
| 128 | 1.251–1.478× | 0.928–1.134× |
| 256 | 1.285–1.291× | 1.128–1.133× |
| 512 | 1.258–1.263× | 1.108–1.115× |

Ranges retain all four process observations. One128-row trained-layer0
reverse observation is variable and the fused candidate loses against the
stronger paired/profile32 control there. Do not remove it or claim a uniform
128-row gain. The512-row result motivates validation with actual native
compiler settings and full model execution. GPU durations
come from Metal command start/end timestamps; they do not prove occupancy,
bandwidth saturation or a whole-model speedup. Wall intervals and all raw
samples remain in the saved report. Ten timing corruptions reject.

The first timing campaign is preserved but excluded. Its control dispatched
SiLU over512 rows even in128/256-row cases and redundantly cleared dense
fully selected outputs. That overstates the benefit relative to native
compact families. `host3.mm` corrects the pointwise extent to actual rows and
removes those clears; it uses a fresh balanced campaign. The first samples
are not included in the table. `timing1-assessment.json` records this history.

All source/asset hashing and compilation precede the full timing campaign;
postflight runs after every child terminates. The original11 dirty checkout
paths remain unchanged. The two packaged auditors reproduce the retained
reports exactly. Results and scope are recorded in `qualification1.json`.

## Remaining work

This is an operator candidate. Real trained MLP activations, complete down
projection agreement, safe compiled-graph fusion eligibility, observable
intermediates/aliases, asynchronous failures and recovery, memory leases,
complete model states, output tokens and prefix-return/batch transitions
still require validation before native serving admission. The existing
strict coding result remains0/8. No new TTFT, decode, aggregate throughput,
ANE/Gemma, multimodal or Qwen qualification is established. The serving table
at checkpoint `d6486e5a3` remains the current end-to-end result.

Audit saved artifacts without inference, with fresh report destinations:

```sh
python3 -B audit_correctness1.py ABSOLUTE_STUDY FRESH_REPORT
python3 -B audit_timing2.py ABSOLUTE_STUDY FRESH_REPORT
```

To prepare a genuinely new concern/change on the same pinned workspace,
choose an unused sibling study directly under `out/research/mistral7b-local-coder`:

```sh
python3 -B prepare1.py FRESH_STUDY
python3 -B compile1.py FRESH_STUDY
python3 -B FRESH_STUDY/run_correctness1.py
python3 -B audit_correctness1.py FRESH_STUDY
python3 -B FRESH_STUDY/run_timing2.py
python3 -B audit_timing2.py FRESH_STUDY
```

Preparation needs the existing asset Python environment with NumPy and
safetensors. Hosts require Xcode27.1/Metal4 on Apple10/M5. Paths intentionally
refer to the pinned local studies. No automatic inference retry or baseline
execution occurs. Do not repeat this unchanged screen after its gates pass.
