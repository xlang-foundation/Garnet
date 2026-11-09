# Native half-prefill GPU attribution

This is an instrumented diagnostic of the native half-prefill candidate from
checkpoint `644b5bb6d7a4d3186b03a0dcfd22c57ac1edffc0`. It changes no native
runtime code and is not a new benchmark, TTFT or throughput result. The retained
study is `out/research/mistral7b-local-coder/mistral-half-prefill-attribution1`.

One fresh serial M5 engine runs four active owners, fifteen actual prefix
chunks, three warmups and one offset1920 diagnostic call through all32 layers.
Native encoder timestamp counters are prepared before claiming request rows.
Nineteen body records and one head record are retained. The final body record
has1,026 encoders, including32 packing and32 half-attention encoders. There are
no executed MPS attention products. Counter calibration, bounds, command IDs,
duration consistency and complete route counts are audited. Seven deliberate
counter corruptions reject. All complete KV/hidden digests,2,097,152 hidden
values and next predictions match the uninstrumented candidate. Native owners
close to zero and device allocation remains under4MiB.

The final instrumented body spans950.38ms. Timestamp intervals cover947.38ms
as a union, with5.87ms of cross-category overlap and3.00ms of uncovered gaps
inside the encoder envelope. Encoder durations are not summed as elapsed time.

| Category | Exclusive share of timestamp-covered union |
|---|---:|
| Projection work, including its gathers | 74.64% |
| Half attention | 7.66% |
| SiLU | 6.60% |
| Q/KV half packing | 2.95% |

Other exclusive categories and unassigned cross-category overlap account for
the remainder. These shares describe this instrumented timeline. They do not
prove occupancy, bandwidth saturation, isolated causal speedups or proportions
of uninstrumented serving time. The separate validated prefill-body gain
remains1.18–1.22×; the2× and half-TTFT targets remain unmet.

Projection work is the next priority. The source comparison verifies the
pinned vLLM-Metal checkout and its installed MLX0.32.4 NAX headers. Garnet's
packed half-input/half-weight/float-accumulator descriptors hardcode strict
precision; the inspected MLX NAX fragment descriptors use relaxed precision.
Both stage dequantized weights in threadgroup memory. An isolated descriptor
experiment must verify the actual cooperative maps, complete guarded outputs
and fixed numerical bounds before timing. This code difference alone is not
proof of the frozen baseline's per-operator kernel selection or its speed.

The subsequent [packed descriptor screen](../mistral_packed_descriptor_study/README.md)
passes those correctness checks but finds no useful speed gain in two balanced
cohorts. The descriptor-only hypothesis is closed for these kernels; keep the
runtime unchanged and do not repeat it unchanged.

The comparison also inventories prior packed64, full-tile and loop-control
results, which do not establish the needed512-row MLP gain. Do not rerun those
unchanged candidates. Full-model coding quality, prefix return, continuous
batching, HTTP serving, ANE/Gemma and the original Qwen objectives remain open.
No vLLM/SGLang inference was repeated.

`audit1.py STUDY [FRESH_REPORT]` audits retained data without inference. The
runner and worker retain the exact diagnostic recipe and bindings; they need
a fresh study destination. Instrumentation changes execution timing, so its
results must remain separate from the uninstrumented body comparison.
