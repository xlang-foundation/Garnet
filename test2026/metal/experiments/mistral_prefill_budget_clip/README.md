# Native prefill budget clipping experiment

The completion-first scheduler previously selected a 256-token prefill shape
when one to three ready decoders left 511, 510, or 509 actual tokens of its
512-token budget. This optional native overlay separates the prepared tensor
extent from the actual work: it can execute 509 tokens using the prepared
512-token profile while three requests decode. No model operators, precision
rules, workspace limits, or physical row identities change.

`GARNET_METAL_PREFILL_BUDGET_CLIP` defaults to OFF. Native
`set_prefill_budget_clip(true)` requires completion-first scheduling, four leased
rows, and qualified wide half-attention profiles. Ready decode runs first and
actual work remains bounded by 512 tokens. This directory checkpoints an isolated
source overlay; it does not enable or integrate the option in the production
runtime. It is based on the frozen wide-half engine, without the N4 weight cache.

Validation passed: 53/53 native tests with the option ON and OFF, 7,877 portable
checks, and complete half-attention output/KV-parent comparisons for 509/510/511
active rows. A new 32-output-token fixture keeps three decoders alive: all 128
output IDs/text, 520 per-request state digests, and 32,768 hidden values match the
original native control. Four executions of each residual count were observed.
The extended control also reproduces the saved eight-token prefixes and prompt
states. Fourteen model-evidence corruptions are rejected.

The serial HTTP admission produced 12 responses and 1,536 exact saved FP32 output
IDs/text; twelve evidence corruptions are rejected. The vLLM-Metal and SGLang
baselines were not rerun. Input/source hashing and builds ran outside collection.

| Candidate warm wave | TTFT (s) | Decode/request (tokens/s) | Aggregate output (tokens/s) | Maximum delivery gap (s) |
|---|---:|---:|---:|---:|
| Clipping, warm 1 | 9.266 | 8.577 | 20.672 | 0.986 |
| Clipping, warm 2 | 9.593 | 8.391 | 20.131 | 1.005 |
| Saved wide policy, warm 1 | 9.340 | 8.899 | 20.813 | 0.522 |
| Saved wide policy, warm 2 | 9.614 | 8.596 | 20.149 | 0.543 |

These are one-process diagnostics from different collections with uncontrolled
OS/thermal conditions, not a causal speedup test. Prefill invocations fell from
90 to 57 across three waves, but this does not establish a throughput gain.
Keep clipping disabled: longer mixed-stage delivery gaps give another reason
to use measured execution cost and decode latency, rather than utilization alone.

Saved per-token SSE timestamps also distinguish the common streaming window
after every first token and before the first final token. Per-request rates in
that window are 12.897–13.202 tokens/s for clipping, 13.138–13.503 for the saved
wide policy, and 13.674–14.006 for N4. This window includes HTTP/CPU work; it is
not isolated GPU decode timing. Five malformed-timestamp/counter controls reject.

Fresh native source reproduction, without building or inference:

```sh
python -B prepare_native1.py /Users/shawn/CantorAI/out/research/mistral7b-local-coder/FRESH-NAME
```

This verifies all 26,296 source files against the recorded content digest, then
prepares build/model runners and absolute fixture/inventory paths. The build
runner executes both native suites serially. Serving snapshots remain bound to
the saved study; a fresh build needs fresh library/profile bindings before any
new serving campaign.

Replay all three full saved audits without inference:

```sh
python -B replay_saved1.py /Users/shawn/CantorAI/out/research/mistral7b-local-coder/FRESH-REPORTS
```

`qualification1.json` pins the source, library, audits, and decision. The saved
study is `/Users/shawn/CantorAI/out/research/mistral7b-local-coder/mistral-prefill-budget-clip1`.
Full native prefix lineage for this engine and repeated serving qualification
remain pending. Strict coding stays at 0/8; no 2× target or global goal is met.
