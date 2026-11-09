# Optional native completion-first prefill policy

The frozen four-client Mistral campaign has markedly different first-token
distributions. Median TTFT by sorted request rank is approximately
14.37/14.37/14.71/14.71 seconds for Garnet,5.66/9.67/13.47/15.62 for
vLLM-Metal and4.59/8.92/13.04/16.29 for SGLang-MLX. All use a512-token
prefill budget. Reference submissions within each wave are nearly simultaneous.
Native logs prove four-owner128-token prefill steps. Reference first-token
staggering motivates this experiment; it does not dynamically prove their
internal schedule or per-operator kernel choice. TTFT is not isolated prefill
kernel time. The saved-data audit retains all bound input hashes.

This candidate adds an optional C++ scheduling policy to the frozen native
engine, without altering kernels or xModel operators. One bounded prefill
owner resumes at the front of the queue until its prompt finishes. Ready
decode batches still execute first and may contain all four leased rows.
The default remains the existing round-robin prefill policy.

The API is `set_prefill_completion_priority(bool)`, exposed by the Metal native
scheduler. It can be configured only before any request is tracked. It requires
bounded chunks, disallows unbounded chunk reconfiguration, and is incompatible
with grouped bulk prefill. Scheduler status reports the configured policy.
No physical rows move, and row epochs, token-prefix restore, acknowledgment,
completion and cancellation retain their existing ownership checks.

## Qualification performed

The six-file native patch is generated against the frozen fused-MLP source.
The complete26,294-file source is reproduced exactly by the packaged generator.
It adds the portable scheduler field/setter and queue behavior, the Metal API
bridge/status/backend marker, and tests in the existing portable fixture.
No build uses the dirty Garnet working tree.

The new portable tests cover the disabled policy, bounded-chunk configuration,
FIFO prompt resumption, ready decoder priority, partial/final acknowledgments,
cancellation, abandon/retry, stable leases, epoch changes on row reuse, full
prefix hits, live configuration rejection and bulk incompatibility. The isolated
Metal build passes all52 native tests serially, including the unchanged tests.

One fresh candidate engine then serves the unchanged frozen2K-input/four-client/
128-output requests in three sequential rounds: cold,warm1,warm2. All12 responses
and1,536 token IDs/text match the saved FP32 native control exactly. The independent
audit verifies full native/client stream correspondence, row/epoch/progress
lifecycle, projection/attention/fusion policies, actual four-row decode batches,
FIFO prefill owner order1..12, and148 prefill steps preceded by all ready decoders.
The run executes6,144 native fused MLP blocks. Owners close cleanly within the
existing4MiB allocation-close limit. Eight deliberate admission-audit corruptions
reject, and the earlier scheduling audit rejects four corrupted inputs.

Source/model hashing happens before the full admission collection and after
terminal shutdown. No profiling or state-witness copying is added to the run.
The HTTP client retains the existing small request-body hash after submission.
No vLLM/SGLang inference is executed.

## Observations and decision

The table contains candidate observations only, from one fresh process. These
are admission diagnostics, not repeated or matched-control performance claims.

| Round | Median TTFT s | Worst request TTFT s | Decode/request tokens/s | Aggregate tokens/s | Delivery-gap p95 s |
|---|---:|---:|---:|---:|---:|
| cold |11.88 |19.83 |7.99 |17.23 |0.325 |
| warm1 |12.50 |21.03 |7.46 |16.24 |0.358 |
| warm2 |12.86 |21.53 |7.29 |15.86 |0.363 |

First tokens are staggered, as intended. The later request waits substantially
longer, and throughput/decode observations are worse than the saved Garnet
campaign. Different campaign times and uncontrolled thermal/OS conditions
prevent attributing a quantitative change solely to scheduling. There is no
evidence to promote this128-token policy as a general performance improvement.
**The default remains unchanged.** A lower median TTFT alone would not satisfy
the speed, throughput, tail-latency and quality objectives.

Canonical GPU KV contents have not been independently witnessed for this new
schedule. Real cached-prefix lineage and coding-quality qualification are still
pending. Portable prefix-hit tests and exact output IDs do not substitute for
those gates. The native code remains in a reproducible experimental engine;
it is not integrated into the production source at HEAD.

The next distinct candidate can use the already-supported prepared owner
profiles up to512 tokens under the same global512-token bound, then qualify
model/cache lineage and compare the full latency/throughput distribution.
That candidate has not been run by this checkpoint.

The latest qualified serving observations remain the
[fused MLP campaign](../mistral_fused_mlp_serving_benchmark/README.md).
Strict coding remains0/8. The2×/half-TTFT, production integration, ANE/Gemma
and original Qwen objectives remain open.

## Reproduction

The generator requires pinned sibling research studies, the Apple build tools,
and a fresh directory under `out/research/mistral7b-local-coder`. All build and
GPU steps are serial relative to inference. Do not rerun unchanged inference
to verify saved evidence.

```sh
python3 -B prepare1.py /absolute/path/to/mistral7b-local-coder/FRESH-STUDY
python3 -B /absolute/path/to/FRESH-STUDY/build_native1.py
python3 -B /absolute/path/to/FRESH-STUDY/prepare_admission1.py
python3 -B /absolute/path/to/FRESH-STUDY/run_admission1.py
python3 -B audit_admission1.py /absolute/path/to/FRESH-STUDY
```

The two auditors accept `STUDY FRESH-REPORT.json`. Both reproduce their complete
saved reports without inference. The generator is separately checked against
all26,294 qualified source file hashes and the complete native patch.
Local evidence is `mistral-prefill-completion-policy1`, with replay studies
`mistral-prefill-completion-replay1` and
`mistral-prefill-completion-replay-source1`. The original11 dirty working-tree
files remain byte-identical.
