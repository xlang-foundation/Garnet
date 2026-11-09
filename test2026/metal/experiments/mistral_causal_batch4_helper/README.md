# Causal four-row helper drafting

Experimental native Mistral-7B target verification with a Qwen2.5-Coder-0.5B
helper. This checkpoint qualifies batching and ownership; it does not enable
unconditional helper drafting as a production default or establish a 2× win.

## Implementation

`helper-native1.patch` applies to the frozen `mistral-gqa16-value-native1`
source, which includes the previously merged upstream main checkpoint. It:

- Admits four dense Qwen request rows when the bounded dense-prefill extension
  is compiled. The helper uses C512 and a prepared T16 prefill family.
- Makes the optional diagnostic hidden-bank getter report its actual initialized
  row count and width, allowing the W896 helper to be inspected.

The review patch uses zero context; apply it with `git apply --unidiff-zero`.
`audit_review_patch1.py` checks that it produces exactly the frozen source edits.

The native C++ runtime owns row leases, token reconciliation and target
verification. The existing xModel operators and inference arithmetic source
remain unchanged. Python constructs graphs and orchestrates the experiment.

## Correctness

- Isolated native build: all 56 tests passed.
- Helper lifecycle: 13 committed-prefix queries, request replacement, stale
  epoch rejection, selective updates and replay of the last retained token.
  Serial and joint execution produced the same 42 helper IDs; identical-batch
  repeats matched 2,932 whole KV/hidden fingerprints. Sixteen audit corruptions
  were rejected.
- Existing FP32 chunk attention switches scalar/MPS arithmetic at a total of
  eight active tokens. Serial and joint helper execution differed in 1,266
  internal state fingerprints. This checkpoint explicitly does **not** claim
  serial/batch helper bit-state equivalence.
- Productive target qualification: four original ~2K prompts, 32 output tokens
  per request; all 128 target IDs and 7,865 target KV/hidden fingerprint pairs
  matched ordinary decoding at corresponding progress. The target accepted
  69 of 156 proposals, reducing target decode calls from 32 to 15. Twelve audit
  corruptions were rejected. Proposals use only committed target output.
- The longer timed runs produced 6,144 exact target IDs across twelve bursts;
  their causal traces passed audit. Full-state hashes were disabled during
  those measurements. Nine trace/timer corruptions were rejected.

## Measurement and decision

Apple M5/24 GiB; four strictly serial engines in ordinary/helper/helper/ordinary
order. Each engine ran one cold and two warmed C4 bursts, with the original
2048/2051/2048/2049-token prompts and 128 outputs per request. Prefix retention
was disabled. Helper work is included. These are **native burst** measurements;
first prediction excludes HTTP/SSE transport.

| Warm median | Ordinary | Batched helper |
|---|---:|---:|
| First prediction, seconds | 14.4174 | 14.7261 |
| Decode, tokens/s/request | 9.4728 | 10.3249 |
| Aggregate, tokens/s | 18.3764 | 18.8255 |

The two paired aggregate ratios were 0.9891 and 1.0531; decode ratios were
1.0508 and 1.1268. Each helper burst accepted 321 of 546 proposals, requiring
49 verifier calls and 167 helper calls. Across-run drift was substantial and
thermal/OS conditions were uncontrolled. The aggregate results do not establish
a consistent gain. The 2× target, matched HTTP comparison and coding-quality
qualification remain unmet. No ANE plugin was used in this experiment.

Frozen vLLM-Metal and SGLang measurements were neither rerun nor modified. The
previous matched HTTP table is unchanged. These native numbers must not replace
its TTFT or throughput values.

## Reproduce the CPU audit

`source-reference1.json` binds these review copies to the complete local raw
roots. Those roots retain failed attempts, complete source maps, native build
evidence and every raw result. Model weights and build products are not copied
into Git. Redundant final blank lines in a review copy may be removed; any such
normalization is recorded and its Python AST is checked unchanged.

Run `replay1.py RAW_QUALIFICATION_ROOT FRESH_OUTPUT_DIRECTORY` with the asset
Python environment. The replay reconstructs all 26,561 source files from the
prior frozen source plus the two-file patch, checks provenance and reruns the
helper, productive-target and measurement audits without inference. It must
produce reports identical to the frozen originals. All engines must remain
serial when collecting any new inference results.
