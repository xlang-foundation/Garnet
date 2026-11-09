# Experimental native scheduled dense verification

The opt-in `GARNET_METAL_DENSE_VERIFY_SCHEDULER` extension connects the target
verifier to Garnet's native request scheduler. The new C++ API is exported as
`continuous_scheduler.verify_decode_json(hidden, positions, slots, active,
tokens, stops)`. It derives known tokens, remaining budgets and row epochs from
the native schedule, checks proposal masks/positions, runs exact target
verification and rollback, proves accepted token-ledger progress, captures an
eligible prefix, and commits the emitted span before releasing terminal leases.
It adds no xModel operators or weight/GPU-workspace copies. The option defaults
off and requires verification, continuous rows and token-prefix tracking.

The new B1/T4 split-score path passed 24 extra ownership cases: unsubmitted and
partially submitted Abort, foreign pending writers, rejected route changes,
complete recovery, and cleanup. Both scheduler builds passed 53 native tests.
Managed full-model qualification matched every capped target ID and all 65
complete KV/hidden digests against current ordinary decoding, while 12 invalid
API calls preserved scheduler usability. The 128-token state witness uses one
native output-token headroom and is cancelled after capture. A separate actual
budget 2 request proves terminal release. This qualification is untimed.

The persistent B1 HTTP worker uses the same native scheduler API, native C++
tokenizers, the full target prompt and a fixed bounded helper context. Two fresh
engines served six serial 2K-input/128-output requests. All 768 output IDs matched
the native control. Independent audits check helper causality/resident prefix
matching, scheduler epochs/budgets, accepted rollback and raw SSE timing.

| Warmed single-request diagnostic | TTFT | Decode | Overall |
|---|---:|---:|---:|
| Garnet B1 scheduled speculation | 3.512 s | 23.773 tokens/s | 14.438 tokens/s |
| Frozen vLLM-Metal reference | 4.219 s | 20.751 tokens/s | 12.345 tokens/s |
| Frozen SGLang-MLX reference | 4.159 s | 20.737 tokens/s | 12.518 tokens/s |

These are diagnostic comparisons. The new engine admits one native owner at a
time; the references permit four concurrent sequences. HTTP arrival buffering
is not native C4 batching. The prior full serving comparison is not promoted.
No C4 speculative throughput or HTTP full-KV witness is claimed. Reference
inference was never rerun. Neither 2x throughput/decode nor half TTFT is achieved.
Coding quality and new-bridge returning/append/changed-prefix lifecycle remain
open. Four requests with four verification tokens require an up-to-16 physical
row profile; the current bound remains four. Compact owner prefill coexistence
also needs qualification. Keep this extension experimental and default off.

`native/native1.patch` applies to the pinned split-score native snapshot. All
26,300 source-file contents were reconstructed from the base and the six-file
patch. Full source, build commands, binaries, raw streams, earlier harness
failures and model bindings remain in frozen local studies listed by
`qualification1.json`. Large artifacts and model weights are not committed.
Run `replay_saved1.py FRESH-REPORT-DIRECTORY` with the asset-environment Python
to reproduce the three complete saved audits without inference.
