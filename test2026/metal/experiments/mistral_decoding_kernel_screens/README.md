# Mistral decoding kernel screens

Two isolated Metal operator experiments on the Apple M5/24 GiB Mac. These
results do not qualify a new inference runtime, coding quality, or serving
speed. Frozen vLLM-Metal/SGLang references were neither changed nor rerun.
The native runtime remains the prior wide-lifecycle checkpoint.

## Result

The strict FP32 affine 8-row by 2-column tile is rejected. Its median paired
speed ratios versus the current 4-row by 4-column tile were 0.922/0.924 at
8 selected rows and 0.740/0.755 at 16 rows across the two fresh cohorts. The
619,158,144 whole-parent word comparisons were exact, including sparse and
permuted maps, selected counts 0 through 16, offsets and row padding. It
passed 432 independent FP64 samples and 11 corruption checks. Increased
register pressure is a hypothesis; this screen did not measure occupancy.

The GQA16 candidate shares one 32-key by 128-dimension tile (16 KiB) across
four query tokens and four query heads per KV head. It writes raw dot
products into the existing probability buffer and applies the original
scale when loading the unchanged full-row softmax. Per-lane accumulation,
SIMD reduction order, and the separate FP32 probability boundary remain.
Compiler options match native attention (`nil`/default).

| Verification score + softmax, ~2K context, four query tokens | Current ms, cohorts 1 / 2 | GQA16 ms, cohorts 1 / 2 | Paired speed ratio, cohorts 1 / 2 |
|---|---:|---:|---:|
| B1 | 0.8142 / 0.8142 | 0.1278 / 0.1296 | 6.37 / 6.29 |
| B4 | 4.6383 / 4.6471 | 0.5443 / 0.5455 | 8.51 / 8.52 |

These are complete score-and-softmax command timings, not token speeds.
The B1 current route uses the native split-128 score path when its maximum
visible prefix reaches 512; B4 current uses native generic verification
scores. Both are also checked against the original single-dispatch score
kernel. The timing protocol retains 17 rounds after four warmups per cell,
rotates three methods, and reverses method and batch order in the second
fresh process. Hashes, builds and file I/O stay outside timed collection.
Thermal and OS conditions are uncontrolled; no end-to-end gain is implied.

GQA16 passed 468 cases per fresh process over B1/B4 and two layouts,
including all uniform four-token masks, mixed owners, unequal first slots,
inactive rows and capacity edges. All 2,541,870,720 parent word pairs match
native exactly; inputs and guards remain unchanged. The independent audit
checks 1,200 FP64 probability head samples and rejects 12 corruptions,
including shared wrong probabilities that preserve row sums. Fresh process
outputs repeat exactly. Full-model state, rollback/ownership, native ON/OFF
suites, causal drafting and matched C1/C4 serving remain required.

## Reproduce

Raw buffers, binaries and logs remain at the immutable roots in
`source-reference.json`; they are intentionally excluded from Git. The
copied `provenance1.json` indexes bind every raw file. Trained affine inputs
are additionally bound to the earlier frozen projection screen; GQA inputs
are generated deterministically by the recorded native harness.

Run `replay_audits1.py FRESH_OUTPUT_DIR` with a Python environment containing
NumPy. It verifies the package and all raw evidence, executes the complete
CPU auditors with only report destinations redirected, and compares entire
reports with the frozen originals. It never starts Metal or a model.

For a new GPU collection, create a new study directory, copy the recorded
sources, bind the same external inputs, then run that study's `run1.py`.
The drivers refuse output reuse and require idle model/benchmark workers.
Never rerun collection into these frozen raw roots.

See `gqa16/decision1.json` for native integration requirements. Keep the goal
active: the 2x decode/throughput and half-TTFT targets are not met by these
operator results. Gemma/ANE, coding-quality qualification and the return to
Qwen 27B remain open.
