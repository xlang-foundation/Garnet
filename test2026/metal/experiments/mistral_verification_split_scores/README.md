# Experimental Mistral split verification scores

The full 2K target context remains intact. A fixed 192-token user-content tail,
rendered with the helper chat template, makes the 512-token Qwen2.5-Coder-0.5B
helper usable on this workload. The helper base has 221 tokens. Only already
generated target text is appended; the native target verifies every proposal.

The new native option, `GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES`, defaults off
and requires the prepared Mistral verification layout. At visible length >=512,
it splits B1/T4/H32/KV8/D128/C4096 QK scores into 128-position tiles, then runs
the original full-row softmax. It reuses the probability buffer, preserves the
per-key FP32 reduction and compiler policy, and allocates no extra GPU workspace
or weight copy. Other shapes and shorter prefixes use the original path.

Two fresh isolated operator runs checked 440 mask/layout/boundary cases and
230,704,320 complete probability-parent word pairs exactly. Both native builds
passed 53 tests. Full-model qualification and four fresh OFF/ON, ON/OFF stages
checked every target ID and 520 complete KV/hidden digest pairs against the
current ordinary target. Causal helper conditioning, resident prefix matching,
acceptance and rollback are independently audited with corruption controls.

| Native bounded stage | Split off | Split on | Decode rate ratio |
|---|---:|---:|---:|
| Pair 1 | 20.34 tokens/s | 24.69 tokens/s | 1.214x |
| Pair 2 | 20.97 tokens/s | 24.27 tokens/s | 1.157x |

These are native-stage measurements on one frozen 2K case, not a matched HTTP
comparison. No new serving TTFT or C4 throughput is admitted. The frozen
vLLM-Metal/SGLang baselines were not rerun. Coding quality, serving integration,
new-build prefix lifecycle and the 2x goal remain open. Keep the option off.

`native/native1.patch` applies to the pinned prior native source. The complete
26,300-file content map was reconstructed from the base and the eight-file patch.
`qualification1.json` binds local frozen raw evidence, source snapshots, libraries
and complete audits. Large buffers, captures, binaries and models stay in those
frozen local studies. Earlier preparation/replay failures are retained.

Run `replay_saved1.py FRESH-REPORT-DIRECTORY` with the pinned asset environment
Python to reproduce all six complete saved audits without inference. Native
build commands and exact configuration are retained in the frozen native study;
build that isolated snapshot, with GPU tests and model stages strictly serial.
