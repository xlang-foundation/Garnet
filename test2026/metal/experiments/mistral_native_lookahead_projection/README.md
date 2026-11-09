# Separate native lookahead projection profile

The default-off projection entry prepares fixed dimensions, strides and up-to-B*16 physical row capacity. Actual selected occupancy is runtime data derived from immutable request views. It retains the existing four-row weight reuse, dequantization, FP32 FMA and SIMD reduction body, with a separate specialized pipeline/cache and no inference-time compiler fallback. Inactive outputs are initialized, including the empty batch.

The original indexed decode API keeps its 4/16-row limits. A preprocessing check found identical legacy wide-kernel tokens with the new shader mode absent. Actual native tests additionally passed both enabled/disabled wide and lookahead-attention regressions. The new test passed 820 cases over all five supported Mistral projection shapes, B1..4 and each selected occupancy through B*16, using ordinary native FP32 decode as the bitwise reference. It covers poisoned inactive rows, pending/partial flush, abort, invalid controls and owner cleanup.

This component does not validate a Session's actual row epochs or model generations. It is not yet used by a full model's lookahead graph and makes no speed, coding or full-model state claim. `qualification.json` records exact scope and local raw evidence hashes.
