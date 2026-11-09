# Bounded native lookahead attention

This default-off component specializes prefix/local attention for a fixed native lookahead layout. Canonical prefix K/V stays read-only; projection-order local K/V is separate. All active paths are compacted to logical causal key order before the existing FP32 probability/value boundary. The model-specific graph operations remain unchanged.

Enable `GARNET_METAL_LOOKAHEAD_ATTENTION` together with `GARNET_METAL_VERIFY_ATTENTION`, then build and run `garnet_lookahead_attention_test` in a coordinated serial GPU slot. Preparation specializes batch/capacity/window/levels/guesses; execution validates immutable profile views, shapes, causal closure, and operand ownership before encoding. A future session must enforce actual live row epochs.

The actual native test passed all 82 cases including all 42 admitted geometries, B1/B4, missing owners, capacity/budget tails, and poisoned inactive inputs. Ordinary native attention is the bitwise reference; FP64 samples are secondary checks. The original numerical failure and corrected run are both retained in the pinned local evidence. See `qualification.json` for exact scope and hashes.

This is an attention component, not an integrated decoding algorithm or a performance result.
