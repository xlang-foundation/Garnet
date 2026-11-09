# Experimental compact prefill and verification coexistence

`GARNET_METAL_COMPACT_VERIFY_COEXISTENCE` defaults off. It permits the prepared
compact-owner prefill families to coexist with the native B4/T1 target verifier
on the tested Mistral envelope: 32 layers, width 4096, H32/KV8/D128, capacity 4096
and vocabulary 32768. Both preparation orders are supported before any row claim
or execution. Other geometries and preparation after a claim remain rejected.
No shader arithmetic, xModel operators, weight copies or public API signatures
change. Verification bypasses compact prefill routing; the existing compact
swap restores the complete base session before verification runs.

| Qualification | Evidence |
|---|---|
| Native C4/T1 ownership | All four decode occupancies; 104 exact output IDs; 2,340 full-state digest pairs |
| Compact then verify, and reverse order | 102 output IDs per arm; 5,200 full-state digest pairs versus current ordinary |
| Rejected preparation | Default-off combinations, late preparation, unsupported capacity; 1,300 unchanged-state digest pairs |
| B2 interleave | Rows 1/2 compact spans while row 0 decodes; 12 exact IDs and 780 full-state digest pairs |
| Native builds | 53 tests passed with extension enabled; 53 passed with default off |

The engine prepares all 17 compact families through 512 during initialization.
Actual native chunks include 4, 128, 256 and 512; B1 uses 256/512, and B2 uses 128/256.
Every tick stays within the 512 actual-token global budget. No request-time
profile compilation occurs. Lifecycle witnesses cover terminal release,
queued refill, cancellation and surviving requests, using all four complete
KV/hidden rows. The first fixture missed four-way decode because A finished too
early; its failed coverage audit is retained, and a changed A budget covers the
missing path. The extra B2 geometry probe uses an empty stop list and bounded
outputs; primary model qualification uses the model EOS token 2.

This is a correctness checkpoint. There are no new serving speed measurements.
C4/T4 speculation needs an FP32 profile for up to 16 physical token rows, including its
vocabulary head. Raising bounds must not route target verification through FP16
MPP. Prefix retention and its profile identity still need new-bridge lifecycle
qualification. The current fixture uses tracking only. Coding quality, the 2x
performance targets and production promotion remain open; keep the option off.

`native/native1.patch` applies to the pinned scheduler-bridge native source.
All 26,300 source-file contents were reconstructed from the base and five-file
patch. Frozen local studies retain full source/builds, raw witnesses, model
bindings and failed coverage evidence; `qualification1.json` binds those paths
and the curated snapshots. Run `replay_saved1.py FRESH-REPORT-DIRECTORY` using the
asset-environment Python to reproduce all three complete audits without inference.
