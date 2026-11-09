# Actual-model KV hoist failure recovery

This extends the experimental early-commit stage at16cb55b32. It does not change
attention arithmetic or implement two-product attention. The new
`GARNET_METAL_KV_HOIST_TEST_FAULTS` option defaults OFF and requires the hoist
experiment. Compile it only for qualification. All native shader literals,
xModel operators, layouts and GPU allocations remain unchanged.

Prepare a clean tree with
`python3 prepare_native_source.py REPOSITORY ADMITTED_SOURCE ADMITTED_MANIFEST NEW_SOURCE`.
The preceding stage's pinned source preparation is reused, then the seven-file
`test-faults1.patch` is applied and every changed source hash verified. Never
build the dirty worktree. Exact configurations/builds and binaries are retained
in `out/research/mistral7b-local-coder/mistral-kv-hoist-failure1`.
Both fault-enabled and disabled builds pass all50 native tests each.

Test-only session APIs are `diagnostic_arm_kv_hoist_fault(mode)` and
`diagnostic_kv_hoist_fault_witness_json()`. Modes0/1/2 disarm, throw after encoding
the two moved commits without submission, or flush that work then throw.
Arming requires a ready session/drained batch and is single-shot. Full reset
also disarms an unfired request. The witness retains encoded/submitted/drained/
completed counts and pending-work status through native abort. Its GPU duration
is an execution-completion witness, never a performance measurement.

Run through XLang:
`run_fault_model.py LIB SOURCE MODEL FROZEN_FIXTURE NEW_OUTPUT`.
The actual32-layer Mistral engine prepares B4/C4096/T128. It records healthy
128/256-token states and predictions, caches all four128-token prefixes, then
runs both failure modes. Eleven follow-up generation/read/reset/release/cache/
arming calls reject each failed session. Tracked-token/prefill queries return
invalid-1 sentinels; they do not throw. All four cache entries and metadata remain
unchanged. Full reset clears every physical KV matrix/hidden bank, old epochs
are rejected, cached prefixes restore under fresh epochs, and continuation
exactly matches healthy full state/predictions. A separate arm-then-reset check
proves no latent fault fires on the next valid request. All native owners close.

The saved initial worker incorrectly expected a tracked-count exception, stopped
in mode1 and exited1. The corrected worker requires the documented-1 sentinels,
records partial results on errors and completes in a new process. No native
source/math/guard was changed for this correction. Raw first-run evidence is
retained. The initial saved-data auditor also assumed an explicit `head_dim` in
Mistral's config; the final auditor derives128 from4096/32. That correction starts
no GPU work and changes no collected data.

`python3 audit_fault_model.py ABSOLUTE_STUDY` independently verifies the saved
qualification. It binds source/library/worker/fixtures, checks both test modes,
and anchors the healthy128-token state to the prior qualified engine. It checks
65 prior state digests,325 complete-state zero hashes against independent IEEE
F32-zero byte references,520 exact prefix-restoration/continuation hash pairs,
15 stale-epoch/uninitialized-prediction rejections, and seven corrupted-record
negative controls. Final native owners are zero; post-close Metal bytes1,900,544.
The existing framework CoreAnalytics warnings remain unexplained in raw logs.

This covers two explicit software fault sites at the first eligible layer.
Other hardware errors/cancellation, end-to-end serving failure propagation,
cache pressure and general nonfinite arithmetic are not qualified. No timing,
coding quality, default promotion, ANE/Gemma/multimodal or Qwen admission follows.
The production fault option remains OFF. The next substantive stage is a
separately identified zero-copy two-product numerical profile and its primitive,
model, cache/state, quality and serial serving measurements.

A separate native `fault_context_guards.cpp` links the admitted fault-enabled
static library using its exact CMake compiler definitions. It verifies invalid,
open-batch and partly-submitted arming rejects without changing the pending arm,
then disarms and successfully reuses the context after abort. Exact compile/run
commands and zero-owner success are retained. The final auditor also checks it.
