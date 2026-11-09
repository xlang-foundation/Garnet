# Within-engine contiguous softmax diagnostic

This diagnostic extends the default-off experiment retained at `3667eeb38`.
Apply `diagnostic-switch1.patch` to the separately admitted native source from
`mistral-contiguous-softmax-native1/source1`, after verifying that study's source
manifest. Do not apply to or build the dirty working checkout.

The four-file patch adds a per-context boolean for selecting the two existing
precompiled kernels. It changes no shader or xModel operator. The session
exposes `diagnostic_contiguous_softmax_set_enabled(bool)` and a read-only
`diagnostic_contiguous_softmax_enabled` property. Session mutex/readiness guards
reject closed/poisoned sessions. Context guards reject open/partially submitted
batches before changing the flag. Compiled-out builds reject enabling it.
The profile's compile-time default remains OFF. Compiled-ON contexts retain their
previous enabled default. Other contexts keep independent selection state.

Native tests cover defaults, per-context isolation, invalid open/partial changes,
exact words across OFF/ON/OFF changes, abort recovery, unchanged prepared counts
and compiled-out rejection. All50 ON tests and the relevant OFF native test pass.

Run through XLang with positional arguments:
`run_paired_body.py LIB SOURCE FIXTURE NEW_OUTPUT OWNERS INITIAL_ORDER`.
OWNERS is1 or4 and INITIAL_ORDER is0 or1. Compile all pipelines and seed15 actual
chunks before timing. Three warmup pairs precede nine alternating measured pairs
for the same128-token chunk at1920. The collector must serialize all engines,
finish hashing/builds beforehand, and retain process ordering and raw samples.
No baseline inference or profiling may overlap collection.

`python3 audit_paired_body.py ABSOLUTE_STUDY` verifies the saved four-engine
collection without GPU work. It checks process order/bindings,72 body calls,
actual route counters, stable allocation/cleanup, and520 complete KV/hidden
hash witnesses against the previous frozen original-kernel control.

Paired median original/new body ratios are1.021/1.035 at one active owner and
1.018/1.027 at four. Some individual pairs regress. These are small descriptive
body effects, excluding embedding/head/hashes/HTTP/scheduling. They do not prove
full-prefill or serving speed, half TTFT,2x throughput, useful coding quality,
prefix-return/cache-pressure admission, ANE, Gemma or multimodal qualification.
The extension remains default-off and the saved serving table is unchanged.
