# Native contiguous-control softmax experiment

Default-off native extension, following the standalone prototype in
`../mistral_contiguous_softmax/`. It retains the original FP32 arithmetic.
The prepared bucket pipeline can use the new kernel only after completed host
controls prove a contiguous active prefix for each owner. Unsupported geometry,
gaps, overwrites, pending controls and full-request execution keep the old path.
No xModel operator changes are required.

`native-candidate2.patch` applies to the admitted source under
`out/research/mistral7b-local-coder/mistral-prefill-prefix-policy1/source1`,
not directly to the current dirty checkout. The retained native study records
its clean Git archive, the 28 verified admitted overlays, all nine candidate
file hashes, exact configure/build commands and full native test logs.
Build only a separately prepared and verified clean source tree.
Set `GARNET_METAL_MPS_CONTIGUOUS_SOFTMAX=ON` to compile the experiment;
its dependencies are prepared MPS attention, context buckets and index32.
The default is OFF. Both build modes passed all 50 native tests.

The research worker patch applies to the verified `worker2.py` in the trained
Metal limiter study; it adds read-only route and state witnesses. The final
`audit_model_state.py STUDY` verifies the saved OFF/ON runs without inference.
It checks 32 generated IDs/text per engine, the saved control, native dispatch
routing, full physical KV/hidden digest witnesses at four boundaries, owner
cleanup and native test evidence. First-decode call17 precedes the sparse
final-prefill call18; the retained initial auditor incorrectly classified
call18 and was corrected without editing or recollecting raw results.

`run_body.py` runs through XLang with positional arguments:
`LIB SOURCE FIXTURE NEW_OUTPUT OWNERS ENABLED`.
OWNERS is 1 or 4; ENABLED is the compiled variant, 0 or 1.
It prepares all profiles, seeds 15 actual prompt chunks, then replays the same
128-token chunk at offset1920 (3 warmups, 9 measured forwards).
The collector must serialize fresh engines; no source hashing, builds,
profiling or other inference may overlap timing.
`audit_body.py BODY_STUDY` verifies the retained 8-engine collection,
all route/state/owner witnesses and process ordering without inference.
Body spans exclude embedding, head, state hashes, HTTP and scheduling;
they cannot substitute for serving TTFT or throughput.

Batch4 body ratios were only 1.007x / 1.043x in reversed-order cohorts.
Batch1 ratios were 0.982x / 1.042x. Drift exceeds the small effect, so no
robust whole-model speedup or production promotion follows. The latest saved
serving table and coding failure remain unchanged. Cached returns, long-cache
pressure, logits, useful coding quality, ANE and multimodal admission are open.
Exact source/library bindings and failed attempts remain in the retained study.
