# Native half-prefill experiment: regression passes, serving qualification open

This opt-in C++/Objective-C++/Metal overlay integrates the qualified standalone
P16 attention policy into the Mistral compiled graph. The selected serving
engine is unchanged. `GARNET_METAL_CHUNK_HALF_ATTENTION` defaults OFF. It requires
Apple10/M5 hardware and the recorded bounded Mistral preparation recipe.

The kernel follows the pinned vLLM-Metal NAX implementation's relaxed MPP
descriptors and vectorized register loads. Q/K/V round to half; accumulation,
online softmax and output remain float. Original FP32 canonical KV, generic
public attention behavior, existing shader sources and all 137 xModel files
remain unchanged. The new policy does not satisfy the old FP32 numerical
contract; its fixed half-QKV budget is recorded separately. Full-model quality
is not implied by primitive bounds or a short output regression.

The native profile is up to B4, T128, C4096, H32/KV8/D128 and 512 physical token
rows. Three fixed half arenas request 68MiB plus384 guard bytes. Q and the
visible KV extent are packed on GPU; scratch is reused under the existing
shared attention lease. A shared4-byte range flag follows that lease across
graph preparation and decoder execution. Only its owning batch can reset or
inspect it, and Flush retains earlier failures. Invalid finite-domain inputs
invalidate all pending outputs before publication. Canonical KV remains FP32;
this experiment does not establish model-memory savings.

All bounded half-prefill calls, including tiny and empty tails, use the same
policy. The earlier scalar fallback made a request's arithmetic depend on
unrelated batch occupants: the retained diagnostic found32,765 differing words
for the same eight-token request alone versus batched. The new native test
requires exact output agreement across occupancy and context-bucket changes.
Its producer namespace is `fp16-prefill-all-p16-gpupack-v1`. Prefix-return and
continuous-batch qualification of this new policy remain open.

Validation on the M5/24GiB Mac:

- Enabled52/52 and default-off50/50 serial native suites pass. Tests cover
  complete output bounds, independent FP64 samples, offset parents, scratch
  poison/guards/tails, canonical-cache writes, controls, aliases, asynchronous
  range failures, Abort/recovery, graph/session handoff and failure isolation.
- Three actual trained attention inputs from layers0/15/31 pass across
  6,291,456 output words and18,432 independent FP64 values. The largest sampled
  error consumes0.310 of the fixed half-policy allowance. Original MPS controls
  remain exact to the captured native output. Old FP32 failures stay recorded.
- Fresh serial OFF/ON full-model runs produce the same32 generated IDs and
  text as the frozen control. The OFF engine matches260 prior physical-state
  digests. The ON engine executes544 half-prefill calls and no MPS products;
  its changed hidden/KV values are descriptive evidence, not a full-model
  numerical oracle. Both close with zero native owners and under4MiB allocated.
- Eight further fresh serial engines run two reversed-order cohorts, one/four
  active owners,15 real prefix chunks,3 warmups and9 measured calls each. The
  measured call replays offset1920 through all32 layers. Embedding, head,
  hashing, scheduling and HTTP are outside the body span. Within-policy state
  and predictions are exact across cohorts; cross-policy next predictions
  agree. Four OFF body states and8,388,608 hidden values remain exact to the
  frozen controls.

| Active requests | FP32 body wall median, cohort range | Half body wall median, cohort range | Ratio in each cohort |
|---|---:|---:|---:|
| 1 | 273.45–307.50ms | 230.11–254.64ms | 1.188–1.208× |
| 4 | 1134.40–1189.72ms | 963.30–976.32ms | 1.178–1.219× |

These are ratios of separate-process medians, not within-engine paired samples.
The stronger standalone attention gain does not extend to the complete body.
No new TTFT, decode, throughput, HTTP, ANE, Gemma or Qwen qualification is claimed.
The original strict coding result remains0/8; no2× or half-TTFT target is met.
No vLLM/SGLang inference was repeated and their frozen artifacts are unchanged.
Next is attribution of the remaining body cost and full-model quality/cache
validation before any serving admission.

Follow-up [serving regression gates](../mistral_half_prefill_serving_gates/README.md)
now pass the nine-request return/edit/queued-fifth protocol, exact target and
survivor state checks, and an80MiB cache-pressure run with14 evictions. All3,052
coding tokens/text remain exact to the FP32 control, while strict quality is
still0/8. These bounded correctness results do not qualify serving speed,
general failure paths or the full model-quality target.

The retained local study is
`out/research/mistral7b-local-coder/mistral-half-attention-native1`.
It preserves commands, source generations, failed tests, binaries, raw results,
model/profile bindings and audit controls. Early test failures include an
uninitialized scratch read, MPS reference-process allocation retention, the
graph/session range-flag ownership defect, and missing test-fixture restoration
after intentionally failed KV writes. Corrected results do not replace those
failures. CoreAnalytics context-leak warnings remain in both model logs;
zero native owners do not explain them.

Prepare an exact fresh source tree without building the dirty checkout:

```sh
python3 prepare_native_source.py REPOSITORY ADMITTED_SOURCE ADMITTED_MANIFEST FRESH_SOURCE
```

The helper delegates the pinned committed-KV/hoist preparation, checks the
existing foundation, applies only this nine-path overlay and verifies its
hashes. The committed-MPS and old flash selectors stay OFF. The retained
configure JSON files give the complete ON/OFF recipes; their absolute paths
identify this local study. Build into fresh directories, then run CTest
serially. The native test also supports separate `--reference-trained` and
`--trained` processes; keeping MPS controls separate preserves the absolute
4MiB post-close gate without changing the numerical bounds.

Saved-data auditors take the absolute retained study directory; body auditors
take its `body7` directory. An optional second argument selects a fresh report
path when re-auditing retained evidence. NumPy is required for
`audit_trained7.py`. Audits intentionally refuse to overwrite reports.
Workers and the model runner are retained research tools bound to the recorded
fixtures, profiles and library bindings, not public serving APIs. The
qualification JSON records results and hashes without bundling large payloads.
