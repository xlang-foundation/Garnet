# Native fused Mistral MLP: exact regression, speed unmeasured

This opt-in native Metal path combines the Mistral gate/up projections and
SiLU, writes their product directly to a private half-precision input, and
runs the original Safe packed down projection. It avoids materializing the
three FP32 intermediate values and the down-input gather. Existing logical
graph slots remain allocated; this is not an overall memory-reduction claim.
The additional private arena is 14,680,064 bytes.

`GARNET_METAL_FUSED_MISTRAL_MLP` defaults **OFF**. The selected experiment
uses the existing M5 half-prefill B4/C4096/T128 profile, with a global maximum
of512 prefill tokens. Fusion handles128–512 selected rows; smaller calls and
decode retain the original path. No xModel operator, model, tokenizer or
frozen vLLM/SGLang artifact changes.

The implementation is a clean experimental native-source overlay on the
frozen half-prefill engine. `native-candidate1.patch` records its eleven
changed/new paths relative to that engine. As with the preceding half-policy
checkpoints, production integration still requires bringing the prerequisite
overlays into the normal build; this checkpoint does not enable a root-build
production feature.

## Runtime safeguards

- Graph preparation proves that the gate/up/SiLU values each have exactly
  one expected writer and reader, using immutable logical alias roots before
  physical scratch reuse. Inputs, outputs, output parts and other consumers
  prevent fusion. Thirteen malformed logical-use cases reject.
- Pipelines, the cooperative-layout GPU probe and the private arena are
  prepared once. The execution method neither compiles nor allocates GPU
  storage. It validates devices, spans, shapes, selection uniqueness, native
  command ownership, prepared bounds and checkpoint/private-scratch aliases.
- Normalized input may share its arena with the down output. Input gathering
  precedes inactive-output clearing. Output/private-arena pending ownership
  is registered before writes, including Abort and partial-submit recovery.
- The producer uses the previously qualified MSL4 Fast fused shader. Half
  operands and the original coefficient/reduction policy are retained; the
  original Safe down pipeline is reused. The original native SiLU's pinned
  configuration uses nil/default compiler options. The earlier standalone
  explicit-MSL3.1 control is a surrogate and supplies no serving speed claim.
- Cache identity adds `fused-mistral-mlp-fast-v1`; backend information and
  research profiles assert the same compiled policy. There is no runtime
  policy toggle that could mix cache producers.
- `fused_mlp_sites` and `fused_mlp_blocks` expose preparation/execution
  witnesses. Existing packed/affine counters retain their logical-product
  convention: one fused block counts three products, but its producer and
  down stage use two compute dispatches, plus gathering/clearing work.

## Retained results

The local studies are `mistral-fused-mlp-native1` and
`mistral-fused-mlp-prefix-http1`, under `out/research/mistral7b-local-coder`.

| Gate | Result |
|---|---|
| Actual native method, trained layers0/15/31 |54 cases pass |
| Complete native parent/output guard comparisons |113,688,576 FP32 words |
| Actual compact half output and zero-tail comparisons |242,135,040 half words |
| Invalid native operations |81 reject |
| Aliases / partial-submit Abort recovery / native owner cleanup |Pass |
| Full32-layer model |32 output IDs and complete text exact |
| Full physical KV/hidden state at calls1/16/18/25 |260 digests and65,536 hidden values exact |
| Actual fused model blocks |512; tail/decode0 |
| Enabled native suite |52/52 pass, serial |
| New option default-off build and suite |OFF confirmed;52/52 pass, serial |
| Model audit corruptions |14 reject |
| Cached and uncached return/edit/queued-fifth runs |752 IDs/text per run exact to frozen half control |
| Target-state comparisons to frozen half control |1,170 digest pairs per run exact |
| Survivor-state observations |1,950 per run, preserved |
| Prefix hits: exact A / append / late edit / early edit |211 /128 /128 /0 tokens |
| Fused blocks in uncached / cached protocol |224 /128 |
| Cache audit corruptions |8 inherited +5 fused-specific reject |

These are correctness regressions on the pinned fixtures, not a general
numerical oracle or coding-quality qualification. Canonical KV/decode remains
FP32, unlike the frozen references' FP16 state. Native owners return to zero
and post-close Metal allocations remain below4MiB. CoreAnalytics context-leak
messages remain in the raw logs and are not explained by those owner counts.

The initial native preparation is retained. A pre-inference review corrected
input/output-alias ordering and pending-write registration before selecting
source2. A first cache audit completed its checks but failed at report writing
because the wrapper assumed an argv-selected filename. The original wrapper
is preserved; audit2 uses the existing generic auditor and passes without any
inference rerun or changed validation criterion.

**No new TTFT, decode or aggregate-throughput result is established here.**
Cache pressure and the unchanged eight-task coding collection still need
regression on this producer before a new serving campaign. The saved strict
coding score remains0/8. The2×/half-TTFT targets, ANE/Gemma and the original
Qwen objectives remain open. No comparator inference is repeated.

The subsequent [pressure/coding regressions](../mistral_fused_mlp_serving_gates/README.md)
and [serving measurements](../mistral_fused_mlp_serving_benchmark/README.md)
now cover this same native producer. They preserve the strict0/8 coding
score and leave the2×/half-TTFT targets unmet.

## Reproduction

Preparation requires the pinned sibling half-policy, standalone fused-screen
and real trained-capture studies. Use an absolute, fresh destination directly
under the local research directory. It reproduces all26,294 source files and
asserts the canonical manifest hash. The packaged preparation was itself
replayed without inference and reproduced the complete selected source.

```sh
python3 -B prepare_native1.py /absolute/path/to/mistral7b-local-coder/FRESH-NATIVE
python3 -B /absolute/path/to/FRESH-NATIVE/build_native1.py
python3 -B /absolute/path/to/FRESH-NATIVE/run_model1.py
python3 -B /absolute/path/to/FRESH-NATIVE/audit_model1.py
python3 -B /absolute/path/to/FRESH-NATIVE/run_suite1.py
python3 -B /absolute/path/to/FRESH-NATIVE/prepare_prefix1.py /absolute/path/to/mistral7b-local-coder/FRESH-PREFIX
python3 -B /absolute/path/to/FRESH-PREFIX/launch1.py
python3 -B /absolute/path/to/FRESH-PREFIX/audit_fused1.py
```

These tools retain local evidence/dependency paths. They refuse to overwrite
collected runs. All model and native-suite execution is serial; saved baseline
audits perform no inference. Build only a clean prepared source, never the
dirty Garnet root. The original11 dirty files remain byte-identical; overlapping
feature changes are checkpointed through the Git index without rewriting them.
