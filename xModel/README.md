# xModel

**xModel** is Garnet's backend-neutral model programming format and package
contract. An xModel package contains XLang3 Tensor Expression programs written
as `.py` source with Python-compatible syntax, an xModel manifest (`model.json`),
backend profiles, and contracts for external assets.

Each xModel manifest selects `.py` entrypoints; relative imports connect shared
xModel helpers. The migrated `.x` copies have been removed.

The 27 xModel programs cover text, vision-language, speech recognition, and
speech synthesis. Backend profiles remain separate from xModel semantics.
Checkpoint and tokenizer assets are external and are not bundled here.

`test/xlang3/models/run_production_capture.py` imports all programs and captures
40 graphs with tiny symbolic profiles. This validates capture, not pretrained
inference or complete backend operator coverage. See the production-capture
coverage document under that test directory for the verified limits.

## Image generation

FLUX.2 Klein 4B supports text-to-image and reference-image editing on
TensorRT/CUDA. Its manifest identifies the external checkpoint layout and
compiled tensor programs. Install the [native image provider](../src/media/README.md)
and run the driver with XLang3 from the Garnet binary directory.

```text
xlang3 <repo>/tools/generation/generate_flux2.py CHECKPOINT CACHE OUTPUT_PNG "PROMPT" 512 42
xlang3 <repo>/tools/generation/generate_flux2.py CHECKPOINT CACHE OUTPUT_PNG "EDIT PROMPT" 512 42 REFERENCE_PNG
```

See the [generation checks](../test/xlang3/generation/README.md) for test setup.
