# Image generation checks

Run with the CUDA build of Garnet and XLang3. Set
`GARNET_FLUX2_TEST_CHECKPOINT` to register pretrained FLUX.2 checks.
Also enable `GARNET_TEST_PRETRAINED_GENERATION` and install the native
image provider to generate and decode a real PNG.

```text
cmake --build BUILD --target garnet garnet_prepare_precision
ctest --test-dir BUILD --output-on-failure -R "(generation|flux2)"
```

The tests cover graph capture, numerical comparisons, native noise and
tensor primitives, tokenizer and prompt parity, and pretrained output.
Keep checkpoints and generated artifacts outside the source tree.
