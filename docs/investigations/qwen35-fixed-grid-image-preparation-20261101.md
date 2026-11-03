# Qwen3.5 CUDA image preparation for a loaded grid

The existing 4B serving profile builds a 12 by 20 patch vision engine at startup.
Request preparation previously ran smart resize independently for every image.
Different aspect ratios produced another grid, which `_build_vision` correctly
rejected. A dynamic minimum of 65,536 pixels could also exceed the loaded
320 by 192 image area (61,440 pixels). This was a request preparation defect;
changing the pipeline visualization could not correct it.

When a vision engine has a grid, `Model._prepare` now calls the native
`cuda_tensor_io.prepare_image_grid(path, grid_h, grid_w, max_pixels)` operator.
The nvJPEG image remains on the GPU. The existing fused bicubic resize,
normalization, and patch-layout kernel fits the image proportionally inside
the fixed canvas with centered, neutral-gray padding. It allocates exactly
`grid_h * grid_w * 1536 * sizeof(float)` output bytes, with no intermediate
canvas or per-request engine compilation. Grids must be positive, even, and
fit the admitted maximum (at most 1,048,576 pixels for the tensor I/O API).

An explicit/loaded grid is authoritative; the dynamic minimum applies when
selecting the first grid for a profile without one. Existing dynamic resize
entry points and their output are preserved. A profile without an explicit
grid still selects its engine on the first image, then fits later images to
that engine's grid.

Verification on 2026-11-01:

- Native image suite passed, including fixed landscape/portrait/square grids
  on both local CUDA devices, independent padding/layout checks, invalid grid
  rejection, existing JPEG parity, stream/device ownership, and readiness.
- Native XLang3 tensor I/O test passed three grids and four invalid inputs.
- All eight prompt-packing cases passed, including reused vision buffers.
- Existing SRV-001 Cantor was replaced and restarted with the existing
  configuration. Public HTTPS image inference returned HTTP 200 for landscape
  PNG, square PNG, portrait WebP, and wide JPEG. All four named the red square
  and blue circle with an explicit color-and-shape prompt.
- The original short prompt also succeeded for the original failing landscape
  PNG. Its square variant returned the correct shapes but omitted colors;
  that receipt is retained separately rather than counted as a color pass.
- Only the three owned test failures proven to occur before model prefill
  were reconciled as zero token execution using signed database settlements.
  Limits and unrelated reservations were unchanged. Other unknown failures
  still require evidence before reconciliation.

Evidence: `D:/CantorStorage/.deployment/cantorserve-regression-20261101/`.
This is a functional regression check, not throughput or broad model accuracy
qualification. Wyze required renewed vendor sign-in after the native restart.
