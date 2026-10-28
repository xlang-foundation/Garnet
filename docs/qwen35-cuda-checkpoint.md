# Qwen3.5 CUDA checkpoint for WorldSense

This checkpoint combines the current Qwen3.5-2B/4B CUDA work with the Qwen27B
branch snapshot `b0606e372`. The optimization goal is paused and its recurring
schedule has been deleted. The user requested a short inference check so
WorldSense can perform the broader feature validation next.

## Checks completed on October 28, 2026

- Combined native build: all 61 steps passed; existing XLang3 SDK unchanged.
- 29 CUDA component checks passed, including memory, race and synchronization
  fixtures, verifier coverage and default/joined sessions at batches 1/2/4/8.
- Qwen3.5-2B and 4B each passed three real-image requests through
  `garnet.serve_model` / `garnet.infer_json`, using their actual checkpoints.
- Each model processed image A, image B, then image A again. Generated IDs,
  request/stop accounting and speculative work matched exactly for the replay.
  Prefix, image and answer caches were off, and both models closed successfully.
- Python syntax checks, fake-native adapter controls and 135 independent CPU
  math cases passed. The stale research adapter AST guard was corrected and
  competing native-draft ownership is rejected before construction.

The original 72-role regression was stopped after the current check at the
user's request to shorten verification. It is **not a full regression pass**.
Qwen27B code is included; no Qwen27B model validation ran. WorldSense UI/camera
feature testing, independent quality, full-model sanitizer closure, fresh
performance/memory comparisons and the 2x optimization target remain separate.

## Using the tested build

The tested private DLL is
`D:/CantorAI/work/qwen35-cuda/build-vs18/bin/garnet.dll`, SHA256
`91498fec58814f625bf94cbb61fd79925d89b6bf424f8fc15de0b1f09604ab9f`.
The primary checkout and the existing WorldSense app bundle were not updated.

[The checkpoint record](research/qwen35-cuda-checkpoint.json) contains exact
batch-1 model profiles, checkpoint/input identities, environment and evidence.
Apply the environment before loading Garnet, and keep the checkout available
for the `runtime/` and `xModel/` imports. Ordinary CPython does not provide the
native Garnet/tensor packages; use the existing XLang3 runtime and CUDA/TensorRT
installation. The public API setup used the checkout's `xModel` directory and
the explicit profile from the record. Reload the model to change its static
batch, context capacity or image grid.

The tested profiles use a 12x20 image patch grid, 512-token capacity and
32-token requests. Every request resets state and re-encodes its JPEG. These
checks establish the observed inference/reset behavior, not broader feature or
quality acceptance. The older benchmark charts remain historical measurements.

## Reproducing the local check

Exact workflow script snapshots are retained under `test2026/cuda/`:
`run-worldsense-demo-final-build.py`, `run-worldsense-demo-full-regression.py`,
`run-worldsense-demo-smoke.py` and `worldsense-demo-public-inference-worker.py`.
They are local workflow snapshots: copy them to the existing
`D:/CantorAI/work/qwen35-cuda` workflow root before use; they expect its existing
SDK, model, input and baseline evidence. Preserve old attempts instead of
reusing output names. The full-regression script is included for traceability;
the shorter smoke script implements the final requested scope.

Raw requests, results, exact process exits and verified evidence archives remain
in that workflow root. Model weights, images, vendor binaries and generated
engines are excluded from the source checkpoint. The historical October 23
record is retained inside the JSON, and the detailed research report/log preserve
all earlier failures, interruptions, rejected trials and measurement limits.
