# Qwen3-TTS text projection uses SiLU

The deployed Qwen3-TTS 12Hz 0.6B CustomVoice checkpoint declares
`talker_config.hidden_act = "silu"`. Qwen's `Qwen3TTSResizeMLP` selects
`ACT2FN[config.hidden_act]` for the text projection. Garnet's TensorRT prompt
and autoregressive embedding operators instead used GELU_ERF.

Both native projection paths now compute `x * sigmoid(x)`. The 0.6B and 1.7B
`talker_common.py` dependency content changes deliberately invalidate cached
prompt/decode graphs made with the old lowering.

Reference: [Qwen's model implementation](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/core/models/modeling_qwen3_tts.py).

## Regression

Run `test/xlang3/tts_projection_checks.py` using native XLang3 with a CUDA
TensorRT build. The fixture contains tiny synthetic safetensors weights and
compares both real native operators with independent scalar CPU SiLU math.
It needs no model download, CPython package, or Cantor node. On the tested
build, prompt maximum error was 0.002432 and decode error was below 0.000001,
against a 0.035 bound. The previous GELU artifact failed the same fixture
with errors 0.6634 and 0.2737.

## Scope of the evidence

This correction alone does not establish complete TTS quality. The preserved
31-case sampled speech trial passed 23 automatic ASR roundtrips. It was not
promoted to the production profile.

Offline CPU reference diagnosis used the existing deployed checkpoint,
verified by SHA-256, and the official Qwen package with its pinned Transformers
version. That reference is a diagnostic tool and is never a serving provider.
For an identical prompt and three teacher-forced frames, the native main
transformer logits had cosine similarities above 0.99989. With identical hidden
state and first code, all 16 greedy predictor codes matched. On identical
128-frame codec inputs, waveform cosine similarity was 0.999995 and normalized
RMSE was 0.00310. These observations narrow the remaining audio-quality work;
they do not substitute for end-to-end speech acceptance.

The isolated executable test artifact linked the changed TensorRT builder with
the previously qualified serving objects. It is not evidence that a complete
build of every current shared source file passed.
