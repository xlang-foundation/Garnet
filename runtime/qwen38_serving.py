# SPDX-License-Identifier: Apache-2.0
"""Bounded Qwen3.8-27B text orchestration using native mixed NVFP4/FP8 kernels.

The initial profile deliberately retains the verified B1/T32/T4 4096-slot
decoder contract. Larger contexts require separate native admission and tests.
No sampling, speculative tokens, image encoder, or application-side cache is
silently substituted for the selected model.
"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path


def validate_profile(profile):
    allowed = {"batch", "maxInputTokens", "maxOutputTokens", "prefillChunk", "decodeChunk", "deviceId", "cudaGraph"}
    if set(profile) - allowed:
        raise ValueError("Unsupported Qwen3.8 serving profile option")
    for name in ("batch", "maxInputTokens", "maxOutputTokens", "prefillChunk", "decodeChunk", "deviceId"):
        if type(profile.get(name)) is not int:
            raise ValueError("Qwen3.8 profile requires explicit integer " + name)
    if (profile["batch"], profile["prefillChunk"], profile["decodeChunk"], profile["deviceId"]) != (1, 32, 4, 0):
        raise ValueError("Qwen3.8 serving requires B1/T32/T4 on logical device0")
    if profile.get("cudaGraph") is not True:
        raise ValueError("Qwen3.8 serving requires the verified CUDA graph route")
    if not 1 <= profile["maxOutputTokens"] <= 1024 or profile["maxInputTokens"] < 1:
        raise ValueError("Qwen3.8 input/output profile is invalid")
    if profile["maxInputTokens"] + profile["maxOutputTokens"] != 4096:
        raise ValueError("Qwen3.8 initial native context admission is exactly4096 slots")


class Model:
    def __init__(self, root, profile, name):
        validate_profile(profile)
        if name != "Qwen3.8-27B":
            raise ValueError("Qwen3.8 worker model identity mismatch")
        # Pin the same projection/top1 routes as the independently replayed
        # native B1 consistency proof, in this model's own process only.
        for flag in ("GARNET_FP8_FULL_ATTN_M32_CANONICAL", "GARNET_FP8_GDN_M32_CANONICAL", "GARNET_PARALLEL_BATCH_TOP1"):
            if os.environ.get(flag) not in (None, "1"):
                raise ValueError("Conflicting native Qwen3.8 policy: " + flag)
            os.environ[flag] = "1"
        import garnet
        from xModel.qwen38.vl_27b.model import create_cuda_bf16_low_batch_study_decoder, validate_config

        self.root, self.name, self.profile = root, name, dict(profile)
        self.lock, self.decoder = threading.RLock(), None
        self.max_input, self.max_output = profile["maxInputTokens"], profile["maxOutputTokens"]
        self.config = json.loads(Path(root, "config.json").read_text())
        validate_config(self.config)
        self.tokenizer = garnet.tokenizer()
        self.tokenizer.open(root)
        self.stops = [self.tokenizer.token_id("<|im_end|>"), self.tokenizer.token_id("<|endoftext|>")]
        if any(type(token) is not int or token < 0 for token in self.stops):
            raise ValueError("Checkpoint stop tokens are unavailable")
        try:
            self.decoder = create_cuda_bf16_low_batch_study_decoder(root, self.config, capacity=4096, batch=1)
            self.decoder.enable_cuda_graph(True)
        except BaseException:
            self.close()
            raise

    def _prepare(self, prompt, image):
        if image:
            raise ValueError("This Qwen3.8 serving profile accepts text only")
        ids = list(self.tokenizer.encode(prompt))
        if not ids or len(ids) > self.max_input:
            raise ValueError("Chat input exceeds loaded Qwen3.8 token profile")
        embeddings = self.decoder.io.embedding(ids)
        return [self.decoder.io.row(embeddings, i) for i in range(len(ids))], [[i] * 3 for i in range(len(ids))], ids, len(ids), 0., 0

    def infer(self, prompt, image, limit, on_token=None):
        if type(limit) is not int or not 1 <= limit <= self.max_output:
            raise ValueError("Output limit exceeds loaded Qwen3.8 profile")
        with self.lock:
            if self.decoder is None:
                raise RuntimeError("Qwen3.8 model is stopped")
            started = time.monotonic()
            self.decoder.reset()
            hidden, positions, ids, current, _, _ = self._prepare(prompt, image)
            input_done = time.monotonic()
            prediction = self.decoder.prefill_sequence(hidden, [positions], [len(ids)])
            first = time.monotonic()
            generated, sampled, stop, steps = [], 0, None, 0
            for index in range(limit):
                token = prediction[0]
                sampled += 1
                if on_token is not None:
                    on_token(0, token, sampled, len(ids))
                if token in self.stops:
                    stop = token
                    break
                generated.append(token)
                if index + 1 == limit:
                    break
                prediction = self.decoder.step(prediction, [1], [[current] * 3])
                current += 1
                steps += 1
            done = time.monotonic()
            text = self.tokenizer.decode(generated)
            return {"status": "ok", "model_id": self.name, "backend": "Garnet CUDA/TensorRT Qwen3.8", "batch": 1,
                "text": text, "texts": [text], "prompt_token_ids": [ids], "prompt_tokens": [len(ids)],
                "generated_token_ids": [generated], "generated_tokens": [len(generated)],
                "sampled_tokens_including_stop": [sampled], "stop_token_ids": [stop],
                "finish_reasons": ["eos" if stop is not None else "length"], "decode_steps": steps,
                "visual_tokens": 0, "visual_tokens_per_row": [0], "profile": self.profile,
                "ttft_seconds": first - started, "input_seconds": input_done - started,
                "prefill_seconds": first - input_done, "decode_seconds": done - first,
                "total_seconds": done - started, "source": __file__,
                "prompt_sha256": [hashlib.sha256(prompt.encode()).hexdigest()],
                "cache_policy": {"prefix": False, "image": False, "answer": False},
                "arithmetic": {"weights": "mixed NVFP4/FP8", "head": "NVFP4 W4A4", "kv_cache": "BF16",
                    "equivalent_to_transformers_full_bf16": False},
                "speculative": {"enabled": False},
                "decoder_profiles": {"physical_batch": 1, "prefill_chunk": 32, "decode_chunk": 4, "capacity": 4096}}

    def close(self):
        with self.lock:
            if self.decoder is not None:
                self.decoder.close()
                self.decoder = None

    def memory_info(self):
        with self.lock:
            if self.decoder is None:
                raise RuntimeError("Qwen3.8 model is stopped")
            return self.decoder.memory_info()
