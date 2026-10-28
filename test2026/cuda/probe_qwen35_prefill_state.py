"""Observe ordinary prefill state at the first prediction, B1/B2 externally serialized."""
import hashlib
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen35_prefill_state_observer import snapshot

sys.path.insert(0, os.environ["GARNET_RUNTIME_ROOT"])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import garnet
from runtime.cuda_serving import Model

prior_path, output_path = map(Path, sys.argv[1:3])
binary_path = output_path.with_suffix(".f32")
assert not output_path.exists() and not binary_path.exists()
prior = json.loads(prior_path.read_text())
assert prior["diagnostic_complete"] and len(prior["trace"]) == 27
source_path = Path(prior["source"])
assert hashlib.sha256(source_path.read_bytes()).hexdigest() == prior["source_sha256"]
source = json.loads(source_path.read_text())
request, profile = source["request"], prior["profile"]
batch = profile["batch"]
assert request["model_id"] == "Qwen3.5-4B" and batch in (1, 2)
assert profile == source["ordinary_profile"] and profile["speculativeMTP"] == 0 and not profile["compactKVOutputs"]
case = next(c for c in request["cases"] if c["style"] == "uniform")
assert [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in case["images"]] == case["input_sha256"]
expected = prior["trace"][0]
evidence = {"batch": batch, "profile": profile, "prior_trace": str(prior_path.resolve()),
    "prior_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(), "events": [],
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "diagnostic_complete": False,
    "root_cause_resolved": False, "scope": "Two reset ordinary limit1 requests; full finite canonical state after prefill. Not timing, independent quality, a production change or a causal acceptance."}


def save():
    output_path.write_text(json.dumps(evidence, indent=2))


def event(name):
    evidence["events"].append({"event": name, "unix_seconds": time.time()})
    save()
    print("PREFILL_STATE_EVENT", batch, name, flush=True)


def flat(value):
    return [v for part in value for v in flat(part)] if isinstance(value, list) else [float(value)]


def hidden(decoder):
    raw = flat(garnet.tensor_to_cpu(decoder.last_hidden).tolist())
    normalized = flat(garnet.tensor_to_cpu(decoder.head.run_device([decoder.last_hidden])).tolist())
    width = decoder.config["hidden_size"]
    assert len(raw) == len(normalized) == batch * width and all(math.isfinite(v) for v in raw + normalized)
    return {"raw_hidden_rows": [raw[r * width:(r + 1) * width] for r in range(batch)],
        "normalized_hidden_rows": [normalized[r * width:(r + 1) * width] for r in range(batch)]}


def check_hidden(values):
    for key in ("raw_hidden_rows", "normalized_hidden_rows"):
        for actual, saved in zip(values[key], expected[key]):
            assert struct.pack("<" + "f" * len(actual), *actual) == struct.pack("<" + "f" * len(saved), *saved), "First hidden changed from prior trace"


event("load_begin")
model = Model(request["checkpoint"], profile, request["model_id"])
try:
    event("load_complete")
    decoder = model.decoder
    control = model.infer_batch(case["prompts"], case["images"], 1)
    evidence["control"] = control
    evidence["control_hidden"] = hidden(decoder)
    evidence["control_state"] = snapshot(decoder, batch)
    save()
    assert control["generated_token_ids"] == [[token] for token in expected["prediction"]]
    check_hidden(evidence["control_hidden"])
    event("control_complete")
    original_prefill, original_chunk = decoder.prefill_sequence, decoder.chunk_hidden
    captured = 0

    def traced_prefill(values, positions, lengths=None):
        hashes = [hashlib.sha256() for _ in range(batch)]
        width = decoder.config["hidden_size"]
        for value in values:
            data = decoder.io.finite_f32_bytes(value)
            assert len(data) == batch * width * 4
            for row in range(batch):
                hashes[row].update(data[row * width * 4:(row + 1) * width * 4])
        evidence["actual_prefill_inputs"] = {"fp32_row_sha256": [h.hexdigest() for h in hashes],
            "tokens": len(values), "words_per_row": len(values) * width, "positions": positions, "lengths": lengths,
            "hash_order": "Time then hidden channel, complete finite FP32 input rows"}
        save()
        assert evidence["actual_prefill_inputs"] == prior["actual_prefill_inputs"]
        return original_prefill(values, positions, lengths)

    def traced_chunk(*args, **kwargs):
        global captured
        prediction = original_chunk(*args, **kwargs)
        predict = args[3] if len(args) > 3 else kwargs.get("predict", True)
        if predict:
            captured += 1
            evidence["prediction"] = list(prediction)
            save()
            assert captured == 1 and list(prediction) == expected["prediction"]
            event("state_capture_begin")
            evidence["captured_state"] = snapshot(decoder, batch, binary_path)
            evidence["captured_hidden"] = hidden(decoder)
            save()
            check_hidden(evidence["captured_hidden"])
            for key in ("batch", "row_width_words", "finite_fp32_words_total", "row_sha256", "components", "format"):
                assert evidence["captured_state"][key] == evidence["control_state"][key], "Reset/capture changed complete canonical state"
            event("state_capture_complete")
        return prediction

    decoder.prefill_sequence, decoder.chunk_hidden = traced_prefill, traced_chunk
    result = model.infer_batch(case["prompts"], case["images"], 1)
    evidence["instrumented"] = result
    save()
    assert captured == 1
    for key in ("prompt_token_ids", "generated_token_ids", "sampled_tokens_including_stop", "finish_reasons", "stop_token_ids", "visual_tokens_per_row"):
        assert result[key] == control[key]
    event("instrumented_complete")
except BaseException as error:
    evidence["failure"] = {"type": type(error).__name__, "message": str(error)}
    save()
    raise
finally:
    model.close()
    event("closed")
evidence["diagnostic_complete"] = True
save()
print("PREFILL_STATE_COMPLETE", batch, flush=True)
