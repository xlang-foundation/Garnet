"""Diagnostic-only extra layer0 output; reject unless original states/hidden match."""
import hashlib
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen35_layer_debug_observer import RowCapture, make_forward
from qwen35_prefill_state_observer import snapshot
sys.path.insert(0, os.environ["GARNET_RUNTIME_ROOT"])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import garnet
from runtime.cuda_serving import Model
from xModel.qwen35.vl_0_8b import cuda_packed_decoder as decoder_module
from xModel.qwen35.vl_0_8b import cuda_packed_program as program_module

prior_path, output_path = map(Path, sys.argv[1:3])
input_path, layer_path = output_path.with_suffix(".input.f32"), output_path.with_suffix(".layer0.f32")
assert not any(p.exists() for p in (output_path, input_path, layer_path))
prior = json.loads(prior_path.read_text())
assert prior["diagnostic_complete"]
trace_path = Path(prior["prior_trace"])
assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == prior["prior_sha256"]
trace = json.loads(trace_path.read_text())
source_path = Path(trace["source"])
assert hashlib.sha256(source_path.read_bytes()).hexdigest() == trace["source_sha256"]
source = json.loads(source_path.read_text())
request, profile = source["request"], prior["profile"]
batch = profile["batch"]
assert batch in (1, 2) and request["model_id"] == "Qwen3.5-4B"
assert profile == source["ordinary_profile"]
assert not any(profile[k] for k in ("speculativeMTP", "internalPackedStores", "compactKVOutputs", "sharePackedOutputStorage"))
case = next(c for c in request["cases"] if c["style"] == "uniform")
assert [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in case["images"]] == case["input_sha256"]
debug_forward, ast_proof = make_forward(program_module)


class ObservedProgram(program_module.CUDAPackedDecoderProgram):
    def _forward(self):
        assert not self.internal_packed_stores and not self.compact_kv_outputs
        self.debug_layer0_offset = self.output_width
        self.output_width += self.size * self.config["hidden_size"]
        return debug_forward(self)


# Applies only inside this diagnostic process; no production file is modified.
decoder_module.CUDAPackedDecoderProgram = ObservedProgram
evidence = {"batch": batch, "profile": profile, "events": [], "diagnostic_complete": False,
    "prior_capture": str(prior_path.resolve()), "prior_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "ast_proof": ast_proof,
    "chunks": [], "instrumented_graph_admitted_against_original": False, "root_cause_resolved": False,
    "scope": "Diagnostic-only appended layer0 hidden output. Exact original complete canonical-state/first-hidden gates precede observation. Compiler tactics can change with extra outputs; no model quality, serving, timing, memory saving or causal acceptance."}


def save():
    output_path.write_text(json.dumps(evidence, indent=2))


def event(name):
    evidence["events"].append({"event": name, "unix_seconds": time.time()})
    save()
    print("LAYER0_DEBUG_EVENT", batch, name, flush=True)


def flat(value):
    return [v for part in value for v in flat(part)] if isinstance(value, list) else [float(value)]


def hidden(decoder):
    raw = flat(garnet.tensor_to_cpu(decoder.last_hidden).tolist())
    normalized = flat(garnet.tensor_to_cpu(decoder.head.run_device([decoder.last_hidden])).tolist())
    width = decoder.config["hidden_size"]
    assert len(raw) == len(normalized) == batch * width and all(math.isfinite(v) for v in raw + normalized)
    return {"raw_hidden_rows": [raw[r * width:(r + 1) * width] for r in range(batch)],
        "normalized_hidden_rows": [normalized[r * width:(r + 1) * width] for r in range(batch)]}


def original_gates(state, values):
    for key in ("batch", "row_width_words", "finite_fp32_words_total", "row_sha256", "components", "format"):
        assert state[key] == prior["captured_state"][key], "Extra graph output changed original canonical state; reject observer"
    for key in ("raw_hidden_rows", "normalized_hidden_rows"):
        for actual, expected in zip(values[key], prior["captured_hidden"][key]):
            assert struct.pack("<" + "f" * len(actual), *actual) == struct.pack("<" + "f" * len(expected), *expected), "Extra graph output changed original first hidden; reject observer"


writers = {}
context = {}
event("load_begin")
model = Model(request["checkpoint"], profile, request["model_id"])
try:
    event("load_complete")
    decoder = model.decoder
    control = model.infer_batch(case["prompts"], case["images"], 1)
    evidence["control"] = control
    evidence["control_state"] = snapshot(decoder, batch)
    evidence["control_hidden"] = hidden(decoder)
    save()
    original_gates(evidence["control_state"], evidence["control_hidden"])
    assert control["generated_token_ids"] == prior["control"]["generated_token_ids"]
    evidence["instrumented_graph_admitted_against_original"] = True
    event("original_graph_controls_passed")
    original_prefill, original_chunk = decoder.prefill_sequence, decoder.chunk_hidden
    program = decoder.programs[decoder.prefill_chunk_size]
    original_run = program.run
    width, size = decoder.config["hidden_size"], program.size
    evidence["debug_layout"] = {"layer": 0, "offset_words": program.debug_layer0_offset,
        "tokens_per_chunk": size, "hidden_size": width, "complete_output_width_words": program.output_width}

    def prefill(values, positions, lengths=None):
        tokens = len(values)
        writers["input"] = RowCapture(input_path, batch, tokens * width)
        writers["layer0"] = RowCapture(layer_path, batch, tokens * width)
        for value in values:
            raw = decoder.io.finite_f32_bytes(value)
            assert len(raw) == batch * width * 4
            writers["input"].append([raw[r * width * 4:(r + 1) * width * 4] for r in range(batch)])
        evidence["input_capture"] = writers["input"].finish()
        evidence["actual_prefill_inputs"] = {"fp32_row_sha256": evidence["input_capture"]["row_sha256"],
            "tokens": tokens, "words_per_row": tokens * width, "positions": positions, "lengths": lengths,
            "hash_order": "Time then hidden channel, complete finite FP32 input rows"}
        save()
        assert evidence["actual_prefill_inputs"] == prior["actual_prefill_inputs"]
        return original_prefill(values, positions, lengths)

    def chunk(*args, **kwargs):
        assert decoder.prefill_mode
        context["count"] = len(args[0])
        return original_chunk(*args, **kwargs)

    def run(inputs):
        packed = original_run(inputs)
        count = context["count"]
        assert 1 <= count <= size
        value = decoder.io.slice(packed, program.debug_layer0_offset, [batch, size * width])
        raw = decoder.io.finite_f32_bytes(value)
        assert len(raw) == batch * size * width * 4
        writers["layer0"].append([raw[r * size * width * 4:r * size * width * 4 + count * width * 4] for r in range(batch)])
        evidence["chunks"].append({"count": count, "prefill": decoder.prefill_mode})
        save()
        return packed

    decoder.prefill_sequence, decoder.chunk_hidden, program.run = prefill, chunk, run
    result = model.infer_batch(case["prompts"], case["images"], 1)
    evidence["instrumented"] = result
    evidence["layer0_capture"] = writers["layer0"].finish()
    evidence["captured_state"] = snapshot(decoder, batch)
    evidence["captured_hidden"] = hidden(decoder)
    save()
    original_gates(evidence["captured_state"], evidence["captured_hidden"])
    for key in ("prompt_token_ids", "generated_token_ids", "sampled_tokens_including_stop", "finish_reasons", "stop_token_ids", "visual_tokens_per_row"):
        assert result[key] == control[key]
    assert sum(c["count"] for c in evidence["chunks"]) == evidence["actual_prefill_inputs"]["tokens"]
    event("observation_complete")
except BaseException as error:
    evidence["failure"] = {"type": type(error).__name__, "message": str(error)}
    save()
    raise
finally:
    for writer in writers.values():
        writer.close()
    model.close()
    event("closed")
evidence["diagnostic_complete"] = True
save()
print("LAYER0_DEBUG_COMPLETE", batch, flush=True)
