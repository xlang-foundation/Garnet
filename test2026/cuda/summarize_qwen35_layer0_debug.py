"""Audit admitted diagnostic graph observations of actual layer0 hidden values."""
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
from summarize_qwen35_prefill_state import differences


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


root = Path(sys.argv[1]).resolve()
read = lambda p: json.loads(p.read_text(encoding="utf-8-sig"))
output = root / "qwen35-layer0-debug-v1-summary.json"
assert not output.exists()
identity_path = root / "qwen35-layer0-debug-v1-before-run-identity.json"
identity, epoch = read(identity_path), digest(identity_path)
design = read(root / "qwen35-layer0-debug-v1-prospective-design.json")
proof = read(root / "qwen35-layer-debug-observer-selfcheck-v1.json")["ast_proof"]
records, arrays, sources = {}, {}, {}
for batch in (1, 2):
    tag = f"qwen35-layer0-debug-b{batch}-v1"
    path, exit_path = root / (tag + ".json"), root / (tag + "-exit.json")
    data, terminal = read(path), read(exit_path)
    assert terminal["exit_code"] == 0 and terminal["execution_identity_sha256"].lower() == epoch
    assert terminal["garnet_dll_sha256"].lower() == identity["garnet_binary_sha256"].lower()
    assert terminal["gpu_uuid"] == data["cuda_visible_devices"] == design["gpu_uuid"]
    assert data["batch"] == batch and data["diagnostic_complete"] and data["instrumented_graph_admitted_against_original"]
    assert data["ast_proof"] == proof and not data["root_cause_resolved"]
    prior_path = root / f"qwen35-prefill-state-b{batch}-v1.json"
    assert digest(prior_path) == data["prior_sha256"]
    prior = read(prior_path)
    assert data["profile"] == prior["profile"] and data["actual_prefill_inputs"] == prior["actual_prefill_inputs"]
    for role in ("control", "captured"):
        for key in ("batch", "row_width_words", "finite_fp32_words_total", "row_sha256", "components", "format"):
            assert data[role + "_state"][key] == prior["captured_state"][key]
        for key in ("raw_hidden_rows", "normalized_hidden_rows"):
            for a, b in zip(data[role + "_hidden"][key], prior["captured_hidden"][key]):
                assert struct.pack("<" + "f" * len(a), *a) == struct.pack("<" + "f" * len(b), *b)
    width = data["debug_layout"]["hidden_size"]
    tokens = data["actual_prefill_inputs"]["tokens"]
    assert width == 2560 and tokens == design["actual_prompt_tokens"] == 85
    size = data["profile"]["prefillChunk"]
    assert data["debug_layout"]["tokens_per_chunk"] == size == 64
    assert data["debug_layout"]["offset_words"] == prior["captured_state"]["row_width_words"] + size * width
    assert data["debug_layout"]["complete_output_width_words"] == prior["captured_state"]["row_width_words"] + 2 * size * width
    assert [c["count"] for c in data["chunks"]] == [64, 21] and all(c["prefill"] for c in data["chunks"])
    arrays[batch] = {}
    for key, suffix in (("input_capture", ".input.f32"), ("layer0_capture", ".layer0.f32")):
        capture = data[key]
        binary = root / (tag + suffix)
        assert Path(capture["binary_file"]).resolve() == binary.resolve()
        assert capture["rows"] == batch and capture["words_per_row"] == tokens * width
        assert capture["binary_bytes"] == binary.stat().st_size == batch * tokens * width * 4 <= design["max_bytes_per_binary"]
        assert digest(binary) == capture["binary_sha256"]
        values = np.memmap(binary, dtype="<f4", mode="r", shape=(batch, tokens * width))
        assert np.isfinite(values).all()
        assert [hashlib.sha256(row.tobytes()).hexdigest() for row in values] == capture["row_sha256"]
        if key == "input_capture":
            assert capture["row_sha256"] == prior["actual_prefill_inputs"]["fp32_row_sha256"]
        arrays[batch][key] = values
        sources[binary.name] = digest(binary)
    records[batch] = data
    for p in (path, exit_path):
        sources[p.name] = digest(p)
a, b = records[1], records[2]
pa, pb = dict(a["profile"]), dict(b["profile"])
assert pa.pop("batch") == 1 and pb.pop("batch") == 2 and pa == pb
inputs_a, inputs_b = arrays[1]["input_capture"], arrays[2]["input_capture"]
assert np.array_equal(inputs_a[0].view("<u4"), inputs_b[0].view("<u4"))
assert np.array_equal(inputs_b[0].view("<u4"), inputs_b[1].view("<u4"))
out_a, out_b = arrays[1]["layer0_capture"], arrays[2]["layer0_capture"]
metrics = differences(out_a[0], out_b[0])
per_token = []
for token in range(85):
    start = token * 2560
    per_token.append({"token_index_zero_based": token,
        "difference": differences(out_a[0, start:start + 2560], out_b[0, start:start + 2560])})
first = next((p for p in per_token if p["difference"]["fp32_words_different"]), None)
if first is not None:
    token = first["token_index_zero_based"]
    first["prompt_token_id"] = a["control"]["prompt_token_ids"][0][token]
    first["position"] = a["actual_prefill_inputs"]["positions"][0][token]
result = {"diagnostic_evidence_complete": True, "root_cause_resolved": False,
    "instrumented_graph_matches_original_complete_states_and_first_hidden": True,
    "actual_input_words_identical_between_batches_and_duplicate_rows": True,
    "layer0_output_difference": metrics, "first_differing_layer0_prompt_token": first,
    "per_token_layer0_differences": per_token,
    "batch2_duplicate_layer0_row_difference": differences(out_b[0], out_b[1]),
    "sources_sha256": sources, "execution_identity_sha256": epoch,
    "garnet_dll_sha256": identity["garnet_binary_sha256"], "production_model_modified": False,
    "scope": "Actual layer0 output from an extra-output diagnostic graph admitted only after exact original full-state/first-hidden controls. Not an operator oracle or proof that all unseen intermediates match. No quality, serving, latency, memory saving, batch invariance or 2x acceptance."}
output.write_text(json.dumps(result, indent=2) + "\n")
print("LAYER0_DEBUG_EVIDENCE_COMPLETE", "first_prompt_token", None if first is None else first["token_index_zero_based"])
