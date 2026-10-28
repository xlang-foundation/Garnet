"""Audit complete row-major canonical captures and localize observed differences."""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


def differences(a, b):
    assert len(a) == len(b) > 0
    different, first, maximum, numerator, denominator = 0, None, 0.0, [], []
    exact_bf16_a = exact_bf16_b = 0
    for start in range(0, len(a), 131072):
        x, y = a[start:start + 131072], b[start:start + 131072]
        assert np.isfinite(x).all() and np.isfinite(y).all()
        wx, wy = x.view("<u4"), y.view("<u4")
        mask = wx != wy
        count = int(np.count_nonzero(mask))
        if count and first is None:
            first = start + int(np.flatnonzero(mask)[0])
        different += count
        exact_bf16_a += int(np.count_nonzero((wx & 65535) == 0))
        exact_bf16_b += int(np.count_nonzero((wy & 65535) == 0))
        dx, dy = x.astype(np.float64), y.astype(np.float64)
        delta = dx - dy
        maximum = max(maximum, float(np.max(np.abs(delta))))
        numerator.append(float(np.sum(delta * delta, dtype=np.float64)))
        denominator.append(float(np.sum(dx * dx, dtype=np.float64)))
    den = math.fsum(denominator)
    return {"words": len(a), "fp32_words_different": different, "first_differing_word_in_component": first,
        "max_absolute_difference": maximum, "relative_l2": math.sqrt(math.fsum(numerator) / den) if den else None,
        "words_exactly_representable_as_bf16": {"reference": exact_bf16_a, "candidate": exact_bf16_b}}


def main(root):
    output = root / "qwen35-prefill-state-v1-summary.json"
    assert not output.exists()
    read = lambda p: json.loads(p.read_text(encoding="utf-8-sig"))
    def digest(p):
        with p.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    identity_path = root / "qwen35-prefill-state-v1-before-run-identity.json"
    identity = read(identity_path)
    epoch = digest(identity_path)
    design = read(root / "qwen35-prefill-state-v1-prospective-design.json")
    config = read(root / "models/Qwen3.5-4B/config.json")["text_config"]
    sources, records, arrays = {}, {}, {}
    for batch in (1, 2):
        tag = f"qwen35-prefill-state-b{batch}-v1"
        path, exit_path = root / (tag + ".json"), root / (tag + "-exit.json")
        data, terminal = read(path), read(exit_path)
        assert terminal["exit_code"] == 0 and terminal["execution_identity_sha256"].lower() == epoch
        assert terminal["garnet_dll_sha256"].lower() == identity["garnet_binary_sha256"].lower()
        assert terminal["gpu_uuid"] == data["cuda_visible_devices"] == design["gpu_uuid"]
        assert data["diagnostic_complete"] and not data["root_cause_resolved"] and data["batch"] == batch
        prior_path = root / f"qwen35-4b-batch-drift-b{batch}-v1.json"
        assert digest(prior_path) == data["prior_sha256"]
        prior = read(prior_path)
        assert data["profile"] == prior["profile"]
        assert data["actual_prefill_inputs"] == prior["actual_prefill_inputs"]
        for key in ("raw_hidden_rows", "normalized_hidden_rows"):
            assert data["captured_hidden"][key] == data["control_hidden"][key] == prior["trace"][0][key]
        capture = data["captured_state"]
        for key in ("batch", "row_width_words", "finite_fp32_words_total", "row_sha256", "components", "format"):
            assert capture[key] == data["control_state"][key]
        width = capture["row_width_words"]
        assert capture["finite_fp32_words_total"] == batch * width
        binary = Path(capture["binary_file"])
        assert binary.resolve() == (root / (tag + ".f32")).resolve()
        assert binary.stat().st_size == capture["binary_bytes"] == batch * width * 4 <= design["max_capture_bytes_per_case"]
        assert digest(binary) == capture["binary_sha256"]
        values = np.memmap(binary, mode="r", dtype="<f4", shape=(batch, width))
        row_hashes = [hashlib.sha256() for _ in range(batch)]
        cursor = 0
        components = capture["components"]
        assert len(components) == 64
        for i, part in enumerate(components):
            assert part["layer"] == i // 2 and part["part"] == i % 2 + 1
            kind = config["layer_types"][i // 2]
            assert part["kind"] == kind
            if kind == "linear_attention":
                channels = 2 * config["linear_num_key_heads"] * config["linear_key_head_dim"] + config["linear_num_value_heads"] * config["linear_value_head_dim"]
                expected_width = channels * config["linear_conv_kernel_dim"] if i % 2 == 0 else config["linear_num_value_heads"] * config["linear_key_head_dim"] * config["linear_value_head_dim"]
            else:
                capacity = data["profile"]["maxInputTokens"] + data["profile"]["maxOutputTokens"]
                expected_width = config["num_key_value_heads"] * capacity * config["head_dim"]
            assert part["width_words"] == expected_width
            assert part["offset_words"] == cursor and part["width_words"] > 0
            n = part["width_words"]
            for row in range(batch):
                chunk = values[row, cursor:cursor + n]
                assert len(chunk) == n and np.isfinite(chunk).all()
                raw = chunk.tobytes()
                assert hashlib.sha256(raw).hexdigest() == part["row_sha256"][row]
                row_hashes[row].update(raw)
            cursor += n
        assert cursor == width and [h.hexdigest() for h in row_hashes] == capture["row_sha256"]
        for p in (path, exit_path, binary):
            sources[p.name] = digest(p)
        records[batch], arrays[batch] = data, values
    a, b = records[1], records[2]
    pa, pb = dict(a["profile"]), dict(b["profile"])
    assert pa.pop("batch") == 1 and pb.pop("batch") == 2 and pa == pb
    ia, ib = a["actual_prefill_inputs"], b["actual_prefill_inputs"]
    assert all(h == ia["fp32_row_sha256"][0] for h in ib["fp32_row_sha256"])
    assert ia["positions"][0] == ib["positions"][0] and ia["lengths"][0] == ib["lengths"][0]
    assert ia["words_per_row"] == ib["words_per_row"]
    comparisons = []
    for x, y in zip(a["captured_state"]["components"], b["captured_state"]["components"]):
        assert {k: v for k, v in x.items() if k != "row_sha256"} == {k: v for k, v in y.items() if k != "row_sha256"}
        start, n = x["offset_words"], x["width_words"]
        part = {k: x[k] for k in ("layer", "kind", "part", "offset_words", "width_words")}
        part["batch1_vs_batch2_row0"] = differences(arrays[1][0, start:start + n], arrays[2][0, start:start + n])
        part["batch2_row0_vs_row1"] = differences(arrays[2][0, start:start + n], arrays[2][1, start:start + n])
        comparisons.append(part)
    first = next((p for p in comparisons if p["batch1_vs_batch2_row0"]["fp32_words_different"]), None)
    result = {"diagnostic_evidence_complete": True, "root_cause_resolved": False,
        "same_actual_prefill_inputs_and_positions": True,
        "reset_and_observation_complete_state_equal": True, "first_prediction_hidden_matches_prior_trace": True,
        "first_differing_canonical_component": first, "components": comparisons,
        "batch2_identical_rows_all_canonical_words_equal": all(p["batch2_row0_vs_row1"]["fp32_words_different"] == 0 for p in comparisons),
        "sources_sha256": sources, "execution_identity_sha256": epoch,
        "garnet_dll_sha256": identity["garnet_binary_sha256"], "production_model_modified": False,
        "scope": "Complete finite canonical state after prefill at the first prediction. Captured convolution state covers only its retained window; it is not an all-prompt projection trace. Differences localize an observation, not an operator cause, independent quality, batch invariance, speed or memory acceptance."}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("PREFILL_STATE_EVIDENCE_COMPLETE", "first_component", None if first is None else (first["layer"], first["kind"], first["part"]))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
