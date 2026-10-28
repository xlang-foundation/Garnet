"""Audit exact original controls, then compare observed BF16 GEMM boundaries."""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from summarize_qwen35_prefill_state import differences

root = Path(sys.argv[1]).resolve()
read = lambda p: json.loads(p.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bf16(words):
    return (np.asarray(words, dtype="<u2").astype("<u4") << 16).view("<f4")


output = root / "qwen35-layer0-stage-v1-summary.json"
assert not output.exists()
identity_path = root / "qwen35-layer0-stage-v1-before-run-identity.json"
identity, epoch = read(identity_path), digest(identity_path)
fixture_path = root / "qwen35-layer0-stage-v1-fixtures.json"
fixture = read(fixture_path)
build = read(root / "qwen35-layer0-stage-v1-build-exit.json")
executable = root / "qwen35-layer0-stage-probe-v1.exe"
assert build["exit_code"] == 0 and build["execution_identity_sha256"].lower() == epoch
assert build["component_binary_sha256"].lower() == digest(executable)
gpu = read(root / "qwen35-layer0-stage-v1-prospective-design.json")["gpu_uuid"]
weights = {w["suffix"]: w for w in fixture["tensors"]}
for item in fixture["tensors"]:
    assert digest(Path(item["file"])) == item["sha256"]
data_by_batch, stage, sources = {}, {}, {}
for batch in (1, 2):
    tag = f"qwen35-layer0-stage-b{batch}-v1"
    path, exit_path = root / (tag + ".json"), root / (tag + "-exit.json")
    data, terminal = read(path), read(exit_path)
    assert terminal["exit_code"] == 0 and terminal["execution_identity_sha256"].lower() == epoch
    assert terminal["component_binary_sha256"].lower() == build["component_binary_sha256"].lower()
    assert data["batch"] == batch and data["gpu_uuid"] == gpu and data["sm"] == 89
    assert data["admitted_exact_original_reference"] and data["cuda_graph_capture"] and not data["whole_program_cuda"]
    assert [(c["repeat"], c["start"], c["count"], c["output_word_mismatches"]) for c in data["chunks"]] == [(r, s, n, 0) for r in (0, 1) for s, n in ((0, 64), (64, 21))]
    case = next(c for c in fixture["cases"] if c["batch"] == batch)
    for repeat in (0, 1):
        assert data["repeat_state_mismatches"][str(repeat)] == 0
        for label, key, suffix in (("actual_output_files", "expected_output", f".actual-r{repeat}.f32"),
                                  ("actual_state_files", "expected_state", f".state-r{repeat}.f32")):
            actual = Path(data[label][str(repeat)])
            expected = Path(case["files"][key]["file"])
            assert actual.resolve() == (root / (tag + suffix)).resolve()
            assert digest(expected) == case["files"][key]["sha256"]
            assert actual.read_bytes() == expected.read_bytes()
            assert np.isfinite(np.fromfile(actual, dtype="<f4")).all()
            sources[actual.name] = digest(actual)
    binary = Path(data["stage_file"])
    assert binary.resolve() == (root / (tag + ".stages.u16")).resolve()
    values = np.memmap(binary, dtype="<u2", mode="r")
    assert len(values) == data["stage_words"]
    entries, cursor = {}, 0
    for entry in data["observations"]:
        key = entry["repeat"], entry["chunk_start"], entry["row"], entry["projection"], entry["part"]
        assert key not in entries and entry["word_offset"] == cursor and entry["gemm_rows"] == batch * 64
        assert entry["words"] == (entry["inner"] if entry["part"] == "input" else entry["columns"])
        entries[key] = values[cursor:cursor + entry["words"]]
        assert np.isfinite(bf16(entries[key])).all()
        cursor += entry["words"]
    projections = ("qkv", "z", "b", "a", "out", "gate", "up", "down")
    expected = {(r, s, b, p, k) for r in (0, 1) for s in (0, 64) for b in range(batch) for p in projections for k in ("input", "output")}
    assert set(entries) == expected and cursor == len(values)
    for _, s, b, p, k in entries:
        assert np.array_equal(entries[0, s, b, p, k], entries[1, s, b, p, k])
    for p in (path, exit_path, binary):
        sources[p.name] = digest(p)
    data_by_batch[batch], stage[batch] = data, entries
rows, dot_checks = [], []
suffixes = {p: ("mlp." if p in ("gate", "up", "down") else "linear_attn.") + (p + "_proj.weight" if p in ("gate", "up", "down", "out") else "in_proj_" + p + ".weight") for p in projections}
for start in (0, 64):
    for name in projections:
        a, b = stage[1], stage[2]
        query_a, query_b = a[0, start, 0, name, "input"], b[0, start, 0, name, "input"]
        output_a, output_b = a[0, start, 0, name, "output"], b[0, start, 0, name, "output"]
        weight = weights[suffixes[name]]
        assert weight["shape"] == [len(output_a), len(query_a)] == [len(output_b), len(query_b)]
        row = {"prompt_token_index": start, "projection": name,
            "input_difference": differences(bf16(query_a), bf16(query_b)),
            "output_difference": differences(bf16(output_a), bf16(output_b)),
            "duplicate_batch2_input_equal": bool(np.array_equal(query_b, b[0, start, 1, name, "input"])),
            "duplicate_batch2_output_equal": bool(np.array_equal(output_b, b[0, start, 1, name, "output"]))}
        rows.append(row)
        changed = np.flatnonzero(output_a != output_b)
        if np.array_equal(query_a, query_b) and len(changed):
            matrix = np.memmap(weight["file"], dtype="<u2", mode="r", shape=tuple(weight["shape"]))
            query = bf16(query_a)
            for index in changed[:8]:
                coefficients = bf16(matrix[int(index)])
                products = [float(x) * float(y) for x, y in zip(query, coefficients)]
                dot, absolute = math.fsum(products), math.fsum(abs(v) for v in products)
                gamma = len(query) * 2**-24 / (1 - len(query) * 2**-24)
                bound = gamma * absolute + (abs(dot) + gamma * absolute) / 256 + 2**-133
                observed = [float(bf16([output_a[index]])[0]), float(bf16([output_b[index]])[0])]
                dot_checks.append({"prompt_token_index": start, "projection": name, "output_channel": int(index),
                    "same_exact_bf16_query_and_weight": True, "fp64_dot": dot, "sum_abs_products": absolute,
                    "original_fixed_fp32_accum_bf16_output_bound": bound, "observed_batch1_batch2": observed,
                    "original_fixed_bound_fractions": [abs(v - dot) / bound for v in observed]})
result = {"diagnostic_evidence_complete": True, "root_cause_resolved": False,
    "all_original_output_and_state_words_exact_both_repeats": True,
    "all_observed_bf16_words_replay_identically": True, "projection_comparisons": rows,
    "sampled_same_query_fp64_checks": dot_checks,
    "sampled_original_fixed_bounds_passed": all(f <= 1 for c in dot_checks for f in c["original_fixed_bound_fractions"]),
    "sources_sha256": sources, "execution_identity_sha256": epoch, "component_binary_sha256": digest(executable),
    "reference_garnet_native_epoch_sha256": fixture["reference_native_epoch_sha256"],
    "scope": "Admitted actual firstlayer component only. BF16 projection IO at first tokens of two chunks; operator-local observations do not prove a unique internal algorithm, all-token causal attribution, independent model quality, a batch-invariant production policy or speed/memory/2x acceptance."}
output.write_text(json.dumps(result, indent=2) + "\n")
assert result["sampled_original_fixed_bounds_passed"], "Original sampled FP64 projection bound failed; preserve evidence"
print("LAYER0_STAGE_EVIDENCE_COMPLETE", [(r["prompt_token_index"], r["projection"], r["input_difference"]["fp32_words_different"], r["output_difference"]["fp32_words_different"]) for r in rows])
