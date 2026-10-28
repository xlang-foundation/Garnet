"""Freeze an extra-output diagnostic after completed canonical-state localization."""
import ast
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
repo = root / "Garnet"
read = lambda p: json.loads(p.read_text(encoding="utf-8-sig"))
output = root / "qwen35-layer0-debug-v1-before-run-identity.json"
assert not output.exists()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


assert read(root / "qwen35-prefill-state-v1-controller-exit.json")["exit_code"] == 0
summary = read(root / "qwen35-prefill-state-v1-summary.json")
assert summary["diagnostic_evidence_complete"] and summary["same_actual_prefill_inputs_and_positions"]
first = summary["first_differing_canonical_component"]
assert (first["layer"], first["part"]) == (1, 1)
assert all(p["batch1_vs_batch2_row0"]["fp32_words_different"] == 0 for p in summary["components"][:2])
prior_path = root / "qwen35-prefill-state-v1-before-run-identity.json"
subprocess.run([sys.executable, str(repo / "test2026/cuda/validate_qwen35_execution_identity.py"), str(prior_path)], check=True)
identity = read(prior_path)
identity["capture_provenance"] = {"parent_manifest": prior_path.name, "parent_manifest_sha256": digest(prior_path),
    "parent_captured_at_utc": identity["captured_at_utc"],
    "checkpoint_full_hashes": "Inherited validated parent full digests with current size/mtime guards revalidated; no fresh payload scan."}
identity["captured_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
checks = read(root / "qwen35-layer-debug-observer-selfcheck-v1.json")
assert checks["all_known_value_checks_passed"] and checks["ast_proof"]["removal_of_three_observer_additions_recovers_original_ast"]
for name, expected in checks["source_sha256"].items():
    path = repo / ("xModel/qwen35/vl_0_8b/" if name == "cuda_packed_program.py" else "test2026/cuda/") / name
    assert digest(path) == expected, "AST observer changed after CPU proof"
for name in ("qwen35_layer_debug_observer.py", "probe_qwen35_layer0_debug.py", "summarize_qwen35_layer0_debug.py", "freeze_qwen35_layer0_debug.py"):
    path = repo / "test2026/cuda" / name
    ast.parse(path.read_text())
    identity["source_sha256"]["test2026/cuda/" + name] = digest(path)
names = ["run-qwen35-layer0-debug.ps1", "qwen35-layer0-debug-v1-prospective-design.json",
    "check-qwen35-layer-debug-observer-v1.py", "qwen35-layer-debug-observer-selfcheck-v1.json",
    "qwen35-layer-debug-observer-selfcheck-v1.f32", "qwen35-layer0-debug-method-v1.py.txt",
    "qwen35-prefill-state-v1-controller-exit.json", "qwen35-prefill-state-v1-summary.json"]
for batch in (1, 2):
    for suffix in (".json", ".log", "-exit.json"):
        names.append(f"qwen35-prefill-state-b{batch}-v1" + suffix)
for name in names:
    identity["orchestration_sha256"][name] = digest(root / name)
output.write_text(json.dumps(identity, indent=2) + "\n")
print("LAYER0_DEBUG_SOURCE_EPOCH_READY", output)
