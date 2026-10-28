"""Pin the standalone source/fixtures before compilation and serialized GPU cases."""
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
output = root / "qwen35-layer0-stage-v1-before-run-identity.json"
assert not output.exists()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


assert read(root / "qwen35-layer0-debug-audit-v2-exit.json")["exit_code"] == 0
summary = read(root / "qwen35-layer0-debug-v2-summary.json")
assert summary["diagnostic_evidence_complete"] and summary["instrumented_graph_matches_original_complete_states_and_first_hidden"]
assert summary["first_differing_layer0_prompt_token"]["token_index_zero_based"] == 0
prior_path = root / "qwen35-layer0-debug-audit-v2-before-run-identity.json"
subprocess.run([sys.executable, str(repo / "test2026/cuda/validate_qwen35_execution_identity.py"), str(prior_path)], check=True)
identity = read(prior_path)
identity["capture_provenance"] = {"parent_manifest": prior_path.name, "parent_manifest_sha256": digest(prior_path),
    "parent_captured_at_utc": identity["captured_at_utc"], "reference_native_epoch_sha256": summary["execution_identity_sha256"],
    "scope": "Standalone component compiled after this snapshot; exact executable digest recorded in build/case exits. Model DLL unchanged. Inherited checkpoint full digests/current stat guards validated; no fresh full payload scan."}
identity["captured_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
for name in ("probe_qwen35_layer0_stages.cu", "summarize_qwen35_layer0_stages.py", "freeze_qwen35_layer0_stages.py"):
    path = repo / "test2026/cuda" / name
    if path.suffix == ".py":
        ast.parse(path.read_text())
    identity["source_sha256"]["test2026/cuda/" + name] = digest(path)
names = ["run-qwen35-layer0-stages.ps1", "build-layer0-stage-probe-v1.cmd", "prepare-qwen35-layer0-stage-v1.py",
    "qwen35-layer0-stage-v1-prospective-design.json", "qwen35-layer0-stage-v1-fixtures.json",
    "qwen35-layer0-debug-v2-summary.json", "qwen35-layer0-debug-audit-v2-exit.json"]
fixture = read(root / "qwen35-layer0-stage-v1-fixtures.json")
assert fixture["reference_debug_summary_sha256"] == digest(root / "qwen35-layer0-debug-v2-summary.json")
for tensor in fixture["tensors"]:
    path = Path(tensor["file"])
    assert path.stat().st_size == tensor["bytes"] and digest(path) == tensor["sha256"]
    names.append(path.relative_to(root).as_posix())
for case in fixture["cases"]:
    for value in case["files"].values():
        path = Path(value["file"])
        assert path.stat().st_size == value["bytes"] and digest(path) == value["sha256"]
        names.append(path.relative_to(root).as_posix())
for name in set(names):
    identity["orchestration_sha256"][name] = digest(root / name)
output.write_text(json.dumps(identity, indent=2) + "\n")
print("LAYER0_STAGE_SOURCE_EPOCH_READY", output)
