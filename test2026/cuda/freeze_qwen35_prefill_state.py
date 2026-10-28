"""Capture a new observation epoch after the completed identical-prefix study."""
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
repo = root / "Garnet"
output = root / "qwen35-prefill-state-v1-before-run-identity.json"
assert not output.exists()
read = lambda p: json.loads(p.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


for name in ("qwen35-after-batches-v1-exit.json", "qwen35-4b-batch-drift-v1-controller-exit.json"):
    assert read(root / name)["exit_code"] == 0
summary = read(root / "qwen35-4b-batch-drift-v1-summary.json")
assert summary["diagnostic_evidence_complete"] and summary["same_actual_prefill_inputs_and_positions"]
assert summary["first_raw_hidden_fp32_difference_prediction"] == 0
assert read(root / "optimized-batches-cross-mode-final-v1.json")["all_six_profiles_cross_mode_equal"]
prior_path = root / "qwen35-4b-batch-drift-v1-before-run-identity.json"
subprocess.run([sys.executable, str(repo / "test2026/cuda/validate_qwen35_execution_identity.py"), str(prior_path)], check=True)
identity = read(prior_path)
identity["capture_provenance"] = {"parent_manifest": prior_path.name, "parent_manifest_sha256": digest(prior_path),
    "parent_captured_at_utc": identity["captured_at_utc"],
    "checkpoint_full_hashes": "Inherited validated parent full digests with revalidated current size/mtime guards; payloads not freshly hashed."}
identity["captured_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
for name in ("qwen35_prefill_state_observer.py", "probe_qwen35_prefill_state.py", "summarize_qwen35_prefill_state.py", "freeze_qwen35_prefill_state.py"):
    relative = "test2026/cuda/" + name
    identity["source_sha256"][relative] = digest(repo / relative)
checks = read(root / "qwen35-prefill-state-observer-selfcheck-v2.json")
assert checks["all_known_value_checks_passed"] and not checks["gpu_executed"]
for name, expected in checks["source_sha256"].items():
    assert digest(repo / "test2026/cuda" / name) == expected, "Observer changed after known-value checks"
names = ["run-qwen35-prefill-state.ps1", "qwen35-prefill-state-v1-prospective-design.json",
    "check-qwen35-prefill-observer-v1.py", "qwen35-prefill-state-observer-selfcheck-v1.json",
    "qwen35-prefill-state-observer-selfcheck-v1.f32", "qwen35-prefill-state-observer-selfcheck-v2.json",
    "qwen35-prefill-state-observer-selfcheck-v2.f32", "qwen35-after-batches-v1-exit.json",
    "qwen35-4b-batch-drift-v1-controller-exit.json", "qwen35-4b-batch-drift-v1-summary.json",
    "audit-qwen35-batch-cross-mode-v1.py", "optimized-batches-cross-mode-final-v1.json",
    "audit-qwen35-state-allocation-v1.py", "qwen35-state-allocation-v1.json"]
for batch in (1, 2):
    for suffix in (".json", ".log", "-exit.json"):
        names.append(f"qwen35-4b-batch-drift-b{batch}-v1" + suffix)
for name in names:
    identity["orchestration_sha256"][name] = digest(root / name)
output.write_text(json.dumps(identity, indent=2) + "\n")
print("PREFILL_STATE_SOURCE_EPOCH_READY", output)
