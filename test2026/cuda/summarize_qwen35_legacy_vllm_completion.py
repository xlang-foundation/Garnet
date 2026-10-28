"""Preserve the old serial4B completion evidence without inventing a sidecar."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
a = p.parse_args()
output = a.root / 'vllm-4b-b8-saved-completion-admission.json'
assert not output.exists(), 'Preserve legacy admission evidence'
names = ['matched512-vllm-4b-b8.json', 'matched512-vllm-4b-b8.config.json',
         'matched512-vllm-4b-b8.log', 'large-4b-resume-stages.json',
         'resume-4b-large-matched-matrix.ps1']
blobs = {name: (a.root / name).read_bytes() for name in names}
rows = json.loads(blobs[names[0]])
config = json.loads(blobs[names[1]])
log = blobs[names[2]].decode('utf-8-sig')
stages = json.loads(blobs[names[3]])
controller = blobs[names[4]].decode('utf-8-sig').replace('\r\n', '\n')
selected = [s['stage'] for s in stages if s['model'] == '4b' and s['batch'] == 8]
checks = {
    '28_unique_requests': len(rows) == len({(x['trial'], x['workload_group']) for x in rows}) == 28,
    'completion_marker': 'QWEN35_VLLM_BENCHMARK_COMPLETE Qwen3.5-4B batch 8 requests 28' in log,
    'ordered_post_exit_stages': selected == ['garnet_begin', 'vllm_begin', 'lmstudio_begin', 'profile_complete'],
    'post_exit_guard_in_retained_controller': "if($LASTEXITCODE -ne 0){throw 'vLLM4B failed'}\n    Stage 'lmstudio_begin' $batch" in controller,
    'speculation_none_in_saved_log': 'speculative_config=None' in log,
    'no_speculation_in_saved_options': 'speculative_config' not in config['options'],
}
result = {
    'all_passed': all(checks.values()), 'checks': checks,
    'sources': [{'file': name, 'sha256': hashlib.sha256(blob).hexdigest()} for name, blob in blobs.items()],
    'original_per_run_exit_sidecar_available': False,
    'scope': 'Derived admission from complete raw cohort, terminal benchmark marker and retained controller '
             'stage after checked native exit. This is not a reconstructed original exit sidecar, original '
             'complete-host isolation proof or a rerun. Saved rows/config/log/controller are unchanged.',
}
output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(checks, indent=2))
assert result['all_passed']
