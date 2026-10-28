"""Render only completed six-process memory results and exact saved-work controls."""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(sys.argv[1]).resolve()
sources = {}


def read(path):
    raw = path.read_bytes()
    sources[path.name] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))


stem = 'progress-parent-mtp-memory-v1'
for suffix in ('.png', '.svg', '-sources.json'):
    assert not (root / (stem + suffix)).exists(), 'Preserve rendered evidence'
assert read(root / 'qwen35-parent-mtp-memory-v1-controller-exit.json')['exit_code'] == 0
data = read(root / 'qwen35-parent-mtp-memory-v1-summary.json')
assert data['all_six_memory_controls_passed'] and len(data['models']) == 2
timing = read(root / 'qwen35-shared-mtp-study-v1-summary.json')
assert timing['all_completed_raw_controls_passed']
lm = {}
for model in data['models']:
    size, batch = model['model'], model['batch']
    for mode in ('d4', 'split', 'parent'):
        events = read(root / f'memory-parent-mtp-{size}-b{batch}-{mode}-v1-session.json')
        assert events[-1]['event'] == 'complete'
        selected = [e for e in events if 'input_sha256' in e]
        role = 'a1' if mode == 'd4' else 'b1'
        path = root / f'matched512-garnet-{size}-b{batch}-study-shared-mtp-v1-{role}.json'
        reference = read(path)
        assert sources[path.name] == timing['sources_sha256'][str(path)]
        assert len(selected) == 6
        for event, ref in zip(selected, reference):
            assert event['input_sha256'] == ref['input_sha256']
            assert event['speculative_metrics'] == ref['speculative_metrics'], 'Diagnostic discrete work differs from its completed timing route'
    summary = read(root / f'memory-lmstudio-{size}-b{batch}-512-partitioned-summary.json')
    assert summary['adapter']['uuid'] == 'GPU-472bb21a-0ca8-9d60-47a0-c527bb3d2d81'
    lm[size] = summary['phases']['steady_after']['bytes']['TotalCommitted']['median']

face = '#f7f9fc'
palette = ['#8195a5', '#c49443', '#159b83', '#567fb2']
fig, axes = plt.subplots(2, 2, figsize=(14.2, 8.8), facecolor=face)
fig.suptitle('Parent-pool sharing — isolated GPU-memory diagnostics', x=.055, y=.98,
             ha='left', fontsize=19, weight='bold')
fig.text(.055, .925, 'Six fresh sequential processes on the same RTX4080; same target profiles and six input/output groups per process.', fontsize=10)
for column, model in enumerate(data['models']):
    size, batch, modes = model['model'], model['batch'], model['modes']
    process = [modes[m]['warm_process_counter_bytes']['TotalCommitted'] / 2**30 for m in ('d4', 'split', 'parent')]
    process.append(lm[size] / 2**30)
    owned = [modes[m]['retained_owned_model_bytes'] / 2**30 for m in ('d4', 'split', 'parent')]
    for row, values in enumerate((process, owned)):
        ax = axes[row, column]
        labels = ['Original\nMTP D4', 'Split MTP\nP16/D1/C4', 'Parent-pool\nsharing'] + (['LM Studio\nsaved'] if row == 0 else [])
        bars = ax.bar(range(len(values)), values, width=.57, color=palette[:len(values)])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value + max(values)*.025, f'{value:.3f}', ha='center', fontsize=10, weight='bold')
        ax.set_xticks(range(len(values)), labels, fontsize=9)
        ax.set_ylim(0, max(values)*1.19)
        ax.set_ylabel('GiB', fontsize=10)
        ax.set_title(f'Qwen3.5 {size.upper()} · batch {batch}\n' +
                     ('Warmed Windows TotalCommitted counter' if row == 0 else 'Retained model-owned allocations'),
                     loc='left', fontsize=12, weight='bold', pad=14)
        ax.set_facecolor(face)
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.17)
        for spine in ax.spines.values():
            spine.set_visible(False)
    d4_delta = model['control_minus_parent_counter_bytes']['d4']['TotalCommitted'] / 2**20
    split_delta = model['control_minus_parent_counter_bytes']['split']['TotalCommitted'] / 2**20
    removed = (modes['split']['retained_owned_model_bytes'] - modes['parent']['retained_owned_model_bytes']) / 2**20
    axes[0, column].text(0, -.30, f'Original − parent: {d4_delta:+.1f} MiB; split − parent: {split_delta:+.1f} MiB',
                         transform=axes[0, column].transAxes, fontsize=9)
    axes[1, column].text(0, -.30, f'Exclusive MTP pools removed versus split: {removed:.2f} MiB',
                         transform=axes[1, column].transAxes, fontsize=9)
fig.text(.055, .095, 'Process counters and owned allocations are different measures. Sampled committed bytes are not physical residency or lifetime peaks.', fontsize=9)
fig.text(.055, .06, 'LM Studio counters are historical; vLLM process totals are unavailable. Polling and phase holds invalidate latency comparisons.', fontsize=9)
fig.text(.055, .025, 'Experimental memory evidence only; no serving promotion or 2× speed acceptance. Independent quality and full profile coverage remain open.', fontsize=9, color='#a52b2b')
fig.subplots_adjust(left=.065, right=.97, top=.83, bottom=.22, hspace=.85, wspace=.25)
for suffix in ('png', 'svg'):
    fig.savefig(root / (stem + '.' + suffix), dpi=160, facecolor=face)
plt.close(fig)
provenance = {'sources_sha256': sources, 'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'execution_identity_sha256': data['execution_identity_sha256'], 'garnet_dll_sha256': data['garnet_dll_sha256'],
    'all_six_case_discrete_work_matches_completed_routes': True, 'primary_latency_accepted': False,
    'scope': 'Completed warmed process counters and retained ownership, separately labeled; historical saved LM Studio.'}
(root / (stem + '-sources.json')).write_text(json.dumps(provenance, indent=2) + '\n')
print(root / (stem + '.png'))
