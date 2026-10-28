"""Plot audited diagnostic boundaries without relabeling primary benchmarks."""
import hashlib
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

root = Path(sys.argv[1]).resolve()
stem = root / 'progress-first-token-boundaries-v2'
for extension in ('.png', '.svg', '-sources.json'):
    assert not Path(str(stem) + extension).exists()
sources = {}


def read(path):
    raw = path.read_bytes()
    sources[path.name] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))


summary = read(root / 'qwen35-first-token-probe-v2-summary.json')
assert summary['all56_boundary_and_input_output_work_controls_passed']
assert summary['process_observation_qualified'] and not summary['primary_latency_accepted']
cases = []
names = ('target_prediction_ready', 'speculative_prefill_returned', 'complete_answer_returned')
for case in summary['cases']:
    size, batch = case['model'], case['batch']
    rows = read(root / f'first-token-probe-{size}-b{batch}-v2-requests.json')
    warm = [r for r in rows if not r['warmup']]
    assert len(rows) == 28 and len(warm) == 20
    medians = [statistics.median(r['first_token_boundary_diagnostic']['timestamps_ns_from_probe_entry'][n] / 1e6 for r in warm) for n in names]
    cases.append({'model': size, 'batch': batch, 'boundary_medians_ms': dict(zip(names, medians))})

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none'})
fig = plt.figure(figsize=(12, 5.5), facecolor='white')
ax = fig.add_axes([0.14, 0.40, 0.79, 0.35])
colors = ('#2473b8', '#ed9b32', '#d3d9e0')
for y, case in zip((1, 0), cases):
    target, mtp, answer = case['boundary_medians_ms'].values()
    for start, end, color in zip((0, target, mtp), (target, mtp, answer), colors):
        ax.barh(y, end-start, left=start, height=0.33, color=color, edgecolor='white', linewidth=0.4)
    ax.text(answer+8, y, f'{answer:.1f} ms', va='center', weight='bold', color='#26364a')
ax.set_yticks([1, 0], ['2B / batch 1', '4B / batch 8'])
ax.set_xlim(0, 700)
ax.set_ylim(-0.6, 1.6)
ax.set_xlabel('Milliseconds from probe entry')
ax.set_axisbelow(True)
ax.grid(axis='x', color='#e8ebef')
ax.tick_params(axis='both', length=0, pad=8)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.text(0.055, 0.94, 'First prediction is ready well before the complete answer returns', fontsize=17, weight='bold', color='#172b43')
fig.text(0.055, 0.885, 'Qwen3.5 CUDA · original MTP · warmed diagnostic · different batch sizes', fontsize=11, color='#526174')
fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(colors,
    ('Until target prediction', 'Remaining MTP setup', 'Until complete answer return'))],
    loc='upper left', bbox_to_anchor=(0.14, 0.835), ncol=3, frameon=False, fontsize=10)
table = fig.add_axes([0.14, 0.16, 0.79, 0.15]);table.axis('off')
cells = [[f"{c['model'].upper()} / B{c['batch']}"] + [f'{v:.2f} ms' for v in c['boundary_medians_ms'].values()] for c in cases]
tab = table.table(cellText=cells, colLabels=['Profile', 'Target prediction ready', 'MTP prefill returned', 'Complete answer returned'],
    cellLoc='center', colLoc='center', loc='center', colWidths=[0.15, 0.27, 0.27, 0.31])
tab.auto_set_font_size(False);tab.set_fontsize(10);tab.scale(1, 1.45)
for (row, col), cell in tab.get_celld().items():
    cell.set_edgecolor('#e5e9ee')
    cell.set_facecolor('#edf2f8' if row == 0 else '#ffffff')
    if row == 0:cell.set_text_props(weight='bold', color='#26364a')
fig.text(0.055, 0.065, 'Each boundary is the median of 20 warmed CPU timestamps. Bars show differences between median boundaries.', fontsize=9, color='#526174')
fig.text(0.055, 0.030, 'All 56 workload/control checks passed. Timestamp hooks add overhead; client streaming and primary speed gains are not established.', fontsize=9, color='#526174')
fig.savefig(str(stem)+'.png', dpi=160)
fig.savefig(str(stem)+'.svg')
plt.close(fig)
Path(str(stem)+'-sources.json').write_text(json.dumps({'sources_sha256': sources, 'cases': cases,
    'execution_identity_sha256': summary['execution_identity_sha256'], 'primary_latency_accepted': False,
    'scope': 'Diagnostic median boundaries from a common probe-entry clock, not primary latency or client streaming.'}, indent=2)+'\n')
print(str(stem)+'.png')
