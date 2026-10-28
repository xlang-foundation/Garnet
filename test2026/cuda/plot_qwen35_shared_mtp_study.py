"""New chart files from the completed shared-MTP study; preserve old charts."""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(sys.argv[1]).resolve()
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
summary_path = root / 'qwen35-shared-mtp-study-v1-summary.json'
terminal_path = root / 'qwen35-shared-mtp-study-v1-controller-exit.json'
assert read(terminal_path)['exit_code'] == 0, 'Only completed timing studies may be plotted'
data = read(summary_path)
assert data['all_completed_raw_controls_passed'] and len(data['models']) == 2
provisional = any(m['timing_provisional_due_to_process_observation'] for m in data['models'])
files = ['progress-shared-mtp-baselines-v1', 'progress-shared-mtp-all20-v1']
for name in files:
    for suffix in ('.png', '.svg', '-sources.json'):
        assert not (root / (name + suffix)).exists(), 'Preserve prior rendered evidence'

face = '#f7f9fc'
colors = {'garnet_a': '#8fb1a9', 'garnet_b': '#159b83', 'lmstudio': '#8293aa',
          'vllm_serial': '#4688c4', 'fp32': '#dd9c40', 'bf16': '#8b6ab0'}
provenance = {'summary': summary_path.name, 'summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
              'terminal_sha256': hashlib.sha256(terminal_path.read_bytes()).hexdigest(),
              'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'execution_identity_sha256': data['execution_identity_sha256'],
              'garnet_dll_sha256': data['garnet_dll_sha256'],
              'timing_provisional_due_to_process_observation': provisional,
              'goal_2x_achieved': False, 'independent_quality_accepted': False,
              'process_memory_measured': False}


def finish(fig, stem, source):
    for suffix in ('png', 'svg'):
        fig.savefig(root / (stem + '.' + suffix), dpi=160, facecolor=face)
    (root / (stem + '-sources.json')).write_text(json.dumps(dict(provenance, analysis=source), indent=2) + '\n')
    plt.close(fig)
    print(root / (stem + '.png'))


fig, axes = plt.subplots(1, 2, figsize=(16.4, 8.4), facecolor=face)
fig.suptitle('Shared-weight MTP sizing — request latency' + (' · PROVISIONAL' if provisional else ''),
             x=.04, y=.98, ha='left', fontsize=19, weight='bold')
fig.text(.04, .915, 'Same image hashes and 32 sampled tokens per row within each panel; sequential fresh-process Garnet ABBA.', fontsize=10)
for ax, model in zip(axes, data['models']):
    common = model['historical_common_equal32']
    saved = common['saved_competitors_ms']
    values = [common['current_control_ms'], common['current_candidate_ms']]
    labels = ['Garnet control · repeated', 'Garnet split MTP · repeated']
    palette = [colors['garnet_a'], colors['garnet_b']]
    for name, value in saved.items():
        values.append(value)
        if name == 'lmstudio':
            labels.append('LM Studio · saved'); palette.append(colors['lmstudio'])
        elif name == 'vllm_serial':
            labels.append('vLLM serial · saved'); palette.append(colors['vllm_serial'])
        else:
            bf16 = 'convbfloat16' in name
            labels.append('vLLM MTP3 · ' + ('BF16 conv' if bf16 else 'FP32 conv') + ' setting')
            palette.append(colors['bf16' if bf16 else 'fp32'])
    bars = ax.barh(range(len(values)), values, height=.59, color=palette)
    ax.set_yticks(range(len(values)), labels, fontsize=9)
    ax.invert_yaxis()
    for bar, value in zip(bars, values):
        ax.text(value + max(values) * .018, bar.get_y() + bar.get_height() / 2, f'{value:.1f}',
                va='center', fontsize=10, weight='bold')
    target = common['half_fastest_baseline_ms']
    ax.axvline(target, linestyle='--', color='#b52d2d', linewidth=1.5)
    ax.text(target, 1.025, f'2× latency target\n{target:.1f} ms', transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=10, color='#b52d2d')
    qualifier = '\nObserved overlap / incomplete observation' if model['timing_provisional_due_to_process_observation'] else ''
    ax.set_title(f'Qwen3.5 {model["model"].upper()} · batch {model["batch"]}\n'
                 f'{common["groups"]} common equal 32 groups\n'
                 f'Fastest saved baseline / candidate: {common["fastest_baseline_over_candidate_ratio"]:.3f}×{qualifier}',
                 loc='left', fontsize=12, weight='bold', pad=61)
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xlabel('API-return group median · ms (lower is better)', fontsize=10)
    ax.set_facecolor(face); ax.set_axisbelow(True); ax.grid(axis='x', alpha=.15)
    for spine in ax.spines.values(): spine.set_visible(False)
fig.text(.04, .13, 'Garnet: median of two repeats per group, then groups. All 20 warm groups are retained in the separate within-Garnet chart.', fontsize=9)
screen_text = ('Both sizing-only candidates missed the prospective 1.05× speed screen; no serving promotion.'
               if not any(m['experimental_speed_screen_passed'] for m in data['models'])
               else 'Experimental step-screen results are separate from full-goal acceptance; no automatic serving promotion.')
fig.text(.04, .175, screen_text, fontsize=10, color='#a52b2b', weight='bold')
fig.text(.04, .09, 'Competitors: saved cohorts. APIs, intermediate arithmetic and speculative settings differ; sampled observation cannot prove isolation.', fontsize=9)
fig.text(.04, .045, 'Independent quality, full batch/capacity coverage, new process-memory measurements and full-goal acceptance remain open.', fontsize=9, color='#a52b2b')
fig.subplots_adjust(left=.19, right=.97, top=.69, bottom=.28, wspace=.86)
finish(fig, files[0], [m['historical_common_equal32'] for m in data['models']])

fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.6), facecolor=face)
fig.suptitle('Shared-weight MTP sizing — all 20 warm groups' + (' · PROVISIONAL' if provisional else ''),
             x=.04, y=.97, ha='left', fontsize=18, weight='bold')
fig.text(.04, .875, 'Same 28 input/output/stop signatures; eight warmups excluded. Original target P64/D4; split drafter P16/D1/C4.', fontsize=10)
for ax, model in zip(axes, data['models']):
    values = model['all20_group_median_ms']
    bars = ax.bar([0, 1], [values['a'], values['b']], width=.55, color=[colors['garnet_a'], colors['garnet_b']])
    maximum = max(values.values())
    for x, letter in enumerate(('a', 'b')):
        for n, offset in [('1', -.09), ('2', .09)]:
            ax.scatter(x + offset, model['per_run'][letter + n]['median_group_ms'], color='#263247', s=32, zorder=3)
        ax.text(x, values[letter] + maximum * .04, f'{values[letter]:.1f} ms', ha='center', fontsize=12, weight='bold')
    ax.set_title(f'Qwen3.5 {model["model"].upper()} · batch {model["batch"]}\n'
                 f'Control / candidate: {model["all20_control_over_candidate_ratio"]:.3f}×', loc='left', fontsize=12, weight='bold')
    ax.text(.02, .94, 'Prospective 1.05× speed screen: ' + ('passed' if model['experimental_speed_screen_passed'] else 'not met'),
            transform=ax.transAxes, fontsize=10, color='#a52b2b')
    ax.set_xticks([0, 1], ['Current control', 'Split MTP candidate'])
    ax.set_ylabel('API-return group median · ms (lower is better)')
    ax.set_ylim(0, maximum * 1.26); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True); ax.set_facecolor(face)
    for spine in ax.spines.values(): spine.set_visible(False)
fig.text(.04, .13, 'Bars: median of per-group repeat medians. Dots: each run’s median, not confidence intervals. Fresh sequential A1 B1 B2 A2.', fontsize=9)
fig.text(.04, .085, 'All 20 includes EOS groups omitted from equal 32 comparator subsets. No new process-VRAM or lifetime-peak measurement here.', fontsize=9)
fig.text(.04, .04, 'The 1.05× experimental step screen is separate from the full 2× objective and independent quality; no automatic default promotion.', fontsize=9, color='#a52b2b')
fig.subplots_adjust(left=.08, right=.97, top=.75, bottom=.24, wspace=.27)
finish(fig, files[1], [{'model': m['model'], 'batch': m['batch'], 'values': m['all20_group_median_ms'],
                       'per_run': m['per_run']} for m in data['models']])
