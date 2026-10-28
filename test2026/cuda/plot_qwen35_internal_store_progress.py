"""Static real-model internal-store charts with explicit measurement scopes."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('--memory', action='store_true')
p.add_argument('--vision-graph', action='store_true')
p.add_argument('--compact-kv', action='store_true')
a = p.parse_args()
assert not (a.vision_graph and a.compact_kv)
study = 'compact-kv' if a.compact_kv else 'vision-graph' if a.vision_graph else 'internal-stores'
method = 'Compact KV outputs' if a.compact_kv else 'Vision CUDA graph replay' if a.vision_graph else 'Internal packed stores'
sources = {}


def read(name):
    blob = (a.root / name).read_bytes()
    sources[name] = hashlib.sha256(blob).hexdigest()
    return json.loads(blob)


fig, axes = plt.subplots(1, 2, figsize=(15.8, 7.2), facecolor='#f7f9fc')
records = []
provisional = False
for ax, (size, batch) in zip(axes, (('2b', 1), ('4b', 8))):
    ax.set_facecolor('#f7f9fc')
    if a.memory:
        if a.compact_kv:
            parent=read('compact-kv-observed-memory-v2-exit.json')
            observation=read('compact-kv-observed-memory-v2-process-audit.json')
            assert parent['exit_code']==0 and observation['all_four_cases_covered']
            assert observation['known_overlap_count']==0 and not observation['observer_metadata_errors']
        pair = read(f'{study}-memory-{size}-b{batch}-pair-summary.json')
        lm = read(f'memory-lmstudio-{size}-b{batch}-512-partitioned-summary.json')
        phase = lm['phases']['steady_after']
        assert phase['samples'] >= 3
        values = [pair['profiles'][mode]['warm_process_counter_bytes'] for mode in ('control', 'candidate')]
        values += [{key: phase['bytes'][key]['median'] for key in ('DedicatedUsage', 'SharedUsage', 'TotalCommitted')}]
        gib = 1024 ** 3
        for shift, key, label, color in ((-.24, 'DedicatedUsage', 'Dedicated', '#3479b7'),
                                         (0, 'TotalCommitted', 'Total committed', '#159b83'),
                                         (.24, 'SharedUsage', 'Shared', '#efa846')):
            bars = ax.bar([i + shift for i in range(3)], [row[key] / gib for row in values],
                          width=.22, color=color, label=label)
            if key == 'TotalCommitted':
                for bar, row in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, row[key] / gib + .15,
                            f'{row[key] / gib:.2f}', ha='center', fontsize=11, weight='bold')
        ax.set_xticks(range(3), ['Garnet\ncontrol', 'Garnet\ncompact KV' if a.compact_kv else 'Garnet\nvision replay' if a.vision_graph else 'Garnet\ninternal stores', 'LM Studio\nsaved'])
        ax.set_ylabel('Warmed committed process counters · GiB')
        ax.set_ylim(0, max(row['TotalCommitted'] / gib for row in values) * 1.20)
        ax.set_title(f'Qwen3.5 {size.upper()} · batch {batch}', loc='left', weight='bold')
        delta = pair['control_minus_candidate_counter_bytes']['TotalCommitted'] / gib
        delta_label = f'{delta * 1024:+.1f} MiB' if abs(delta) < .1 else f'{delta:+.2f} GiB'
        ax.text(.5, -.22, f'Control − candidate total: {delta_label}',
                ha='center', transform=ax.transAxes, fontsize=11)
        records.append({'model': size, 'batch': batch, 'counter_bytes': values,
                        'garnet_dll_sha256': pair['garnet_dll_sha256']})
    else:
        data = read(f'packed-study-{size}-b{batch}-{study}-all-baselines.json')
        audit_prefix = 'vision-graph-observed-v1' if a.vision_graph else 'internal-store-observed-v1'
        audit = read(data['process_observation_audit_file'] if a.compact_kv else f'{audit_prefix}-{size}-b{batch}-process-audit.json')
        provisional |= (not audit['all_four_roles_covered'] or audit['known_overlap_count'] > 0
                        or bool(audit['observer_metadata_errors']))
        values = data['median_group_ms']
        mtps = sorted(key for key in values if key.startswith('vllm-mtp3'))
        names = ('garnet_a', 'garnet_b', 'lmstudio', 'vllm_serial', *mtps)
        labels = ('Garnet control · repeated', 'Garnet compact KV · repeated' if a.compact_kv else 'Garnet vision replay · repeated' if a.vision_graph else 'Garnet internal stores · repeated',
                  'LM Studio · saved', 'vLLM serial · saved',
                  *('vLLM MTP3 · BF16 conv setting' if 'convbfloat16' in key else
                    'vLLM MTP3 · FP32 conv setting' for key in mtps))
        colors = ('#90b3a9', '#159b83', '#8293aa', '#4688c4',
                  *('#8b6ab0' if 'convbfloat16' in key else '#dd9c40' for key in mtps))
        walls = [values[key] for key in names]
        bars = ax.barh(range(len(names)), walls, color=colors, height=.60)
        ax.invert_yaxis()
        ax.set_yticks(range(len(names)), labels, fontsize=9)
        for bar, value in zip(bars, walls):
            ax.text(value + max(walls) * .018, bar.get_y() + bar.get_height() / 2,
                    f'{value:.1f}', va='center', fontsize=10, weight='bold')
        target = data['half_fastest_baseline_ms']
        ax.axvline(target, color='#a52b2b', linestyle='--', linewidth=1.5)
        ax.text(target, 1.02, f'2× target\n{target:.1f} ms', transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', color='#a52b2b', fontsize=9)
        ax.set_title(f'Qwen3.5 {size.upper()} · batch {batch}\n'
                     f'{data["common_equal32_groups"]} matched 32-token groups\n'
                     f'Fastest saved baseline / candidate: {data["garnet_b_speed_ratio_vs_fastest"]:.2f}×',
                     loc='left', weight='bold', fontsize=11, pad=52)
        ax.set_xlabel('API-return group median · ms (lower is better)' if a.compact_kv else 'Full-request group median · ms (lower is better)')
        ax.set_xlim(0, max(walls) * 1.18)
        records.append({'model': size, 'batch': batch, 'values_ms': values,
                        'common_equal32_groups': data['common_equal32_groups'],
                        'half_fastest_baseline_ms': target, 'process_audit': audit})
    ax.grid(axis='y' if a.memory else 'x', alpha=.18)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

if a.memory:
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(.5, .90), ncol=3, frameon=False)
    title = method + ' — warmed Windows process memory'
    fig.text(.04, .12, 'Separate sequential control/candidate processes; six matched input/output groups and exact timing profiles.', fontsize=10)
    fig.text(.04, .075, 'Dedicated/shared/total medians are independent. LM Studio counters are historical; vLLM process totals are unavailable.', fontsize=9)
    fig.text(.04, .035, 'Sampled committed counters are not physical residency or lifetime peaks. These diagnostics do not measure primary latency.', fontsize=9, color='#444444')
    fig.subplots_adjust(left=.07, right=.98, bottom=.31, top=.81, wspace=.25)
else:
    title = method + (' — request latency' if a.compact_kv else ' — full-request latency') + (' · PROVISIONAL' if provisional else '')
    subtitle = ('Same image hashes / 32 tokens per row; sequential ABBA. Internal stores fixed off for 2B and on for 4B.' if a.vision_graph else
                'Same real-image hashes and 32 sampled tokens per row within each panel; sequential fresh-process Garnet ABBA.')
    fig.text(.04, .91, subtitle, fontsize=10)
    fig.text(.04, .13, 'Garnet: two repeats per group, then median across groups. All 20 warm groups are retained in the separate within-Garnet study.', fontsize=9)
    fig.text(.04, .085, 'Competitors: saved single cohorts. Convolution labels describe settings; APIs, arithmetic and speculation differ. Sampled observation cannot prove isolation.', fontsize=9)
    fig.text(.04, .04, ('External overlap/observer errors recorded; timings remain provisional. ' if provisional else '')
             + 'Independent quality and the full 2× goal remain open.', fontsize=9, color='#a52b2b')
    fig.subplots_adjust(left=.19, right=.98, bottom=.23, top=.70, wspace=.78)
fig.suptitle(title, x=.04, ha='left', fontsize=16, weight='bold')
prefix = 'progress-compact-kv' if a.compact_kv else 'progress-vision-graph' if a.vision_graph else 'progress-internal-store'
stem = prefix + ('-memory' if a.memory else '-baselines')
for extension in ('png', 'svg'):
    fig.savefig(a.root / f'{stem}.{extension}', dpi=160, facecolor=fig.get_facecolor())
plt.close(fig)
(a.root / f'{stem}-sources.json').write_text(json.dumps({
    'sources_sha256': sources, 'records': records, 'provisional_due_to_observation': provisional,
    'goal_2x_achieved': False, 'independent_quality_accepted': False,
}, indent=2), encoding='utf-8')
print(a.root / f'{stem}.png')
