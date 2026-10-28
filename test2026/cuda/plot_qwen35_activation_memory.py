"""Plot isolated activation-storage memory pairs and saved LM Studio counters."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    sources = {}

    def read(name):
        blob = (root / name).read_bytes()
        sources[name] = hashlib.sha256(blob).hexdigest()
        return json.loads(blob)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))
    records = []
    gib = 1024 ** 3
    for ax, (size, batch) in zip(axes, (('2b', 1), ('4b', 8))):
        pair = read(f'activation-memory-{size}-b{batch}-pair-summary.json')
        lm = read(f'memory-lmstudio-{size}-b{batch}-512-partitioned-summary.json')
        phase = lm['phases']['steady_after']
        assert phase['samples'] >= 3
        lm_counters = {key: phase['bytes'][key]['median']
                       for key in ('DedicatedUsage', 'SharedUsage', 'TotalCommitted')}
        profiles = pair['profiles']
        counters = [profiles[key]['warm_process_counter_bytes']
                    for key in ('control', 'candidate')] + [lm_counters]
        dedicated = [value['DedicatedUsage'] / gib for value in counters]
        shared = [value['SharedUsage'] / gib for value in counters]
        totals = [value['TotalCommitted'] / gib for value in counters]
        positions = list(range(3))
        # Separate medians need not add, even though the counters add per sample.
        ax.bar([x - .23 for x in positions], dedicated, color='#3479b7', width=.21, label='Dedicated')
        ax.bar(positions, totals, color='#14967f', width=.21, label='Total committed')
        ax.bar([x + .23 for x in positions], shared, color='#efa846', width=.21, label='Shared')
        for x, total in zip(positions, totals):
            ax.text(x, total + .12, f'{total:.2f}', ha='center', fontsize=11)
        for x, value in zip(positions, shared):
            ax.text(x + .23, value + .08, f'{value:.2f}', ha='center', fontsize=8, color='#966014')
        ax.set_xticks(list(positions), ['Garnet\ncontrol', 'Garnet\nshared arena', 'LM Studio\nsaved cohort'])
        ax.set_title(f'Qwen3.5 {size.upper()} · batch {batch}', fontsize=13)
        ax.set_ylabel('Warmed committed process counters (GiB)')
        ax.set_ylim(0, max(totals) * 1.19)
        ax.grid(axis='y', alpha=.22)
        ax.set_axisbelow(True)
        owned = [profiles[key]['retained_owned_model_bytes'] / gib
                 for key in ('control', 'candidate')]
        delta = pair['control_minus_candidate_counter_bytes']['TotalCommitted'] / gib
        ax.text(.5, -.24, f'Garnet retained allocations: {owned[0]:.2f} → {owned[1]:.2f} GiB\n'
                f'Process counter reduction: {delta:+.2f} GiB',
                transform=ax.transAxes, ha='center', va='top', fontsize=10)
        records.append({'model': size, 'batch': batch, 'counter_bytes': counters,
                        'garnet_retained_owned_gib': owned,
                        'garnet_dll_sha256': pair['garnet_dll_sha256']})
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(.5, .91), ncol=3, frameon=False, fontsize=10)
    fig.suptitle('Shared decoder activation storage — warmed process memory', fontsize=15)
    fig.text(.5, .025, 'Same-DLL Garnet pairs; LM Studio is historical. Windows sampled counters are not peak or physical residency.\n'
             'Retained allocations use a separate accounting method; APIs and intermediate arithmetic differ. External host activity was recorded.',
             ha='center', fontsize=9, color='#444444')
    fig.subplots_adjust(top=.84, bottom=.33, left=.075, right=.98, wspace=.26)
    for extension in ('png', 'svg'):
        fig.savefig(root / f'progress-activation-memory.{extension}', dpi=170)
    plt.close(fig)
    external = root / 'activation-external-gpu1-observation.json'
    if external.exists():
        sources[external.name] = hashlib.sha256(external.read_bytes()).hexdigest()
    result = {'sources_sha256': sources, 'records': records,
              'scope': 'Two separate same-build control/candidate processes per model. Historical LM counters. '
              'Sampled warmed committed process counters, not true peak, physical residency or causal significance. '
              'Retained allocations are a distinct measure. No inference-quality or latency acceptance.'}
    (root / 'progress-activation-memory-sources.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(root / 'progress-activation-memory.png')


if __name__ == '__main__':
    main()
