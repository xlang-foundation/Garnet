#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit serial real-weight operator evidence; numerical differences are ungated.

Collection integrity, exact weight representation and Metal sample oracles do
not establish Core ML numerical/coding quality or observed ANE execution.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(root):
    root = root.resolve()
    status = json.loads((root/'status.json').read_text())
    require(status['completed'] is True, 'incomplete collection')
    require(sha(Path(status['probe'])) == status['probe_sha256'], 'changed native probe')
    exports = {Path(r['argv'][3]) for r in status['runs']}
    models = {p: json.loads((p/'manifest.json').read_text()) for p in exports}
    require(len(models) in (1, 2), 'unexpected representation count')
    primary = Path(next(r for r in status['runs'] if r['engine'] == 'metal')['argv'][3])
    # The first single-representation collector predates the exports list.
    # Its primary manifest digest is still required; multi-export collections
    # must bind every representation explicitly.
    entries = status.get('exports', [dict(root=str(primary),
        manifest_sha256=status['model_manifest_sha256'], representation=models[primary]['representation'])])
    declared = {Path(e['root']): e for e in entries}
    require(len(entries) == len(declared) and declared.keys() == models.keys(),
            'declared exports differ from executed exports')
    require(status['model_manifest_sha256'] == sha(primary/'manifest.json'), 'changed primary manifest')
    base = models[primary]
    require(base['rows'] in (1, 128, 512) and base['hidden'] == 4096 and base['intermediate'] == 14336
            and base['checkpoint_sha256'] == 'd7182d886f93a9efa73d6d556ee33ffe839a4e09e15965eeef46f295ab5b503d', 'wrong model dimensions/pin')
    files = {str(root/'status.json'): sha(root/'status.json'), str(Path(status['probe'])): status['probe_sha256']}
    for path, model in models.items():
        require(declared[path]['manifest_sha256'] == sha(path/'manifest.json')
                and declared[path]['representation'] == model['representation'], 'changed export manifest')
        require(model['input_sha256'] == base['input_sha256'] and model['rows'] == base['rows']
                and [w['native_rounded_sha256'] for w in model['weights']] == [w['native_rounded_sha256'] for w in base['weights']], 'different input/weights between representations')
        require(sum(w['native_fp16_words_matched'] for w in model['weights']) == 176160768, 'missing native weight parity')
        if model.get('inner_tile', 0):
            require(model['remapped_tile_weight_words_matched'] == 176160768, 'missing remapped tile weight parity')
        for rel, digest in model['files'].items():
            require(sha(path/rel) == digest, 'changed model/input/profile asset: '+rel)
        files[str(path/'manifest.json')] = sha(path/'manifest.json')
    expected = {(cohort, 'metal', primary) for cohort in (0, 1)}
    expected |= {(cohort, engine, model) for cohort in (0, 1)
                 for engine in ('coreml_cpu', 'coreml_cpu_ane') for model in models}
    runs, outputs, previous, pids = {}, {}, 0, set()
    for run in status['runs']:
        model = Path(run['argv'][3])
        key = run['cohort'], run['engine'], model
        require(key in expected and key not in runs and run['returncode'] == 0, 'wrong/duplicate/failed run')
        require(run['started_ns'] >= previous and run['finished_ns'] > run['started_ns']
                and run['pid'] not in pids, 'overlap or reused process identity')
        previous = run['finished_ns']
        pids.add(run['pid'])
        out = Path(run['output']).resolve()
        require(out.parent == root and run['argv'][4] == str(out), 'run output path differs')
        report = json.loads((out/'report.json').read_text())
        require(report['rows'] == base['rows'] and report['warmups'] == 3 and report['repetitions'] == 9
                and report['output_words'] == base['rows']*4096 and report['metal_native_owners_after_close'] == 0,
                'profile/count/cleanup mismatch')
        metrics = ('host_to_host_ms', 'gpu_ms', 'command_encode_wait_ms') if key[1] == 'metal' else (
            'host_to_host_ms', 'native_predict_ms', 'framework_prediction_and_output_copy_ms')
        for metric in metrics:
            require(len(report[metric]) == 9 and all(type(v) in (int, float) and math.isfinite(v) and v > 0
                                                    for v in report[metric]), 'invalid latency samples')
        planned_ne = None
        if key[1] == 'metal':
            require(report['engine'] == 'metal' and report['profile_misses'] == report['runtime_pipeline_compiles'] == 0,
                    'Metal profile not prepared')
            for name in ('gate', 'up', 'down'):
                oracle = report['oracle'][name]
                require(oracle['samples'] == 12 and math.isfinite(oracle['max_error_fraction'])
                        and 0 <= oracle['max_error_fraction'] <= 1, 'Metal independent FP64 sample failed')
            require(report['oracle']['activation_words'] == base['rows']*14336, 'missing activation oracle')
        else:
            native = report['native_report']
            policy = 'cpu' if key[1] == 'coreml_cpu' else 'cpu_ane'
            require(report['engine'] == 'coreml' and report['policy'] == native['compute_units'] == policy
                    and native['predictions'] == 12 and native['declared_state_bytes'] == 0
                    and native['stateful'] is False, 'Core ML call/state policy mismatch')
            require(report.get('arithmetic', 'fp16') == models[model].get('arithmetic', 'fp16'),
                    'Core ML arithmetic differs from export')
            require(native['anticipated_operation_devices'].get('gpu', 0) == 0, 'unexpected preferred GPU operation')
            if 'anticipated_operations' in native:
                ops = native['anticipated_operations']
                require(dict(collections.Counter(op['preferred'] for op in ops)) == native['anticipated_operation_devices'],
                        'per-operation placement count differs')
                convolutions = [op for op in ops if op['operator'].endswith('.conv')]
                require(len(convolutions) == models[model].get('projection_convolutions', 3), 'missing projection placement')
                planned_ne = sum(op['preferred'] == 'neural_engine' for op in convolutions)
        words = np.fromfile(out/'output.f32', dtype='<f4')
        require(words.size == base['rows']*4096 and np.isfinite(words).all(), 'output word count/nonfinite value')
        runs[key] = report
        outputs[key] = words
        files[str(out/'output.f32')] = sha(out/'output.f32')
        files[str(out/'report.json')] = sha(out/'report.json')
        report['planned_neural_engine_projections'] = planned_ne
    require(runs.keys() == expected, 'missing serial cohort or engine')
    comparisons = []
    for key, report in runs.items():
        cohort, engine, model = key
        gold = outputs[cohort, 'metal', primary].astype(np.float64)
        got = outputs[key].astype(np.float64)
        difference = got-gold
        rms = float(np.sqrt(np.mean(gold*gold)))
        require(rms > 0, 'zero reference signal')
        comparisons.append(dict(cohort=cohort, engine=engine, representation=models[model]['representation'],
            export_root=str(model), inner_tile=models[model].get('inner_tile', 0),
            palette_row_group=models[model].get('palette_row_group', 1),
            arithmetic=models[model].get('arithmetic', 'fp16'),
            input_mode=models[model].get('input_mode', 'embedding'),
            host_to_host_median_ms=statistics.median(report['host_to_host_ms']),
            metal_relative_host_speedup=statistics.median(runs[cohort, 'metal', primary]['host_to_host_ms'])/statistics.median(report['host_to_host_ms']),
            planned_neural_engine_projections=report['planned_neural_engine_projections'],
            reference_rms=rms, normalized_rmse=float(np.sqrt(np.mean(difference*difference))/rms),
            maximum_absolute_error=float(np.max(np.abs(difference))),
            exact_output_words=int(np.count_nonzero(outputs[key].view(np.uint32) == outputs[cohort, 'metal', primary].view(np.uint32)))))
    for engine, model in {(e, m) for _, e, m in expected}:
        require(np.array_equal(outputs[0, engine, model].view(np.uint32), outputs[1, engine, model].view(np.uint32)),
                'output changed between fresh cohorts')
    files[str(Path(__file__).resolve())] = sha(Path(__file__))
    return dict(collection_integrity_passed=True, native_fp16_weight_words_matched=176160768,
                processes=len(runs), comparisons=comparisons, files=files,
                coreml_numerical_quality_established=False, observed_ane_execution_established=False,
                full_model_or_serving_speed_established=False, scope=__doc__)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = audit(a.campaign)
    with a.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('collection_integrity_passed', 'processes', 'native_fp16_weight_words_matched')}))
