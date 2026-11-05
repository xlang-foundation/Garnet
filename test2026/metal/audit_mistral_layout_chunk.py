#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit serial real-weight late-chunk latency and exact complete-state hashes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def require(value, message):
    if not value:
        raise ValueError(message)


def audit(root):
    root = root.resolve();files = {}
    def sha(path):
        with path.open('rb') as stream:
            files[str(path)] = hashlib.file_digest(stream,'sha256').hexdigest()
        return files[str(path)]
    def read(path):
        sha(path)
        return json.loads(path.read_text())
    status = read(root/'collector-status.json')
    require(status['completed'] is True and len(status['runs']) == 8, 'incomplete native paired collection')
    identities = [(0,1,'control'),(0,1,'candidate'),(0,4,'control'),(0,4,'candidate'),
                  (1,4,'candidate'),(1,4,'control'),(1,1,'candidate'),(1,1,'control')]
    libraries = {'control': '850c5d4cb7c7b2b67e20d0085d79beecfa98297cb0e77ac1c431cb57ebd8cec6',
                 'candidate': 'f2d3069b119eb135232edb8a8c7ef879feb763e97efcb3db4ad7a02ec5e4f2a5'}
    fixture = root.parent/'mistral-coding-context-fixture1/fixture.json'
    require(sha(fixture) == '6847ca35e1b1dac2bf04bb8e4f2cf358a7f7a0acb4da3fc5b3a1fff4fdbc35ad', 'changed frozen prompts')
    previous, pids, results = 0, set(), {}
    for run, identity in zip(status['runs'],identities):
        cohort, owners, variant = identity
        require((run['cohort'],run['owners'],run['variant']) == identity and run['returncode'] == 0
                and type(run['pid']) is int and run['pid'] > 0 and run['pid'] not in pids
                and previous <= run['started_ns'] < run['finished_ns'], 'failed/overlapping native process')
        previous = run['finished_ns'];pids.add(run['pid'])
        args = run['argv'];path = root/f'cohort{cohort}-owners{owners}-{variant}.json'
        require(len(args) == 8 and args[1] == '-B' and args[2] == str(root/'tools/run_mistral_stage_probe.py')
                and args[5:] == [str(fixture),str(path),str(owners)]
                and sha(Path(args[3])/'libgarnet.dylib') == run['library_sha256'] == libraries[variant],
                'changed native command/library/fixture')
        item = read(path);sha(path.with_suffix('.log'))
        require(item['completed'] is True and item['library'] == args[3]+'/libgarnet.dylib'
                and (item['case_id'],item['batch'],item['capacity'],item['chunk'],item['owners'],item['offset'],item['warmups'])
                    == ('context-2048',4,4096,128,owners,1920,3), 'changed native chunk contract')
        tags = {s.strip() for s in item['backend'].split(';')}
        require(('chunk_layout_vector4' in tags) == (variant == 'candidate')
                and {'chunk_kv_scatter_contiguous','mistral_decode_reuse_four_rows','affine_coefficient_dtype=fp16'} <= tags,
                'wrong native layout/precision policy')
        require(len(item['seeds']) == 15 and [s['offset'] for s in item['seeds']] == list(range(0,1920,128))
                and all(math.isfinite(s['wall_seconds']) and s['wall_seconds'] > 0 for s in item['seeds']), 'invalid prompt seed coverage')
        require(len(item['measured']) == 9, 'missing native timing samples')
        for index, sample in enumerate(item['measured']):
            require(sample['repeat'] == index and sample['layout_vector_dispatches'] == (256 if variant == 'candidate' else 0)
                    and math.isfinite(sample['wall_seconds']) and sample['wall_seconds'] > 0
                    and math.isfinite(sample['gpu_ms']) and 0 < sample['gpu_ms'] <= sample['wall_seconds']*1000+1,
                    'invalid duration or vector dispatch route')
        require(len(item['state_sha256']) == 32 and all(len(layer) == 2 and all(re.fullmatch('[0-9a-f]{64}',h) for h in layer)
                for layer in item['state_sha256']) and re.fullmatch('[0-9a-f]{64}',item['hidden_sha256']), 'invalid complete-state witness')
        before, after, closed = item['prepared'],item['after'],item['cleanup']
        require(before['metal_allocated_bytes'] == after['metal_allocated_bytes']
                and 0 <= after['metal_allocated_bytes'] <= after['metal_recommended_working_set_bytes']-256*1024*1024
                and 0 <= closed['metal_allocated_bytes'] <= 4*1024*1024, 'native allocation bound changed')
        for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):
            require(before[key] == after[key] and closed[key] == 0, 'native owner counts changed/retained')
        results[identity] = item
    comparisons = []
    for owners in (1,4):
        control_state = results[0,owners,'control']
        for cohort in (0,1):
            for variant in ('control','candidate'):
                item = results[cohort,owners,variant]
                require((item['state_sha256'],item['hidden_sha256']) == (control_state['state_sha256'],control_state['hidden_sha256']),
                        'complete cache or hidden differs across variants/cohorts')
            a,b = results[cohort,owners,'control'],results[cohort,owners,'candidate']
            wall_a,wall_b = [statistics.median(s['wall_seconds'] for s in i['measured']) for i in (a,b)]
            gpu_a,gpu_b = [statistics.median(s['gpu_ms'] for s in i['measured']) for i in (a,b)]
            comparisons.append(dict(cohort=cohort,owners=owners,control_wall_seconds=wall_a,candidate_wall_seconds=wall_b,
                                    wall_speedup=wall_a/wall_b,control_gpu_ms=gpu_a,candidate_gpu_ms=gpu_b,gpu_speedup=gpu_a/gpu_b))
    for name, digest in read(root/'tools/manifest.json').items():
        require(sha(root/'tools'/name) == digest, 'changed native driver')
    sha(Path(__file__).resolve())
    return dict(passed=True,processes=8,measured_chunk_forwards=72,full_state_layers_per_run=32,
                comparisons=comparisons,files=files,scope=__doc__,serving_speed_established=False,
                note='Same late prompt chunk replayed; warm steady body duration excludes embedding, head, token emission, hashes and request scheduling. No vLLM/SGLang process was run.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collection', type=Path);parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args();report = audit(args.collection)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({k:report[k] for k in ('passed','processes','comparisons')}))
