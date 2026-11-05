#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Freeze matched Mistral serving fixtures before any comparator is started."""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from prepare_local_coder_cases import TASKS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model-manifest', type=Path, required=True)
    parser.add_argument('--native-fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True, trust_remote_code=False)
    native = json.loads(args.native_fixture.read_text())
    if Path(native['model']).resolve() != args.checkpoint.resolve():
        raise ValueError('Native/comparator checkpoint paths differ')
    served = 'local-mistral-7b'
    args.output.mkdir(parents=True, exist_ok=False)

    def render(content):
        messages = [dict(role='user', content=content)]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        return messages, text, ids

    def freeze(identity, prompts, budget, target=None):
        folder = args.output/identity
        folder.mkdir()
        bodies, records = [], []
        for index, content in enumerate(prompts):
            messages, text, ids = render(content)
            if len(ids)+budget > 4096:
                raise ValueError('Request exceeds matched context capacity')
            bodies.append(dict(model=served, messages=messages, temperature=0, max_tokens=budget,
                               stream=True, stream_options=dict(include_usage=True, continuous_usage_stats=True)))
            records.append(dict(row=index, prompt_tokens=len(ids), token_ids=ids, rendered_prompt=text,
                                rendered_prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                requested_prompt_tokens=target))
        write(folder/'cases.json', bodies)
        workload = dict(concurrency=len(bodies), max_output_tokens=budget, requests=records,
                        cases_sha256=digest(folder/'cases.json'),
                        tokenizer_sha256=digest(args.checkpoint/'tokenizer.json'),
                        tokenizer_config_sha256=digest(args.checkpoint/'tokenizer_config.json'),
                        scope='Controlled text serving load; quality is evaluated separately')
        write(folder/'workload.json', workload)
        return dict(id=identity, cases=str((folder/'cases.json').resolve()),
                    workload=str((folder/'workload.json').resolve()),
                    cases_sha256=workload['cases_sha256'])

    plans = []
    for target in (256, 2048):
        for concurrency in (1, 2, 4):
            prompts = []
            for row in range(concurrency):
                lead = 'Benchmark case 1 job '+str(row)+'.\n'
                unit = '\n# Reference module\ndef identity(value):\n    return value\n'
                def content(count):
                    return lead+unit*count+'\nTask: '+TASKS[row % len(TASKS)]
                low, high = 0, 1
                while len(render(content(high))[2]) < target:
                    low, high = high, high*2
                while low+1 < high:
                    middle = (low+high)//2
                    if len(render(content(middle))[2]) < target:
                        low = middle
                    else:
                        high = middle
                prompts.append(content(high))
            plans.append(freeze('p'+str(target)+'-c'+str(concurrency)+'-o128', prompts, 128, target))
    manifest = dict(schema=1, scope='Mistral7B text, capacity4096, nominal256/2048 inputs, concurrency1/2/4; coding quality is a separate failing gate until proven otherwise',
                    model_asset_manifest_sha256=digest(args.model_manifest),
                    native_fixture_sha256=digest(args.native_fixture),
                    warmup_waves_per_cell=2, measured_waves_per_cell=5,
                    required_fresh_process_cohorts=3, plans=plans,
                    policy='One engine at a time; retain failures/raw data/paging; no builds, installations or downloads while measuring. No automatic speed or quality admission.')
    write(args.output/'benchmark-manifest.json', manifest)
    # The exact existing coding fixture is replayed independently, without
    # adding a system instruction or changing its output/format contract.
    quality = []
    for case in native['cases']:
        if case['kind'] != 'coding':
            continue
        content = case['messages'][0]['content']
        _, text, ids = render(content)
        if text != case['rendered'] or ids != case['prompt_ids']:
            raise ValueError('Existing native coding fixture changed')
        quality.append(freeze(case['id'], [content], case['max_tokens']))
    write(args.output/'quality-manifest.json', dict(plans=quality, suite_sha256=native['suite_sha256'],
                                                   scope='Eight unchanged code-only coding contracts, greedy1024-token budgets; retain all failures'))
    print(json.dumps(dict(performance_cells=len(plans), coding_tasks=len(quality),
                          manifest_sha256=digest(args.output/'benchmark-manifest.json'))))


if __name__ == '__main__':
    main()
