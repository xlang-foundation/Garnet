#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Freeze unchanged coding contracts and bounded context probes before inference."""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite', type=Path, default=Path(__file__).with_name('local_coder_quality_suite.json'))
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)

    def render(identity, kind, content, budget):
        messages = [dict(role='user', content=content)]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        return dict(id=identity, kind=kind, messages=messages, rendered=rendered, prompt_ids=ids,
                    prompt_tokens=len(ids), max_tokens=budget,
                    rendered_sha256=hashlib.sha256(rendered.encode()).hexdigest())

    cases = [render(task['id'], 'coding', task['prompt'], suite['sampling']['max_tokens'])
             for task in suite['tasks']]
    # Controlled context fixtures test inference state, not general coding quality.
    unit = '\n# Reference module\ndef identity(value):\n    return value\n'
    task = 'Write Python merge_intervals(intervals): merge overlapping closed intervals, return sorted tuples, preserve the input, and handle empty input. Return only code.'
    for target in (256, 2048):
        def materialize(count):
            return render('context-'+str(target), 'context',
                          'Context probe.\n'+unit*count+'\nTask: '+task, 64)
        low, high = 0, 1
        while materialize(high)['prompt_tokens'] < target:
            low, high = high, high*2
        while low+1 < high:
            middle = (low+high)//2
            if materialize(middle)['prompt_tokens'] < target:
                low = middle
            else:
                high = middle
        row = materialize(high)
        row['requested_prompt_tokens'] = target
        cases.append(row)
    assert len(cases) == 10 and len({row['id'] for row in cases}) == 10
    assert max(row['prompt_tokens']+row['max_tokens'] for row in cases) <= 4096
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'suite.json').write_bytes(args.suite.read_bytes())
    fixture = dict(schema=1, model=str(args.model.resolve()), context_capacity=4096,
                   suite_sha256=digest(args.suite), temperature=0,
                   stops=sorted(set([tokenizer.eos_token_id])), cases=cases,
                   metadata_sha256={name: digest(args.model/name) for name in
                                    ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'download-verification.json')},
                   scope='Eight unchanged coding contracts plus two controlled context probes; no performance claim')
    (args.output/'fixture.json').write_text(json.dumps(fixture, ensure_ascii=True, indent=2)+'\n')
    print(json.dumps(dict(cases=[(r['id'], r['prompt_tokens'], r['max_tokens']) for r in cases],
                          fixture_sha256=digest(args.output/'fixture.json'))))


if __name__ == '__main__':
    main()
