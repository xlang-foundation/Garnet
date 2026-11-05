#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture logits at a fixed reference continuation for precision diagnosis."""
import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    case = next(row for row in fixture['cases'] if row['id'] == args.case)
    source = args.reference/args.case/'request-0.result.json'
    reference = json.loads(source.read_text())
    if not 1 <= args.steps <= len(reference['generated_ids']):
        raise ValueError('Teacher continuation exceeds reference')
    args.output.mkdir(exist_ok=False)
    model, tokenizer = load(fixture['model'], tokenizer_config={'trust_remote_code': False})
    assert tokenizer.encode(case['rendered'], add_special_tokens=False) == case['prompt_ids']
    cache = make_prompt_cache(model)
    tokens = mx.array([case['prompt_ids']])
    rows = []
    for step in range(args.steps):
        logits = model(tokens, cache=cache)[:, -1, :]
        mx.eval(logits)
        rows.append(logits.astype(mx.float32))
        tokens = mx.array([[reference['generated_ids'][step]]])
    array = mx.concatenate(rows, axis=0)
    mx.eval(array)
    mx.save(str(args.output/'logits.npy'), array)
    status = dict(model=fixture['model'], case=case, steps=args.steps,
                  forced_ids=reference['generated_ids'][:args.steps],
                  fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  scope='Teacher-forced logits diagnosis; no quality/performance admission')
    (args.output/'reference.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(dict(case=args.case, steps=args.steps)), flush=True)


if __name__ == '__main__':
    main()
