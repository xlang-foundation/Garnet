#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate frozen MLX functional sequences; run before the native process."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    model, tokenizer = load(fixture['model'], tokenizer_config={'trust_remote_code': False})
    status = dict(status='running', fixture=str(args.fixture.resolve()),
                  fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                  versions={name: importlib.metadata.version(name) for name in ('mlx', 'mlx-lm', 'transformers')},
                  scope='MLX functional reference, not a vLLM/SGLang serving or speed baseline', cases=[])
    for case in fixture['cases']:
        assert tokenizer.encode(case['rendered'], add_special_tokens=False) == case['prompt_ids']
        folder = args.output/case['id']
        folder.mkdir()
        cache = make_prompt_cache(model)
        tokens = mx.array([case['prompt_ids']])
        generated = []
        began = time.monotonic()
        for step in range(case['max_tokens']):
            logits = model(tokens, cache=cache)[:, -1, :]
            next_token = mx.argmax(logits, axis=-1)
            mx.eval(logits, next_token)
            if step == 0:
                mx.save(str(folder/'first-logits.npy'), logits.astype(mx.float32))
            value = int(next_token.item())
            generated.append(value)
            if value in fixture['stops']:
                break
            tokens = mx.array([[value]])
        result = dict(status='completed', finish_reason='stop' if generated[-1] in fixture['stops'] else 'length',
                      text=tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                      generated_ids=generated, prompt_ids=case['prompt_ids'],
                      elapsed_seconds=time.monotonic()-began)
        (folder/'request-0.result.json').write_text(json.dumps(result, ensure_ascii=True, indent=2)+'\n')
        status['cases'].append(dict(id=case['id'], prompt_tokens=len(case['prompt_ids']),
                                   output_tokens=len(generated), finish_reason=result['finish_reason']))
        (args.output/'status.json').write_text(json.dumps(status, indent=2)+'\n')
        print(json.dumps(status['cases'][-1]), flush=True)
        del cache, logits, next_token, tokens
        gc.collect()
        mx.clear_cache()
    status['status'] = 'completed'
    (args.output/'status.json').write_text(json.dumps(status, indent=2)+'\n')


if __name__ == '__main__':
    main()
