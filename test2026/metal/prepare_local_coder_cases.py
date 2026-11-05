"""Create deterministic, token-counted coding request waves before measuring.

Requires tokenizers and jinja2 in the preparation environment, not the engine.
Server-reported prompt counts must subsequently match the frozen manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer


TASKS = [
    'Write Python merge_intervals(intervals): merge overlapping closed intervals, '
    'return sorted tuples, preserve the input, and handle empty input. Return only code.',
    'Write Python stable_unique(items, key): retain the first item for each hashable '
    'key(item), preserving order. Handle generators and empty input. Return only code.',
    'Write Python topological_sort(nodes, edges): return a deterministic ordering; '
    'raise ValueError on cycles. Edges are (source, destination). Return only code.',
    'Write Python retry_delay(attempt, base, cap): bounded exponential backoff '
    'without jitter or overflow; reject negative attempts and invalid bounds. Return only code.',
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='local-coder-27b')
    p.add_argument('--prompt-tokens', type=int, default=2048)
    p.add_argument('--output-tokens', type=int, default=128)
    p.add_argument('--concurrency', type=int, choices=(1, 2, 4, 8), default=1)
    p.add_argument('--seed', type=int, default=1)
    a = p.parse_args()
    if a.prompt_tokens < 256 or a.output_tokens < 2 or a.seed < 1:
        raise ValueError('Invalid workload budget')
    a.output.mkdir(parents=True, exist_ok=False)
    tokenizer_path = a.checkpoint/'tokenizer.json'
    template_path = a.checkpoint/'chat_template.jinja'
    tok = Tokenizer.from_file(str(tokenizer_path))
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    def fail(message):
        raise ValueError(message)
    env.globals['raise_exception'] = fail
    template = env.from_string(template_path.read_text())
    cases, records = [], []
    for i in range(a.concurrency):
        task = TASKS[i % len(TASKS)]
        lead = 'Benchmark case '+str(a.seed)+' job '+str(i)+'.\n'
        def materialize(filler):
            messages = [dict(role='user', content=lead+filler+'\nTask: '+task)]
            rendered = template.render(messages=messages, add_generation_prompt=True,
                                       enable_thinking=False, tools=None)
            ids = tok.encode(rendered, add_special_tokens=False).ids
            return messages, rendered, ids
        # Repeated source context is a controlled load fixture, not a quality corpus.
        unit = '\n# Reference module\ndef identity(value):\n    return value\n'
        low, high = 0, 1
        while len(materialize(unit*high)[2]) < a.prompt_tokens:
            low, high = high, high*2
        while low+1 < high:
            middle = (low+high)//2
            if len(materialize(unit*middle)[2]) < a.prompt_tokens:
                low = middle
            else:
                high = middle
        # Token-count every final request, rather than treating characters as tokens.
        messages, rendered, ids = materialize(unit*high)
        body = dict(model=a.model, messages=messages, temperature=0,
                    max_tokens=a.output_tokens, stream=True,
                    stream_options=dict(include_usage=True, continuous_usage_stats=True),
                    chat_template_kwargs=dict(enable_thinking=False))
        cases.append(body)
        records.append(dict(row=i, prompt_tokens=len(ids), requested_prompt_tokens=a.prompt_tokens,
                            rendered_prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                            token_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                            token_ids=ids))
    wire = json.dumps(cases, indent=2, ensure_ascii=False)+'\n'
    (a.output/'cases.json').write_text(wire)
    manifest = dict(seed=a.seed, model=a.model, concurrency=a.concurrency,
                    max_output_tokens=a.output_tokens, requests=records,
                    tokenizer_sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                    template_sha256=hashlib.sha256(template_path.read_bytes()).hexdigest(),
                    cases_sha256=hashlib.sha256(wire.encode()).hexdigest(),
                    scope='Controlled code-context performance fixture; not a coding-quality evaluation')
    (a.output/'workload.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(cases=str(a.output/'cases.json'),
                         prompt_tokens=[r['prompt_tokens'] for r in records])))


if __name__ == '__main__':
    main()
